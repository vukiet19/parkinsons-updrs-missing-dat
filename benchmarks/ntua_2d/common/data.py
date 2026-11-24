"""Load complete subjects and preserve the fixed subject-level folds."""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parents[1]
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


# One record represents one patient, including all selected MRI images.
@dataclass(frozen=True)
class Subject:
    subject_id: str
    mri: tuple[Path, ...]
    dat: Path
    updrs: float
    mmse: float
    outer_fold: int
    inner_folds: tuple[int | None, ...]


def read_subjects(root: Path = DATA_ROOT) -> list[Subject]:
    with (root / 'data' / 'subjects.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    subjects = []
    for row in rows:
        subject = Subject(
            row['subject_id'], tuple(root / p for p in row['mri_paths'].split(';')),
            root / row['dat_path'], float(row['updrs']), float(row['mmse']),
            int(row['outer_fold']),
            tuple(int(row[f'inner_{k}']) if row[f'inner_{k}'] else None for k in range(3)),
        )
        if not np.isfinite([subject.updrs, subject.mmse]).all():
            raise ValueError(f'Non-finite target: {subject.subject_id}')
        if subject.updrs < 0 or not 0 <= subject.mmse <= 30:
            raise ValueError(f'Invalid clinical score: {subject.subject_id}')
        if not 1 <= len(subject.mri) <= 3:
            raise ValueError(f'Expected 1-3 MRI images: {subject.subject_id}')
        if subject.outer_fold not in range(3):
            raise ValueError(f'Invalid outer fold: {subject.subject_id}')
        for k, value in enumerate(subject.inner_folds):
            if (k == subject.outer_fold and value is not None) or (k != subject.outer_fold and value not in range(3)):
                raise ValueError(f'Invalid inner fold: {subject.subject_id}, outer={k}')
        # Prevent an edited manifest from reading files outside the supplied dataset.
        for path in (*subject.mri, subject.dat):
            if not path.resolve().is_relative_to((root / 'dataset_224_preprocessed').resolve()):
                raise ValueError(f'Image path is outside the dataset: {path}')
            if not path.is_file():
                raise FileNotFoundError(path)
        subjects.append(subject)
    if len(subjects) != 15 or len({s.subject_id for s in subjects}) != 15:
        raise ValueError('This experiment requires exactly 15 unique subjects.')
    if [sum(s.outer_fold == k for s in subjects) for k in range(3)] != [5, 5, 5]:
        raise ValueError('Outer folds must contain 5, 5, and 5 subjects.')
    return subjects


def data_fingerprint(subjects: list[Subject], root: Path = DATA_ROOT) -> str:
    digest = hashlib.sha256((root / 'data' / 'subjects.csv').read_bytes())
    for subject in subjects:
        for path in (*subject.mri, subject.dat):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


# UPDRS statistics are fitted only on the current training partition.
@dataclass(frozen=True)
class TargetScale:
    mean: float
    std: float

    @classmethod
    def fit(cls, subjects: list[Subject]):
        values = np.asarray([s.updrs for s in subjects], dtype=np.float64)
        return cls(float(values.mean()), max(float(values.std()), 1e-6))

    def normalize(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.mean) / self.std

    def restore(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


def read_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        if image.size != (224, 224):
            raise ValueError(f'Expected 224x224 without another crop/resize: {path}')
        array = np.array(image.convert('RGB'), dtype=np.float32, copy=True) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - MEAN) / STD


class Images(Dataset):
    """Cache decoded images, never frozen CNN features."""

    def __init__(self, subjects: list[Subject], include_dat: bool):
        self.rows = []
        for subject in subjects:
            row = dict(subject_id=subject.subject_id,
                       mri=torch.stack([read_image(p) for p in subject.mri]),
                       target=torch.tensor(subject.updrs, dtype=torch.float32),
                       mmse=torch.tensor(subject.mmse / 30.0, dtype=torch.float32))
            if include_dat:
                row['dat'] = read_image(subject.dat)
            self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_subjects(rows: list[dict]) -> dict:
    """Flatten MRI images and remember their owners for equal-patient pooling."""
    batch = dict(subject_id=[row['subject_id'] for row in rows],
                 mri=torch.cat([row['mri'] for row in rows]),
                 owners=torch.repeat_interleave(torch.arange(len(rows)),
                           torch.tensor([len(row['mri']) for row in rows])),
                 target=torch.stack([row['target'] for row in rows]),
                 mmse=torch.stack([row['mmse'] for row in rows]))
    if 'dat' in rows[0]:
        batch['dat'] = torch.stack([row['dat'] for row in rows])
    return batch


def subject_mean(values: torch.Tensor, owners: torch.Tensor, count: int) -> torch.Tensor:
    """Average images inside each patient before averaging patients in a loss."""
    result = values.new_zeros((count, *values.shape[1:]))
    result.index_add_(0, owners, values)
    counts = torch.bincount(owners, minlength=count).to(values.dtype)
    return result / counts.reshape(count, *([1] * (values.ndim - 1))).clamp_min(1)
