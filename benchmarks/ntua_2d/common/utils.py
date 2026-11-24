"""Small reproducibility and artifact helpers."""

import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Install a CUDA-enabled PyTorch or use --device cpu.')
    return torch.device(name)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f'Refusing to write an empty table: {path}')
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def new_directory(base: Path, requested: Path | None, prefix: str) -> Path:
    path = requested or base / f'{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_hash(root: Path) -> str:
    """Record the experiment implementation without hashing dataset or outputs."""
    digest = hashlib.sha256()
    paths = sorted(p for directory in ('common', 'kd_0', 'kd_1')
                   for p in (root / directory).rglob('*.py') if 'runs' not in p.parts)
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
