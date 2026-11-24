"""Evaluate saved outer-fold checkpoints without fitting or selecting anything."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import TargetScale, data_fingerprint, read_subjects
from .experiment import baseline_modules
from .fit import make_loader, move_batch
from .utils import new_directory, read_json, write_csv, write_json


def metrics(rows: list[dict]) -> dict:
    errors = np.asarray([row['y_pred'] - row['y_true'] for row in rows], dtype=np.float64)
    return dict(n_subjects=len(rows), mae=float(np.abs(errors).mean()),
                rmse=float(np.sqrt(np.square(errors).mean())))


def summarize(rows: list[dict], full_cv: bool) -> list[dict]:
    result = []
    for role in sorted({row['role'] for row in rows}):
        selected = [row for row in rows if row['role'] == role]
        if len({row['subject_id'] for row in selected}) != len(selected):
            raise ValueError('OOF predictions contain duplicate subjects.')
        for fold in sorted({row['fold'] for row in selected}):
            group = [row for row in selected if row['fold'] == fold]
            result.append(dict(role=role, scope='fold', fold=fold, **metrics(group)))
        result.append(dict(role=role, scope='pooled_oof' if full_cv else 'partial_oof',
                           fold='all', **metrics(selected)))
    return result


@torch.inference_mode()
def run_test(baseline, run: Path, device, role='student', output=None):
    run = run.resolve()
    metadata = read_json(run / 'run.json')
    if metadata['baseline'] != baseline or metadata['status'] != 'complete':
        raise ValueError('The run must be a completed training run of this baseline.')
    subjects = read_subjects()
    fingerprint = data_fingerprint(subjects)
    if fingerprint != metadata['data_fingerprint']:
        raise ValueError('Dataset or subject manifest differs from the training run.')
    models, _ = baseline_modules(baseline)
    rows = []
    roles = ['teacher', 'student'] if role == 'both' else [role]
    for fold in metadata['folds']:
        held_out = [subject for subject in subjects if subject.outer_fold == fold]
        training = [subject for subject in subjects if subject.outer_fold != fold]
        expected_scale = TargetScale.fit(training)
        for current_role in roles:
            path = run / f'fold_{fold}' / f'{current_role}.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            expected = dict(baseline=baseline, role=current_role, outer_fold=fold,
                            data_fingerprint=fingerprint, format_version=1)
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Checkpoint identity mismatch: {path}')
            if saved['config'] != metadata['config']:
                raise ValueError(f'Checkpoint configuration mismatch: {path}')
            if (set(saved['train_subjects']) != {s.subject_id for s in training}
                    or set(saved['test_subjects']) != {s.subject_id for s in held_out}
                    or saved['scale'] != asdict(expected_scale)):
                raise ValueError(f'Checkpoint split or target scaler mismatch: {path}')
            model_class = models.Teacher if current_role == 'teacher' else models.Student
            # Checkpoint contains the weights; test must not download pretrained weights.
            model = model_class(weights='none').to(device)
            model.load_state_dict(saved['state_dict'], strict=True)
            model.eval()
            scale = TargetScale(**saved['scale'])
            # MRI-only student evaluation does not open a single DaT image.
            loader = make_loader(held_out, current_role == 'teacher', metadata['config'], 0, False)
            for batch in loader:
                batch = move_batch(batch, device)
                predictions = scale.restore(model(batch)['prediction']).cpu().tolist()
                for sid, actual, predicted in zip(batch['subject_id'], batch['target'].cpu().tolist(), predictions):
                    if not np.isfinite(predicted):
                        raise FloatingPointError(f'Non-finite prediction: {sid}')
                    rows.append(dict(fold=fold, role=current_role, subject_id=sid,
                                     y_true=actual, y_pred=predicted, absolute_error=abs(actual - predicted)))
            del model, saved
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    full_cv = sorted(metadata['folds']) == [0, 1, 2]
    expected_count = 15 if full_cv else len(metadata['folds']) * 5
    for current_role in roles:
        if sum(row['role'] == current_role for row in rows) != expected_count:
            raise ValueError('Held-out prediction count does not match the requested folds.')
    folder = new_directory(run / 'tests', output, 'test')
    scores = summarize(rows, full_cv)
    write_csv(folder / 'predictions.csv', rows)
    write_csv(folder / 'metrics.csv', scores)
    write_json(folder / 'test.json', dict(run=str(run), baseline=baseline,
        data_fingerprint=fingerprint, purpose=metadata['purpose'], full_cv=full_cv,
        role=role, folds=metadata['folds'], status='complete'))
    write_json(run / 'latest_test.json', dict(test=str(folder)))
    for row in scores:
        print(f"{row['role']:7} {row['scope']:11} fold={row['fold']} n={row['n_subjects']:2} "
              f"MAE={row['mae']:.3f} RMSE={row['rmse']:.3f}", flush=True)
    print(f'Test results: {folder}', flush=True)
    return folder
