"""Eval-unit level prediction shared by validation (train.py) and testing (evaluate.py)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from data.dataset import to_device


@torch.no_grad()
def predict_units(method, stage: str, model, loader, scaler, device) -> list[dict]:
    """Average row predictions (e.g. 2D slices) of each eval unit; return original-scale values."""
    model.eval()
    preds, truth, subject = defaultdict(list), {}, {}
    for batch in loader:
        batch = to_device(batch, device)
        z = method.predict(stage, model, batch).float().cpu()
        for unit, sid, y, value in zip(batch["unit"], batch["subject_id"], batch["y"].cpu().tolist(), z.tolist()):
            preds[unit].append(scaler.restore(value))
            truth[unit], subject[unit] = y, sid
    out = []
    for unit in sorted(preds):
        value = float(np.mean(preds[unit]))
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite prediction for {unit}")
        out.append(dict(unit=unit, subject_id=subject[unit], y_true=float(truth[unit]), y_pred=value,
                        n_rows=len(preds[unit])))
    return out


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    """Fixed teacher: no parameter updates and no BatchNorm statistic updates (Alg. 1 line 14)."""
    model.eval()
    model.requires_grad_(False)
    return model
