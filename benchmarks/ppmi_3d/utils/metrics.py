"""Regression metrics shared by validation (train.py) and testing (evaluate.py)."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def regression_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and of equal length.")
    if not np.isfinite(p).all():
        raise FloatingPointError("Non-finite predictions.")
    err = p - y
    out = dict(n=int(y.size), mae=float(np.abs(err).mean()), rmse=float(np.sqrt(np.square(err).mean())),
               bias=float(err.mean()))
    sst = float(np.square(y - y.mean()).sum())
    out["r2"] = float(1.0 - np.square(err).sum() / sst) if sst > 0 else math.nan
    if y.size >= 3 and np.std(y) > 0 and np.std(p) > 0:
        out["pearson_r"] = float(stats.pearsonr(y, p)[0])
        out["spearman_rho"] = float(stats.spearmanr(y, p)[0])
    else:  # undefined for constant predictions (e.g. the train-mean reference)
        out["pearson_r"] = math.nan
        out["spearman_rho"] = math.nan
    return out
