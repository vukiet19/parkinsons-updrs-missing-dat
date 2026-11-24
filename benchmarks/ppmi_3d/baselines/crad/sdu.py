"""Smooth Distillation Unit (SDU): detached confidence weights (Sec. 3.4-3.5).

All errors are measured on the train-fold z-scored target, i.e. in units of the training SD.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def layer_confidence(cafr_t: torch.Tensor, aux: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """S^k = 1 - clip(||Y_c'^k - Y_c||^2), Eq. 14, with Y_c the aux score normalised to [0, 1].
    Samples without an aux score get S^k = 1 [Assumption]. Shapes: (B, K), (B,), (B,) -> (B, K)."""
    error = (cafr_t - torch.nan_to_num(aux)[:, None]).square()
    conf = 1.0 - error.clamp(0.0, 1.0)
    return torch.where(mask[:, None], conf, torch.ones_like(conf))


@torch.no_grad()
def output_confidence(pred_t: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    """S^C = 1 - clip(dis(Y_T', Y)), Eq. 11, with dis = |y_T - y| / scale [Proposal]. (B,)"""
    return 1.0 - ((pred_t - y).abs() / scale).clamp(0.0, 1.0)


@torch.no_grad()
def psi(pred_t: torch.Tensor, pred_s: torch.Tensor, y: torch.Tensor, mode: str = "per_sample",
        eps: float = 1e-8) -> torch.Tensor:
    """psi, Eq. 12: (e_S / (e_T + e_S + eps))^2, softening KD where the student already beats
    the teacher; psi = 1 where the student is worse (Sec. 3.4 text).

    per_sample [Paper text] | batch: decide on batch means as orig_repo/train_test.py:65-79 [Code]
    always: soft formula everywhere | hard: 0/1 gate ("Soften", Table 7) | none: psi = 1. (B,)
    """
    e_t = (pred_t - y).abs()
    e_s = (pred_s - y).abs()
    soft = (e_s / (e_t + e_s + eps)).square()
    ones = torch.ones_like(soft)
    if mode == "per_sample":
        return torch.where(e_s <= e_t, soft, ones)
    if mode == "batch":
        return ones if e_s.mean() > e_t.mean() else soft
    if mode == "always":
        return soft
    if mode == "hard":
        return torch.where(e_s <= e_t, torch.zeros_like(soft), ones)
    if mode == "none":
        return ones
    raise ValueError(f"Unknown psi_mode: {mode}")
