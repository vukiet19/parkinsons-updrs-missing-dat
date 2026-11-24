"""CReg-KD loss terms adapted to regression (see notes/CREGKD_adaptation.md).

All targets/predictions are in train-fold z-units (protocol `normalization: zscore_train`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .afr import phi


@torch.no_grad()
def gate(pred_t: torch.Tensor, y: torch.Tensor, eta: float) -> torch.Tensor:
    """Gated distillation weight, Eq. 8: psi_k = 1 - clip(dis(o^T_k, o^GT_k) / eta, 0, 1).

    dis = |y_T - y| -- the brain-age form of Sec. 3.3 (|E_T[age] - age|) and of the authors' Gated-KD
    regression code (BrainAgeReg run.py:161); measured in train-fold SD units [Proposal, G3]. (B,)"""
    return 1.0 - ((pred_t - y).abs() / eta).clamp(0.0, 1.0)


def output_kd(pred_s: torch.Tensor, pred_t: torch.Tensor, form: str = "mse") -> torch.Tensor:
    """TASK-BOUND replacement of tau^2 KL(p^T || p^S) (Eq. 9/12) for a scalar regression output.

    mse: (y_S - y_T)^2, = 2 sigma^2 KL(N(y_T, s^2) || N(y_S, s^2)) [Proposal, G4; same as CRAD Q2 / DFTD D4]
    l1 : |y_S - y_T|. The teacher side is a constant (frozen teacher, no_grad). Per sample (B,)."""
    if form == "mse":
        return (pred_s - pred_t).square()
    if form == "l1":
        return (pred_s - pred_t).abs()
    raise ValueError(f"Unknown output KD form: {form}")


def teacher_hints(feats_t: list[torch.Tensor], feats_dat: list[torch.Tensor] | None, source: str):
    """Teacher hint maps h^T_l (G2). mri: MRI-encoder stages (main). concat: channel-concatenation of the
    MRI- and DaT-encoder stages, DaT resized to the MRI grid when they differ [Assumption]."""
    if source == "mri" or feats_dat is None:
        if source == "concat":
            raise ValueError("teacher_hint=concat needs a teacher with a separate DaT encoder")
        return feats_t
    if source != "concat":
        raise ValueError(f"Unknown teacher_hint: {source}")
    out = []
    for fm, fd in zip(feats_t, feats_dat):
        if fd.shape[2:] != fm.shape[2:]:
            mode = "bilinear" if fm.ndim == 4 else "trilinear"
            fd = F.interpolate(fd, size=fm.shape[2:], mode=mode, align_corners=False)
        out.append(torch.cat([fm, fd], dim=1))
    return out


def hint_loss(refined_s: list[torch.Tensor], hints_t: list[torch.Tensor], weight: torch.Tensor,
              phi_mode: str = "pow_avg", reduce_layers: str = "sum") -> tuple[torch.Tensor, torch.Tensor]:
    """L_HT, Eq. 10/12: sum_l psi * ||phi(f(h^S_l)) - phi(h^T_l)||^2.

    Per layer: mean over batch and positions of psi_b (phi_s - phi_t)^2 [Code context.py:47-50, rule C];
    layers summed [Code context.py:43] (reduce_layers=mean: 1/M). Teacher hints are detached
    (context.py:43). Returns (loss, per-layer values (M,))."""
    per_layer = []
    for s, t in zip(refined_s, hints_t):
        diff = (phi(s, phi_mode) - phi(t.detach(), phi_mode)).square()   # (B, N)
        per_layer.append((diff * weight[:, None]).mean())
    per_layer = torch.stack(per_layer)
    if reduce_layers == "sum":
        return per_layer.sum(), per_layer.detach()
    if reduce_layers == "mean":
        return per_layer.mean(), per_layer.detach()
    raise ValueError(f"Unknown reduce_layers: {reduce_layers}")
