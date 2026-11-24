"""CRAD loss terms. Every function returns per-sample values unless stated otherwise."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .pfac import PFAC


def orthogonal_loss(z_m: torch.Tensor, z_p: torch.Tensor, form: str = "code") -> torch.Tensor:
    """L_ORT, Eq. 3 (label-free): intra-modal similarity -> 1, inter-modal similarity -> 0.

    s, d = mean |cosine similarity| over all sample pairs of the batch, including the diagonal.
    ``code``: (1 - (s_mm + s_pp)/2) + d_mp, i.e. ``ort_project_loss2``
    (orig_repo/models/crad_model.py:237-244, orthogonal_type_T=2), the code path matching Eq. 3.
    ``paper_sum``: literal sum over (i,j) in {(m,p),(p,m)} = 2 x the code value.
    Returns a scalar (batch-level loss).
    """
    m = F.normalize(z_m.flatten(1), p=2, dim=1)
    p = F.normalize(z_p.flatten(1), p=2, dim=1)
    s_mm = (m @ m.T).abs().mean()
    s_pp = (p @ p.T).abs().mean()
    d_mp = (m @ p.T).abs().mean()
    code = (1.0 - (s_mm + s_pp) * 0.5) + d_mp
    if form == "code":
        return code
    if form == "paper_sum":
        return 2.0 * code
    raise ValueError(f"Unknown ort.form: {form}")


def gram_kd(z_t: torch.Tensor, z_s: torch.Tensor, mode: str = "gram_discrepancy", tau: float = 1.0) -> torch.Tensor:
    """L_KD^U, Eq. 6: compare channel self-similarity (Gram) matrices of teacher and student.

    Each channel vector is L2-normalised over spatial positions and G = Z_hat Z_hat^T is C x C,
    as in ``DistillationOrthogonalLoss`` applied per sample (orig_repo/utils.py:31-53 via
    train_test.py:50-55). Eq. 6 writes KL(G_T || G_S) but G is not a probability distribution and
    the paper gives no normalisation, so the official formula is the main variant (rule B):

    ``gram_discrepancy`` [Code utils.py:46-51]: |sum_ij exp(G_T,ij) (G_T,ij - G_S,ij)| * tau^2 / C^2
        (``F.kl_div(G_S, G_T, log_target=True)``). It is non-negative and finite, but NOT a KL
        divergence: it matches an exp(G_T)-weighted mean of the Gram entries, so G_S != G_T can
        give zero loss. We therefore call it "Gram discrepancy", not KL.
    ``kl_softmax_rows`` [Proposal, sensitivity]: true KL(P_T || P_S) between row-wise softmax
        distributions of the Gram matrices (as in MiniLM relation KD, Wang et al. 2020a).
    Returns (B,).
    """
    t = F.normalize(z_t.detach().flatten(2), p=2, dim=2)
    s = F.normalize(z_s.flatten(2), p=2, dim=2)
    g_t = torch.bmm(t, t.transpose(1, 2)) / tau
    g_s = torch.bmm(s, s.transpose(1, 2)) / tau
    if mode == "gram_discrepancy":
        c = g_t.shape[-1]
        return (g_t.exp() * (g_t - g_s)).sum((1, 2)).abs() * tau * tau / (c * c)
    if mode == "kl_softmax_rows":
        log_p_t = F.log_softmax(g_t, dim=-1)
        log_p_s = F.log_softmax(g_s, dim=-1)
        return (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1).mean(-1) * tau * tau
    raise ValueError(f"Unknown kd.fd_divergence: {mode}")


def output_kd(pred_s: torch.Tensor, pred_t: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    """L_KD^P adapted to regression [Proposal]. Eq. 7 is a KL between class distributions; for a
    scalar output, KL(N(t, s^2) || N(s_pred, s^2)) = (t - s_pred)^2 / (2 s^2), i.e. an MSE. (B,)"""
    diff = pred_s - pred_t.detach()
    if kind == "mse":
        return diff.square()
    if kind == "l1":
        return diff.abs()
    raise ValueError(f"Unknown output_kd: {kind}")


def attention_kd(feats_s, feats_t, layers, pfac: PFAC, distance: str = "mse") -> torch.Tensor:
    """PF-CMAD, L_KD^F, Eq. 10: || M_S^k Z_S^{k,m} - M_T^k Z_T^{k,m} ||^2 on the MRI stages.

    Each side uses its own parameter-free PFAC map; the teacher side is detached. The distance
    is averaged over elements: ``smooth_l1`` = orig_repo/train_test.py:346-349 [Code, default];
    ``mse`` = squared L2 of Eq. 10 [Paper]. Returns (B, len(layers)); the caller reduces layers.
    """
    out = []
    for k in layers:
        a_s, _ = pfac(feats_s[k])
        a_t, _ = pfac(feats_t[k].detach())
        if distance == "mse":
            d = (a_s - a_t).square()
        elif distance == "smooth_l1":
            d = F.smooth_l1_loss(a_s, a_t, reduction="none")
        else:
            raise ValueError(f"Unknown attention_distance: {distance}")
        out.append(d.flatten(1).mean(1))
    return torch.stack(out, dim=1)


def cafr_loss(pred: torch.Tensor, aux: torch.Tensor, mask: torch.Tensor, kind: str = "mse",
              reduce_layers: str = "mean") -> torch.Tensor:
    """L_MSE of Eq. 2 (1/K sum_k ||Y_c'^k - Y_c||^2); samples without an aux score are masked.
    ``smooth_l1`` + ``sum`` reproduce orig_repo/train_test.py:210-214. Returns a scalar."""
    if not mask.any():
        return pred.sum() * 0.0
    pred, target = pred[mask], aux[mask][:, None].expand_as(pred[mask])
    if kind == "mse":
        diff = (pred - target).square()
    elif kind == "smooth_l1":
        diff = F.smooth_l1_loss(pred, target, reduction="none")
    else:
        raise ValueError(f"Unknown cafr loss: {kind}")
    per_sample = diff.mean(1) if reduce_layers == "mean" else diff.sum(1)
    return per_sample.mean()
