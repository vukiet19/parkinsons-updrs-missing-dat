"""Mutual-information machinery of M_RAD (Sec. III-B): region critics, region weights, adversarial D_s.

* Region critics S_c^a, S_c^b, S_s^a, S_s^b (Eq. 4, 5, 8, 9): three FC layers 256-64-1 with ReLU
  (Sec. IV-C), one critic shared by all regions of a modality [Assumption].
* JSD MI estimator. The paper cites Deep InfoMax [45] (Hjelm et al., ICLR 2019) for the JSD bound
      I_JSD(g; r) >= E_joint[-sp(-T)] - E_product[sp(T)],   sp = softplus,
  so the loss to MINIMISE is  sp(-T(real)) + sp(T(fake)) >= 0  ("jsd_dim", default).
  Eq. 4/5/8/9 as printed read  sp(T(fake)) - sp(T(real)), which is unbounded below (T(real) -> +inf
  drives it to -inf); it is kept as "paper_literal" only so that tests can demonstrate this (rule D).
* Negative pairs (Sec. III-B: "the mini-batch is shuffled"): r_i of another sample k != j. With 2D
  slices several rows belong to the same subject, so by default k is drawn among rows of OTHER subjects
  [Assumption]; rows without such a partner contribute no negative term.
* Region weights omega (Eq. 3, 6, 7): learnable, "updated with other model parameters". An unconstrained
  weight in a minimised loss diverges, so omega = N * softmax(theta) (positive, mean 1) [Assumption];
  "uniform" = omega = 1 (Table IV, Model III: no region awareness).
* D_s^a, D_s^b (Eq. 12-14): classify (c, s) pairs drawn from the joint as fake and from the product of
  marginals as real; the discriminator maximises L_adv, the encoders minimise it (max-min of Eq. 14),
  implemented with a gradient-reversal layer in a single backward pass [Assumption].
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def mlp(dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(d_in, d_out))
        if i < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.coeff * grad, None


def grad_reverse(x: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, coeff)


def negative_index(subject_ids: Sequence[str], mode: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """For every row j pick a partner k for the 'fake' pair. Returns (k, valid) of shape (B,).

    mode "other_subject": k belongs to a different subject; "any": k != j (plain shuffle).
    Uses the global torch RNG, so it is reproducible under seed_everything.
    """
    b = len(subject_ids)
    if mode == "other_subject":
        bad = torch.tensor([[a == c for c in subject_ids] for a in subject_ids], device=device)
    elif mode == "any":
        bad = torch.eye(b, dtype=torch.bool, device=device)
    else:
        raise ValueError(f"Unknown negative sampling mode: {mode}")
    scores = torch.rand(b, b, device=device).masked_fill(bad, -1.0)
    k = scores.argmax(1)
    valid = ~bad[torch.arange(b, device=device), k]
    return k, valid


def jsd_terms(pos: torch.Tensor, neg: torch.Tensor, valid: torch.Tensor, estimator: str) -> torch.Tensor:
    """pos, neg: (B, N) critic scores of real / fake pairs. Returns the per-region loss (N,)."""
    if not bool(valid.any()):                      # no fake pair available: no MI signal in this batch
        return pos.sum(0) * 0.0
    v = valid.float()[:, None]
    neg_mean = F.softplus(neg).mul(v).sum(0) / v.sum()
    if estimator == "jsd_dim":
        return F.softplus(-pos).mean(0) + neg_mean
    if estimator == "paper_literal":               # Eq. 4 as printed: unbounded below (see module doc)
        return neg_mean - F.softplus(pos).mean(0)
    raise ValueError(f"Unknown MI estimator: {estimator}")


class RegionCritic(nn.Module):
    """T(g, r_i) for every region i; g is a global vector (c or f), r the (B, N, C_r) region vectors."""

    def __init__(self, global_dim: int, region_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = mlp([global_dim + region_dim, *hidden, 1])

    def score(self, g: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([g[:, None, :].expand(-1, r.shape[1], -1), r], -1)).squeeze(-1)

    def forward(self, g, r, k, valid, estimator: str) -> torch.Tensor:
        return jsd_terms(self.score(g, r), self.score(g, r[k]), valid, estimator)


class RegionWeights(nn.Module):
    def __init__(self, n: int, mode: str):
        super().__init__()
        if mode not in ("learnable_softmax", "uniform"):
            raise ValueError(f"Unknown region weight mode: {mode}")
        self.mode, self.n = mode, n
        self.theta = nn.Parameter(torch.zeros(n)) if mode == "learnable_softmax" else None
        self.register_buffer("ones", torch.ones(n), persistent=False)

    def forward(self) -> torch.Tensor:
        if self.theta is None:
            return self.ones
        return self.n * torch.softmax(self.theta, 0)


class PairDiscriminator(nn.Module):
    """D_s (Eq. 12-13). Returns the discriminator's binary cross-entropy (= -L_adv per pair).

    Joint pairs (c_j, s_j) -> label 0 ("fake"), product pairs (c_j, s_k) -> label 1 ("real").
    s (and c when ``reverse_c``) pass through a gradient-reversal layer: the discriminator minimises the
    BCE (maximises L_adv), the encoders maximise it (minimise L_adv), i.e. the max-min of Eq. 14.
    """

    def __init__(self, dim_c: int, dim_s: int, hidden: Sequence[int]):
        super().__init__()
        self.net = mlp([dim_c + dim_s, *hidden, 1])

    def forward(self, c, s, k, valid, reverse_c: bool) -> torch.Tensor:
        s = grad_reverse(s)
        c = grad_reverse(c) if reverse_c else c
        joint = self.net(torch.cat([c, s], 1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(joint, torch.zeros_like(joint))
        if bool(valid.any()):
            prod = self.net(torch.cat([c[valid], s[k][valid]], 1)).squeeze(1)
            loss = loss + F.binary_cross_entropy_with_logits(prod, torch.ones_like(prod))
        return loss
