"""Imputation-induced distillation module M_IID (Sec. III-C, Sec. IV-C): U_LIT, student S, teacher T.

Student S (Sec. IV-C): four FC layers 512-256-64-1 on f_a = [c_a, s_a]; the last one is the shared
protocol head (INTERFACE).   o_i^s = ReLU(FC_i^s(o_{i-1}^s)), o_0^s = f_a,  y_s = head_s(o_3^s).

Teacher T (Eq. 16-17, Fig. 1):
    j_0 = Fuse_0(f_a, f_b or 0, c'_b)                            (Eq. 16; zero f_b if DaT is missing)
    o_i^t = ReLU(FC_i^t(j_{i-1})),  j_i = Fuse_i(o_i^s, resize(c'_b), o_i^t),  i = 1..3   (Eq. 17)
    y_t = head_t(j_3)
  Eq. 17 writes o_{i-1}^t next to o_i^s; the three fused vectors must have equal length for a 1x1
  convolution (this is why the paper max-pools U_LIT(c_a) to the size of o_i), so the teacher feature of
  the same width is used [Assumption].
"Fuse" = 1x1 convolution over the vectors stacked as channels (Eq. 16-17, "conv1x1", default): one weight
  per source plus a bias, applied position-wise. In Eq. 16 c'_b (512) is shorter than f (1024); it is
  zero-padded in the s-slot so it is aligned with c_b inside f_b = [c_b, s_b] [Assumption]. "linear" =
  a fully connected layer on the concatenation (the alternative reading used by dftd_adni).
U_LIT (Sec. IV-C, Eq. 15): encoder E and decoder D with three FC layers each, ratio 2, each followed by
  ReLU (c_b >= 0 because it is a GAP of post-ReLU features).  resize = max-pooling (Sec. III-C.2).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.heads import build_head


def resize_vec(x: torch.Tensor, size: int) -> torch.Tensor:
    """Max-pool a (B, D) vector to (B, size) (Sec. III-C.2)."""
    if x.shape[1] == size:
        return x
    if x.shape[1] < size:
        raise ValueError("U_LIT output is shorter than a teacher layer; max-pooling cannot up-sample.")
    if x.shape[1] % size == 0:     # non-overlapping windows; deterministic backward (adaptive pool's is not)
        return x.view(x.shape[0], size, x.shape[1] // size).amax(-1)
    return F.adaptive_max_pool1d(x[:, None], size)[:, 0]


class LateralTransition(nn.Module):
    """U_LIT = D(E(c_a)) -> c'_b."""

    def __init__(self, dim: int, depth: int = 3):
        super().__init__()
        dims = [dim // (2 ** i) for i in range(depth + 1)]
        if dims[-1] < 1:
            raise ValueError("U_LIT is too deep for this feature width.")
        enc, dec = [], []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(d_in, d_out), nn.ReLU(inplace=True)]
        rev = dims[::-1]
        for d_in, d_out in zip(rev[:-1], rev[1:]):
            dec += [nn.Linear(d_in, d_out), nn.ReLU(inplace=True)]
        self.E, self.D = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, c_a: torch.Tensor) -> torch.Tensor:
        return self.D(self.E(c_a))


class Fuse(nn.Module):
    def __init__(self, sizes: Sequence[int], out: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "conv1x1":
            if len(set(sizes)) != 1 or sizes[0] != out:
                raise ValueError(f"conv1x1 fusion needs equal lengths, got {sizes} -> {out}")
            self.op = nn.Conv1d(len(sizes), 1, kernel_size=1)
        elif mode == "linear":
            self.op = nn.Linear(sum(sizes), out)
        else:
            raise ValueError(f"Unknown fusion: {mode}")

    def forward(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.mode == "conv1x1":
            return self.op(torch.stack(list(parts), 1)).squeeze(1)
        return self.op(torch.cat(list(parts), 1))


class StudentBranch(nn.Module):
    def __init__(self, protocol: dict, d_f: int, widths: Sequence[int]):
        super().__init__()
        dims = [d_f, *widths]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:])])
        self.head = build_head(protocol, widths[-1])

    def forward(self, f_a: torch.Tensor):
        h, feats = f_a, []
        for layer in self.layers:
            h = F.relu(layer(h))
            feats.append(h)
        return self.head(h).squeeze(-1), feats


class IID(nn.Module):
    def __init__(self, protocol: dict, dim_c: int, widths: Sequence[int], fusion: str, lit_depth: int):
        super().__init__()
        self.dim_c, self.d_f = dim_c, 2 * dim_c
        dims = [self.d_f, *widths]
        self.student = StudentBranch(protocol, self.d_f, widths)
        n0 = [self.d_f, self.d_f, self.d_f] if fusion == "conv1x1" else [self.d_f, self.d_f, dim_c]
        self.t_fuse0 = Fuse(n0, self.d_f, fusion)
        self.t_layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:])])
        self.t_fuse = nn.ModuleList([Fuse([w, w, w], w, fusion) for w in widths])
        self.t_head = build_head(protocol, widths[-1])
        self.lit = LateralTransition(dim_c, lit_depth)
        self.fusion = fusion

    def forward(self, f_a, f_b, c_a, detach_student_into_teacher: bool = False) -> dict:
        """f_b: (B, d_f) real global DaT representation, already zeroed where DaT is missing."""
        cb_hat = self.lit(c_a)
        pred_s, s_feats = self.student(f_a)
        cb0 = torch.cat([cb_hat, torch.zeros_like(cb_hat)], 1) if self.fusion == "conv1x1" else cb_hat
        j = self.t_fuse0([f_a, f_b, cb0])
        t_feats = []
        for layer, fuse, o_s in zip(self.t_layers, self.t_fuse, s_feats):
            o_t = F.relu(layer(j))
            t_feats.append(o_t)
            o_s = o_s.detach() if detach_student_into_teacher else o_s
            j = fuse([o_s, resize_vec(cb_hat, o_t.shape[1]), o_t])
        return dict(pred_s=pred_s, pred_t=self.t_head(j).squeeze(-1), s_feats=s_feats, t_feats=t_feats,
                    cb_hat=cb_hat)
