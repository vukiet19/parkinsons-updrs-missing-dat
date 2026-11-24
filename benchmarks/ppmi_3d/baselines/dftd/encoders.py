"""Encoders of the region-aware disentanglement module M_RAD (Sec. III-B, Sec. IV-C).

Paper: each of F^a o H_c^a, F^a o H_s^a, F^b o H_c^b, F^b o H_s^b is a 5-conv network; F^a / F^b are its
first three layers and produce the intermediate map m; H_c / H_s are the remaining layers and produce the
inter-modality relevant representation c and the intra-modality specific representation s.

Here (INTERFACE = shared backbone): every F o H path is exactly the shared ResNet-18. F = stem + stages
1-3 (m: 256 channels, stride 16); H_c and H_s are two copies of stage 4 followed by global average
pooling, so c and s are 512-d vectors (the 512-d embedding of Sec. IV-J). [Assumption] split point.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from models.backbones import build_backbone


def gap(z: torch.Tensor) -> torch.Tensor:
    return z.flatten(2).mean(-1)


def region_vectors(m: torch.Tensor, grid: int) -> torch.Tensor:
    """Split m (B, C, *S) into grid^dims NON-overlapping blocks and average each (Sec. III-B: N = 4x4x4).

    Uses torch.tensor_split so blocks never overlap even when S is not divisible by ``grid``
    (adaptive pooling would overlap, e.g. 14 -> 4 bins). Returns (B, N, C). [Assumption] mean pooling.
    """
    dims = m.ndim - 2
    blocks = [m]
    for axis in range(2, 2 + dims):
        if m.shape[axis] < grid:
            raise ValueError(f"Feature map size {tuple(m.shape[2:])} is smaller than the region grid {grid}.")
        blocks = [part for block in blocks for part in torch.tensor_split(block, grid, dim=axis)]
    return torch.stack([b.flatten(2).mean(-1) for b in blocks], dim=1)


class DisentangleEncoder(nn.Module):
    """x -> m (F), c = GAP(H_c(m)), s = GAP(H_s(m)), r = region vectors of m."""

    def __init__(self, protocol: dict, in_channels: int, grid: int):
        super().__init__()
        base = build_backbone(protocol, in_channels)
        self.stem = base.stem
        self.trunk = nn.Sequential(*base.stages[:3])           # F
        self.head_c = base.stages[3]                           # H_c
        self.head_s = copy.deepcopy(base.stages[3])            # H_s (same initialisation) [Assumption]
        self.grid = int(grid)
        self.spatial_dims = base.spatial_dims
        self.region_dim = base.channels[2]
        self.out_dim = base.channels[3]

    @property
    def n_regions(self) -> int:
        return self.grid ** self.spatial_dims

    def forward(self, x: torch.Tensor) -> dict:
        m = self.trunk(self.stem(x))
        return dict(m=m, c=gap(self.head_c(m)), s=gap(self.head_s(m)), r=region_vectors(m, self.grid))
