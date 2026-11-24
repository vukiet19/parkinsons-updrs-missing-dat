"""Attentive Feature Refinement (AFR) layers and the channel-pooling function phi.

AFR (Sec. 2.2.2, Eq. 10, Fig. 1): the student's hint map of layer l is refined, h^S_l -> f(h^S_l),
before both sides are pooled to a spatial attention map by phi.

form="code"  : depthwise k=3 conv -> pointwise k=1 conv -> BatchNorm, no ReLU
               [Code orig_repo/models/context.py:10-33] (main, rule C)
form="paper" : k=3 conv -> k=1 conv -> ReLU  ("two convolution operators with a kernel size of 3 and 1
               respectively followed by a ReLU activation function", caption of Fig. 1) (sensitivity)
"""

from __future__ import annotations

import torch
from torch import nn


def _ops(dims: int):
    if dims == 2:
        return nn.Conv2d, nn.BatchNorm2d
    if dims == 3:
        return nn.Conv3d, nn.BatchNorm3d
    raise ValueError("spatial_dims must be 2 or 3")


def afr_block(dims: int, channels: int, form: str) -> nn.Module:
    conv, bn = _ops(dims)
    if form == "code":
        return nn.Sequential(conv(channels, channels, 3, 1, 1, bias=False, groups=channels),
                             conv(channels, channels, 1, 1, 0, bias=False), bn(channels))
    if form == "paper":   # conv biases: PyTorch default [Assumption]; groups not stated -> full conv
        return nn.Sequential(conv(channels, channels, 3, 1, 1), conv(channels, channels, 1, 1, 0),
                             nn.ReLU(inplace=True))
    raise ValueError(f"Unknown AFR form: {form}")


class AFR(nn.Module):
    """One refinement block per distilled stage (M = 4 for ResNet-18). Training-time only."""

    def __init__(self, dims: int, channels, form: str):
        super().__init__()
        self.blocks = nn.ModuleList([afr_block(dims, int(c), form) for c in channels])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return [block(f) for block, f in zip(self.blocks, feats)]


def phi(f: torch.Tensor, mode: str) -> torch.Tensor:
    """Channel pooling R^{C x spatial} -> R^{spatial}, flattened to (B, N) [Paper Sec. 2.2.2;
    Code context.py:51-60]. pow_avg = sum_i |x_i|^2 / C (main, Fig. 4); avg_pow = (sum_i x_i / C)^2;
    mean = sum_i x_i / C; raw = x (no pooling). No normalisation (context.py:61-62 is commented out)."""
    if mode == "pow_avg":
        out = f.pow(2).mean(1)
    elif mode == "avg_pow":
        out = f.mean(1).pow(2)
    elif mode == "mean":
        out = f.mean(1)
    elif mode == "raw":
        out = f
    else:
        raise ValueError(f"Unknown phi: {mode}")
    return out.reshape(f.shape[0], -1)
