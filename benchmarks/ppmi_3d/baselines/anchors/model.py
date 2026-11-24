"""Networks of the common anchors, built only from the shared backbone and head."""

from __future__ import annotations

import torch
from torch import nn

from models.backbones import build_backbone
from models.heads import build_head


def gap(x: torch.Tensor) -> torch.Tensor:
    """Global average pooling over all spatial dims: (B, C, *S) -> (B, C)."""
    return x.flatten(2).mean(-1)


class MRIOnlyNet(nn.Module):
    """build_backbone(MRI) -> GAP of the last stage -> build_head(512)."""

    def __init__(self, protocol: dict):
        super().__init__()
        self.encoder = build_backbone(protocol, int(protocol["preprocessing"]["mri_channels"]))
        self.head = build_head(protocol, self.encoder.channels[-1])

    def forward(self, mri: torch.Tensor) -> dict:
        feats = self.encoder(mri)
        return dict(pred=self.head(gap(feats[-1])).squeeze(-1), feats=feats)


class MMTeacherNet(nn.Module):
    """Late fusion: MRI encoder + DaT encoder -> GAP each -> concat -> build_head(1024).

    Same architecture as the CReg-KD late-fusion teacher (baselines/cregkd/model.py:CRegTeacher), written
    separately so the anchors never depend on a baseline's code."""

    def __init__(self, protocol: dict):
        super().__init__()
        pre = protocol["preprocessing"]
        self.mri_encoder = build_backbone(protocol, int(pre["mri_channels"]))
        self.dat_encoder = build_backbone(protocol, int(pre["dat_channels"]))
        self.head = build_head(protocol, self.mri_encoder.channels[-1] + self.dat_encoder.channels[-1])

    def forward(self, mri: torch.Tensor, dat: torch.Tensor) -> dict:
        # CORE: pool the two modalities separately, then concatenate their features.
        feats, feats_dat = self.mri_encoder(mri), self.dat_encoder(dat)
        pooled = torch.cat([gap(feats[-1]), gap(feats_dat[-1])], dim=1)
        return dict(pred=self.head(pooled).squeeze(-1), feats=feats, feats_dat=feats_dat)
