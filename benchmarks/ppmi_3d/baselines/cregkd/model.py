"""CReg-KD-dagger networks on the shared backbone/head.

Original CReg-KD is SELF-distillation: teacher and student are the same network on the same T1w input
(Sec. 2.1, Fig. 1). Here the teacher must see DaT (TASK-BOUND, decision G1 in notes/CREGKD_adaptation.md):

  fusion=late  (main) : MRI encoder + DaT encoder (shared ResNet-18 family) -> GAP each -> concat -> head
  fusion=early        : one encoder on the channel-concatenation MRI (+) DaT (same spatial size required)

The student is the plain MRI backbone + head (the paper's "Baseline" network) plus, at training time only,
the AFR layers on its four stages (orig_repo/run_student_gkd.py:88-95, models/context.py).
"""

from __future__ import annotations

import torch
from torch import nn

from models.backbones import build_backbone
from models.heads import build_head

from .afr import AFR


def gap(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).mean(-1)


class CRegTeacher(nn.Module):
    def __init__(self, protocol: dict, cfg: dict):
        super().__init__()
        pre = protocol["preprocessing"]
        self.fusion = cfg["teacher"]["fusion"]
        if self.fusion == "late":
            self.mri_encoder = build_backbone(protocol, int(pre["mri_channels"]))
            self.dat_encoder = build_backbone(protocol, int(pre["dat_channels"]))
            c = self.mri_encoder.channels[-1]
            self.head = build_head(protocol, 2 * c)
        elif self.fusion == "early":
            self.encoder = build_backbone(protocol, int(pre["mri_channels"]) + int(pre["dat_channels"]))
            self.head = build_head(protocol, self.encoder.channels[-1])
        else:
            raise ValueError(f"Unknown teacher fusion: {self.fusion}")

    def forward(self, mri: torch.Tensor, dat: torch.Tensor) -> dict:
        if self.fusion == "late":
            feats = self.mri_encoder(mri)
            feats_dat = self.dat_encoder(dat)
            pooled = torch.cat([gap(feats[-1]), gap(feats_dat[-1])], dim=1)
            return dict(pred=self.head(pooled).squeeze(-1), feats=feats, feats_dat=feats_dat)
        if dat.shape[2:] != mri.shape[2:]:
            raise ValueError("fusion=early needs MRI and DaT on the same grid (co-registered)")
        feats = self.encoder(torch.cat([mri, dat], dim=1))
        return dict(pred=self.head(gap(feats[-1])).squeeze(-1), feats=feats, feats_dat=None)


class CRegStudent(nn.Module):
    def __init__(self, protocol: dict, cfg: dict):
        super().__init__()
        self.encoder = build_backbone(protocol, int(protocol["preprocessing"]["mri_channels"]))
        self.head = build_head(protocol, self.encoder.channels[-1])
        kd = cfg["kd"]
        # AFR exists only when distilling; it is never used by predict() (no extra inference parameters).
        self.afr = (AFR(int(protocol["data"]["spatial_dims"]), [self.encoder.channels[l] for l in kd["layers"]],
                        cfg["afr"]["form"]) if kd["enabled"] and cfg["afr"]["enabled"] else None)

    def forward(self, mri: torch.Tensor) -> dict:
        feats = self.encoder(mri)
        return dict(pred=self.head(gap(feats[-1])).squeeze(-1), feats=feats)
