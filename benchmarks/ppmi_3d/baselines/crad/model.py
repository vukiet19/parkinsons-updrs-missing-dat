"""CRAD teacher (MRI + DaT) and student (MRI only) built on the shared backbone/head.

Teacher: orig_repo/models/crad_model.py:294-338; Student: :341-384; Fig. 1 and Table A.1.
Backbone and head come from the protocol (INTERFACE); projection, fusion, OP and CAFR are CORE.
"""

import torch
import torch.nn.functional as F
from torch import nn

from models.backbones import build_backbone
from models.heads import build_head

from .cafr import CAFR, gap
from .modules import fusion, student_projection, student_refine, teacher_projection


class CRADTeacher(nn.Module):
    def __init__(self, protocol: dict, cfg: dict):
        super().__init__()
        dims = int(protocol["data"]["spatial_dims"])
        pre = protocol["preprocessing"]
        self.mri_encoder = build_backbone(protocol, int(pre["mri_channels"]))   # separate encoders,
        self.dat_encoder = build_backbone(protocol, int(pre["dat_channels"]))   # train_test.py:106-107
        c = self.mri_encoder.channels[-1]
        self.om_proj = teacher_projection(dims, c)                              # OP, Z_T^m
        self.op_proj = teacher_projection(dims, c)                              # OP, Z_T^p
        self.fusion = fusion(dims, c)                                           # Z_T^u
        self.head = build_head(protocol, c)
        self.cafr = (CAFR(self.mri_encoder.channels, cfg["cafr"]["layers"], cfg["cafr"]["pool_input"])
                     if cfg["cafr"]["enabled"] else None)

    def forward(self, mri: torch.Tensor, dat: torch.Tensor) -> dict:
        feats = self.mri_encoder(mri)
        z_m = self.om_proj(feats[-1])
        z_p = self.op_proj(self.dat_encoder(dat)[-1])
        if z_p.shape[2:] != z_m.shape[2:]:   # different MRI/DaT input sizes (3D) [Assumption]
            mode = "bilinear" if z_m.ndim == 4 else "trilinear"
            z_p = F.interpolate(z_p, size=z_m.shape[2:], mode=mode, align_corners=False)
        z_u = self.fusion(torch.cat([z_m, z_p], dim=1))
        return dict(pred=self.head(gap(z_u)).squeeze(-1), feats=feats, z_m=z_m, z_p=z_p, z_u=z_u,
                    cafr=self.cafr(feats) if self.cafr is not None else None)


class CRADStudent(nn.Module):
    def __init__(self, protocol: dict, cfg: dict):
        super().__init__()
        dims = int(protocol["data"]["spatial_dims"])
        self.mri_encoder = build_backbone(protocol, int(protocol["preprocessing"]["mri_channels"]))
        c = self.mri_encoder.channels[-1]
        self.om_proj, self.op_proj = student_projection(dims, c), student_projection(dims, c)
        self.fim, self.fip = student_refine(dims, c), student_refine(dims, c)
        self.fusion = fusion(dims, c)                                           # Z_S^u
        self.head = build_head(protocol, c)

    def forward(self, mri: torch.Tensor) -> dict:
        feats = self.mri_encoder(mri)
        z_m = self.fim(self.om_proj(feats[-1]))      # crad_model.py:367-368
        z_p = self.fip(self.op_proj(feats[-1]))
        z_u = self.fusion(torch.cat([z_m, z_p], dim=1))
        return dict(pred=self.head(gap(z_u)).squeeze(-1), feats=feats, z_m=z_m, z_p=z_p, z_u=z_u)
