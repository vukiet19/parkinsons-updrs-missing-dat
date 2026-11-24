"""DFTD model = M_RAD (two disentangling encoders + MI critics + region weights + D_s) + M_IID.

Modality a = MRI (always available), modality b = DaT (missing at test time).
``DFTDStudentOnly`` is the MRI path of DFTD (F^a, H_c^a, H_s^a, S) without DaT, teacher or MI terms:
the same-architecture student-only reference.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoders import DisentangleEncoder
from .iid import IID, StudentBranch
from .mi import PairDiscriminator, RegionCritic, RegionWeights


def _mri_encoder(protocol: dict, cfg: dict) -> DisentangleEncoder:
    return DisentangleEncoder(protocol, int(protocol["preprocessing"]["mri_channels"]), cfg["regions"]["grid_mri"])


class DFTDModel(nn.Module):
    RAD_PARTS = ("enc_a", "enc_b", "w_a", "w_b", "S_c_a", "S_c_b", "S_s_a", "S_s_b", "D_s_a", "D_s_b")

    def __init__(self, protocol: dict, cfg: dict):
        super().__init__()
        pre, reg, hid = protocol["preprocessing"], cfg["regions"], list(cfg["critic"]["hidden"])
        self.enc_a = _mri_encoder(protocol, cfg)
        self.enc_b = DisentangleEncoder(protocol, int(pre["dat_channels"]), reg["grid_dat"])
        d, ra, rb = self.enc_a.out_dim, self.enc_a.region_dim, self.enc_b.region_dim
        self.w_a = RegionWeights(self.enc_a.n_regions, reg["weights"])     # omega_a (Eq. 3, 7)
        self.w_b = RegionWeights(self.enc_b.n_regions, reg["weights"])     # omega_b
        self.S_c_a = RegionCritic(d, rb, hid)          # c_a  vs regions of x_b   (Eq. 4)
        self.S_c_b = RegionCritic(d, ra, hid)          # c_b  vs regions of x_a   (Eq. 5)
        self.S_s_a = RegionCritic(2 * d, ra, hid)      # f_a  vs regions of x_a   (Eq. 8)
        self.S_s_b = RegionCritic(2 * d, rb, hid)      # f_b  vs regions of x_b   (Eq. 9)
        self.D_s_a = PairDiscriminator(d, d, hid)      # I(c_a; s_a)              (Eq. 12)
        self.D_s_b = PairDiscriminator(d, d, hid)      # I(c_b; s_b)              (Eq. 13)
        iid = cfg["iid"]
        self.iid = IID(protocol, d, list(iid["widths"]), iid["fusion"], int(iid["lit_depth"]))
        self.rad_frozen = False

    def rad_modules(self):
        return [getattr(self, name) for name in self.RAD_PARTS]

    def freeze_rad(self):
        """Stage 2 (Alg. 1 lines 13-19): 'fine-tune M_IID ... while fixing M_RAD'."""
        self.rad_frozen = True
        for module in self.rad_modules():
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.rad_frozen:            # keep BatchNorm statistics of the fixed M_RAD untouched
            for module in self.rad_modules():
                module.eval()
        return self


class DFTDStudentOnly(nn.Module):
    def __init__(self, protocol: dict, cfg: dict):
        super().__init__()
        self.enc_a = _mri_encoder(protocol, cfg)
        self.student = StudentBranch(protocol, 2 * self.enc_a.out_dim, list(cfg["iid"]["widths"]))

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        e = self.enc_a(mri)
        return self.student(torch.cat([e["c"], e["s"]], 1))[0]
