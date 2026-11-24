"""Cognitive Awareness Feature Refinement (CAFR) heads of the teacher.

Paper: Sec. 3.2.1, Eq. 2 -- Y_c'^k = h(GAP(Z_T^k); W^k) on every intermediate MRI stage k.
Code:  orig_repo/models/crad_model.py:147-164 -- h = Linear(C, 1), applied to GAP of the
       PFAC-weighted map; attached to the MRI encoder only (fea_kd_mode=1, train_test.py:210-214).
Default here: GAP of the raw map (paper); ``pool_input="pfac"`` reproduces the code.
TASK-BOUND: the auxiliary score is a clinical score normalised to [0, 1] (protocol.data.aux).
"""

import torch
from torch import nn

from .pfac import PFAC


def gap(z: torch.Tensor) -> torch.Tensor:
    return z.flatten(2).mean(-1)


class CAFR(nn.Module):
    def __init__(self, channels, layers, pool_input: str = "raw"):
        super().__init__()
        if pool_input not in ("raw", "pfac"):
            raise ValueError("cafr.pool_input must be 'raw' or 'pfac'")
        self.layers = list(layers)
        self.pool_input = pool_input
        self.pfac = PFAC()
        self.heads = nn.ModuleList([nn.Linear(channels[k], 1) for k in self.layers])

    def forward(self, features) -> torch.Tensor:
        """features: list of stage maps -> (B, len(layers)) predicted normalised aux score."""
        out = []
        for k, head in zip(self.layers, self.heads):
            z = features[k] if self.pool_input == "raw" else self.pfac(features[k])[0]
            out.append(head(gap(z)).squeeze(-1))
        return torch.stack(out, dim=1)
