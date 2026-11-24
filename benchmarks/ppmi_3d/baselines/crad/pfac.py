"""Parameter-Free Attention Converter (PFAC).

Paper: Eq. 9 and Fig. 4 -- channel-normalise Z, GAP and GMP, combine, M = Sigmoid(diag(Z' Z'^T)).
Code:  orig_repo/models/crad_model.py:34-50 (``ChannelAttention_wn``, the default "PFAC_wn",
       main.py:19). The ReLU after GAP+GMP comes from the code (no-op for non-negative features).
diag(q q^T) == q**2, so the C x C matrix is never materialised. Works for 2D and 3D maps.
"""

import torch
import torch.nn.functional as F
from torch import nn


class PFAC(nn.Module):
    def forward(self, z: torch.Tensor):
        """z: (B, C, *S) -> (M * z, M) with M: (B, C, 1, ..., 1). No trainable parameters."""
        normalized = F.normalize(z, p=2, dim=1)                    # crad_model.py:44
        spatial = tuple(range(2, z.ndim))
        q = F.relu(normalized.mean(spatial) + normalized.amax(spatial))   # GAP + GMP, :45-47
        weights = torch.sigmoid(q * q)                              # diag(q q^T), :48-50
        weights = weights.view(*weights.shape, *([1] * len(spatial)))
        return weights * z, weights
