"""Networks of the kd_basic anchor (STEWARD D-041).

Teacher : exactly the ``mm_teacher`` architecture, imported (not copied) from baselines.anchors.model:
          two encoders (MRI, DaT) -> GAP each -> concat -> shared head.
Student : exactly the ``mri_only`` architecture (encoder -> GAP of the last stage -> head), so that
          the benchmark can isolate the effect of KD. The added regressors R_k are used in training.

FitNets hint regressor R (Romero et al., ICLR 2015, Sec. 2.2: "a regressor on top of the guided layer"
trained with an L2 loss against the hint). Here R_k is a bias-free 1x1(x1) convolution C_k -> C_k
(RepDistiller ConvReg, `models/util.py:131-154`, supports grid alignment and uses batch norm and ReLU).
Student and teacher-MRI features share a grid. R_k is saved with the model and bypassed by predict().
"""

from __future__ import annotations

import torch
from torch import nn

from baselines.anchors.model import MMTeacherNet, MRIOnlyNet  # reused architectures, see module docstring

__all__ = ["MMTeacherNet", "KDBasicStudent"]


def build_regressor(spatial_dims: int, channels: int) -> nn.Module:
    """1x1 (2D) or 1x1x1 (3D) convolution C -> C without bias; dimensionality follows data.spatial_dims."""
    conv = {2: nn.Conv2d, 3: nn.Conv3d}.get(int(spatial_dims))
    if conv is None:
        raise ValueError(f"data.spatial_dims must be 2 or 3, got {spatial_dims}")
    return conv(channels, channels, kernel_size=1, bias=False)


class KDBasicStudent(MRIOnlyNet):
    """MRIOnlyNet + training-only hint regressors (one per hint layer)."""

    def __init__(self, protocol: dict, cfg: dict):
        super().__init__(protocol)
        kd = cfg["kd"]
        self.hint_layers = [int(k) for k in kd["hint_layers"]] if kd["enabled"] else []
        dims = int(protocol["data"]["spatial_dims"])
        self.regressors = nn.ModuleDict(
            {str(k): build_regressor(dims, int(self.encoder.channels[k])) for k in self.hint_layers})

    def warmup_features(self, mri: torch.Tensor) -> list[torch.Tensor]:
        """Run the encoder only through the deepest guided layer (FitNets Sec. 2.3)."""
        x = self.encoder.stem(mri)
        features = []
        for stage in self.encoder.stages[:max(self.hint_layers) + 1]:
            x = stage(x)
            features.append(x)
        return features

    def regress(self, feats: list[torch.Tensor], idx: torch.Tensor | None = None) -> list[torch.Tensor]:
        """R_k(h^S_k) for every hint layer k (optionally on the rows ``idx`` only)."""
        return [self.regressors[str(k)](feats[k] if idx is None else feats[k][idx]) for k in self.hint_layers]
