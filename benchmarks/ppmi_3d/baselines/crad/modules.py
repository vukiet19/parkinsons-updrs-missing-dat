"""CRAD projection / fusion blocks (dimension-agnostic copies of the original Conv3d blocks).

Teacher OP projections : orig_repo/models/crad_model.py:300-303 (Conv1x1, no bias, + ReLU per modality)
Fusion Fi              : orig_repo/models/crad_model.py:204-210 (Conv1x1 2C->C, Conv3x3, ReLU, Conv1x1);
                         the trailing AdaptiveAvgPool is applied by the caller so that the spatial
                         fused map Z^u stays available for Eq. 6 (Table A.1: Z^u is (B, 512, 392)).
Student projection     : orig_repo/models/crad_model.py:346-357 (two pseudo-modality branches
                         om_proj/op_proj = Conv3x3-ReLU-Conv1x1, then fim/fip = Conv1x1-ReLU-Conv1x1).
"""

from torch import nn


def conv(dims: int, cin: int, cout: int, kernel: int) -> nn.Module:
    layer = nn.Conv2d if dims == 2 else nn.Conv3d
    return layer(cin, cout, kernel, padding=kernel // 2, bias=False)


def teacher_projection(dims: int, c: int) -> nn.Module:
    return nn.Sequential(conv(dims, c, c, 1), nn.ReLU())


def fusion(dims: int, c: int) -> nn.Module:
    return nn.Sequential(conv(dims, 2 * c, c, 1), conv(dims, c, c, 3), nn.ReLU(), conv(dims, c, c, 1))


def student_projection(dims: int, c: int) -> nn.Module:
    return nn.Sequential(conv(dims, c, c, 3), nn.ReLU(), conv(dims, c, c, 1))


def student_refine(dims: int, c: int) -> nn.Module:
    return nn.Sequential(conv(dims, c, c, 1), nn.ReLU(), conv(dims, c, c, 1))
