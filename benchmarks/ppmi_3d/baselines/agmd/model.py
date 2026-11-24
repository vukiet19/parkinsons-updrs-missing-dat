"""AGMD teacher (MRI + DaT) and student (MRI only) on the shared backbone/head.

Teacher: orig_repo/model.py:256-315 (Fig. 1a). Student: :318-349 ("the same workflow but without the CMT
module", Sec. 2.1). Backbone (code: 3D VGG-16, :46-97) and output head (code: 2-layer MLP classifier, :25-43)
are INTERFACE and come from the protocol; CMT, graph learning and the graph transformer are CORE.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.backbones import build_backbone
from models.heads import build_head

from .graph import BrainGraph


def _conv(dims: int):
    return nn.Conv2d if dims == 2 else nn.Conv3d


class CrossModalAttention(nn.Module):
    """Cross-Modal Attentive Fusion Transformer, one layer (Sec. 2.2, Eq. 1; orig_repo/model.py:109-141).

    Joint channel attention on F_MRI (+) F_PET (code: SE block on the sum, applied to both modalities);
    Q = Conv(F_MRI), K = V = Conv(F_PET); Attn = softmax(Q K^T / sqrt(C)) over PET positions;
    F_fused = Conv(Attn V + F_MRI), Conv = 3x3 conv C->C/2, ReLU, BN, 1x1 conv C/2->C [Code].
    ``scale``: "sqrt_c" [Paper Eq. 1] | "none" [Code model.py:138].
    """

    def __init__(self, dims: int, c: int, scale: str = "sqrt_c"):
        super().__init__()
        conv = _conv(dims)
        bn = nn.BatchNorm2d if dims == 2 else nn.BatchNorm3d
        pool = nn.AdaptiveAvgPool2d if dims == 2 else nn.AdaptiveAvgPool3d
        self.query, self.key, self.value = conv(c, c, 1), conv(c, c, 1), conv(c, c, 1)
        self.channel_attn = nn.Sequential(pool(1), conv(c, c // 2, 1), nn.ReLU(), conv(c // 2, c, 1), nn.Sigmoid())
        self.conv1 = nn.Sequential(conv(c, c // 2, 3, padding=1), nn.ReLU(inplace=True), bn(c // 2),
                                   conv(c // 2, c, 1))
        if scale not in ("sqrt_c", "none"):
            raise ValueError(f"Unknown cmt.scale: {scale}")
        self.scale = 1.0 / math.sqrt(c) if scale == "sqrt_c" else 1.0
        self.last_attention = None            # kept only when ``keep_attention`` is set (checks)
        self.keep_attention = False

    def forward(self, mri: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        w = self.channel_attn(mri + pet)
        mri, pet = mri * w, pet * w
        q = self.query(mri).flatten(2)                    # (B, C, N)
        k = self.key(pet).flatten(2)
        v = self.value(pet).flatten(2)
        attn = torch.softmax(torch.einsum("bcn,bcm->bnm", q, k) * self.scale, dim=-1)
        if self.keep_attention:
            self.last_attention = attn.detach()
        fused = torch.einsum("bnm,bcm->bcn", attn, v).view_as(mri)
        return self.conv1(fused + mri)


class GraphReadout(nn.Module):
    """Graph transformer over the multiscale graphs, G = GraphTransformer(H^{l-1}, H^l) (Sec. 2.3), then the
    code's projection Linear(N*hidden -> 512) + ReLU (orig_repo/model.py:245-253).

    Rule D (orig_repo/model.py:248, 306, 340): the code feeds a 2D tensor (B, N*64) to a batch_first=False
    TransformerEncoder, i.e. one unbatched sequence whose tokens are the SUBJECTS of the batch, so every
    subject attends to the others and a prediction depends on the batch it is in. Fixed per the paper's text:
    the tokens are the graph nodes of ONE subject (batch_first=True), same depth/heads/FFN as the code.
    """

    def __init__(self, n_nodes: int, cfg: dict):
        super().__init__()
        t = cfg["transformer"]
        d = int(cfg["graph"]["gcn_hidden"])
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=int(t["nhead"]), dim_feedforward=int(t["dim_feedforward"]),
                                           dropout=float(t["dropout"]), batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(t["num_layers"]), enable_nested_tensor=False)
        self.down = nn.Sequential(nn.Linear(n_nodes * d, int(t["global_dim"])), nn.ReLU(inplace=True))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.down(self.encoder(tokens).flatten(1))


class _GraphBody(nn.Module):
    """Shared tail of teacher and student: graph learning -> graph transformer -> head."""

    def _init_body(self, protocol: dict, cfg: dict, channels, graph_layout=None):
        self.graph = BrainGraph(protocol, cfg, channels, layout=graph_layout)
        self.readout = GraphReadout(self.graph.n_nodes, cfg)
        self.head = build_head(protocol, int(cfg["transformer"]["global_dim"]))

    def _body(self, feats: dict) -> dict:
        # P1b 4.3: the layout precomputes node masks for resnet_stage_sizes(input_size) (stride-32 topology, D-024);
        # a different encoder topology or input size must fail loudly instead of pooling the wrong cells.
        lay = self.graph.layout
        for l, size in zip(lay.layers, lay.stage_sizes):
            got = list(feats[l].shape[2:])
            if got != list(size):
                raise ValueError(f"Feature map of stage {l} has size {got}, graph layout expects {list(size)}")
        tokens, adjs = self.graph(feats)
        g = self.readout(tokens)
        return dict(graph=tokens, global_=g, pred=self.head(g).squeeze(-1), adj=adjs)


class AGMDTeacher(_GraphBody):
    def __init__(self, protocol: dict, cfg: dict):
        super().__init__()
        dims = int(protocol["data"]["spatial_dims"])
        pre = protocol["preprocessing"]
        self.mri_encoder = build_backbone(protocol, int(pre["mri_channels"]))
        self.dat_encoder = build_backbone(protocol, int(pre["dat_channels"]))
        ch = self.mri_encoder.channels
        self.cmt_layers = [int(l) for l in cfg["cmt"]["layers"]]
        graph_layers = [int(l) for l in cfg["graph"]["layers"]]
        if not set(graph_layers) <= set(self.cmt_layers):
            raise ValueError("The teacher builds its graphs on fused features: graph.layers must be in cmt.layers")
        self.cmt = nn.ModuleDict({str(l): CrossModalAttention(dims, ch[l], cfg["cmt"]["scale"])
                                  for l in self.cmt_layers})
        self._init_body(protocol, cfg, ch)

    def forward(self, mri: torch.Tensor, dat: torch.Tensor) -> dict:
        fm, fp = self.mri_encoder(mri), self.dat_encoder(dat)
        fused = {}
        for l in self.cmt_layers:
            p = fp[l]
            if p.shape[2:] != fm[l].shape[2:]:      # different MRI/DaT input sizes (3D) [Assumption, as CRAD]
                mode = "bilinear" if p.ndim == 4 else "trilinear"
                p = F.interpolate(p, size=fm[l].shape[2:], mode=mode, align_corners=False)
            fused[l] = self.cmt[str(l)](fm[l], p)
        out = self._body(fused)
        out["feats"] = fused
        return out


class AGMDStudent(_GraphBody):
    def __init__(self, protocol: dict, cfg: dict, graph_layout=None):
        super().__init__()
        self.mri_encoder = build_backbone(protocol, int(protocol["preprocessing"]["mri_channels"]))
        self._init_body(protocol, cfg, self.mri_encoder.channels, graph_layout=graph_layout)

    def forward(self, mri: torch.Tensor) -> dict:
        f = self.mri_encoder(mri)
        out = self._body({l: f[l] for l in range(len(f))})
        out["feats"] = {l: f[l] for l in range(len(f))}
        return out
