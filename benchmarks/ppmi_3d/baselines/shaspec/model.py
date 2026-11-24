"""ShaSpec network (Wang et al., CVPR 2023, Sec. 3.1-3.3; official code billhhh/ShaSpec @ 57a04fa).

Modules (paper notation -> here):
  f_sha   shared encoder, ONE network applied to every modality (Eq. 1)        -> ``shared``  (protocol ResNet-18)
  f_spec  one specific encoder per modality (Eq. 1)                             -> ``specific[m]`` (protocol ResNet-18)
  f_proj  projection of [r, s]; its output is added to r (Eq. 2, residual fusion) -> ``proj``
  f_dec   decoder on the fused features of ALL modality slots (Eq. 3)          -> ``head`` on concat(f_mri, f_dat)
  f_dco   domain (modality) classifier on specific features (Eq. 5)            -> ``dom_cls``
  missing modality n: f^(n) = 1/(N-1) sum_{i != n} r^(i) over available i (Eq. 4)

Official-code anchors (DualNet_SS.py @ 57a04fa):
  shared encoder runs on all modalities stacked into one batch      :399-400
  specific encoders                                                  :359-362, 417-420
  CompositionalLayer: conv(cat(r, s)) + r, no bias                   :312-338 (conv 3x3x3, 2C -> C)
  missing feature = shared feature of the 1st available modality     :411 (Eq. 4 mean is commented out, :412-414);
                    for N = 2 modalities both are identical (only one modality is available when one is missing)
  decoder input = concat of the N fused features (channel axis)      :487
  domain classifier Linear(C, N) on the pooled specific feature      :357, 422-426, 493-494

Fusion level (config ``fusion.type``):
  "linear" [Paper Sec. 3.1 "linear projection f_proj"; Sec. 4.2 classification: "fusion ... happens at the layer
           before the FC layers"]: r, s = GAP of the last backbone stage (C = 512), proj = Linear(2C, C).
  "conv"   [Code, BraTS segmentation]: fusion on the last-stage feature MAPS with a 3x3 (x3) conv, DAO on maps,
           then GAP before the head (sensitivity check).
"""

from __future__ import annotations

import torch
from torch import nn

from models.backbones import build_backbone
from models.heads.regression import build_head

MODALITIES = ("mri", "dat")       # fixed decoder slot order


def _gap(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).mean(2) if x.ndim > 2 else x


def _bcast(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return mask.view(-1, *([1] * (like.ndim - 1))).to(like.dtype)


class ShaSpecModel(nn.Module):
    def __init__(self, protocol: dict, cfg: dict, specific: tuple[str, ...] = MODALITIES, dom_cls: bool = True):
        super().__init__()
        pre = protocol["preprocessing"]
        channels = dict(mri=int(pre["mri_channels"]), dat=int(pre["dat_channels"]))
        if len(set(channels[m] for m in specific)) != 1:
            # f_sha is ONE network for every modality, so inputs must share a format (issue #15 of the
            # official repo: "align the dimension first").
            raise ValueError(f"ShaSpec needs equal channel counts for its shared encoder, got {channels}.")
        self.specific_modalities = tuple(specific)
        self.shared = build_backbone(protocol, channels["mri"])
        self.specific = nn.ModuleDict({m: build_backbone(protocol, channels[m]) for m in specific})
        dims = int(protocol["data"]["spatial_dims"])
        c = self.shared.channels[-1]
        f = cfg["fusion"]
        self.fusion = f["type"]
        if self.fusion == "linear":
            self.proj = nn.Linear(2 * c, c, bias=bool(f["bias"]))
        elif self.fusion == "conv":
            conv = nn.Conv2d if dims == 2 else nn.Conv3d
            self.proj = conv(2 * c, c, kernel_size=3, padding=1, bias=bool(f["bias"]))
        else:
            raise ValueError(f"Unknown fusion.type: {self.fusion}")
        self.missing_feature = cfg["missing_feature"]
        if self.missing_feature not in ("mean_shared", "first_available"):
            raise ValueError(f"Unknown missing_feature: {self.missing_feature}")
        self.dom_cls = nn.Linear(c, len(MODALITIES)) if dom_cls else None
        self.head = build_head(protocol, len(MODALITIES) * c)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Last-stage map (conv fusion) or its GAP vector (linear fusion)."""
        return x if self.fusion == "conv" else _gap(x)

    def forward(self, images: dict, avail: dict, enter: dict | None = None) -> dict:
        """images: {m: (B, C, *S)} (missing samples may hold anything; zero-filled when they enter).
        avail: {m: bool (B,)} modality m is available for the sample (used for fusion / Eq. 4).
        enter: {m: bool (B,)} sample passes through the encoders of m (default: = avail, i.e. the paper;
               the official code feeds zero-filled missing modalities too -> enter = all True).
        Returns pred (B,), shared r / specific s features per modality (zeros where not entered) and masks."""
        enter = avail if enter is None else enter
        mods = [m for m in MODALITIES if m in images]
        batch = avail["mri"].shape[0]
        x = {m: images[m] * _bcast(avail[m], images[m]) for m in mods}       # code DualNet_SS.py:389-392
        idx = {m: enter[m].nonzero().flatten() for m in mods}

        # Eq. 1 (shared): one forward over all modalities stacked in one batch (code :399-400).
        parts = [x[m].index_select(0, idx[m]) for m in mods if idx[m].numel()]
        shared_all = self._features(self.shared(torch.cat(parts))[-1])
        r, start = {}, 0
        for m in mods:
            n = idx[m].numel()
            r[m] = shared_all.new_zeros((batch,) + shared_all.shape[1:]).index_put((idx[m],), shared_all[start:start + n])
            start += n
        # Eq. 1 (specific)
        s = {}
        for m in self.specific_modalities:
            if m in mods and idx[m].numel():
                part = self._features(self.specific[m](x[m].index_select(0, idx[m]))[-1])
                s[m] = part.new_zeros((batch,) + part.shape[1:]).index_put((idx[m],), part)
            else:
                s[m] = shared_all.new_zeros((batch,) + shared_all.shape[1:])

        # Eq. 4: generated feature of a missing modality from the available shared features.
        avail_f = {m: avail[m].to(shared_all.dtype) if m in mods else avail["mri"].new_zeros(batch, dtype=shared_all.dtype)
                   for m in MODALITIES}
        ref = r[mods[0]]
        fused = []
        for m in MODALITIES:
            others = [k for k in MODALITIES if k != m and k in r]
            if self.missing_feature == "mean_shared":
                count = sum((_bcast(avail_f[k], ref) for k in others), torch.zeros_like(ref[:, :1]))
                generated = sum((_bcast(avail_f[k], ref) * r[k] for k in others), torch.zeros_like(ref))
                generated = generated / count.clamp_min(1.0)
            else:   # first available modality in slot order (code :411)
                generated, taken = torch.zeros_like(ref), torch.zeros_like(avail_f["mri"])
                for k in others:
                    pick = avail_f[k] * (1 - taken)
                    generated = generated + _bcast(pick, ref) * r[k]
                    taken = taken + pick
            if m in r and m in s:
                # Eq. 2: f = f_proj(r, s) + r (code CompositionalLayer :335-336)
                own = self.proj(torch.cat([r[m], s[m]], 1)) + r[m]
                a = _bcast(avail_f[m], ref)
                fused.append(a * own + (1 - a) * generated)
            else:
                fused.append(generated)
        no_input = sum(avail_f[m] for m in MODALITIES) == 0
        if bool(no_input.any()):
            raise ValueError("ShaSpec received a sample with no available modality.")
        # Eq. 3: decoder over the fused features of every slot (code :487 concatenates the slots)
        pred = self.head(torch.cat([_gap(f) for f in fused], 1)).squeeze(1)
        return dict(pred=pred, r=r, s=s, enter=enter, avail=avail)

    def domain_logits(self, s: torch.Tensor) -> torch.Tensor:
        """f_dco on the pooled specific feature (code :422-426, 493-494)."""
        return self.dom_cls(_gap(s))
