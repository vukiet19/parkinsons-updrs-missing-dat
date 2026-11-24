"""Backbone factory driven by ``protocol.backbone``.

Every encoder of every method is created here, so they all share one initialisation (STEWARD D-020):
    3D, pretrained: none                     -> Kaiming initialisation (unchanged since P0a)
    3D, pretrained: medicalnet_r18_23dataset -> MedicalNet resnet_18_23dataset.pth (D-023), needs
                                                backbone.weights_path and backbone.weights_sha256
    2D                                       -> torchvision (pretrained: imagenet | none), unchanged
"""

from __future__ import annotations

import torch

from utils.config import resolve_path

from .resnet import MEDICALNET_R18, ResNetEncoder, load_medicalnet, medicalnet_state_dict


def _medicalnet_source(cfg: dict) -> tuple[str, str]:
    path, sha = cfg.get("weights_path"), cfg.get("weights_sha256")
    if not path or not sha:
        raise ValueError(f"backbone.pretrained={MEDICALNET_R18} needs backbone.weights_path and backbone.weights_sha256")
    return str(resolve_path(path)), str(sha)


def build_backbone(protocol: dict, in_channels: int) -> ResNetEncoder:
    cfg = protocol["backbone"]
    if cfg["name"] != "resnet18":
        raise ValueError(f"Unsupported backbone: {cfg['name']}")
    dims = int(protocol["data"]["spatial_dims"])
    pretrained = cfg.get("pretrained", "none")
    medicalnet = dims == 3 and pretrained == MEDICALNET_R18
    # D-024: shortcut defaults to "B" so protocols without the key (NTUA) build the original network.
    # The encoder is first built exactly as with pretrained: none (same RNG use, so heads built afterwards
    # get the same initialisation in the pretrained protocol and in the scratch control), then overwritten.
    encoder = ResNetEncoder(spatial_dims=dims, in_channels=in_channels,
                            pretrained="none" if medicalnet else pretrained, shortcut=cfg.get("shortcut", "B"))
    encoder.load_report = None
    if medicalnet:
        encoder.load_report = load_medicalnet(encoder, *_medicalnet_source(cfg))
    return encoder


def backbone_load_report(protocol: dict) -> dict | None:
    """Load report recorded in run.json (None without MedicalNet): file, SHA-256, tensors loaded/skipped.

    Computed for a 1-channel reference encoder from the cached file; an encoder with in_channels != 1 is
    adapted by load_medicalnet (logged when it is built) following ``in_channels_rule``."""
    cfg = protocol["backbone"]
    if not (int(protocol["data"]["spatial_dims"]) == 3 and cfg.get("pretrained", "none") == MEDICALNET_R18):
        return None
    path, sha = _medicalnet_source(cfg)
    mapped, skipped = medicalnet_state_dict(path, sha)
    with torch.random.fork_rng(devices=[]):        # building the reference must not move the global RNG
        reference = ResNetEncoder(3, 1, "none", shortcut=cfg.get("shortcut", "B"))
    own = reference.state_dict()
    return dict(source=MEDICALNET_R18, path=path, sha256=sha.lower(), n_encoder_tensors=len(own),
                n_loaded=sum(k in mapped for k in own), n_skipped=len(skipped), skipped=skipped,
                kept_initial=[k for k in own if k not in mapped],
                in_channels_rule="stem kernel repeated over input channels / in_channels when in_channels != 1 [Assumption]")


__all__ = ["ResNetEncoder", "build_backbone", "backbone_load_report", "MEDICALNET_R18"]
