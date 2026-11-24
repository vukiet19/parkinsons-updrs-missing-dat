"""ResNet-18 (He et al., CVPR 2016) exposing its four residual stages, in 2D or 3D.

2D: torchvision ``resnet18`` (optionally ImageNet weights). 3D: the same topology with
Conv3d/BatchNorm3d/MaxPool3d (random initialisation). ``avgpool``/``fc`` are not used.
Output: list of 4 feature maps with 64/128/256/512 channels at strides 4/8/16/32.

3D shortcut (STEWARD D-024), used where a block changes stride or width:
    "B" (default; NTUA runs): 1x1x1 conv + BatchNorm (parameter names downsample.0 / downsample.1).
    "A" (PPMI protocol): parameter-free -- stride-subsample (avg_pool3d, kernel 1) + zero-padded channels,
        the MedicalNet ResNet-18 shortcut. Unlike MedicalNet (models/resnet.py:downsample_basic_block
        concatenates ``out.data``, which cuts autograd), gradients DO flow through it (He et al. 2016).
2D (torchvision) only supports "B".

3D pretrained weights (STEWARD D-020, D-023): ``pretrained: medicalnet_r18_23dataset`` loads MedicalNet
(Tencent) ``resnet_18_23dataset.pth`` (Chen, Ma, Zheng, Med3D, arXiv:1904.00625) into EVERY encoder via
``load_medicalnet``. Known differences from the pretraining set-up, recorded and kept on purpose:
    - topology (D-024): MedicalNet layer3/layer4 use stride 1 + dilation 2/4 (overall stride 8); we keep
      stride 2 (overall stride 32). Kernel shapes are identical, so the weights load, but the receptive
      field of stages 3-4 differs from the pretrained network;
    - shortcut A: MedicalNet concatenates ``out.data`` (no gradient through the shortcut); ours keeps it;
    - intensity (D-017): MedicalNet z-scored non-zero voxels and filled the background with N(0, 1) noise;
      we feed the PPMI MRI z-score / DaT ratio as is (``intensity: none``).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

MEDICALNET_R18 = "medicalnet_r18_23dataset"
# Hard-coded MedicalNet -> ResNetEncoder name map (D-023). MedicalNet keys come from models/resnet.py of
# Tencent/MedicalNet (conv1/bn1/layer1..4), saved under DataParallel (prefix "module.", stripped first).
_MEDICALNET_PATTERNS = (
    (re.compile(r"^conv1\.weight$"), lambda m: "stem.0.weight"),
    (re.compile(r"^bn1\.(weight|bias|running_mean|running_var|num_batches_tracked)$"), lambda m: f"stem.1.{m[1]}"),
    (re.compile(r"^layer([1-4])\.([01])\.(conv1|bn1|conv2|bn2)\.(.+)$"),
     lambda m: f"stages.{int(m[1]) - 1}.{m[2]}.{m[3]}.{m[4]}"),
)
_MEDICALNET_SKIP = re.compile(r"^conv_seg\.")   # segmentation head of MedicalNet: not part of the encoder
_WEIGHTS_CACHE: dict[tuple[str, str], tuple[dict, list[str]]] = {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def medicalnet_state_dict(path, sha256: str) -> tuple[dict, list[str]]:
    """(mapped encoder state dict, skipped keys) of a MedicalNet checkpoint; checked and read once per process.

    Raises on a SHA-256 mismatch or on any key that is neither mapped nor ``conv_seg.*``."""
    path = Path(path).resolve()
    key = (str(path), str(sha256).lower())
    if key in _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE[key]
    if not path.is_file():
        raise FileNotFoundError(f"MedicalNet weights not found: {path}")
    actual = file_sha256(path)
    if actual != key[1]:
        raise ValueError(f"SHA-256 mismatch for {path}: file {actual}, protocol weights_sha256 {key[1]}")
    obj = torch.load(path, map_location="cpu", weights_only=True)   # never weights_only=False
    raw = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    mapped, skipped, unknown = {}, [], []
    for name, tensor in raw.items():
        name = name[len("module."):] if name.startswith("module.") else name
        if _MEDICALNET_SKIP.match(name):
            skipped.append(name)
            continue
        for pattern, target in _MEDICALNET_PATTERNS:
            match = pattern.match(name)
            if match:
                mapped[target(match)] = tensor
                break
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"Unexpected keys in {path} (not mapped, not conv_seg.*): {unknown}")
    _WEIGHTS_CACHE[key] = (mapped, skipped)
    return mapped, skipped


def load_medicalnet(encoder: "ResNetEncoder", path, sha256: str) -> dict:
    """Strictly copy MedicalNet weights into a 3D ``encoder``; return the load report (D-020, D-023).

    Every parameter and buffer of the encoder must come from the file with the same shape, except a
    ``num_batches_tracked`` absent from the file (then kept at its initial value, listed in the report).
    [Assumption] in_channels != 1 (e.g. CReg-KD early fusion MRI (+) DaT): the 1-channel stem kernel is
    repeated over the input channels and divided by in_channels, the rule of the 2D ImageNet branch."""
    if encoder.spatial_dims != 3:
        raise ValueError("MedicalNet weights are 3D; the 2D encoder uses torchvision/ImageNet.")
    mapped, skipped = medicalnet_state_dict(path, sha256)
    own = encoder.state_dict()
    state, kept_init = {}, []
    adapted = None
    for name, current in own.items():
        if name not in mapped:
            if name.endswith("num_batches_tracked"):   # optional BN counter: keep the initial value, logged
                state[name] = current
                kept_init.append(name)
                continue
            raise ValueError(f"MedicalNet file has no tensor for encoder '{name}'")
        tensor = mapped[name]
        if name == "stem.0.weight" and tensor.shape[1] == 1 and current.shape[1] != 1:
            adapted = int(current.shape[1])
            tensor = tensor.repeat(1, adapted, 1, 1, 1) / adapted
        if tuple(tensor.shape) != tuple(current.shape):
            raise ValueError(f"Shape mismatch for '{name}': file {tuple(tensor.shape)}, encoder {tuple(current.shape)}")
        state[name] = tensor.to(current.dtype)
    extra = sorted(set(mapped) - set(own))
    if extra:
        raise ValueError(f"MedicalNet tensors with no encoder counterpart (shortcut must be 'A'): {extra}")
    encoder.load_state_dict(state, strict=True)
    report = dict(source=MEDICALNET_R18, path=str(Path(path).resolve()), sha256=str(sha256).lower(),
                  n_encoder_tensors=len(own), n_loaded=len(own) - len(kept_init), n_skipped=len(skipped),
                  skipped=skipped, kept_initial=kept_init, in_channels=int(own["stem.0.weight"].shape[1]),
                  in_channels_adapted=adapted is not None)
    print(f"backbone: MedicalNet {Path(path).name} (sha256 {report['sha256'][:12]}...) -> {report['n_loaded']}/"
          f"{report['n_encoder_tensors']} encoder tensors loaded, {report['n_skipped']} skipped {skipped or ''}"
          + (f", kept initial {kept_init}" if kept_init else "")
          + (f"; stem repeated over {adapted} input channels / {adapted} [Assumption]" if adapted else ""), flush=True)
    return report


class _ShortcutA3d(nn.Module):
    """Parameter-free shortcut: subsample by ``stride`` and zero-pad the missing channels (D-024)."""

    def __init__(self, cout: int, stride: int):
        super().__init__()
        self.cout, self.stride = cout, stride

    def forward(self, x):
        out = F.avg_pool3d(x, kernel_size=1, stride=self.stride)
        missing = self.cout - out.shape[1]
        if missing < 0:
            raise ValueError("shortcut A cannot reduce the number of channels")
        if missing:
            # No .data / detach: the zero channels are constants, the identity part keeps its gradient.
            out = torch.cat([out, out.new_zeros(out.shape[0], missing, *out.shape[2:])], dim=1)
        return out
from torchvision.models import ResNet18_Weights, resnet18


class _BasicBlock3d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, shortcut: str = "B"):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != cout:
            if shortcut == "A":
                self.downsample = _ShortcutA3d(cout, stride)
            elif shortcut == "B":   # unchanged: keeps the parameter names of every NTUA checkpoint
                self.downsample = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))
            else:
                raise ValueError(f"Unknown shortcut type: {shortcut!r} (expected 'A' or 'B')")

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + identity)


class ResNetEncoder(nn.Module):
    channels = (64, 128, 256, 512)

    def __init__(self, spatial_dims: int = 2, in_channels: int = 3, pretrained: str = "none",
                 shortcut: str = "B"):
        super().__init__()
        if shortcut not in ("A", "B"):
            raise ValueError(f"Unknown shortcut type: {shortcut!r} (expected 'A' or 'B')")
        if spatial_dims == 2:
            if shortcut != "B":   # torchvision ResNet-18 uses the projection (type B) shortcut
                raise ValueError("2D (torchvision) ResNet-18 only supports shortcut 'B'.")
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained == "imagenet" else None
            base = resnet18(weights=weights)
            if in_channels != 3:
                conv = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
                with torch.no_grad():  # sum over RGB keeps the response to a replicated grey image
                    conv.weight.copy_(base.conv1.weight.sum(1, keepdim=True).repeat(1, in_channels, 1, 1)
                                      / in_channels)
                base.conv1 = conv
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.stages = nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4])
        elif spatial_dims == 3:
            if pretrained not in ("none", None):   # MedicalNet is loaded by build_backbone after construction
                raise ValueError(f"3D pretrained must be 'none' here (got {pretrained!r}); "
                                 f"'{MEDICALNET_R18}' is loaded by models.backbones.build_backbone.")
            self.stem = nn.Sequential(nn.Conv3d(in_channels, 64, 7, 2, 3, bias=False), nn.BatchNorm3d(64),
                                      nn.ReLU(inplace=True), nn.MaxPool3d(3, 2, 1))
            stages, cin = [], 64
            for cout, stride in zip(self.channels, (1, 2, 2, 2)):
                stages.append(nn.Sequential(_BasicBlock3d(cin, cout, stride, shortcut),
                                            _BasicBlock3d(cout, cout, 1, shortcut)))
                cin = cout
            self.stages = nn.ModuleList(stages)
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        else:
            raise ValueError("spatial_dims must be 2 or 3")
        self.spatial_dims = spatial_dims
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
