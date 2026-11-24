"""Image loading (2D PNG or 3D NIfTI), dataset and collate function shared by all methods."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .manifest import Row

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGE_CACHE: dict = {}


def _normalize(x: torch.Tensor, mode: str) -> torch.Tensor:
    """x: (C, *S) float in [0, 1] for PNG, raw intensities for NIfTI."""
    if mode == "none":  # D-017: PPMI MRI is already brain z-scored and DaT is a reference ratio
        return x
    if mode == "imagenet":
        c = x.shape[0]
        mean = torch.tensor(IMAGENET_MEAN if c == 3 else (sum(IMAGENET_MEAN) / 3,) * c)
        std = torch.tensor(IMAGENET_STD if c == 3 else (sum(IMAGENET_STD) / 3,) * c)
        shape = (c,) + (1,) * (x.ndim - 1)
        return (x - mean.view(shape)) / std.view(shape)
    if mode == "zscore":  # per image, over non-zero (foreground) voxels
        fg = x[x != 0]
        fg = fg if fg.numel() > 1 else x.flatten()
        return (x - fg.mean()) / fg.std().clamp_min(1e-6)
    if mode == "minmax":
        return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)
    raise ValueError(f"Unknown intensity normalisation: {mode}")


def load_image(path: Path, channels: int, size, intensity: str, spatial_dims: int) -> torch.Tensor:
    """Return a float tensor of shape (channels, *size)."""
    suffixes = "".join(path.suffixes).lower()
    if spatial_dims == 2:
        with Image.open(path) as image:
            image = image.convert("RGB" if channels == 3 else "L")
            array = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(array.copy())
        x = x.permute(2, 0, 1) if x.ndim == 3 else x.unsqueeze(0)
    elif suffixes.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        array = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
        if array.ndim == 4:
            array = array[..., 0]
        x = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported 3D image format: {path}")
    if tuple(x.shape[1:]) != tuple(size):
        mode = "bilinear" if spatial_dims == 2 else "trilinear"
        x = F.interpolate(x.unsqueeze(0), size=tuple(size), mode=mode, align_corners=False).squeeze(0)
    if x.shape[0] != channels:
        x = x.mean(0, keepdim=True).repeat(channels, *([1] * spatial_dims))
    x = _normalize(x, intensity).contiguous()
    if not torch.isfinite(x).all():  # checked here (not in _normalize) so the message names the file
        raise FloatingPointError(f"Non-finite voxels in {path}")
    return x


class ImageRows(Dataset):
    """One item per manifest row. DaT is only opened when ``load_dat`` is True."""

    def __init__(self, rows: list[Row], protocol: dict, load_dat: bool):
        self.rows = rows
        self.load_dat = load_dat
        pre, data = protocol["preprocessing"], protocol["data"]
        self.size = tuple(pre["input_size"])
        self.dims = int(data["spatial_dims"])
        self.mri_channels, self.dat_channels = int(pre["mri_channels"]), int(pre["dat_channels"])
        self.intensity = pre["intensity"]
        self.aux_max = float(data["aux"]["max_value"])
        # Decoded images are shared by every dataset of the process (small datasets only).
        self.cache = _IMAGE_CACHE if data.get("cache_in_memory", True) else None

    def __len__(self):
        return len(self.rows)

    def _image(self, path: Path, channels: int) -> torch.Tensor:
        key = (str(path), channels, self.size, self.intensity)
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        image = load_image(path, channels, self.size, self.intensity, self.dims)
        if self.cache is not None:
            self.cache[key] = image
        return image

    def __getitem__(self, index):
        row = self.rows[index]
        item = dict(mri=self._image(row.mri_path, self.mri_channels), y=float(row.target),
                    aux=row.aux / self.aux_max if math.isfinite(row.aux) else math.nan,
                    unit=row.unit, subject_id=row.subject_id, visit_id=row.visit_id,
                    has_dat=bool(self.load_dat and row.dat_path is not None))
        if item["has_dat"]:
            item["dat"] = self._image(row.dat_path, self.dat_channels)
        return item


def collate(items: list[dict]) -> dict:
    batch = dict(mri=torch.stack([it["mri"] for it in items]),
                 y=torch.tensor([it["y"] for it in items], dtype=torch.float32),
                 aux=torch.tensor([it["aux"] for it in items], dtype=torch.float32),
                 has_dat=torch.tensor([it["has_dat"] for it in items], dtype=torch.bool),
                 unit=[it["unit"] for it in items], subject_id=[it["subject_id"] for it in items])
    batch["aux_mask"] = torch.isfinite(batch["aux"])
    if any(it["has_dat"] for it in items):
        reference = next(it["dat"] for it in items if it["has_dat"])
        batch["dat"] = torch.stack([it["dat"] if it["has_dat"] else torch.zeros_like(reference)
                                    for it in items])
    return batch


def make_loader(rows: list[Row], protocol: dict, load_dat: bool, shuffle: bool, seed: int,
                dataset: ImageRows | None = None) -> DataLoader:
    dataset = dataset or ImageRows(rows, protocol, load_dat)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(protocol["training"]["batch_size"]), shuffle=shuffle,
                      generator=generator, num_workers=int(protocol["training"]["num_workers"]),
                      collate_fn=collate, drop_last=False)


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
