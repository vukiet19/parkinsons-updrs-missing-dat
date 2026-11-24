"""Seeding, device choice and provenance hashes."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT


def seed_everything(seed: int, deterministic: bool = True) -> None:
    # Must be set before the first cuBLAS call for deterministic matmuls.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    # warn_only: a few CUDA kernels (e.g. max-pool backward) have no deterministic variant.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; use --device cpu.")
    return torch.device(name)


def source_hash() -> str:
    """Hash every .py and .yaml file of the implementation (not data, runs or notes)."""
    digest = hashlib.sha256()
    for folder in ("baselines", "configs", "data", "losses", "models", "scripts", "utils"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*")):
            if path.suffix in {".py", ".yaml"} and "__pycache__" not in path.parts:
                digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    for name in ("train.py", "evaluate.py", "infer.py", "export_inference.py", "compare.py"):
        digest.update(name.encode())
        digest.update((PROJECT_ROOT / name).read_bytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_run_checkpoint(path: Path, meta: dict, stage: str, checkpoint: str = "best") -> dict:
    info = meta["stages"][stage]
    if checkpoint == "last":
        info = info.get("last", {})
    expected_sha = info.get("checkpoint_sha256")
    if expected_sha is not None and file_sha256(path) != expected_sha:
        raise ValueError(f"{path}: checkpoint SHA-256 mismatch.")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    keys = ("method", "stage", "fold", "seed")
    identity = (meta["method"], stage, meta["fold"], meta["seed"])
    if (tuple(ckpt.get(k) for k in keys) != identity
            or ("data_fingerprint" in ckpt and ckpt["data_fingerprint"] != meta["data_fingerprint"])):
        raise ValueError(f"{path}: checkpoint identity mismatch.")
    return ckpt
