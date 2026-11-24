"""Load YAML configs, apply ``key.sub=value`` overrides and resolve project paths."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def config_hash(value: dict) -> str:
    """Hash a parsed configuration independent of YAML formatting and key order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def method_config_path(protocol: dict, method: str) -> Path:
    folder = PROJECT_ROOT / "configs" / "baselines"
    ppmi = folder / f"{method}_ppmi.yaml"
    return ppmi if "qc" in protocol["data"] and ppmi.is_file() else folder / f"{method}.yaml"


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    """Return a copy of ``config`` with ``a.b.c=value`` overrides (value parsed as YAML)."""
    config = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value: {item}")
        key, raw = item.split("=", 1)
        node = config
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise KeyError(f"Unknown config section '{part}' in override '{item}'")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key '{key}' (typo?)")
        node[parts[-1]] = yaml.safe_load(raw)
    return config


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def project_relative(path: str | Path) -> str:
    """Store paths inside the project relative to its root, so the folder can be moved/renamed."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
