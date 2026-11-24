"""Shared regression head: pooled feature vector -> one real value (z-scored target)."""

from torch import nn


def build_head(protocol: dict, in_dim: int) -> nn.Module:
    cfg = protocol["head"]
    if cfg["type"] == "linear":
        return nn.Linear(in_dim, 1)
    if cfg["type"] == "mlp":
        return nn.Sequential(nn.Linear(in_dim, int(cfg["hidden"])), nn.ReLU(inplace=True),
                             nn.Dropout(float(cfg.get("dropout", 0.0))), nn.Linear(int(cfg["hidden"]), 1))
    raise ValueError(f"Unknown head type: {cfg['type']}")
