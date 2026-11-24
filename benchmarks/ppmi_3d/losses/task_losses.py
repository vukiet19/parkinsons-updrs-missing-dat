"""Task loss on the z-scored target, selected by ``protocol.loss.task_loss``."""

import torch
import torch.nn.functional as F


def task_loss_per_sample(protocol: dict, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the per-sample loss, shape (B,), so methods can weight or mask samples."""
    kind = protocol["loss"]["task_loss"]
    if kind == "mse":
        return (pred - target).square()
    if kind == "l1":
        return (pred - target).abs()
    if kind == "huber":
        return F.huber_loss(pred, target, reduction="none", delta=float(protocol["loss"]["huber_delta"]))
    raise ValueError(f"Unknown task loss: {kind}")
