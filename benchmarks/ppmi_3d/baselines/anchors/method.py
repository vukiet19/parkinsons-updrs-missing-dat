"""Common anchors plugged into the shared train.py / evaluate.py (interface: baselines/__init__.py).

mri_only   : one stage "student";  L = mean task loss on MRI-only predictions. Never opens DaT.
mm_teacher : one stage "teacher";  L = mean task loss on MRI + DaT predictions. A QC-approved cohort
             requires a pair for every row. Other protocols use paired rows only.
             Compare it with `compare.py --role_a teacher`.
No method-specific hyperparameters: configs/baselines/{mri_only,mm_teacher}.yaml only name the method.
"""

from __future__ import annotations

import torch

from losses.task_losses import task_loss_per_sample

from .model import MMTeacherNet, MRIOnlyNet


class MRIOnlyMethod:
    stages = ("student",)
    primary_role = "student"

    def __init__(self, protocol: dict, cfg: dict):
        self.protocol, self.cfg = protocol, cfg

    def build(self, stage: str) -> torch.nn.Module:
        return MRIOnlyNet(self.protocol)

    def train_rows(self, stage: str, rows):
        return rows

    def needs_dat(self, stage: str, phase: str) -> bool:
        return False                                      # MRI only, in training and at evaluation

    def training_step(self, stage: str, model, frozen: dict, batch: dict):
        task = task_loss_per_sample(self.protocol, model(batch["mri"])["pred"], batch["y_z"]).mean()
        return task, dict(task=task)

    def predict(self, stage: str, model, batch) -> torch.Tensor:
        return model(batch["mri"])["pred"]


class MMTeacherMethod:
    stages = ("teacher",)
    primary_role = "teacher"
    teacher_sections = ()       # the config has no hyperparameters; --teacher_from compares protocol/split/scaler

    def __init__(self, protocol: dict, cfg: dict):
        self.protocol, self.cfg = protocol, cfg

    def build(self, stage: str) -> torch.nn.Module:
        return MMTeacherNet(self.protocol)

    def train_rows(self, stage: str, rows):
        # INTERFACE: the shared trainer and evaluator both use this hook for the teacher role.
        if "qc" in self.protocol["data"]:
            missing = sum(row.mri_path is None or row.dat_path is None for row in rows)
            if missing:
                raise ValueError(f"mm_teacher data contract: {missing} QC-approved row(s) lack an MRI–DaT pair.")
            return rows
        return [r for r in rows if r.dat_path is not None]

    def needs_dat(self, stage: str, phase: str) -> bool:
        return True

    def training_step(self, stage: str, model, frozen: dict, batch: dict):
        # TASK-BOUND CORE: regress the protocol's standardized clinical target.
        task = task_loss_per_sample(self.protocol, model(batch["mri"], batch["dat"])["pred"], batch["y_z"]).mean()
        return task, dict(task=task)

    def predict(self, stage: str, model, batch) -> torch.Tensor:
        return model(batch["mri"], batch["dat"])["pred"]
