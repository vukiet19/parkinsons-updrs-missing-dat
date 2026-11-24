"""kd_basic training logic plugged into the shared train.py / evaluate.py (interface: baselines/__init__.py).

Sources: Hinton et al., arXiv:1503.02531, Sec. 2 (output KD); Romero et al., ICLR 2015 (FitNets Secs. 2.2–2.3);
Lopez-Paz et al., ICLR 2016 (DaT as privileged information, argument only). Design: notes/KDBASIC_adaptation.md
(STEWARD D-041). Reference code (read only, not copied): HobbitLong/RepDistiller @ b84f547,
helper/loops.py:126-129 and :184, distiller_zoo/FitNet.py:6-14.

Stage "teacher": L_T = L_task(teacher(MRI, DaT)), rows with DaT only; then frozen (eval, no grad).
Stage "student" (one stage, no hint pre-training unless kd.hint_warmup_epochs > 0):
    L = L_task + lambda_out * mean[(y_S - y_T)^2] + lambda_hint * sum_k mean[(R_k(h^S_k) - h^T_k)^2]
y in z-units (D-006: Gaussian KL with fixed, equal variance is proportional to squared error).
h^T_k = feature map k of the teacher's MRI encoder, detached. Distillation terms use only rows with DaT.
Warm-up variant (FitNets two-stage, `sens_fitnets_two_stage`): during the first N epochs the loss is
lambda_hint * L_hint only; afterwards the full loss. The shared loop attempts one optimizer step per
batch, so epoch boundaries follow ceil(rows / protocol batch_size), including the last partial batch.
"""

from __future__ import annotations

import math

import torch

from losses.task_losses import task_loss_per_sample

from .model import KDBasicStudent, MMTeacherNet


class KDBasicMethod:
    primary_role = "student"
    teacher_sections = ("teacher",)   # config keys that influence teacher training (train.py --teacher_from)

    def __init__(self, protocol: dict, cfg: dict):
        self.protocol, self.cfg = protocol, cfg
        self.kd_enabled = bool(cfg["kd"]["enabled"])
        self.stages = ("teacher", "student") if self.kd_enabled else ("student",)
        self.warmup_epochs = int(cfg["kd"].get("hint_warmup_epochs", 0))
        self._steps_per_epoch: int | None = None   # set by train_rows("student", ...)
        self._student_batches = 0

    # ---- construction / data needs -------------------------------------------------------
    def build(self, stage: str) -> torch.nn.Module:
        if stage == "teacher":
            return MMTeacherNet(self.protocol)
        self._student_batches = 0
        return KDBasicStudent(self.protocol, self.cfg)

    def train_rows(self, stage: str, rows):
        if stage == "teacher" or not self.cfg["student"]["train_on_samples_without_dat"]:
            selected = [r for r in rows if r.dat_path is not None]
        else:
            selected = rows
        if stage == "student":
            batch_size = int(self.protocol["training"]["batch_size"])
            self._steps_per_epoch = max(1, math.ceil(len(selected) / batch_size))
        return selected

    def needs_dat(self, stage: str, phase: str) -> bool:
        if stage == "teacher":
            return True
        return phase == "train" and self.kd_enabled          # the student never opens DaT at evaluation

    def predict(self, stage: str, model, batch) -> torch.Tensor:
        if stage == "teacher":
            return model(batch["mri"], batch["dat"])["pred"]
        return model(batch["mri"])["pred"]                    # regressors are not on the inference path

    def describe(self, stage: str, model) -> list[str]:
        n = sum(p.numel() for p in model.parameters())
        if stage == "teacher":
            return [f"kd_basic teacher = mm_teacher architecture (MMTeacherNet), {n:,} parameters"]
        reg = sum(p.numel() for p in model.regressors.parameters())
        kd = self.cfg["kd"]
        return [f"kd_basic student = mri_only architecture + {reg:,} training-only regressor parameters "
                f"(hint stages {model.hint_layers}, channels {[model.encoder.channels[k] for k in model.hint_layers]}); "
                f"lambda_out={kd['lambda_out']} lambda_hint={kd['lambda_hint']} warmup_epochs={self.warmup_epochs}"
                if self.kd_enabled else f"kd_basic student with kd.enabled=false: plain mri_only, {n:,} parameters"]

    # ---- losses ---------------------------------------------------------------------------
    def training_step(self, stage: str, model, frozen: dict, batch: dict):
        if stage == "teacher":
            task = task_loss_per_sample(self.protocol, model(batch["mri"], batch["dat"])["pred"], batch["y_z"]).mean()
            return task, dict(task=task)
        return self._student_step(model, frozen.get("teacher"), batch)

    def _in_warmup(self) -> bool:
        if self.warmup_epochs <= 0:
            return False
        if self._steps_per_epoch is None:
            raise RuntimeError("hint_warmup_epochs > 0 needs train_rows('student', ...) to be called first")
        return self._student_batches // self._steps_per_epoch < self.warmup_epochs

    def _student_step(self, student, teacher, batch):
        if self.kd_enabled:
            warm = self._in_warmup()
            self._student_batches += 1
            if warm:
                return self._warmup_step(student, teacher, batch)

        kd = self.cfg["kd"]
        out = student(batch["mri"])
        task = task_loss_per_sample(self.protocol, out["pred"], batch["y_z"]).mean()
        if not self.kd_enabled:
            return task, dict(task=task)

        lambda_out, lambda_hint = float(kd["lambda_out"]), float(kd["lambda_hint"])
        zero = task.new_zeros(())
        terms = dict(task=task, kd_output=zero, kd_hint=zero, warmup=zero)
        if not lambda_out and not lambda_hint:
            return task, terms
        idx = batch["has_dat"].nonzero(as_tuple=True)[0]
        if not idx.numel():
            return task, terms

        with torch.no_grad():
            out_t = teacher(batch["mri"][idx], batch["dat"][idx])

        l_out = (out["pred"][idx] - out_t["pred"]).square().mean() if lambda_out else zero
        l_hint = self._hint_loss(student, out["feats"], out_t["feats"], idx) if lambda_hint else zero
        loss = task + lambda_out * l_out + lambda_hint * l_hint
        terms.update(kd_output=l_out.detach(), kd_hint=l_hint.detach())
        return loss, terms

    def _warmup_step(self, student, teacher, batch):
        zero = batch["y_z"].new_zeros(())
        terms = dict(task=zero, kd_output=zero, kd_hint=zero, warmup=zero.new_ones(()))
        idx = batch["has_dat"].nonzero(as_tuple=True)[0]
        lambda_hint = float(self.cfg["kd"]["lambda_hint"])
        if not idx.numel() or not student.hint_layers or not lambda_hint:
            # A standalone loss supports backward() without giving Adam any parameter gradients.
            return zero.detach().requires_grad_(), terms

        mri = batch["mri"][idx]
        with torch.no_grad():
            teacher_feats = teacher.mri_encoder(mri)
        l_hint = self._hint_loss(student, student.warmup_features(mri), teacher_feats)
        terms["kd_hint"] = l_hint.detach()
        return lambda_hint * l_hint, terms

    @staticmethod
    def _hint_loss(student, student_feats, teacher_feats, idx=None):
        # FitNets Eq. (3), using mean squared error and summing over the selected stages.
        regressed = student.regress(student_feats, idx)
        return sum(((r - teacher_feats[k]).square().mean()
                    for r, k in zip(regressed, student.hint_layers)), teacher_feats[0].new_zeros(()))
