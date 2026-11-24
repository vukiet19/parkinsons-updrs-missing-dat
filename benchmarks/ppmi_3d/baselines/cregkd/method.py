"""CReg-KD-dagger training logic plugged into the shared train.py / evaluate.py.

Stage "teacher" (Sec. 2.1 "the teacher model is firstly pre-trained"; orig_repo/run_teacher.py:178):
    L_T = L_task(teacher(MRI, DaT))
Stage "student" (Eq. 12; orig_repo/run_student_gkd.py:203-212), teacher frozen (eval, no grad):
    L_S = L_task + lambda_1 * mean_k[psi_k (y_S - y_T)^2] + lambda_2 * sum_l mean[psi (phi(f(h^S_l)) - phi(h^T_l))^2]
    psi_k = 1 - clip(|y_T - y|_z / eta)                                                       (Eq. 8)
Distillation terms use only rows whose DaT is available; other rows get the task loss only.
"""

from __future__ import annotations

import torch

from losses.task_losses import task_loss_per_sample

from .losses import gate, hint_loss, output_kd, teacher_hints
from .model import CRegStudent, CRegTeacher


class CRegKDMethod:
    primary_role = "student"
    teacher_sections = ("teacher",)   # config keys that influence teacher training (train.py --teacher_from)

    def __init__(self, protocol: dict, cfg: dict):
        self.protocol, self.cfg = protocol, cfg
        self.kd_enabled = bool(cfg["kd"]["enabled"])
        self.stages = ("teacher", "student") if self.kd_enabled else ("student",)

    # ---- construction / data needs -------------------------------------------------------
    def build(self, stage: str) -> torch.nn.Module:
        return CRegTeacher(self.protocol, self.cfg) if stage == "teacher" else CRegStudent(self.protocol, self.cfg)

    def train_rows(self, stage: str, rows):
        if stage == "teacher" or not self.cfg["student"]["train_on_samples_without_dat"]:
            return [r for r in rows if r.dat_path is not None]
        return rows

    def needs_dat(self, stage: str, phase: str) -> bool:
        if stage == "teacher":
            return True
        return phase == "train" and self.kd_enabled          # the student never opens DaT at evaluation

    def predict(self, stage: str, model, batch) -> torch.Tensor:
        if stage == "teacher":
            return model(batch["mri"], batch["dat"])["pred"]
        return model(batch["mri"])["pred"]                     # AFR is not on the inference path

    # ---- losses ---------------------------------------------------------------------------
    def training_step(self, stage: str, model, frozen: dict, batch: dict):
        if stage == "teacher":
            out = model(batch["mri"], batch["dat"])
            task = task_loss_per_sample(self.protocol, out["pred"], batch["y_z"]).mean()
            return task, dict(task=task)
        return self._student_step(model, frozen.get("teacher"), batch)

    def _student_step(self, student, teacher, batch):
        cfg, kd = self.cfg, self.cfg["kd"]
        out = student(batch["mri"])
        task = task_loss_per_sample(self.protocol, out["pred"], batch["y_z"]).mean()
        if not self.kd_enabled:
            return task, dict(task=task)
        zero = task.new_zeros(())
        layers = list(kd["layers"])
        terms = dict(task=task, kd_output=zero, kd_hint=zero, gate_mean=zero, gate_zero_frac=zero,
                     **{f"kd_hint_l{l + 1}": zero for l in layers})
        has_dat = batch["has_dat"]
        if not has_dat.any():
            return task, terms
        idx = has_dat.nonzero(as_tuple=True)[0]
        with torch.no_grad():
            out_t = teacher(batch["mri"][idx], batch["dat"][idx])
        pred_s, y = out["pred"][idx], batch["y_z"][idx]

        g = cfg["gate"]
        psi = gate(out_t["pred"], y, float(g["eta"])) if g["enabled"] else torch.ones_like(y)

        l_out = (psi * output_kd(pred_s, out_t["pred"], kd["output_form"])).mean()      # weight_func, :70-75

        feats_s = [out["feats"][l][idx] for l in layers]
        refined = student.afr(feats_s) if student.afr is not None else feats_s           # Eq. 10
        hints = teacher_hints(out_t["feats"], out_t["feats_dat"], cfg["afr"]["teacher_hint"])
        l_hint, per_layer = hint_loss(refined, [hints[l] for l in layers], psi, cfg["afr"]["phi"],
                                      kd["reduce_layers"])

        loss = task + float(kd["lambda_1"]) * l_out + float(kd["lambda_2"]) * l_hint       # Eq. 12
        terms.update(kd_output=l_out, kd_hint=l_hint, gate_mean=psi.mean(),
                     gate_zero_frac=(psi <= 0).float().mean(),
                     **{f"kd_hint_l{l + 1}": v for l, v in zip(layers, per_layer)})
        return loss, terms
