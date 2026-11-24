"""DFTD-dagger training logic plugged into the shared train.py / evaluate.py.

DFTD trains its teacher and student branches TOGETHER in one model (no frozen pre-trained teacher), so
``--teacher_from`` does not apply. Stages follow Alg. 1 and split the training data by DaT availability:

  "complete"   (Alg. 1 lines 2-12) subjects with MRI + DaT:
      L = w_rel L_rel (Eq. 6) + w_spe [L_MI^spe (Eq. 7) + w_adv (BCE of D_s^a + D_s^b) (Eq. 12-14)]
          + L_task(y_s) + L_task(y_t) + alpha L_P + beta L_F (Eq. 21) + w_lit L_LIT (Eq. 15)   (Eq. 22-23)
  "incomplete" (Alg. 1 lines 13-19) subjects with MRI only; starts from the "complete" weights,
      M_RAD fixed, f_b = 0, c'_b = U_LIT(c_a):   L = L_task(y_s) + L_task(y_t) + alpha L_P + beta L_F
      Skipped by train.py when the training fold has no such subject (NTUA: every subject has DaT).

Gradient routing ``gradient_routing``:
  "alg1" (default, [Paper] Alg. 1): each formula updates only the parameters listed in Alg. 1 --
      Eq. 6 -> F, H_c, S_c (line 5); Eq. 14 -> F, H_s, S_s, D_s, not H_c (line 7; Sec. III-B.2 "c_a has
      already been calculated"), so c is detached in L_spe; Eq. 15 -> E, D only (line 9), so c_a, c_b are
      detached in L_LIT; Eq. 22 -> M_IID only (line 11), so f_a, f_b, c_a enter M_IID detached.
  "joint" (sensitivity): Eq. 23 minimised end-to-end, every loss reaches every input it depends on.

TASK-BOUND conversions (regression): L_CE -> protocol task loss on the z-scored target; soft-label KD
D_KL(p_s || p_t) with temperature K -> (y_s - y_t)^2 (KL between equal-variance Gaussians; K drops out).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.task_losses import task_loss_per_sample

from .mi import negative_index
from .model import DFTDModel, DFTDStudentOnly

STUDENT_ALIASES = ("complete", "incomplete", "student")


class DFTDMethod:
    primary_role = "student"

    def __init__(self, protocol: dict, cfg: dict):
        self.protocol, self.cfg = protocol, cfg
        self.mode = cfg["mode"]
        if self.mode == "full":
            self.stages = ("complete", "incomplete") if cfg["stage2"]["enabled"] else ("complete",)
            self.eval_roles = (("student", False), ("teacher", True), ("teacher_imputed", False))
        elif self.mode == "student_only":
            self.stages = ("student",)
            self.eval_roles = (("student", False),)
        else:
            raise ValueError(f"Unknown dftd mode: {self.mode}")
        if cfg["gradient_routing"] not in ("alg1", "joint"):
            raise ValueError(f"Unknown gradient_routing: {cfg['gradient_routing']}")

    # ---- construction / data needs -------------------------------------------------------
    def build(self, stage: str) -> torch.nn.Module:
        if self.mode == "student_only":
            return DFTDStudentOnly(self.protocol, self.cfg)
        model = DFTDModel(self.protocol, self.cfg)
        if stage == "incomplete":
            model.freeze_rad()
        return model

    def init_stage(self, stage: str, model, frozen: dict) -> None:
        """Stage 2 continues from the best stage-1 weights (Alg. 1: the same model is fine-tuned)."""
        if stage == "incomplete":
            model.load_state_dict(frozen["complete"].state_dict())
            model.freeze_rad()

    def train_rows(self, stage: str, rows):
        if stage == "complete":
            paired = [r for r in rows if r.dat_path is not None]
            if not paired:
                raise ValueError("DFTD full training requires at least one MRI-DaT pair in the training fold.")
            return paired
        if stage == "incomplete":
            return [r for r in rows if r.dat_path is None]
        return rows

    def needs_dat(self, stage: str, phase: str) -> bool:
        return stage in ("complete", "teacher") and (phase == "train" or stage == "teacher")

    # ---- inference --------------------------------------------------------------------------
    def predict(self, role: str, model, batch) -> torch.Tensor:
        if self.mode == "student_only":
            return model(batch["mri"])
        ea = model.enc_a(batch["mri"])
        f_a = torch.cat([ea["c"], ea["s"]], 1)
        if role in STUDENT_ALIASES:
            return model.iid.student(f_a)[0]
        if role == "teacher":                       # real DaT where available, zero f_b otherwise
            eb = model.enc_b(batch["dat"])
            f_b = torch.cat([eb["c"], eb["s"]], 1) * batch["has_dat"][:, None].float()
        elif role == "teacher_imputed":             # MRI only: f_b = 0, c'_b = U_LIT(c_a)
            f_b = torch.zeros_like(f_a)
        else:
            raise ValueError(f"Unknown DFTD role: {role}")
        return model.iid(f_a, f_b, ea["c"])["pred_t"]

    # ---- losses -----------------------------------------------------------------------------
    def training_step(self, stage: str, model, frozen: dict, batch: dict):
        y = batch["y_z"]
        if self.mode == "student_only":
            task = task_loss_per_sample(self.protocol, model(batch["mri"]), y).mean()
            return task, dict(task_s=task)
        if stage == "complete":
            return self._complete_step(model, batch)
        if stage == "incomplete":
            return self._incomplete_step(model, batch)
        raise ValueError(stage)

    def _iid_terms(self, out: dict, y: torch.Tensor) -> dict:
        kd = self.cfg["kd"]
        task_s = task_loss_per_sample(self.protocol, out["pred_s"], y).mean()
        task_t = task_loss_per_sample(self.protocol, out["pred_t"], y).mean()
        detach = bool(kd["detach_teacher"])
        target = out["pred_t"].detach() if detach else out["pred_t"]
        if kd["output_kd"] == "mse":
            l_p = (out["pred_s"] - target).square().mean()
        elif kd["output_kd"] == "l1":
            l_p = (out["pred_s"] - target).abs().mean()
        else:
            raise ValueError(f"Unknown output_kd: {kd['output_kd']}")
        zero = task_s.new_zeros(())
        l_f = zero
        for i in kd["fd_layers"]:                  # Eq. 21: sum_i || phi(o_i^t) - phi(o_i^s) ||_2
            o_t, o_s = out["t_feats"][i - 1], out["s_feats"][i - 1]
            o_t = o_t.detach() if detach else o_t
            l_f = l_f + (F.normalize(o_t, dim=1) - F.normalize(o_s, dim=1)).norm(dim=1).mean()
        return dict(task_s=task_s, task_t=task_t, kd_output=l_p, kd_feature=l_f)

    def _iid_loss(self, terms: dict) -> torch.Tensor:
        kd = self.cfg["kd"]
        return terms["task_s"] + terms["task_t"] + float(kd["alpha"]) * terms["kd_output"] \
            + float(kd["beta"]) * terms["kd_feature"]

    def _complete_step(self, model, batch):
        cfg, w = self.cfg, self.cfg["weights"]
        if not bool(batch["has_dat"].all()):
            raise ValueError("Stage 'complete' received rows without DaT.")
        alg1 = cfg["gradient_routing"] == "alg1"
        est = cfg["critic"]["estimator"]
        ea, eb = model.enc_a(batch["mri"]), model.enc_b(batch["dat"])
        k, valid = negative_index(batch["subject_id"], cfg["critic"]["negatives"], batch["mri"].device)
        wa, wb = model.w_a(), model.w_b()

        # Eq. 3-6: inter-modality relevant representations share information with the other modality's regions.
        # 1/N sum_i (w_ai L_i + w_bi L_i) is written as two means, so MRI and DaT may use different grids.
        l_rel = ((wa * model.S_c_b(eb["c"], ea["r"], k, valid, est)).mean()
                 + (wb * model.S_c_a(ea["c"], eb["r"], k, valid, est)).mean())
        # Eq. 7-9: the global representation f = [c, s] carries its own regions
        c_a, c_b = (ea["c"].detach(), eb["c"].detach()) if alg1 else (ea["c"], eb["c"])
        f_a_spe, f_b_spe = torch.cat([c_a, ea["s"]], 1), torch.cat([c_b, eb["s"]], 1)
        l_spe_mi = ((wa * model.S_s_a(f_a_spe, ea["r"], k, valid, est)).mean()
                    + (wb * model.S_s_b(f_b_spe, eb["r"], k, valid, est)).mean())
        # Eq. 11-14: adversarial minimisation of I(c; s) (gradient reversal on s, and on c if joint)
        l_adv = (model.D_s_a(c_a, ea["s"], k, valid, reverse_c=not alg1)
                 + model.D_s_b(c_b, eb["s"], k, valid, reverse_c=not alg1))

        # Eq. 16-22: M_IID on (f_a, f_b, c'_b)
        f_a, f_b = torch.cat([ea["c"], ea["s"]], 1), torch.cat([eb["c"], eb["s"]], 1)
        c_in = ea["c"]
        if alg1:
            f_a, f_b, c_in = f_a.detach(), f_b.detach(), c_in.detach()
        out = model.iid(f_a, f_b, c_in, bool(cfg["kd"]["detach_student_into_teacher"]))
        terms = self._iid_terms(out, batch["y_z"])
        target = eb["c"].detach() if alg1 else eb["c"]
        l_lit = (out["cb_hat"] - target).norm(dim=1).mean()          # Eq. 15 (Euclidean distance)

        loss = (float(w["rel"]) * l_rel + float(w["spe"]) * (l_spe_mi + float(w["adv"]) * l_adv)
                + self._iid_loss(terms) + float(w["lit"]) * l_lit)
        terms.update(rel=l_rel, spe_mi=l_spe_mi, adv_bce=l_adv, lit=l_lit,
                     w_a_max=wa.detach().max(), w_b_max=wb.detach().max())
        return loss, terms

    def _incomplete_step(self, model, batch):
        with torch.no_grad():                      # M_RAD is fixed (Alg. 1 line 18 updates M_IID only)
            ea = model.enc_a(batch["mri"])
        f_a = torch.cat([ea["c"], ea["s"]], 1)
        out = model.iid(f_a, torch.zeros_like(f_a), ea["c"], bool(self.cfg["kd"]["detach_student_into_teacher"]))
        terms = self._iid_terms(out, batch["y_z"])  # no L_LIT: c_b does not exist for these subjects
        return self._iid_loss(terms), terms
