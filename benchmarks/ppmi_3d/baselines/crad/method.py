"""CRAD-dagger training logic plugged into the shared train.py / evaluate.py.

Phase 1 (teacher, Alg. 1 lines 1-12):  L_T = L_task + gamma_1 L_CAFR (Eq. 2) + gamma_2 L_ORT (Eq. 3)
Phase 2 (student, Alg. 1 lines 13-21): teacher frozen (eval, no grad);
    L_S = L_task + lambda * [ w_a sum_k S^k * L_F^k  +  psi S^C * (w_p L_P + w_u L_U) ]
          (Eq. 8 weighting, dual SDU of Eq. 15; see notes/CRAD_adaptation.md)
"""

from __future__ import annotations

import math

import torch

from losses.task_losses import task_loss_per_sample

from .losses import attention_kd, cafr_loss, gram_kd, orthogonal_loss, output_kd
from .model import CRADStudent, CRADTeacher
from .pfac import PFAC
from .sdu import layer_confidence, output_confidence, psi


ATTENTION_CONFIDENCE = ("cafr", "output", "none")


class CRADMethod:
    primary_role = "student"

    def __init__(self, protocol: dict, cfg: dict):
        self.protocol, self.cfg = protocol, cfg
        self.kd_enabled = bool(cfg["kd"]["enabled"])
        self.stages = ("teacher", "student") if self.kd_enabled else ("student",)
        self.pfac = PFAC()
        self._aux_reported: set = set()
        self._check_config()

    def _check_config(self) -> None:
        """Refuse configurations in which S^M would silently become 1 (STEWARD D-021, review Q8).

        New keys are read with .get so configs/baselines/crad.yaml (NTUA) behaves exactly as before."""
        sdu, cafr_on = self.cfg["sdu"], bool(self.cfg["cafr"]["enabled"])
        mode = sdu.get("attention_confidence", "cafr")
        if mode not in ATTENTION_CONFIDENCE:
            raise ValueError(f"sdu.attention_confidence must be one of {ATTENTION_CONFIDENCE}, got {mode!r}.")
        if sdu.get("enabled", True) and mode == "cafr" and not cafr_on:
            raise ValueError("sdu.attention_confidence='cafr' needs cafr.enabled=true (S^M comes from the CAFR heads). "
                             "For a CAFR-off variant, set sdu.attention_confidence to 'none' or 'output' "
                             "(code-faithful S^C*psi, variant sens_attn_output).")

    @property
    def report_name(self) -> str:
        return self.cfg.get("report_name", "CRAD†")

    # ---- construction / data needs -------------------------------------------------------
    def build(self, stage: str) -> torch.nn.Module:
        return CRADTeacher(self.protocol, self.cfg) if stage == "teacher" else CRADStudent(self.protocol, self.cfg)

    def train_rows(self, stage: str, rows):
        if stage == "teacher" or not self.cfg["student"]["train_on_samples_without_dat"]:
            rows = [r for r in rows if r.dat_path is not None]   # teacher learns from paired data only
            if stage == "teacher" and self.cfg["cafr"]["enabled"]:
                self._check_aux(rows)
        return rows

    def _check_aux(self, rows) -> None:
        """With CAFR on, the aux score (MoCA/MMSE) must exist: otherwise L_CAFR = 0 and S^M = 1 silently (D-021).
        Partly missing aux keeps the old per-sample masking but is reported."""
        if not rows:
            return
        missing = sum(1 for r in rows if not math.isfinite(r.aux))
        if missing == len(rows):
            raise ValueError("CAFR enabled but no paired teacher row has a finite aux_score. "
                             "Provide the paired-visit cognitive score (MoCA for OFF30, MMSE for NTUA).")
        key = (len(rows), missing)
        if missing and key not in self._aux_reported:   # train_rows is called several times per stage
            self._aux_reported.add(key)
            print(f"[crad] CAFR: {missing}/{len(rows)} teacher rows have no aux_score; they are masked "
                  f"(no L_CAFR, S^k = 1 for those samples).", flush=True)

    def describe(self, stage: str, model) -> list[str]:
        """One line for train.log (optional train.py hook): CAFR state and the attention-KD confidence source."""
        sdu, kd = self.cfg["sdu"], self.cfg["kd"]
        conf = sdu.get("attention_confidence", "cafr") if sdu.get("enabled", True) else "none (sdu off)"
        return [f"{self.report_name} | cafr={'on' if self.cfg['cafr']['enabled'] else 'off'} | "
                f"attention_confidence={conf} | psi={sdu['psi_mode']} | "
                f"fd={kd['fd_divergence']}({kd['fd_target']}) | kd={'on' if self.kd_enabled else 'off'}"]

    def needs_dat(self, stage: str, phase: str) -> bool:
        if stage == "teacher":
            return True
        return phase == "train" and self.kd_enabled   # the student never opens DaT at evaluation

    def predict(self, stage: str, model, batch) -> torch.Tensor:
        if stage == "teacher":
            return model(batch["mri"], batch["dat"])["pred"]
        return model(batch["mri"])["pred"]

    # ---- losses ---------------------------------------------------------------------------
    def training_step(self, stage: str, model, frozen: dict, batch: dict):
        if stage == "teacher":
            return self._teacher_step(model, batch)
        return self._student_step(model, frozen.get("teacher"), batch)

    def _teacher_step(self, teacher, batch):
        cfg = self.cfg
        out = teacher(batch["mri"], batch["dat"])
        task = task_loss_per_sample(self.protocol, out["pred"], batch["y_z"]).mean()
        zero = task.new_zeros(())
        l_cafr = (cafr_loss(out["cafr"], batch["aux"], batch["aux_mask"], cfg["cafr"]["loss"],
                            cfg["cafr"]["reduce_layers"]) if out["cafr"] is not None else zero)
        l_ort = orthogonal_loss(out["z_m"], out["z_p"], cfg["ort"]["form"]) if cfg["ort"]["enabled"] else zero
        t = cfg["teacher"]
        loss = task + float(t["gamma_cafr"]) * l_cafr + float(t["gamma_ort"]) * l_ort
        return loss, dict(task=task, cafr=l_cafr, ort=l_ort)

    def _student_step(self, student, teacher, batch):
        cfg, kd = self.cfg, self.cfg["kd"]
        out = student(batch["mri"])
        task = task_loss_per_sample(self.protocol, out["pred"], batch["y_z"]).mean()
        if not self.kd_enabled:
            return task, dict(task=task)
        zero = task.new_zeros(())
        terms = dict(task=task, kd_attention=zero, kd_output=zero, kd_feature=zero,
                     conf_attention=zero, conf_output=zero)
        has_dat = batch["has_dat"]
        if not has_dat.any():                      # KD only where the teacher can see DaT
            return task, terms
        idx = has_dat.nonzero(as_tuple=True)[0]
        with torch.no_grad():
            out_t = teacher(batch["mri"][idx], batch["dat"][idx])
        pred_s, y = out["pred"][idx], batch["y_z"][idx]

        # SDU weights (detached). High-level: psi * S^C (Eq. 11-13); intermediate: S^M (Eq. 14).
        sdu = cfg["sdu"]
        ones = torch.ones_like(y)
        if sdu["enabled"]:
            s_c = output_confidence(out_t["pred"], y, float(sdu["error_scale"])) if sdu["output_confidence"] else ones
            w_high = s_c * psi(out_t["pred"], pred_s.detach(), y, sdu["psi_mode"], float(sdu["eps"]))
        else:
            w_high = ones
        layers = list(kd["attention_layers"])
        w_att = torch.ones(len(idx), len(layers), device=y.device)
        mode = sdu.get("attention_confidence", "cafr") if sdu["enabled"] else "none"
        if mode == "cafr" and out_t["cafr"] is not None:
            s_m = layer_confidence(out_t["cafr"], batch["aux"][idx], batch["aux_mask"][idx])
            cafr_layers = list(cfg["cafr"]["layers"])
            for j, k in enumerate(layers):
                if k in cafr_layers:
                    w_att[:, j] = s_m[:, cafr_layers.index(k)]
        elif mode == "output":                     # code behaviour: S^C*psi for every KD term
            w_att = w_high[:, None].expand(-1, len(layers))

        att = w_att * attention_kd([f[idx] for f in out["feats"]], out_t["feats"], layers, self.pfac,
                                   kd["attention_distance"])            # (B, K)
        if kd["attention_reduce_layers"] == "sum":                      # code: sum over layers
            l_att = att.mean(0).sum()
        elif kd["attention_reduce_layers"] == "mean":                   # paper: 1/K
            l_att = att.mean()
        else:
            raise ValueError(f"Unknown attention_reduce_layers: {kd['attention_reduce_layers']}")
        l_out = (w_high * output_kd(pred_s, out_t["pred"], kd["output_kd"])).mean()
        div, tau = kd["fd_divergence"], float(kd["fd_tau"])
        if kd["fd_target"] == "fused":
            fd = gram_kd(out_t["z_u"], out["z_u"][idx], div, tau)
        elif kd["fd_target"] == "branches":
            fd = gram_kd(out_t["z_m"], out["z_m"][idx], div, tau) + gram_kd(out_t["z_p"], out["z_p"][idx], div, tau)
        else:
            raise ValueError(f"Unknown fd_target: {kd['fd_target']}")
        l_feat = (w_high * fd).mean()

        l_kd = float(kd["w_attention"]) * l_att + float(kd["w_output"]) * l_out + float(kd["w_feature"]) * l_feat
        loss = task + float(kd["lambda"]) * l_kd
        terms.update(kd_attention=l_att, kd_output=l_out, kd_feature=l_feat,
                     conf_attention=w_att.mean(), conf_output=w_high.mean())
        return loss, terms
