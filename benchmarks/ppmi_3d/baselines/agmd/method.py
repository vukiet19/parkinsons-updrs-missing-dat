"""AGMD-dagger training logic plugged into the shared train.py / evaluate.py.

Stage "teacher" (orig_repo/train_test.py:189-290): MRI + DaT teacher trained with the task loss only.
Stage "student" (train_test.py:60-186): teacher frozen (eval, no grad); per sample b with DaT
    L_S = L_task + mean_b[ sum_l w_l,b * lambda_l * L_l,b ],  l in {shallow, graph, global, logit}
      L_shallow = sum_{l in cmt.layers} MSE(F_fused^l, F_S^l)        Eq. 2   (code: mid + deep, :31-42)
      L_graph   = MSE(H_T, H_S) over all nodes of both graph layers   Eq. 7   (code :44-45)
      L_global  = 1 - cos(G_T, G_S)                                    Eq. 8   (code :47-51)
      L_logit   = (y_S - y_T)^2  -- code-only "distribution" term (KL of log-softmax, :53-55) [B, TASK-BOUND]
    with w from the Uncertainty-Aware Gating (Eq. 9-10, see uag.py).
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from losses.task_losses import task_loss_per_sample

from .model import AGMDStudent, AGMDTeacher
from .uag import mc_dropout_sigma, uag_weights


def per_sample_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).square().flatten(1).mean(1)


class AGMDMethod:
    primary_role = "student"
    # Config sections that change the teacher; --teacher_from requires them to be identical.
    teacher_sections = ("cmt", "graph", "transformer")

    def __init__(self, protocol: dict, cfg: dict):
        self.protocol, self.cfg = protocol, cfg
        self.kd_enabled = bool(cfg["kd"]["enabled"])
        self.stages = ("teacher", "student") if self.kd_enabled else ("student",)

    def build(self, stage: str) -> torch.nn.Module:
        return AGMDTeacher(self.protocol, self.cfg) if stage == "teacher" else AGMDStudent(self.protocol, self.cfg)

    def train_rows(self, stage: str, rows):
        if stage == "teacher" or not self.cfg["student"]["train_on_samples_without_dat"]:
            return [r for r in rows if r.dat_path is not None]
        return rows

    def needs_dat(self, stage: str, phase: str) -> bool:
        if stage == "teacher":
            return True
        return phase == "train" and self.kd_enabled

    def describe(self, stage: str, model) -> list[str]:
        """Lines written to train.log by train.py (optional hook, P1b): graph node mode, layers, atlas file + SHA-256
        and GraphLayout.node_info() (nodes, k, anatomical edges, key ROIs, key-ROI coverage per layer)."""
        info = model.graph.layout.node_info()
        uag = self.cfg["uag"]
        gate = f"UAG: mode={uag['mode']} reduce={uag['reduce']}"
        if uag["mode"] == "mc_dropout":
            gate += f" mc_samples={uag['mc_samples']} mc_scale={uag['mc_scale']} logit_error_scale={uag['error_scale']}"
        return [f"graph: nodes={info['mode']} layers={model.graph.layout.layers} cmt.layers={self.cfg['cmt']['layers']} "
                f"atlas={info['atlas']} sha256={info['atlas_sha256']}",
                "graph node_info: " + json.dumps(info["layers"]), gate]

    def predict(self, stage: str, model, batch) -> torch.Tensor:
        if stage == "teacher":
            return model(batch["mri"], batch["dat"])["pred"]
        return model(batch["mri"])["pred"]

    def training_step(self, stage: str, model, frozen: dict, batch: dict):
        if stage == "teacher":
            out = model(batch["mri"], batch["dat"])
            task = task_loss_per_sample(self.protocol, out["pred"], batch["y_z"]).mean()
            return task, dict(task=task)
        return self._student_step(model, frozen.get("teacher"), batch)

    def distill_terms(self, out_s: dict, out_t: dict) -> dict:
        """Per-sample distillation losses, each (B,). Teacher tensors are constants (computed under no_grad)."""
        kd = self.cfg["kd"]
        terms = {f"shallow_{l}": per_sample_mse(out_t["feats"][l], out_s["feats"][l])
                 for l in self.cfg["cmt"]["layers"]} if float(kd["w_shallow"]) > 0 else {}
        terms["graph"] = per_sample_mse(out_t["graph"], out_s["graph"])
        terms["global"] = 1.0 - F.cosine_similarity(out_t["global_"], out_s["global_"], dim=-1)
        terms["logit"] = (out_s["pred"] - out_t["pred"]).square()
        return terms

    def _student_step(self, student, teacher, batch):
        cfg, kd = self.cfg, self.cfg["kd"]
        out = student(batch["mri"])
        task = task_loss_per_sample(self.protocol, out["pred"], batch["y_z"]).mean()
        if not self.kd_enabled:
            return task, dict(task=task)
        zero = task.new_zeros(())
        logs = dict(task=task, kd_shallow=zero, kd_graph=zero, kd_global=zero, kd_logit=zero, uag_w=zero)
        uag = cfg["uag"]
        mc_enabled = uag["mode"] == "mc_dropout"
        if mc_enabled:
            logs.update(uag_sigma_mean=zero, uag_sigma_batch_max=zero)
        has_dat = batch["has_dat"]
        if not has_dat.any():                      # KD only where the teacher can see DaT
            return task, logs
        idx = has_dat.nonzero(as_tuple=True)[0]
        with torch.no_grad():
            out_t = teacher(batch["mri"][idx], batch["dat"][idx])
        out_s = {k: (v[idx] if torch.is_tensor(v) else {l: f[idx] for l, f in v.items()} if isinstance(v, dict) else v)
                 for k, v in out.items() if k != "adj"}
        y = batch["y_z"][idx]
        layers = [int(l) for l in cfg["cmt"]["layers"]]
        sigma = None
        if mc_enabled:
            sigma = mc_dropout_sigma(teacher, out_t["graph"], int(uag["mc_samples"]))
            logs.update(uag_sigma_mean=sigma.mean(), uag_sigma_batch_max=sigma.max())
        w = uag_weights(cfg, out_t, out_t["pred"], y, layers, sigma=sigma)
        terms = self.distill_terms(out_s, out_t)
        lam = dict(graph=float(kd["w_graph"]), logit=float(kd["w_logit"]) if kd["logit"] else 0.0,
                   **{"global": float(kd["w_global"])}, **{f"shallow_{l}": float(kd["w_shallow"]) for l in layers})
        weighted = {k: (lam[k] * w[k] * v).mean() for k, v in terms.items() if lam[k] > 0}
        l_kd = sum(weighted.values()) if weighted else zero
        loss = task + l_kd
        logs.update(kd_shallow=sum((v for k, v in weighted.items() if k.startswith("shallow_")), zero),
                    kd_graph=weighted.get("graph", zero), kd_global=weighted.get("global", zero),
                    kd_logit=weighted.get("logit", zero), uag_w=w["global"].mean())
        return loss, logs
