"""ShaSpec-dagger training logic plugged into the shared train.py / evaluate.py.

ShaSpec is ONE multi-modal network that handles missing modalities by generating the missing fused feature
from the available shared features (Eq. 4). There is no separately trained teacher, hence no --teacher_from.
The reported MRI-only "student" is the same network evaluated with DaT missing; "teacher" = same network
with MRI + DaT (reference only).

Stages (mode "full"):
  "warmup" [Code: run.sh first command, 80k iters with all modalities, then reload; README "Model Training";
           issue #7]: rows with DaT only, all modalities available, selection on the full-modality val MAE.
  "main"   [Paper Sec. 4.2 "modalities are randomly dropped"; Code run.sh second command ``--mode=random``]:
           starts from the best warmup weights (init_stage), per-sample random modality availability
           (``drop.mode``), selection on the MRI-only val MAE (the reported role).
Mode "student_only": same backbone/fusion/head, MRI only (f_dat = r_mri by Eq. 4), task loss only.

Loss (Eq. 9): L = L_task + alpha * L_dao + beta * L_dco
  L_task: protocol task loss on the z-scored target (TASK-BOUND: Dice+BCE / CE -> regression loss; INTERFACE).
  L_dao : distribution alignment of shared features, L1 option (Eq. 8 with p = 1; Sec. 4.2 "We choose L1 loss";
          code train_SS.py:250 nn.L1Loss (element mean), cyclic pairs :285-288). For N = 2 modalities the cyclic
          pairs (mri, dat) + (dat, mri) equal Eq. 8's ordered double sum (i = k terms vanish): 2 * L1.
  L_dco : modality classification of specific features with CE (Eq. 5; code loss_Dual.py:244-252). The labels
          are MODALITY ids, not the clinical target, so this term is not task-bound.
  Paper Sec. 3.3.3: the DAO / DCO terms of missing modalities are omitted (``aux_on_missing: false``); the
  official code feeds zero-filled missing modalities through the encoders and keeps their terms (issue #4) ->
  ``aux_on_missing: true`` reproduces it (sensitivity, rule A).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.task_losses import task_loss_per_sample

from .model import MODALITIES, ShaSpecModel

MRI_ONLY_ROLES = ("student", "main")
FULL_ROLES = ("teacher", "warmup")


class ShaSpecMethod:
    primary_role = "student"

    def __init__(self, protocol: dict, cfg: dict):
        self.protocol, self.cfg = protocol, cfg
        self.mode = cfg["mode"]
        if self.mode == "full":
            self.stages = ("warmup", "main") if cfg["warmup"]["enabled"] else ("main",)
            self.eval_roles = (("student", False), ("teacher", True))
        elif self.mode == "student_only":
            self.stages = ("student",)
            self.eval_roles = (("student", False),)
        else:
            raise ValueError(f"Unknown shaspec mode: {self.mode}")
        if cfg["drop"]["mode"] not in ("random_subset", "keep_mri", "dedicated", "none"):
            raise ValueError(f"Unknown drop.mode: {cfg['drop']['mode']}")
        if cfg["loss"]["dao"] not in ("l1", "mse"):
            raise ValueError(f"Unknown loss.dao: {cfg['loss']['dao']}")

    # ---- construction / data needs -------------------------------------------------------
    def build(self, stage: str) -> torch.nn.Module:
        if self.mode == "student_only":
            return ShaSpecModel(self.protocol, self.cfg, specific=("mri",), dom_cls=False)
        return ShaSpecModel(self.protocol, self.cfg)

    def init_stage(self, stage: str, model, frozen: dict) -> None:
        """'main' continues from the best warmup weights (code run.sh: --reload_from_checkpoint=True)."""
        if stage == "main" and "warmup" in frozen:
            model.load_state_dict(frozen["warmup"].state_dict())

    def train_rows(self, stage: str, rows):
        if stage == "warmup":                       # all-modality warm-up needs DaT
            return [r for r in rows if r.dat_path is not None]
        return rows

    def needs_dat(self, stage: str, phase: str) -> bool:
        if self.mode == "student_only" or stage in MRI_ONLY_ROLES and phase == "eval":
            return False
        return stage in FULL_ROLES or stage == "main"

    # ---- availability ---------------------------------------------------------------------
    @staticmethod
    def _present(batch) -> dict:
        b = batch["mri"].shape[0]
        return dict(mri=torch.ones(b, dtype=torch.bool, device=batch["mri"].device),
                    dat=batch["has_dat"].to(batch["mri"].device).bool())

    def _sample_availability(self, present: dict) -> dict:
        """Per-sample modality dropping of the "main" stage (CPU RNG seeded by train.py -> deterministic)."""
        mode = self.cfg["drop"]["mode"]
        b = present["mri"].shape[0]
        dev = present["mri"].device
        if mode == "none":
            return dict(present)
        if mode == "dedicated":                     # Table 2 setting: train with the test-time pattern
            return dict(mri=present["mri"].clone(), dat=torch.zeros_like(present["dat"]))
        if mode == "keep_mri":                      # MRI always kept, DaT dropped with probability p_dat
            keep = torch.rand(b) >= float(self.cfg["drop"]["p_dat"])
            return dict(mri=present["mri"].clone(), dat=present["dat"] & keep.to(dev))
        # "random_subset" (code DualNet_SS.py:378-386): k ~ U{1..#present}, then a random k-subset.
        out = {m: torch.zeros(b, dtype=torch.bool) for m in MODALITIES}
        pres = {m: present[m].cpu() for m in MODALITIES}
        for i in range(b):
            avail = [m for m in MODALITIES if bool(pres[m][i])]
            k = int(torch.randint(1, len(avail) + 1, (1,)))
            for j in torch.randperm(len(avail))[:k].tolist():
                out[avail[j]][i] = True
        return {m: v.to(dev) for m, v in out.items()}

    def _images(self, batch, with_dat: bool) -> dict:
        images = dict(mri=batch["mri"])
        if with_dat:
            images["dat"] = batch["dat"] if "dat" in batch else torch.zeros_like(batch["mri"])
        return images

    # ---- inference --------------------------------------------------------------------------
    def predict(self, role: str, model, batch) -> torch.Tensor:
        present = self._present(batch)
        if self.mode == "student_only" or role in MRI_ONLY_ROLES:
            avail = dict(mri=present["mri"], dat=torch.zeros_like(present["dat"]))
            return model(self._images(batch, False), avail)["pred"]
        if role in FULL_ROLES:                       # real DaT where the row has it, Eq. 4 otherwise
            return model(self._images(batch, True), present)["pred"]
        raise ValueError(f"Unknown ShaSpec role: {role}")

    # ---- losses -----------------------------------------------------------------------------
    def aux_losses(self, model, out: dict, loss_mask: dict) -> dict:
        """L_dao (Eq. 8, L1) and L_dco (Eq. 5) over the samples in ``loss_mask`` per modality."""
        cfg = self.cfg["loss"]
        r, s = out["r"], out["s"]
        mods = [m for m in MODALITIES if m in r]
        zero = out["pred"].new_zeros(())
        dao = zero
        if len(mods) > 1:
            for i, m in enumerate(mods):                       # cyclic pairs (code train_SS.py:285-288)
                k = mods[(i + 1) % len(mods)]
                both = (loss_mask[m] & loss_mask[k]).to(r[m].dtype)
                if float(both.sum()) == 0:
                    continue
                diff = r[m] - r[k]
                elem = diff.abs() if cfg["dao"] == "l1" else diff.square()
                per_sample = elem.flatten(1).mean(1)           # nn.L1Loss element mean over the used samples
                dao = dao + (per_sample * both).sum() / both.sum()
        dco = zero
        if model.dom_cls is not None:
            logits, labels, weights = [], [], []
            for label, m in enumerate(MODALITIES):
                if m not in s:
                    continue
                logits.append(model.domain_logits(s[m]))
                labels.append(torch.full_like(loss_mask[m], label, dtype=torch.long))
                weights.append(loss_mask[m].to(logits[-1].dtype))
            w = torch.cat(weights)
            if float(w.sum()) > 0:
                ce = F.cross_entropy(torch.cat(logits), torch.cat(labels), reduction="none")
                dco = (ce * w).sum() / w.sum()
        return dict(dao=dao, dco=dco)

    def training_step(self, stage: str, model, frozen: dict, batch: dict):
        y = batch["y_z"]
        present = self._present(batch)
        if self.mode == "student_only":
            avail = dict(mri=present["mri"], dat=torch.zeros_like(present["dat"]))
            out = model(self._images(batch, False), avail)
            task = task_loss_per_sample(self.protocol, out["pred"], y).mean()
            return task, dict(task=task)
        if stage == "warmup":
            if not bool(present["dat"].all()):
                raise ValueError("Stage 'warmup' received rows without DaT.")
            avail = present
        elif stage == "main":
            avail = self._sample_availability(present)
        else:
            raise ValueError(stage)
        feed_missing = bool(self.cfg["aux_on_missing"])
        # Code variant: every modality slot enters the encoders, missing ones zero-filled (DualNet_SS.py:389-400).
        enter = {m: torch.ones_like(avail[m]) for m in MODALITIES} if feed_missing else avail
        out = model(self._images(batch, True), avail, enter)
        task = task_loss_per_sample(self.protocol, out["pred"], y).mean()
        aux = self.aux_losses(model, out, enter)
        lw = self.cfg["loss"]
        loss = task + float(lw["alpha"]) * aux["dao"] + float(lw["beta"]) * aux["dco"]
        frac = avail["dat"].float().mean()
        return loss, dict(task=task, dao=aux["dao"], dco=aux["dco"], dat_avail=frac,
                          mri_avail=avail["mri"].float().mean())
