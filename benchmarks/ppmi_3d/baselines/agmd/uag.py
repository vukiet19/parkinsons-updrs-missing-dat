"""Uncertainty-Aware Gating (Sec. 2.4, Eq. 9-10): detached per-sample weights of the distillation losses.

Paper Eq. 9: w = 1 - H(p_T) / log N_class with p_T the teacher's class probabilities -> TASK-BOUND (there is
no class distribution in regression). Modes:

  mc_dropout       [D-039, PPMI] w = 1 - clip(sigma / mc_scale, 0, 1), with sigma the sample standard
                   deviation of teacher predictions on the train-fold z-scale. The gate needs no labels.
  error            [Proposal, NTUA default; PPMI sensitivity] w = 1 - clip(|y_T - y| / kappa, 0, 1): the
                   regression confidence already used by CRAD-dagger (S^C, notes/CRAD_adaptation.md Q3) and by
                   AGMD's own code for its logit term (orig_repo/train_test.py:382-386, cal_weight).
  feature_entropy  [Code model.py:352-368] w_l = 1 - H(softmax(channel means of the teacher's fused feature l))
                   / log C_l for the shallow terms; graph and global terms use the entropy of softmax(G_T)
                   (train_test.py:44-49). Label-free and runs unchanged, but it is not a prediction uncertainty.
  none             w = 1 (the paper's "w/o UAG" ablation; code --no_weight).

``reduce``: per_sample [Paper: "instance-level"] | batch (code: weight_func multiplies a batch-mean loss by
the weight vector and averages, train_test.py:389-392, i.e. only the batch-mean weight matters).
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def error_confidence(pred_t: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    return 1.0 - ((pred_t - y).abs() / scale).clamp(0.0, 1.0)


@torch.no_grad()
def softmax_entropy_weight(x: torch.Tensor) -> torch.Tensor:
    """1 - H(softmax(x)) / log(dim), over the last axis. (B, D) -> (B,). orig_repo/model.py:360-366."""
    p = torch.softmax(x.float(), dim=-1)
    h = -(p * torch.log(p + 1e-10)).sum(-1)
    return 1.0 - h / math.log(x.shape[-1])


@torch.no_grad()
def mc_dropout_sigma(teacher, tokens: torch.Tensor, samples: int) -> torch.Tensor:
    """D-039: sample prediction SD using the teacher's existing graph readout dropout."""
    if samples < 2:
        raise ValueError("uag.mc_samples must be at least 2 for a sample standard deviation")
    readout = teacher.readout
    flags = [(module, module.training) for module in readout.modules()]
    devices = [tokens.get_device()] if tokens.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        try:
            # Training mode enables attention dropout and disables the transformer eval fast path.
            readout.train()
            predictions = [teacher.head(readout(tokens)).squeeze(-1) for _ in range(samples)]
        finally:
            for module, training in flags:
                module.training = training
    return torch.stack(predictions).std(dim=0, correction=1)


@torch.no_grad()
def uag_weights(cfg: dict, out_t: dict, pred_t: torch.Tensor, y: torch.Tensor, shallow_layers,
                *, sigma: torch.Tensor | None = None) -> dict:
    """Return per-sample weights {shallow_<l>, graph, global, logit}, each (B,), detached."""
    u = cfg["uag"]
    mode, ones = u["mode"], torch.ones_like(y)
    w = {}
    if mode == "mc_dropout":
        scale = float(u["mc_scale"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("uag.mc_scale must be finite and positive")
        if sigma is None:
            raise ValueError("uag.mode=mc_dropout requires teacher prediction sigma")
        c = 1.0 - (sigma / scale).clamp(0.0, 1.0)
        w.update({f"shallow_{l}": c for l in shallow_layers}, graph=c, **{"global": c})
    elif mode == "error":
        c = error_confidence(pred_t, y, float(u["error_scale"]))
        w.update({f"shallow_{l}": c for l in shallow_layers}, graph=c, **{"global": c})
    elif mode == "feature_entropy":
        for l in shallow_layers:
            w[f"shallow_{l}"] = softmax_entropy_weight(out_t["feats"][l].flatten(2).mean(-1))
        g = softmax_entropy_weight(out_t["global_"])
        w.update(graph=g, **{"global": g})
    elif mode == "none":
        w.update({f"shallow_{l}": ones for l in shallow_layers}, graph=ones, **{"global": ones})
    else:
        raise ValueError(f"Unknown uag.mode: {mode}")
    # Logit term (code only): weighted by cal_weight unless --no_weight (train_test.py:53-55).
    w["logit"] = ones if mode == "none" else error_confidence(pred_t, y, float(u["error_scale"]))
    if u["reduce"] == "batch":
        w = {k: v.mean().expand_as(v) for k, v in w.items()}
    elif u["reduce"] != "per_sample":
        raise ValueError(f"Unknown uag.reduce: {u['reduce']}")
    return w
