"""Shared evaluation, identical for every method: load checkpoints, predict, score, aggregate.

    python evaluate.py --method crad --run_dir runs\\crad\\fold0_seed0          # one run
    python evaluate.py --method crad --run_dir runs\\crad                       # all folds x seeds
    python evaluate.py --method crad --run_dir runs\\crad --split val           # validation subjects
    python evaluate.py --method mri_only --run_dir runs_ppmi/off30/mri_only --split test --checkpoint last

--checkpoint best (default, the primary result: best val MAE) | last (D-013, sensitivity only: <stage>_last.pt
of every scored stage; methods with eval_roles use the last trained stage's _last.pt; a reused teacher uses
the source run's teacher_last.pt). "last" writes eval_<split>_last/ and *_<split>_last.* and never touches
the best-checkpoint outputs.

The reported model is the method's MRI-only role ("student"); the teacher (MRI + DaT) and the
train-mean predictor are reported as references. Nothing here is fitted or selected.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from baselines import get_method
from data.dataset import make_loader
from data.manifest import Split, TargetScaler, cohort_hash, data_fingerprint, make_split, read_manifest, rows_of, verify_approved_data
from utils.config import config_hash, resolve_path
from utils.inference import predict_units
from utils.io import read_json, write_csv, write_json
from utils.metrics import regression_metrics
from utils.repro import choose_device, load_run_checkpoint

METRICS = ("mae", "rmse", "pearson_r", "spearman_rho", "r2", "bias")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True)
    p.add_argument("--run_dir", required=True, help="one fold*_seed* run, or the folder containing them")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--roles", default="all", help="comma list of stages to score, or 'all'")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow_data_change", action="store_true", help="skip the data-fingerprint check")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"],
                   help="best = best val MAE (primary); last = <stage>_last.pt, sensitivity only (D-013)")
    return p.parse_args()


def evaluate_run(method_name, run_dir: Path, split_name, roles, device, allow_data_change, cache, checkpoint="best"):
    """checkpoint "best" (default, unchanged) or "last" (D-013: <stage>_last.pt, outputs in eval_<split>_last/)."""
    if checkpoint not in ("best", "last"):
        raise ValueError(f"checkpoint must be 'best' or 'last', not {checkpoint!r}")
    suffix = "_last" if checkpoint == "last" else ""
    meta = read_json(run_dir / "run.json")
    if meta["method"] != method_name or meta["status"] != "complete":
        raise ValueError(f"{run_dir}: not a completed '{method_name}' run.")
    protocol, cfg = meta["protocol"], meta["config"]
    for field, value in (("protocol_hash", protocol), ("config_hash", cfg)):
        if field in meta and config_hash(value) != meta[field]:
            raise ValueError(f"{run_dir}: stored {field} differs from its contents.")
    key = cohort_hash(protocol) if "qc" in protocol["data"] else config_hash(protocol["data"])
    if key not in cache:
        stats = {}
        rows = read_manifest(protocol, stats)
        cache[key] = (rows, data_fingerprint(protocol, rows), stats)
    rows, fingerprint, stats = cache[key]
    if stats.get("qc_approval") != meta.get("med_state", {}).get("qc_approval"):
        raise ValueError(f"{run_dir}: current QC approval differs from training.")
    if not allow_data_change:
        verify_approved_data(stats, fingerprint)
    if fingerprint != meta["data_fingerprint"] and not allow_data_change:
        raise ValueError(f"{run_dir}: data/manifest differ from training (pass --allow_data_change to override).")
    split = Split(fold=meta["fold"], **{k: tuple(meta["split"][k]) for k in ("train", "val", "test")})
    if split != make_split(rows, meta["fold"], protocol):
        raise ValueError(f"{run_dir}: stored split differs from the protocol split.")
    scaler = TargetScaler(**meta["scaler"])
    if scaler != TargetScaler.fit(rows_of(rows, split.train)):
        raise ValueError(f"{run_dir}: stored target scaler was not fitted on this fold's training subjects.")
    eval_rows = rows_of(rows, getattr(split, split_name))
    method = get_method(method_name)(protocol, cfg)
    out = []

    def load_stage(stage):
        path = run_dir / f"{stage}{suffix}.pt"
        if not path.exists() and "reused_from" in meta["stages"].get(stage, {}):
            path = resolve_path(meta["stages"][stage]["reused_from"])   # teacher shared with another run
            if suffix:                                                  # its last-epoch sibling (D-013)
                path = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        if suffix and not path.exists():
            raise FileNotFoundError(f"{path} not found: --checkpoint last needs a run trained with "
                                    "training.save_last_checkpoint: true (D-013)")
        ckpt = load_run_checkpoint(path, meta, stage, checkpoint)
        model = method.build(stage)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return model.to(device).eval()

    def score(role, model, role_rows, load_dat):
        # An MRI-only role is evaluated with a loader that never opens DaT.
        loader = make_loader(role_rows, protocol, load_dat, shuffle=False, seed=0)
        for r in predict_units(method, role, model, loader, scaler, device):
            out.append(dict(fold=meta["fold"], seed=meta["seed"], role=role, **r))

    if getattr(method, "eval_roles", None) is None:      # one role per stage (e.g. CRAD teacher/student)
        for stage in [s for s in method.stages if roles == ["all"] or s in roles]:
            model = load_stage(stage)
            score(stage, model, method.train_rows(stage, eval_rows) if stage == "teacher" else eval_rows,
                  method.needs_dat(stage, "eval"))
            del model
    else:                                                # several roles of one model (e.g. DFTD)
        trained = [s for s in method.stages if s in meta["stages"] and not meta["stages"][s].get("skipped")]
        model = load_stage(trained[-1])
        for role, load_dat in method.eval_roles:
            if roles != ["all"] and role not in roles:
                continue
            role_rows = [r for r in eval_rows if r.dat_path is not None] if load_dat else eval_rows
            if role_rows:
                score(role, model, role_rows, load_dat)
        del model
    reference = {r.unit: (r.subject_id, r.target) for r in eval_rows}
    for unit, (sid, y) in sorted(reference.items()):
        out.append(dict(fold=meta["fold"], seed=meta["seed"], role="train_mean", unit=unit, subject_id=sid,
                        y_true=y, y_pred=scaler.mean, n_rows=0))
    folder = run_dir / f"eval_{split_name}{suffix}"
    write_csv(folder / "predictions.csv", out)
    scores = []
    for role in dict.fromkeys(r["role"] for r in out):
        sel = [r for r in out if r["role"] == role]
        scores.append(dict(fold=meta["fold"], seed=meta["seed"], role=role,
                           **regression_metrics([r["y_true"] for r in sel], [r["y_pred"] for r in sel])))
    write_csv(folder / "metrics.csv", scores)
    return out, scores


def mean_std(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(statistics.stdev(values)) if len(values) > 1 else 0.0


def summarize(preds, run_scores, split_name, folds_expected):
    """Primary number: pooled out-of-fold metrics per seed, then mean +- SD over seeds."""
    table, pooled_rows = [], []
    for role in dict.fromkeys(r["role"] for r in preds):
        role_preds = [r for r in preds if r["role"] == role]
        seeds = sorted({r["seed"] for r in role_preds})
        complete = all({r["fold"] for r in role_preds if r["seed"] == s} == set(folds_expected) for s in seeds)
        pooled = {}
        for s in seeds:
            sel = [r for r in role_preds if r["seed"] == s]
            units = [r["unit"] for r in sel]
            if len(units) != len(set(units)):
                raise ValueError(f"Duplicate eval units in pooled predictions ({role}, seed {s}).")
            pooled[s] = regression_metrics([r["y_true"] for r in sel], [r["y_pred"] for r in sel])
            pooled_rows.append(dict(role=role, seed=s, scope="pooled_oof" if complete else "pooled_partial",
                                    **pooled[s]))
        by_unit = defaultdict(list)
        for r in role_preds:
            by_unit[r["unit"]].append(r)
        ens = regression_metrics([v[0]["y_true"] for v in by_unit.values()],
                                 [float(np.mean([x["y_pred"] for x in v])) for v in by_unit.values()])
        per_run = [s for s in run_scores if s["role"] == role]
        row = dict(role=role, scope="pooled_oof" if complete else "pooled_partial", n_seeds=len(seeds),
                   n_units=len(by_unit), n_runs=len(per_run))
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_sd"] = mean_std([pooled[s][m] for s in seeds])
            row[f"{m}_seed_ensemble"] = ens[m]
            row[f"{m}_per_run_mean"], row[f"{m}_per_run_sd"] = mean_std([s[m] for s in per_run])
        table.append(row)
    return table, pooled_rows


def write_markdown(path: Path, method, split_name, table):
    lines = [f"# {method}: {split_name} results", "",
             "Pooled out-of-fold metrics per seed, mean ± SD over seeds (n = eval units per seed).", "",
             "| Role | n units | seeds | MAE ↓ | RMSE ↓ | Pearson r ↑ | Spearman ρ ↑ | R² ↑ | Bias |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        cells = [f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_sd']:.3f}" if np.isfinite(r[f"{m}_mean"]) else "n/a"
                 for m in METRICS]
        lines.append(f"| {r['role']} | {r['n_units']} | {r['n_seeds']} | " + " | ".join(cells) + " |")
    lines += ["", "Per-run (fold × seed) mean ± SD and seed-ensemble values are in the CSV next to this file.",
              "`train_mean` = constant prediction equal to the training-fold mean (reference).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def group_folds(method: str, runs: list[Path]) -> list[int]:
    """Keep pooled OFF30 results on one cohort and allow independently tuned fold configs."""
    if not runs:
        raise ValueError("No run metadata to aggregate.")
    metas = [read_json(run / "run.json") for run in runs]
    base = metas[0]["protocol"]
    folds = base["data"]["split"]["folds"]
    is_off30 = ["qc" in meta["protocol"]["data"] for meta in metas]
    if not any(is_off30):
        return folds
    if not all(is_off30):
        raise ValueError("Run group mixes OFF30 and other protocols.")
    expected_cohort = cohort_hash(base)
    fingerprint = metas[0]["data_fingerprint"]
    approval = metas[0].get("med_state", {}).get("qc_approval")
    if approval is None:
        raise ValueError("OFF30 run group lacks QC approval.")
    observed = set()
    for run, meta in zip(runs, metas):
        identity = (meta["fold"], meta["seed"])
        if (meta["method"] != method or meta["status"] != "complete" or identity in observed
                or meta["fold"] not in folds or cohort_hash(meta["protocol"]) != expected_cohort
                or meta["data_fingerprint"] != fingerprint
                or meta.get("med_state", {}).get("qc_approval") != approval):
            raise ValueError(f"{run}: inconsistent OFF30 cohort or fold/seed identity in run group.")
        observed.add(identity)
    return folds


def main():
    args = parse_args()
    root = resolve_path(args.run_dir)
    runs = [root] if (root / "run.json").exists() else sorted(p.parent for p in root.glob("fold*_seed*/run.json"))
    if not runs:
        raise FileNotFoundError(f"No runs under {root}")
    folds_expected = group_folds(args.method, runs)
    roles = ["all"] if args.roles == "all" else [r.strip() for r in args.roles.split(",")]
    device = choose_device(args.device)
    preds, scores, cache = [], [], {}
    for run_dir in runs:
        p, s = evaluate_run(args.method, run_dir, args.split, roles, device, args.allow_data_change, cache,
                            args.checkpoint)
        preds += p
        scores += s
        for row in s:
            print(f"{run_dir.name:14s} {row['role']:10s} n={row['n']:3d} MAE={row['mae']:7.3f} "
                  f"RMSE={row['rmse']:7.3f} r={row['pearson_r']:.3f}")
    suffix = "_last" if args.checkpoint == "last" else ""
    tag = f"{args.split}{suffix}"            # best: unchanged file names; last: *_<split>_last.*
    if len(runs) == 1:
        print(f"Results: {runs[0] / f'eval_{tag}'}")
        return
    if args.split == "val":  # a subject can be a validation subject of several folds
        preds = [dict(r, unit=f"fold{r['fold']}:{r['unit']}") for r in preds]
    table, pooled = summarize(preds, scores, args.split, folds_expected)
    write_csv(root / f"predictions_{tag}.csv", preds)
    write_csv(root / f"metrics_per_run_{tag}.csv", scores)
    write_csv(root / f"metrics_pooled_per_seed_{tag}.csv", pooled)
    write_csv(root / f"summary_{tag}.csv", table)
    summary = dict(method=args.method, split=args.split, runs=[r.name for r in runs], summary=table)
    if suffix:
        summary["checkpoint"] = "last"
    write_json(root / f"summary_{tag}.json", summary)
    write_markdown(root / f"summary_{tag}.md", args.method, args.split + (" (last-epoch checkpoint)" if suffix else ""), table)
    print()
    print((root / f"summary_{tag}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
