"""Train a variant one fold/seed at a time, evaluate it, then delete its checkpoints (low disk peak).

    python scripts/train_eval_prune.py --method agmd --tag sens_cmt_code --set cmt.scale=none
    python scripts/train_eval_prune.py --method agmd --tag abl_no_uag --teacher_from runs/agmd --set uag.mode=none
    python scripts/train_eval_prune.py --protocol configs/protocol_ppmi.yaml --method mri_only --tag scratch
    python scripts/train_eval_prune.py --protocol configs/protocol_ppmi.yaml --method mri_only \
        --selection_dir runs_ppmi/off30/tuning/mri_only
    python scripts/train_eval_prune.py --protocol configs/protocol_ppmi.yaml --config configs/baselines/crad_ppmi.yaml \
        --method crad --tag sens_psi_batch --teacher_from runs_ppmi/off30/crad --set sdu.psi_mode=batch --pset training.lr=1e-3

Use --selection_dir with the fold selections from scripts/tune.py. Every selection is verified before
training starts. For a fixed configuration, supply --config, --set and --pset as needed.

--checkpoint best|last|both chooses which checkpoint evaluate.py scores (default best; last needs a protocol
with training.save_last_checkpoint, D-013); "last" results go to eval_<split>_last/ and *_last summaries.
Runs go to <protocol output_root>/<method>_<tag> (NTUA: runs/, PPMI: runs_ppmi/off30/); --protocol is passed on
to train.py (evaluate.py reads the protocol from each run.json).

For every fold x seed of the protocol: `train.py` (one run) -> `evaluate.py --split val` and `--split test`
(one run, writes eval_<split>/predictions.csv and metrics.csv) -> delete the run's *.pt (a teacher reused via
--teacher_from lives in the source run and is never touched). The folder-level summaries are then rebuilt
from the saved per-run CSVs with evaluate.py's own `summarize`/`write_markdown`, so the outputs have the same
format as `evaluate.py --run_dir <root>`. 2D runs (NTUA) are bit-for-bit reproducible, so a deleted checkpoint
can be re-created with the same command (plus --keep_ckpt); 3D runs (PPMI) are NOT bit-exact, because 3D
max-pool backward has no deterministic CUDA kernel, so a re-created checkpoint is close but not identical. Shared code is not modified; this only orchestrates it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate as shared_evaluate  # noqa: E402
from data.manifest import cohort_hash, data_fingerprint, read_manifest, verify_approved_data  # noqa: E402
from train import run_identity  # noqa: E402
from utils.config import apply_overrides, config_hash, load_yaml, method_config_path, resolve_path  # noqa: E402
from utils.io import read_csv, read_json, write_csv, write_json  # noqa: E402
from utils.repro import source_hash  # noqa: E402

INT_KEYS = ("fold", "seed", "n", "n_rows")


def _typed(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k in INT_KEYS:
            out[k] = int(float(v))
        elif k in ("role", "unit", "subject_id"):
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
    return out


def aggregate(method: str, root: Path, split: str, suffix: str = "") -> None:
    """suffix "" = best checkpoint, "_last" = last-epoch checkpoint (D-013)."""
    runs = sorted(p.parent for p in root.glob("fold*_seed*/run.json"))
    folds_expected = shared_evaluate.group_folds(method, runs)
    preds, scores = [], []
    for run in runs:
        preds += [_typed(r) for r in read_csv(run / f"eval_{split}{suffix}" / "predictions.csv")]
        scores += [_typed(r) for r in read_csv(run / f"eval_{split}{suffix}" / "metrics.csv")]
    if split == "val":
        preds = [dict(r, unit=f"fold{r['fold']}:{r['unit']}") for r in preds]
    table, pooled = shared_evaluate.summarize(preds, scores, split, folds_expected)
    tag = f"{split}{suffix}"
    write_csv(root / f"predictions_{tag}.csv", preds)
    write_csv(root / f"metrics_per_run_{tag}.csv", scores)
    write_csv(root / f"metrics_pooled_per_seed_{tag}.csv", pooled)
    write_csv(root / f"summary_{tag}.csv", table)
    write_json(root / f"summary_{tag}.json", dict(method=method, split=split, runs=[r.name for r in runs],
                                                  summary=table, rebuilt_from="per-run eval CSVs",
                                                  checkpoint="last" if suffix else "best"))
    shared_evaluate.write_markdown(root / f"summary_{tag}.md", method, split + (" (last-epoch checkpoint)" if suffix else ""), table)
    print((root / f"summary_{tag}.md").read_text(encoding="utf-8"))


def load_fold_selections(directory: Path, method: str, protocol: dict, fingerprint: str) -> dict:
    """Resolve validated per-fold configs and the exact CLI overrides that produced them."""
    current_source = source_hash()
    selections = {}
    for fold in protocol["data"]["split"]["folds"]:
        path = directory / f"fold{fold}" / "selected.yaml"
        selected = load_yaml(path)
        identity = dict(method=method, mode="outer_fold", exploratory=False, folds=[fold],
                        protocol_name=protocol["name"], data_fingerprint=fingerprint, source_hash=current_source)
        if any(selected.get(key) != value for key, value in identity.items()):
            raise ValueError(f"{path}: selection does not match the requested benchmark, source and fold.")
        effective = apply_overrides(protocol, selected["pset"])
        cfg_path = resolve_path(selected["config"])
        config = apply_overrides(load_yaml(cfg_path), selected["set"])
        if (config.get("method", method) != method
                or config_hash(effective) != selected["protocol_hash"] or config_hash(config) != selected["config_hash"]
                or cohort_hash(effective) != cohort_hash(protocol)
                or effective["training"]["seeds"] != protocol["training"]["seeds"]
                or effective.get("output_root") != protocol.get("output_root")):
            raise ValueError(f"{path}: effective protocol or method configuration differs from selection.")
        selections[fold] = (effective, config, cfg_path, selected["pset"], selected["set"])
    return selections


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True)
    p.add_argument("--protocol", default="configs/protocol.yaml")
    p.add_argument("--tag", default="", help="optional output suffix")
    p.add_argument("--selection_dir", help="directory containing foldN/selected.yaml for the method")
    p.add_argument("--teacher_from")
    p.add_argument("--config", help="method config passed to train.py --config (default: configs/baselines/<method>.yaml)")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--pset", nargs="*", default=[], help="protocol overrides passed to train.py --pset")
    p.add_argument("--keep_ckpt", action="store_true")
    p.add_argument("--checkpoint", default="best", choices=["best", "last", "both"], help="checkpoint(s) evaluate.py scores")
    args = p.parse_args()
    if args.selection_dir and (args.config or args.pset or args.set):
        p.error("--selection_dir supplies the protocol overrides and method configuration")
    protocol = apply_overrides(load_yaml(resolve_path(args.protocol)), args.pset)
    name = f"{args.method}_{args.tag}" if args.tag else args.method
    root = resolve_path(protocol.get("output_root", "runs")) / name
    fingerprint = None
    if "qc" in protocol["data"] or args.selection_dir:
        stats = {}
        rows = read_manifest(protocol, stats)
        fingerprint = data_fingerprint(protocol, rows)
        verify_approved_data(stats, fingerprint)
    cfg_path = resolve_path(args.config) if args.config else method_config_path(protocol, args.method)
    if args.selection_dir:
        selections = load_fold_selections(resolve_path(args.selection_dir), args.method, protocol, fingerprint)
    else:
        cfg = apply_overrides(load_yaml(cfg_path), args.set)
        selections = {fold: (protocol, cfg, cfg_path, args.pset, args.set) for fold in protocol["data"]["split"]["folds"]}
    py = sys.executable
    checkpoints = ["best", "last"] if args.checkpoint == "both" else [args.checkpoint]
    suffixes = ["_last" if c == "last" else "" for c in checkpoints]
    for fold, (effective, cfg, cfg_path, psets, sets) in selections.items():
        for seed in protocol["training"]["seeds"]:
            run = root / f"fold{fold}_seed{seed}"
            outputs = [run / f"eval_{s}{x}" / name for s in ("val", "test") for x in suffixes
                       for name in ("metrics.csv", "predictions.csv")]
            done = all(path.is_file() for path in outputs)
            if done:
                if fingerprint is not None:
                    meta = read_json(run / "run.json")
                    expected = run_identity(args.method, effective, cfg, fingerprint, fold, seed)
                    if meta["status"] != "complete" or any(meta.get(k) != v for k, v in expected.items()):
                        raise ValueError(f"{run}: existing results differ from the requested OFF30 run")
                print(f"skip {run.name}: trained and evaluated")
                continue
            cmd = [py, "train.py", "--method", args.method, "--protocol", args.protocol, "--tag", args.tag,
                   "--fold", str(fold), "--seed", str(seed)]
            cmd += ["--config", str(cfg_path)]
            if args.teacher_from:
                cmd += ["--teacher_from", args.teacher_from]
            if sets:
                cmd += ["--set", *sets]
            if psets:
                cmd += ["--pset", *psets]
            subprocess.run(cmd, cwd=ROOT, check=True)
            for split in ("val", "test"):
                for ckpt in checkpoints:
                    subprocess.run([py, "evaluate.py", "--method", args.method, "--run_dir", str(run), "--split", split,
                                    "--checkpoint", ckpt], cwd=ROOT, check=True)
            if not args.keep_ckpt:
                if not all(path.is_file() for path in outputs):
                    raise FileNotFoundError(f"{run}: evaluation outputs are incomplete.")
                for ckpt in run.glob("*.pt"):
                    ckpt.unlink()
    for split in ("val", "test"):
        for suffix in suffixes:
            aggregate(args.method, root, split, suffix)


if __name__ == "__main__":
    main()
