"""Equal-budget tuning on validation only (STEWARD D-007, concretised in D-038).

    python scripts/tune.py --method <m> --protocol configs/protocol_ppmi.yaml [--fold all | N]
           [--grid configs/tuning/<m>.yaml] [--config <path>] [--dry_run] [--limit K] [--keep_ckpt]
           [--out <root>] [--pset key=value ...]

Grid file (configs/tuning/<m>.yaml): ``grid`` maps keys to value lists. Keys starting with ``protocol.`` go to
train.py --pset, the others to --set. The Cartesian product is taken in file order, so configuration 0 is
every first value (the current defaults). Optional ``config`` (method config; --config overrides it) and
``budget`` (number of configurations; default protocol.training.tuning_budget; the anchors use 4).

Select the lowest primary-role validation MAE independently in each outer fold. Each fold has its
own trials.csv and selected.yaml. Use --mode pilot for exploratory pooled validation across folds.
Trial resumption checks the effective configuration, source, fold, dataset and validation files.
The output root defaults to runs_ppmi/off30/tuning for OFF30.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import get_method  # noqa: E402
from data.manifest import data_fingerprint, make_split, read_manifest, verify_approved_data  # noqa: E402
from utils.config import apply_overrides, config_hash, load_yaml, method_config_path, project_relative, resolve_path  # noqa: E402
from utils.io import read_csv, read_json, write_csv, write_json  # noqa: E402
from utils.repro import file_sha256, source_hash  # noqa: E402


def _value(v) -> str:
    return json.dumps(v)          # JSON scalars/lists are valid YAML for apply_overrides


def configurations(grid: dict) -> list[dict]:
    keys = list(grid)
    for key in keys:
        if not isinstance(grid[key], list) or not grid[key]:
            raise ValueError(f"grid key {key!r} needs a non-empty list of values")
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def split_overrides(cfg: dict) -> tuple[list[str], list[str]]:
    psets = [f"{k[len('protocol.'):]}={_value(v)}" for k, v in cfg.items() if k.startswith("protocol.")]
    sets = [f"{k}={_value(v)}" for k, v in cfg.items() if not k.startswith("protocol.")]
    return psets, sets


def _run_trial(cmd, root, identity, primary, keep_ckpt, resume):
    key = root / "trial.json"
    result_path = root / "validation.json"
    if root.exists():
        if not key.is_file() or read_json(key) != identity:
            raise ValueError(f"{root}: trial identity differs. Use a new tuning output root.")
    else:
        root.mkdir(parents=True)
        write_json(key, identity)
    folds = identity["folds"]
    runs = [root / f"fold{fold}_seed0" for fold in folds]
    metrics = [run / "eval_val" / "metrics.csv" for run in runs]
    sources = [run / "run.json" for run in runs]
    def check_runs(require_complete):
        for fold, meta_path in zip(folds, sources):
            if not require_complete and not meta_path.is_file():
                continue
            meta = read_json(meta_path)
            expected = {k: identity[k] for k in
                        ("method", "seed", "protocol_hash", "config_hash", "data_fingerprint", "source_hash")}
            expected.update(fold=fold, split=identity["splits"][str(fold)])
            if any(meta.get(k) != v for k, v in expected.items()):
                raise ValueError(f"{meta_path}: run identity differs from the trial")
            if require_complete and meta["status"] != "complete":
                raise ValueError(f"{meta_path}: validation requires a completed run")

    py = sys.executable
    if result_path.is_file():
        check_runs(require_complete=True)
        result = read_json(result_path)
        if (not all(p.is_file() for p in metrics + sources)
                or result["hashes"] != [file_sha256(p) for p in metrics + sources]):
            raise ValueError(f"{result_path}: cached validation evidence changed")
        return result["scores"], True
    check_runs(require_complete=False)
    subprocess.run([py, *cmd, *(["--overwrite"] if resume else [])], cwd=ROOT, check=True)
    check_runs(require_complete=True)
    if len(folds) == 1:
        subprocess.run([py, "evaluate.py", "--method", identity["method"], "--run_dir", str(runs[0]),
                        "--split", "val", "--roles", primary], cwd=ROOT, check=True)
        score, = (row for row in read_csv(metrics[0]) if row["role"] == primary)
        scores = dict(val_mae_mean=float(score["mae"]), val_mae_sd=0.0, val_rmse_mean=float(score["rmse"]))
    else:
        subprocess.run([py, "evaluate.py", "--method", identity["method"], "--run_dir", str(root),
                        "--split", "val", "--roles", primary], cwd=ROOT, check=True)
        score, = (row for row in read_json(root / "summary_val.json")["summary"] if row["role"] == primary)
        scores = {f"val_{k}": score[k] for k in ("mae_mean", "mae_sd", "rmse_mean")}
    if not math.isfinite(scores["val_mae_mean"]):
        raise ValueError(f"{root}: validation MAE is not finite")
    write_json(result_path, dict(scores=scores, hashes=[file_sha256(p) for p in metrics + sources]))
    if not keep_ckpt:
        for run in runs:
            for ckpt in run.glob("*.pt"):
                ckpt.unlink()
    return scores, False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True)
    p.add_argument("--protocol", default="configs/protocol_ppmi.yaml")
    p.add_argument("--grid", help="default: configs/tuning/<method>.yaml")
    p.add_argument("--config", help="method config (overrides the grid's 'config'); passed to train.py --config")
    p.add_argument("--mode", choices=("outer_fold", "pilot"), default="outer_fold")
    p.add_argument("--fold", default="all", help="outer fold to tune, or all configured folds independently")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--limit", type=int, help="run only the first K configurations (checks)")
    p.add_argument("--keep_ckpt", action="store_true")
    p.add_argument("--out", help="tuning root (default: <protocol output_root>/tuning)")
    p.add_argument("--pset", nargs="*", default=[], help="extra protocol overrides applied to every configuration")
    args = p.parse_args()

    grid_path = resolve_path(args.grid or f"configs/tuning/{args.method}.yaml")
    spec = load_yaml(grid_path)
    if spec.get("method", args.method) != args.method:
        raise ValueError(f"{grid_path} is a grid for '{spec.get('method')}', not '{args.method}'")
    protocol = apply_overrides(load_yaml(resolve_path(args.protocol)), args.pset)
    method_cls = get_method(args.method)
    configs = configurations(spec["grid"])
    budget = int(spec.get("budget", protocol["training"]["tuning_budget"]))
    if len(configs) != budget:
        raise ValueError(f"{grid_path}: {len(configs)} configurations, but the budget is {budget}")

    config_arg = args.config or spec.get("config")
    cfg_path = resolve_path(config_arg) if config_arg else method_config_path(protocol, args.method)
    pending = not cfg_path.exists()
    if pending:
        message = f"method config {project_relative(cfg_path)} does not exist yet (pending P1b or another prompt)"
        if not args.dry_run:
            raise FileNotFoundError(message)
        print(f"WARNING: pending: {message}; method keys not validated")
    method_cfg = None if pending else load_yaml(cfg_path)
    # Validate every override BEFORE any training (unknown keys raise KeyError in apply_overrides).
    effective = []
    for i, cfg in enumerate(configs):
        psets, sets = split_overrides(cfg)
        try:
            trial_protocol = apply_overrides(protocol, psets)
            if trial_protocol["data"] != protocol["data"]:
                raise ValueError(f"{grid_path}, configuration {i}: tuning must use the same cohort and split")
            trial_cfg = apply_overrides(method_cfg, sets) if method_cfg is not None else None
            effective.append((trial_protocol, trial_cfg))
        except KeyError as error:
            raise KeyError(f"{grid_path}, configuration {i}: {error}") from None

    out_root = resolve_path(args.out) if args.out else resolve_path(protocol.get("output_root", "runs")) / "tuning"
    folds = protocol["data"]["split"]["folds"]
    if args.limit is not None and not 1 <= args.limit <= budget:
        p.error(f"--limit must be in [1, {budget}]")
    if args.mode == "pilot" and args.fold != "all":
        p.error("--mode pilot uses all folds")
    if args.fold != "all":
        fold = int(args.fold)
        if fold not in folds:
            p.error(f"--fold must be one of {folds}")
        folds = [fold]
    method_root = (out_root / "pilot" if args.mode == "pilot" else out_root) / args.method
    primary = method_cls.primary_role
    base = ["train.py", "--method", args.method, "--protocol", args.protocol]
    if config_arg:
        base += ["--config", config_arg]
    print(f"{args.method}: {len(configs)} configurations (budget {budget}), primary role '{primary}', "
          f"grid {project_relative(grid_path)}, config {project_relative(cfg_path)}, out {method_root}")
    groups = [("all", folds, method_root)] if args.mode == "pilot" else [
        (str(fold), [fold], method_root / f"fold{fold}") for fold in folds]
    for i, cfg in enumerate(configs):
        print(f"  cfg{i}: {json.dumps(cfg)}")
        for group, _, root in groups:
            cmd = trial_command(base, cfg, args.pset, group, root, i)
            print(f"        python {subprocess.list2cmdline(cmd)}")
    if args.dry_run:
        return

    stats = {}
    rows = read_manifest(protocol, stats)
    fingerprint = data_fingerprint(protocol, rows)
    verify_approved_data(stats, fingerprint)
    source = source_hash()
    for group, group_folds, root in groups:
        splits = {str(f): make_split(rows, f, protocol).as_dict() for f in group_folds}
        common = dict(method=args.method, mode=args.mode, folds=group_folds, splits=splits, seed=0,
                      data_fingerprint=fingerprint, source_hash=source)
        select_group(args, base, group, root, common, primary, configs[:args.limit], effective, grid_path, cfg_path)


def trial_command(base, cfg, extra, group, root=None, index=None):
    psets, sets = split_overrides(cfg)
    cmd = base + ["--fold", group, "--seed", "all" if root is None else "0"]
    if root is not None:
        cmd += ["--out", str(root), "--tag", f"cfg{index}"]
    if sets:
        cmd += ["--set", *sets]
    if psets or extra:
        cmd += ["--pset", *extra, *psets]
    return cmd


def select_group(args, base, group, root, common, primary, configs, effective, grid_path, cfg_path):
    trials = []
    for i, cfg in enumerate(configs):
        start = time.time()
        run_root = root / f"{args.method}_cfg{i}"
        cmd = trial_command(base, cfg, args.pset, group, root, i)
        trial_protocol, trial_cfg = effective[i]
        identity = dict(common, index=i, protocol_hash=config_hash(trial_protocol), config_hash=config_hash(trial_cfg))
        row = dict(index=i, **{k: _value(v) for k, v in cfg.items()}, status="ok", error="",
                   val_mae_mean=None, val_mae_sd=None, val_rmse_mean=None, seconds=None,
                   path=project_relative(run_root))
        try:
            scores, resumed = _run_trial(cmd, run_root, identity, primary, args.keep_ckpt, run_root.exists())
            row.update(scores, status="ok (resumed)" if resumed else "ok", seconds=round(time.time() - start, 1))
        except Exception as error:
            row.update(status="failed", error=repr(error)[:500])
        trials.append(row)
        print(f"  cfg{i}: {row['status']} val MAE ({primary}) = {row['val_mae_mean']}")
    write_csv(root / "trials.csv", trials)
    if any(t["status"] == "failed" for t in trials):
        raise RuntimeError(f"Incomplete tuning group; see {root / 'trials.csv'}")
    best = min(trials, key=lambda t: (t["val_mae_mean"], t["index"]))
    cfg = configs[best["index"]]
    psets, sets = split_overrides(cfg)
    final = ["python", *trial_command(base, cfg, args.pset, group)] if args.mode == "outer_fold" else None
    protocol, method_cfg = effective[best["index"]]
    # Split membership stays in trial/run metadata. Selection records carry the fingerprint and fold IDs.
    selected = dict({k: v for k, v in common.items() if k != "splits"},
                    protocol=args.protocol, protocol_name=protocol["name"],
                    protocol_hash=config_hash(protocol), config_hash=config_hash(method_cfg),
                    grid=project_relative(grid_path), config=project_relative(cfg_path), primary_role=primary,
                    selection="lowest primary-role validation MAE; ties follow grid order",
                    n_trials_run=len(trials), n_trials_ok=len(trials), selected_index=best["index"], values=cfg,
                    val_mae_mean=best["val_mae_mean"], set=sets, pset=[*args.pset, *psets],
                    exploratory=args.mode == "pilot" or args.limit is not None,
                    final_command=subprocess.list2cmdline(final) if final and args.limit is None else None)
    (root / "selected.yaml").write_text(
        "# Written by scripts/tune.py (STEWARD D-007/D-038). Validation only; test was never used.\n"
        + yaml.safe_dump(selected, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"selected cfg{best['index']} {json.dumps(cfg)} (val MAE {best['val_mae_mean']:.3f})")
    print(f"final command: {selected['final_command']}")
    print(f"Wrote {root / 'trials.csv'} and {root / 'selected.yaml'}")


if __name__ == "__main__":
    main()
