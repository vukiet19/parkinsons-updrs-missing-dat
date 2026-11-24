"""Shared training entry point: any registered method, any fold/seed of the shared protocol.

    python train.py --method crad --protocol configs/protocol.yaml --config configs/baselines/crad.yaml --fold 0 --seed 0
    python train.py --method crad --fold all --seed all
    python train.py --method mri_only --protocol configs/protocol_ppmi.yaml      # -> runs_ppmi/off30/mri_only

Output root: --out, else protocol ``output_root`` (PPMI-OFF30: runs_ppmi/off30), else runs/ (NTUA).
Model selection uses the validation subjects only (best val MAE); test subjects are never loaded.
"""

from __future__ import annotations

import argparse
import copy
import math
import platform
import shutil
import time
from pathlib import Path

import torch
import torchvision

from baselines import get_method
from data.dataset import make_loader, to_device
from data.manifest import TargetScaler, data_fingerprint, make_split, read_manifest, rows_of, verify_approved_data
from utils.config import apply_overrides, config_hash, load_yaml, method_config_path, project_relative, resolve_path
from utils.inference import freeze, predict_units
from utils.io import read_json, write_csv, write_json
from utils.metrics import regression_metrics
from utils.repro import choose_device, file_sha256, load_run_checkpoint, seed_everything, source_hash


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True)
    p.add_argument("--protocol", default="configs/protocol.yaml")
    p.add_argument("--config", help="method config (default: configs/baselines/<method>.yaml)")
    p.add_argument("--fold", default="all", help="outer fold id or 'all'")
    p.add_argument("--seed", default="all", help="training seed or 'all' (protocol.training.seeds)")
    p.add_argument("--out", default=None, help="output root (default: protocol output_root, else runs)")
    p.add_argument("--tag", default="", help="suffix of the run folder, e.g. an ablation name")
    p.add_argument("--set", nargs="*", default=[], help="method-config overrides key.sub=value")
    p.add_argument("--pset", nargs="*", default=[], help="protocol overrides key.sub=value")
    p.add_argument("--device", default=None, help="auto | cuda | cpu (default: protocol)")
    p.add_argument("--overwrite", action="store_true", help="re-train runs that already exist")
    p.add_argument("--teacher_from", help="reuse teacher.pt of the same fold/seed from this run folder "
                   "(only allowed when protocol, split and teacher-side config are identical)")
    return p.parse_args()


TEACHER_SECTIONS = ("teacher", "cafr", "ort")   # CRAD default; a method may declare `teacher_sections`


def reusable_teacher(source_root: Path, fold, seed, protocol, cfg, split, scaler, sections=TEACHER_SECTIONS,
                     *, method_name=None, fingerprint=None):
    """Return the source path and metadata after checking the teacher checkpoint.

    ``sections``: method-config keys that influence teacher training (absent keys compare as None)."""
    run = source_root / f"fold{fold}_seed{seed}"
    meta = read_json(run / "run.json")
    method_name = method_name or cfg.get("method")
    if fingerprint is None:
        fingerprint = data_fingerprint(protocol, read_manifest(protocol))
    checks = dict(status=meta["status"] == "complete", protocol=meta["protocol"] == protocol,
                  identity=(meta["method"], meta["fold"], meta["seed"]) == (method_name, fold, seed),
                  data_fingerprint=meta.get("data_fingerprint") == fingerprint,
                  split=meta["split"] == split.as_dict(), scaler=meta["scaler"] == dict(mean=scaler.mean, std=scaler.std),
                  teacher_config=all(meta["config"].get(k) == cfg.get(k) for k in sections))
    if not all(checks.values()):
        raise ValueError(f"Teacher of {run} is not reusable: {[k for k, ok in checks.items() if not ok]}")
    path = run / "teacher.pt"
    if not path.exists():
        path = resolve_path(meta["stages"]["teacher"]["reused_from"])
    return path, meta["stages"]["teacher"], load_run_checkpoint(path, meta, "teacher")


class Logger:
    def __init__(self, path: Path):
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, message: str):
        print(message, flush=True)
        self.handle.write(message + "\n")

    def close(self):
        self.handle.close()


def fit_stage(method, stage, frozen, train_rows, val_rows, scaler, protocol, device, seed, log, keep_last=None):
    """Train one stage; return the BEST-val-MAE model (used for freezing / init_stage), its metrics and history.

    ``keep_last`` (dict, D-013): if given, filled with the last epoch's number, val metrics and state_dict.
    It never changes which model is returned."""
    tr = protocol["training"]
    seed_everything(seed, bool(tr["deterministic"]))
    model = method.build(stage).to(device)
    if hasattr(method, "init_stage"):          # optional hook, e.g. DFTD stage 2 starts from stage 1
        method.init_stage(stage, model, frozen)
    if hasattr(method, "describe"):            # optional hook (AGMD P1b): method-specific lines for train.log
        for line in method.describe(stage, model):
            log(f"  [{stage}] {line}")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    train_loader = make_loader(method.train_rows(stage, train_rows), protocol, method.needs_dat(stage, "train"),
                               shuffle=True, seed=seed)
    val_loader = make_loader(method.train_rows(stage, val_rows) if stage == "teacher" else val_rows, protocol,
                             method.needs_dat(stage, "eval"), shuffle=False, seed=seed)
    best, best_state, history = dict(mae=math.inf, epoch=0), None, []
    for epoch in range(1, int(tr["max_epochs"]) + 1):
        model.train()
        sums, seen = {}, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            batch["y_z"] = scaler.normalize(batch["y"])
            loss, terms = method.training_step(stage, model, frozen, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {stage} loss at epoch {epoch}: {terms}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = len(batch["unit"])
            for key, value in dict(loss=loss, **terms).items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * n
            seen += n
        val = regression_metrics(*zip(*[(r["y_true"], r["y_pred"]) for r in
                                         predict_units(method, stage, model, val_loader, scaler, device)]))
        row = dict(stage=stage, epoch=epoch, **{f"train_{k}": v / seen for k, v in sums.items()},
                   **{f"val_{k}": v for k, v in val.items()})
        history.append(row)
        if val["mae"] < best["mae"]:
            best = dict(epoch=epoch, **val)
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if epoch == 1 or epoch % 10 == 0 or epoch == int(tr["max_epochs"]):
            log(f"  {stage:7s} ep {epoch:3d}  loss {row['train_loss']:.4f}  val MAE {val['mae']:.3f}"
                f"  (best {best['mae']:.3f} @ {best['epoch']})")
    if keep_last is not None:
        keep_last.update(epoch=history[-1]["epoch"], val=dict(val),
                         state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    return model, best, history


def resolved_backbone(protocol: dict) -> dict:
    """Backbone settings actually used, with the defaults of absent keys made explicit (D-024), plus the
    pretrained-weight load report (D-020/D-023; None without MedicalNet)."""
    from models.backbones import backbone_load_report   # local import: P0b only touches this function
    cfg = protocol["backbone"]
    keys = ("shortcut", "pretrained", "weights_path", "weights_sha256")
    return dict(shortcut=cfg.get("shortcut", "B"), pretrained=cfg.get("pretrained", "none"),
                weights_path=cfg.get("weights_path"), weights_sha256=cfg.get("weights_sha256"),
                load_report=backbone_load_report(protocol),
                defaults_used=[k for k in keys if k not in cfg])


def output_root(protocol: dict, out: str | None) -> Path:
    return resolve_path(out if out is not None else protocol.get("output_root", "runs"))


def run_identity(method, protocol, cfg, fingerprint, fold, seed):
    return dict(method=method, fold=fold, seed=seed, data_fingerprint=fingerprint, source_hash=source_hash(),
                protocol_hash=config_hash(protocol), config_hash=config_hash(cfg))


def run_one(method_name, method_cls, protocol, cfg, rows, fingerprint, fold, seed, run_dir, device,
            teacher_from=None, med_state=None):
    """``med_state``: the stats dict filled by read_manifest (D-026); None when the caller did not pass one."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "train.log")
    split = make_split(rows, fold, protocol)
    train_rows, val_rows = rows_of(rows, split.train), rows_of(rows, split.val)
    scaler = TargetScaler.fit(train_rows)          # fitted on training subjects of this fold only
    meta = dict(**run_identity(method_name, protocol, cfg, fingerprint, fold, seed),
                status="running", protocol=protocol, config=cfg, dataset=protocol["data"]["dataset"],
                split=split.as_dict(), scaler=dict(mean=scaler.mean, std=scaler.std),
                python=platform.python_version(),
                torch=torch.__version__, torchvision=torchvision.__version__, device=str(device),
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None, stages={},
                med_state=med_state if med_state is not None else
                dict(column_present=None, note="not supplied by the caller of run_one"),
                backbone_resolved=resolved_backbone(protocol))
    write_json(run_dir / "run.json", meta)
    bb = meta["backbone_resolved"]
    log(f"backbone: shortcut={bb['shortcut']} pretrained={bb['pretrained']} weights_path={bb['weights_path']}"
        + (f" (defaults for absent keys: {bb['defaults_used']})" if bb["defaults_used"] else ""))
    log(f"med_state: {meta['med_state']}")
    log(f"[{method_name}] fold {fold} seed {seed}: train {len(split.train)} / val {len(split.val)} / "
        f"test {len(split.test)} subjects ({len(train_rows)} / {len(val_rows)} train/val rows); "
        f"target mean {scaler.mean:.2f} sd {scaler.std:.2f}")
    method = method_cls(protocol, cfg)
    frozen, history = {}, []
    try:
        if teacher_from is not None and "teacher" not in method.stages:
            raise ValueError(f"--teacher_from needs a separately trained 'teacher' stage; {method_name} "
                             f"has stages {method.stages}.")
        for stage in method.stages:
            if not method.train_rows(stage, train_rows):
                meta["stages"][stage] = dict(skipped=True, reason="no training rows for this stage")
                log(f"  -> {stage}: skipped (no training rows for this stage in this fold)")
                continue
            if stage == "teacher" and teacher_from is not None:
                path, info, ckpt = reusable_teacher(teacher_from, fold, seed, protocol, cfg, split, scaler,
                                                    getattr(method, "teacher_sections", TEACHER_SECTIONS),
                                                    method_name=method_name, fingerprint=fingerprint)
                teacher = method.build("teacher")
                teacher.load_state_dict(ckpt["state_dict"])
                del ckpt
                frozen[stage] = freeze(teacher.to(device))
                meta["stages"][stage] = dict(info, reused_from=project_relative(path))
                log(f"  -> teacher reused from {path} (identical protocol/split/teacher config)")
                if hasattr(method, "describe"):    # optional hook, as in fit_stage
                    for line in method.describe(stage, teacher):
                        log(f"  [{stage}] {line}")
                continue
            start = time.time()
            # D-013: optionally keep the last-epoch weights too (sensitivity only; training is unchanged).
            last = {} if protocol["training"].get("save_last_checkpoint", False) else None
            model, best, hist = fit_stage(method, stage, frozen, train_rows, val_rows, scaler, protocol,
                                          device, seed, log, keep_last=last)
            history += hist
            identity = dict(method=method_name, stage=stage, fold=fold, seed=seed, data_fingerprint=fingerprint)
            path = run_dir / f"{stage}.pt"
            torch.save(dict(**identity, best=best, state_dict={k: v.cpu() for k, v in model.state_dict().items()}), path)
            meta["stages"][stage] = dict(best_epoch=best["epoch"], val=best, seconds=round(time.time() - start, 1),
                                         checkpoint_sha256=file_sha256(path))
            if last is not None:
                last_path = run_dir / f"{stage}_last.pt"
                torch.save(dict(**identity, best=best, last_epoch=last["epoch"], state_dict=last["state_dict"]), last_path)
                meta["stages"][stage]["last"] = {**last["val"], "epoch": last["epoch"],
                                                "checkpoint_sha256": file_sha256(last_path)}
                log(f"  -> {stage}: last epoch {last['epoch']} saved as {stage}_last.pt (val MAE {last['val']['mae']:.3f})")
            log(f"  -> {stage}: best epoch {best['epoch']}, val MAE {best['mae']:.3f}, "
                f"RMSE {best['rmse']:.3f} ({time.time() - start:.0f}s)")
            frozen[stage] = freeze(model)
        meta["status"] = "complete"
    except Exception as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        if history:
            write_csv(run_dir / "history.csv", history)
        write_json(run_dir / "run.json", meta)
        log.close()


def main():
    args = parse_args()
    protocol = apply_overrides(load_yaml(resolve_path(args.protocol)), args.pset)
    cfg_path = resolve_path(args.config) if args.config else method_config_path(protocol, args.method)
    cfg = apply_overrides(load_yaml(cfg_path), args.set)
    if cfg.get("method", args.method) != args.method:
        raise ValueError(f"Config {cfg_path} is for method '{cfg.get('method')}', not '{args.method}'.")
    device = choose_device(args.device or protocol["training"]["device"])
    method_cls = get_method(args.method)
    med_state = {}
    rows = read_manifest(protocol, med_state)
    fingerprint = data_fingerprint(protocol, rows)
    verify_approved_data(med_state, fingerprint)
    folds = protocol["data"]["split"]["folds"] if args.fold == "all" else [int(args.fold)]
    seeds = protocol["training"]["seeds"] if args.seed == "all" else [int(args.seed)]
    root = output_root(protocol, args.out) / (args.method + (f"_{args.tag}" if args.tag else ""))
    print(f"{args.method}: {len(rows)} rows, {len({r.subject_id for r in rows})} subjects; device={device}; "
          f"folds={folds} seeds={seeds}; output={root}")
    for fold in folds:
        for seed in seeds:
            run_dir = root / f"fold{fold}_seed{seed}"
            if run_dir.exists():
                recorded = read_json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
                status = recorded.get("status")
                if args.overwrite:
                    shutil.rmtree(run_dir)
                elif status == "complete":
                    expected = run_identity(args.method, protocol, cfg, fingerprint, fold, seed)
                    if "qc" in protocol["data"] and any(recorded.get(k) != v for k, v in expected.items()):
                        raise ValueError(f"{run_dir}: existing run differs from the requested OFF30 run.")
                    print(f"skip {run_dir.name}: already complete (use --overwrite to retrain)")
                    continue
                else:
                    raise FileExistsError(f"{run_dir} exists with status {status}; use --overwrite.")
            run_one(args.method, method_cls, protocol, cfg, rows, fingerprint, fold, seed, run_dir, device,
                    resolve_path(args.teacher_from) if args.teacher_from else None, med_state)
    summary = []
    for run_json in sorted(root.glob("fold*_seed*/run.json")):
        meta = read_json(run_json)
        for stage, info in meta.get("stages", {}).items():
            if info.get("skipped"):
                continue
            summary.append(dict(fold=meta["fold"], seed=meta["seed"], stage=stage, best_epoch=info["best_epoch"],
                                **{f"val_{k}": v for k, v in info["val"].items() if k != "epoch"}))
    if summary:
        write_csv(root / "val_summary.csv", summary)
        print(f"Validation summary: {root / 'val_summary.csv'}")


if __name__ == "__main__":
    main()
