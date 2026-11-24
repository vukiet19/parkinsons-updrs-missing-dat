"""Opt-in MRI-only smoke on the approved OFF30 cohort. Run from benchmarks/ppmi_3d."""

import argparse
import io
import json
import math
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate
import export_inference
import infer
import train
from baselines.anchors.method import MRIOnlyMethod
from baselines.anchors.model import MRIOnlyNet
from data.dataset import ImageRows
from data.manifest import make_split, read_manifest, rows_of
from models.backbones import resnet
from scripts import tune
from utils.config import config_hash, load_yaml, resolve_path
from utils.io import read_csv, read_json, write_csv, write_json
from utils.repro import choose_device, file_sha256

PROTOCOL = "configs/protocol_ppmi.yaml"
METHOD_ARGS = ("--method", "mri_only")
EPOCHS = 2


def run_cli(module, *args):
    argv = [Path(module.__file__).relative_to(ROOT).as_posix(), *map(str, args)]
    print(subprocess.list2cmdline([sys.executable, *argv]), flush=True)
    with patch.object(sys, "argv", argv):
        module.main()


def check_dry_run(out):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_cli(tune, *METHOD_ARGS, "--protocol", PROTOCOL, "--dry_run")
    log = buffer.getvalue()
    (out / "dry_run.log").write_text(log, encoding="utf-8")
    configs = [line for line in log.splitlines() if line.strip().startswith("cfg")]
    planned_runs = [line for line in log.splitlines() if line.strip().startswith("python train.py")]
    assert len(configs) == 4 and len(planned_runs) == 20
    assert "primary role 'student'" in log and "WARNING" not in log
    return [json.loads(line.partition(":")[2]) for line in configs]


def check_train_and_evaluate(out, protocol, train_rows, val_rows, device):
    opened = set()
    batches, losses = [], []
    gradients = {"encoder_stem": [], "head": []}
    initial_weights = {}
    train_paths = {row.mri_path for row in train_rows}
    val_paths = {row.mri_path for row in val_rows}
    allowed = train_paths | val_paths
    image = ImageRows._image
    forward = MRIOnlyNet.forward
    training_step = MRIOnlyMethod.training_step

    def check_image(dataset, path, channels):
        assert path in allowed and not dataset.load_dat, "Unexpected image request from the student loader"
        opened.add(path)
        return image(dataset, path, channels)

    def check_forward(model, mri):
        assert tuple(mri.shape[1:]) == (1, 97, 115, 97)
        result = forward(model, mri)
        pred = result["pred"]
        assert pred.shape == (len(mri),) and torch.isfinite(pred).all()
        if model.training:
            batches.append({"mri": list(mri.shape), "pred": list(pred.shape)})
        return result

    def record_gradient(name):
        def check_gradient(gradient):
            norm = gradient.norm().item()
            assert math.isfinite(norm) and norm > 0, f"Invalid gradient for {name}: {norm}"
            gradients[name].append(norm)
        return check_gradient

    def check_training_step(method, stage, model, frozen, batch):
        assert stage == "student" and not frozen
        assert "dat" not in batch and not batch["has_dat"].any() and batch["aux_mask"].all()
        weights = {"encoder_stem": model.encoder.stem[0].weight, "head": model.head.weight}
        if not losses:
            load = model.encoder.load_report
            assert load["sha256"] == protocol["backbone"]["weights_sha256"]
            assert load["n_loaded"] == load["n_encoder_tensors"]
            for name, parameter in weights.items():
                initial_weights[name] = parameter.detach().clone()
                parameter.register_hook(record_gradient(name))
        if len(losses) == 1:
            for name, parameter in weights.items():
                assert not torch.equal(initial_weights[name], parameter), f"Adam did not update {name}"
        loss, _ = result = training_step(method, stage, model, frozen, batch)
        assert loss.ndim == 0 and torch.isfinite(loss)
        losses.append(loss.item())
        return result

    run = out / "mri_only/fold0_seed0"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with patch.object(ImageRows, "_image", check_image), patch.object(MRIOnlyNet, "forward", check_forward):
        with patch.object(MRIOnlyMethod, "training_step", check_training_step):
            run_cli(train, *METHOD_ARGS, "--protocol", PROTOCOL, "--fold", 0, "--seed", 0,
                    "--pset", f"training.max_epochs={EPOCHS}", "--out", out, "--device", device)
        assert opened == train_paths | val_paths
        train_vram = ({"allocated_bytes": torch.cuda.max_memory_allocated(),
                       "reserved_bytes": torch.cuda.max_memory_reserved()} if device.type == "cuda" else None)
        opened.clear()
        allowed = val_paths
        for checkpoint in ("best", "last"):
            run_cli(evaluate, *METHOD_ARGS, "--run_dir", run, "--split", "val", "--roles", "student",
                    "--checkpoint", checkpoint, "--device", device)
        assert opened == val_paths
    assert [batch["mri"][0] for batch in batches] == [8, 8, 1] * EPOCHS
    assert all(len(values) == len(batches) for values in gradients.values())

    meta = read_json(run / "run.json")
    assert meta["status"] == "complete" and set(meta["stages"]) == {"student"}
    assert meta["protocol"]["training"] == {**protocol["training"], "max_epochs": EPOCHS}
    train_targets = [row.target for row in train_rows]
    expected_scaler = [np.mean(train_targets), np.std(train_targets)]
    np.testing.assert_allclose([meta["scaler"]["mean"], meta["scaler"]["std"]], expected_scaler, rtol=0, atol=1e-12)
    history = [(int(row["epoch"]), float(row["val_mae"])) for row in read_csv(run / "history.csv")]
    assert len(history) == EPOCHS and all(math.isfinite(mae) for _, mae in history)
    best_epoch, best_mae = min(history, key=lambda row: row[1])
    info = meta["stages"]["student"]
    assert (info["best_epoch"], info["val"]["mae"]) == (best_epoch, best_mae)
    assert (info["last"]["epoch"], info["last"]["mae"]) == history[-1]
    for folder, expected in (("eval_val", best_mae), ("eval_val_last", history[-1][1])):
        student = next(row for row in read_csv(run / folder / "metrics.csv") if row["role"] == "student")
        assert int(student["n"]) == len(val_rows)
        assert math.isclose(float(student["mae"]), expected, rel_tol=0, abs_tol=1e-5)
    return run, {
        "batches": batches, "batch_losses": losses, "gradient_l2": gradients,
        "mri_loader_counts": {"train": len(train_paths), "val": len(val_paths), "dat": 0},
        "train_vram": train_vram, "train_scaler": meta["scaler"], "val_mae_by_epoch": dict(history),
        "best_epoch": best_epoch, "best_val_mae": best_mae,
        **{key: meta[key] for key in ("fold", "seed", "device", "gpu", "data_fingerprint",
                                     "source_hash", "protocol_hash", "config_hash", "backbone_resolved")},
    }


def check_bundle(out, run, val_rows, device):
    manifest = out / "mri_inputs.csv"
    bundle = out / "student_best_bundle.pt"
    prediction = out / "bundle_predictions.csv"
    write_csv(manifest, [dict(subject_id=row.subject_id, visit_id=row.visit_id, mri_path=row.mri_path)
                         for row in val_rows])
    run_cli(export_inference, "--run_dir", run, "--output", bundle)
    expected_images = {row.mri_path for row in val_rows}
    loaded_images = set()
    load_nifti = nib.load
    open_path = Path.open
    readable_paths = expected_images | {manifest, bundle, prediction, prediction.with_suffix(".json")}

    def check_open(path, *args, **kwargs):
        assert path.resolve() in readable_paths, "Bundle inference opened an unexpected file"
        return open_path(path, *args, **kwargs)

    def check_nifti(path, *args, **kwargs):
        path = Path(path).resolve()
        assert path in expected_images, "Bundle inference requested a non-validation MRI image"
        loaded_images.add(path)
        return load_nifti(path, *args, **kwargs)

    with patch.object(Path, "open", check_open), patch.object(nib, "load", check_nifti), patch.object(
        resnet, "medicalnet_state_dict", side_effect=AssertionError("Bundle requested pretrained weights")
    ):
        run_cli(infer, "--bundle", bundle, "--manifest", manifest, "--output", prediction,
                "--batch_size", 8, "--device", device)
    assert loaded_images == expected_images
    expected = {row["unit"]: float(row["y_pred"]) for row in read_csv(run / "eval_val/predictions.csv")
                if row["role"] == "student"}
    actual = {row["unit"]: float(row["y_pred"]) for row in read_csv(prediction)}
    assert actual.keys() == expected.keys() == {row.unit for row in val_rows}
    delta = max(abs(expected[unit] - actual[unit]) for unit in expected)
    assert delta <= 1e-5, f"Evaluator/bundle parity: max absolute difference {delta}"
    report = read_json(prediction.with_suffix(".json"))
    assert report["grid_validation"] == "shape_and_affine"
    assert report["provenance"]["checkpoint_sha256"] == file_sha256(run / "student.pt")
    return bundle, {"n": len(actual), "max_abs_delta_updrs": delta,
                    "tolerance_updrs": 1e-5, "inference": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New smoke output folder under runs_ppmi/off30")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    protocol_path = resolve_path(PROTOCOL)
    protocol = load_yaml(protocol_path)
    out = resolve_path(args.out).resolve()
    assert out.is_relative_to(resolve_path(protocol["output_root"]))
    out.mkdir(parents=True, exist_ok=False)
    config_grid = check_dry_run(out)
    stats = {}
    rows = read_manifest(protocol, stats)
    split = make_split(rows, 0, protocol)
    train_rows, val_rows = rows_of(rows, split.train), rows_of(rows, split.val)
    assert (len(train_rows), len(val_rows), len(split.test)) == (17, 7, 6)
    assert all(row.dat_path and math.isfinite(row.aux) for row in train_rows + val_rows)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    run, checks = check_train_and_evaluate(out, protocol, train_rows, val_rows, device)
    approval = stats["qc_approval"]
    assert checks["data_fingerprint"] == approval["data_fingerprint"]
    bundle, parity = check_bundle(out, run, val_rows, device)
    report = out / "checks.json"
    write_json(report, dict(status="READY", protocol=protocol["name"], protocol_file_sha256=file_sha256(protocol_path),
                            manifest_sha256=approval["manifest_sha256"], qc_sha256=approval["qc_sha256"],
                            base_protocol_hash=config_hash(protocol), grid=config_grid, parity=parity, **checks))
    for checkpoint in (run / "student.pt", run / "student_last.pt", bundle):
        checkpoint.unlink()
    print(f"READY. Evidence: {report}. Removed the two smoke checkpoints and temporary bundle.")


if __name__ == "__main__":
    main()
