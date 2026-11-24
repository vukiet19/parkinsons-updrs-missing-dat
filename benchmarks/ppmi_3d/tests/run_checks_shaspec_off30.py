"""Verify ShaSpec on synthetic inputs or run one approved OFF30 smoke fold."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import warnings
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402
import train  # noqa: E402
from baselines.shaspec.method import ShaSpecMethod  # noqa: E402
from data.dataset import make_loader  # noqa: E402
from data.manifest import TargetScaler, data_fingerprint, read_manifest, rows_of, verify_approved_data  # noqa: E402
from utils.config import apply_overrides, config_hash, load_yaml, resolve_path  # noqa: E402
from utils.inference import predict_units  # noqa: E402
from utils.io import read_csv, read_json, write_csv, write_json  # noqa: E402
from utils.mri_inference import export_run, predict_mri, read_mri_manifest  # noqa: E402
from utils.repro import choose_device, load_run_checkpoint, seed_everything  # noqa: E402


def check_gradients(parameters):
    norms = torch.stack([parameter.grad.detach().norm() for parameter in parameters if parameter.grad is not None])
    assert bool(torch.isfinite(norms).all()) and bool((norms > 0).any()), "Invalid gradient norms"
    return len(norms)


def fixture_checks(protocol, cfg, device):
    seed_everything(0)
    shape = tuple(protocol["preprocessing"]["input_size"])
    images = {modality: torch.randn(2, 1, *shape, device=device) for modality in ("mri", "dat")}
    present = {modality: torch.ones(2, dtype=torch.bool, device=device) for modality in images}
    method = ShaSpecMethod(protocol, cfg)
    split_seed = None
    for seed in range(32):
        torch.manual_seed(seed)
        availability = method._sample_availability(present)
        if availability["mri"].sum() == 1 and torch.equal(availability["mri"], ~availability["dat"]):
            split_seed = seed
            break
    assert split_seed is not None, "Fixture needs one sample for each modality and no complete pair"
    torch.manual_seed(split_seed)
    for modality, mask in method._sample_availability(present).items():
        assert torch.equal(mask, availability[modality])

    model = method.build("main").to(device)
    batch = dict(**images, has_dat=present["dat"], y_z=torch.tensor([-0.5, 0.5], device=device))
    torch.manual_seed(split_seed)
    loss, terms = method.training_step("main", model, {}, batch)
    assert torch.isfinite(loss)
    assert terms["mri_avail"] == terms["dat_avail"] == 0.5
    assert terms["dao"] == 0 and torch.isfinite(terms["dco"]) and terms["dco"] > 0
    loss.backward()
    check_gradients(model.parameters())
    for encoder in model.specific.values():
        check_gradients(encoder.parameters())
    del model, loss, terms

    mri = dict(mri=images["mri"][:1])
    mri_only = dict(mri=present["mri"][:1], dat=~present["dat"][:1])
    for fusion, feature_shape in (("linear", (1, 512)), ("conv", (1, 512, 4, 4, 4))):
        model = ShaSpecMethod(protocol, apply_overrides(cfg, [f"fusion.type={fusion}"])).build("main").to(device).eval()
        head_input = []
        hook = model.head.register_forward_pre_hook(lambda _, inputs: head_input.append(inputs[0]))
        with torch.no_grad():
            output = model(mri, mri_only)
        hook.remove()
        assert output["r"]["mri"].shape == output["s"]["mri"].shape == feature_shape
        assert output["pred"].shape == (1,) and torch.isfinite(output["pred"]).all()
        shared = output["r"]["mri"].reshape(1, 512, -1).mean(dim=2)
        torch.testing.assert_close(head_input[0][:, 512:], shared, rtol=0, atol=0)
        del model, output, head_input
    return dict(status="PASS", shape=shape, one_sample_per_modality_seed=split_seed, device=str(device),
                dao_zero_without_pair=True, dco_and_specific_gradients_finite=True,
                singleton_mri_fusion=["linear", "conv"], weights=protocol["backbone"]["pretrained"])


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy())
    return digest.hexdigest()


def train_profiled(protocol, cfg, run_dir, rows, fingerprint, stats, device, report):
    cuda = device.type == "cuda"
    if cuda:
        # Trigger any allocator support warning before opening the train log.
        with warnings.catch_warnings(record=True) as caught:
            torch.empty(1, device=device)
        report["allocator_warnings"] = [str(entry.message) for entry in caught]
        report["cuda_backend"] = torch.cuda.get_allocator_backend()
        report["cuda_free_before_bytes"], report["cuda_total_bytes"] = torch.cuda.mem_get_info(device)

    class ProfiledShaSpec(ShaSpecMethod):
        def init_stage(self, stage, model, frozen):
            super().init_stage(stage, model, frozen)
            if stage == "main":
                report["main_initial_state_sha256"] = state_digest(model.state_dict())

        def training_step(self, stage, model, frozen, batch):
            size = len(batch["unit"])
            report["pending_step"] = dict(stage=stage, batch_size=size)
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            loss, terms = super().training_step(stage, model, frozen, batch)
            report["pending_step"].update({name: float(value.detach()) for name, value in terms.items()})
            return loss, terms

    def after_adam(optimizer, _args, _kwargs):
        assert isinstance(optimizer, torch.optim.Adam)
        step = report["pending_step"]
        if cuda:
            torch.cuda.synchronize(device)
            step.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        step["gradient_tensors_finite"] = check_gradients(
            parameter for group in optimizer.param_groups for parameter in group["params"]
        )
        assert all(math.isfinite(step[name]) for name in ("task", "dao", "dco"))
        report["steps"].append(step)
        del report["pending_step"]
        print(f"Adam step {len(report['steps'])}: {step}", flush=True)

    hook = register_optimizer_step_post_hook(after_adam)
    try:
        train.run_one("shaspec", ProfiledShaSpec, protocol, cfg, rows, fingerprint, 0, 0, run_dir, device,
                      med_state=stats)
    except torch.OutOfMemoryError:
        report["oom"] = True
        if cuda:
            report["oom_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["oom_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            report["cuda_free_at_oom_bytes"] = torch.cuda.mem_get_info(device)[0]
        raise
    finally:
        hook.remove()


def check_warm_start(run_dir, metadata, report):
    stage = metadata["stages"]["warmup"]
    history = [row for row in read_csv(run_dir / "history.csv") if row["stage"] == "warmup"]
    best = min(history, key=lambda row: float(row["val_mae"]))
    ckpt = load_run_checkpoint(run_dir / "warmup.pt", metadata, "warmup")
    initial_digest = state_digest(ckpt["state_dict"])
    assert stage["best_epoch"] == ckpt["best"]["epoch"] == int(best["epoch"])
    assert initial_digest == report["main_initial_state_sha256"]
    last = load_run_checkpoint(run_dir / "warmup_last.pt", metadata, "warmup", "last")
    last_digest = state_digest(last["state_dict"])
    report.update(warmup_best_epoch=stage["best_epoch"], warmup_last_epoch=last["last_epoch"],
                  warmup_state_matches_main_init=True, warmup_best_differs_from_last=initial_digest != last_digest)


def check_bundle(run_dir, out, protocol, metadata, rows, device):
    predictions, scores = evaluate.evaluate_run("shaspec", run_dir, "val", ["all"], device, False, {})
    assert {score["role"] for score in scores} == {"student", "teacher", "train_mean"}
    expected = {row["unit"]: row["y_pred"] for row in predictions if row["role"] == "student"}
    main_val_mae = metadata["stages"]["main"]["val"]["mae"]
    student_mae = next(score["mae"] for score in scores if score["role"] == "student")
    assert math.isclose(student_mae, main_val_mae, rel_tol=0, abs_tol=1e-5), "Main validation role differs"
    val_rows = rows_of(rows, metadata["split"]["val"])
    # Compare models at the same batch size. CUDA float32 changes rounding when batch size changes.
    singleton_protocol = apply_overrides(protocol, ["training.batch_size=1"])
    singleton_loader = make_loader(val_rows, singleton_protocol, load_dat=False, shuffle=False, seed=0)
    method = ShaSpecMethod(protocol, metadata["config"])
    model = method.build("main")
    model.load_state_dict(load_run_checkpoint(run_dir / "main.pt", metadata, "main")["state_dict"])
    scaler = TargetScaler(**metadata["scaler"])
    singleton_reference = {
        row["unit"]: row["y_pred"]
        for row in predict_units(method, "student", model.to(device), singleton_loader, scaler, device)
    }
    del model

    manifest = out / "validation_mri.csv"
    write_csv(manifest, [dict(subject_id=row.subject_id, visit_id=row.visit_id, mri_path=str(row.mri_path))
                         for row in val_rows])
    bundle_path = out / "student.pt"
    torch.save(export_run(run_dir), bundle_path)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    assert bundle["provenance"]["stage"] == "main" and bundle["provenance"]["checkpoint"] == "best"
    assert bundle["reference_grid"] == dict(shape=protocol["preprocessing"]["input_size"],
                                             affine=protocol["preprocessing"]["affine"])
    assert bundle["protocol"]["data"] == {"spatial_dims": 3}
    assert not any(key.startswith(("specific.dat.", "dom_cls.")) for key in bundle["state_dict"])
    inputs = read_mri_manifest(manifest)
    differences = {}
    for batch_size, reference in ((protocol["training"]["batch_size"], expected), (1, singleton_reference)):
        actual = predict_mri(bundle, inputs, device, batch_size=batch_size)
        assert {row["unit"] for row in actual} == reference.keys() == expected.keys()
        delta = max(abs(row["y_pred"] - reference[row["unit"]]) for row in actual)
        assert delta <= 1e-4, f"{batch_size=}: {delta:g} UPDRS points"
        differences[str(batch_size)] = delta
        write_csv(out / f"bundle_validation_b{batch_size}.csv", actual)
    batch_delta = max(abs(expected[unit] - singleton_reference[unit]) for unit in expected)
    return dict(split="val", matched_units=len(expected), roles=sorted({score["role"] for score in scores}),
                max_evaluator_delta_across_batch_sizes_updrs=batch_delta,
                max_abs_delta_updrs_by_batch=differences, atol_updrs=1e-4)


def smoke(protocol, cfg, out, device, verify_only=False):
    out.mkdir(parents=True, exist_ok=verify_only)
    run_dir = out / "shaspec" / "fold0_seed0"
    report = dict(status="BLOCKED", protocol=protocol["name"], protocol_hash=config_hash(protocol),
                  method_config_hash=config_hash(cfg), fold=0, seed=0, torch=torch.__version__,
                  device=str(device), allocator_env={key: os.environ.get(key) for key in
                                                      ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")},
                  budget=dict(epochs_per_stage=protocol["training"]["max_epochs"],
                              batch_size=protocol["training"]["batch_size"]),
                  oom=False, steps=[])
    if verify_only:
        prior_report = read_json(out / "report.json")
        for key in ("protocol_hash", "method_config_hash", "device", "allocator_env"):
            assert prior_report[key] == report[key], f"Smoke settings differ: {key}"
        prior_report["previous_verification_issue"] = prior_report.pop("reason", None)
        report = prior_report
    try:
        stats = {}
        rows = read_manifest(protocol, stats)
        fingerprint = data_fingerprint(protocol, rows)
        verify_approved_data(stats, fingerprint)
        report.update(data_fingerprint=fingerprint, qc_counts=stats["qc_approval"]["qc_counts"])
        assert len(rows) == stats["qc_approval"]["n_after_qc"] and all(row.dat_path is not None for row in rows)
        if not verify_only:
            train_profiled(protocol, cfg, run_dir, rows, fingerprint, stats, device, report)
        for stage in ("warmup", "main"):
            sizes = [step["batch_size"] for step in report["steps"] if step["stage"] == stage]
            assert sizes == [8, 8, 1] * 2, (stage, sizes)
        metadata = read_json(run_dir / "run.json")
        assert metadata["status"] == "complete" and metadata["data_fingerprint"] == fingerprint
        report["source_hash"] = metadata["source_hash"]
        check_warm_start(run_dir, metadata, report)
        report["bundle_parity"] = check_bundle(run_dir, out, protocol, metadata, rows, device)
        report["status"] = "READY" if device.type == "cuda" else "BLOCKED"
        if device.type != "cuda":
            report["reason"] = "CPU smoke passed, configured-batch VRAM is still unverified"
    except Exception as error:
        report["reason"] = repr(error)
        raise
    finally:
        write_json(out / "report.json", report)
    for path in (run_dir / "warmup.pt", run_dir / "warmup_last.pt", run_dir / "main.pt",
                 run_dir / "main_last.pt", out / "student.pt"):
        path.unlink()
    print(f"{report['status']}: {out / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixture", action="store_true", help="synthetic tensors only, no OFF30 images")
    mode.add_argument("--verify_only", action="store_true", help="reuse completed smoke with retained checkpoints")
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    protocol = load_yaml(resolve_path("configs/protocol_ppmi.yaml"))
    cfg = load_yaml(resolve_path("configs/baselines/shaspec.yaml"))
    device = choose_device(args.device)
    if args.fixture:
        result = fixture_checks(protocol, cfg, device)
        write_json(args.out, result)
        print(result, flush=True)
    else:
        smoke(apply_overrides(protocol, ["training.max_epochs=2"]), cfg, args.out, device, args.verify_only)


if __name__ == "__main__":
    main()
