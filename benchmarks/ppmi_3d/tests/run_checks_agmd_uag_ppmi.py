"""AGMD technical acceptance on synthetic OFF30 images, with a real-protocol tuning dry-run."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate
import run_checks_agmd_ppmi as p1b
import train
from baselines.agmd.graph import GraphLayout
from baselines.agmd.method import AGMDMethod
from baselines.agmd.model import GraphReadout
from baselines.agmd.uag import mc_dropout_sigma, uag_weights
from data.dataset import make_loader, to_device
from data.manifest import TargetScaler, data_fingerprint, read_manifest, rows_of
from utils.config import apply_overrides
from utils.inference import freeze
from utils.io import read_csv, read_json, write_json
from utils.mri_inference import export_run, predict_mri
from utils.repro import file_sha256, seed_everything, source_hash

PROTOCOL, CONFIG, DEVICE = p1b.PROTOCOL, p1b.CONFIG, p1b.DEVICE
TRAIN0 = p1b.TRAIN0
PPMI_ARGS = ["--method", "agmd", "--protocol", p1b.ARGS.protocol, "--config", p1b.ARGS.config]
FIXTURE_ARGS = ["--method", "agmd", "--protocol", str(p1b.FIXTURE_PATH), "--config", p1b.ARGS.config]
SAMPLES = CONFIG["uag"]["mc_samples"]
SCALE = CONFIG["uag"]["mc_scale"]
WEIGHT_KEYS = ("shallow_1", "shallow_2", "graph", "global")
REPORT = p1b.REPORT_DIR / "checks_report_agmd_off30_uag_fixture"
EXPECT_ERROR = TestCase().assertRaisesRegex


def rng_states(device):
    cpu = torch.random.get_rng_state()
    return (cpu, torch.cuda.get_rng_state(device)) if device.type == "cuda" else (cpu,)


@torch.no_grad()
def check_mc_sampling():
    results = {}
    devices = [torch.device("cpu")]
    if DEVICE.type == "cuda":
        devices.append(DEVICE)
    for device in devices:
        seed_everything(0)
        cfg = dict(graph=dict(gcn_hidden=8), transformer=dict(nhead=2, dim_feedforward=16, dropout=0.1,
                                                              num_layers=2, global_dim=8))
        teacher = nn.Module()
        teacher.readout = GraphReadout(6, cfg)
        teacher.head = nn.Linear(8, 1)
        teacher.to(device)
        freeze(teacher)
        tokens = torch.randn(3, 6, 8, device=device, requires_grad=True)
        deterministic = teacher.head(teacher.readout(tokens))
        initial_state = {key: value.clone() for key, value in teacher.state_dict().items()}
        mcs, attention_calls = [], []
        def record_attention(module, _args):
            attention_calls.append(module.training)
        def record_prediction(module, _args, output):
            assert not module.training and not output.requires_grad
            mcs.append(output.squeeze(-1).clone())
        attention = teacher.readout.encoder.layers[0].self_attn
        with ExitStack() as stack:
            stack.callback(attention.register_forward_pre_hook(record_attention).remove)
            stack.callback(teacher.head.register_forward_hook(record_prediction).remove)
            before = rng_states(device)
            sigma = mc_dropout_sigma(teacher, tokens, SAMPLES)
            assert all(torch.equal(a, b) for a, b in zip(before, rng_states(device)))
        assert attention_calls == [True] * SAMPLES
        draws = torch.stack(mcs)
        assert torch.unique(draws, dim=0).shape[0] > 1
        torch.testing.assert_close(sigma, draws.std(dim=0, correction=1), rtol=0, atol=0)
        assert sigma.shape == (3,) and torch.isfinite(sigma).all() and (sigma > 0).all()
        assert not sigma.requires_grad and not any(module.training for module in teacher.modules())
        assert torch.equal(teacher.head(teacher.readout(tokens)), deterministic)
        assert all(torch.equal(v, initial_state[k]) for k, v in teacher.state_dict().items())
        assert all(parameter.grad is None for parameter in teacher.parameters())

        # Isolate attention dropout from the explicit nn.Dropout modules.
        for module in teacher.readout.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        assert (mc_dropout_sigma(teacher, tokens, SAMPLES) > 0).all()
        for layer in teacher.readout.encoder.layers:
            layer.self_attn.dropout = 0.0
        zero = mc_dropout_sigma(teacher, tokens, SAMPLES)
        assert torch.equal(zero, torch.zeros_like(zero))
        labels = torch.zeros_like(zero)
        weights = uag_weights(CONFIG, {}, labels, labels, (1, 2), sigma=zero)
        assert all(torch.equal(weights[key], torch.ones_like(zero)) for key in WEIGHT_KEYS)

        teacher.readout.encoder.layers[0].dropout1.train()
        flags = [module.training for module in teacher.modules()]
        mc_dropout_sigma(teacher, tokens, SAMPLES)
        assert [module.training for module in teacher.modules()] == flags
        attention.dropout = 0.1
        before = rng_states(device)
        with patch.object(teacher.head, "forward", side_effect=RuntimeError("injected head failure")), \
             EXPECT_ERROR(RuntimeError, "injected head failure"):
            mc_dropout_sigma(teacher, tokens, SAMPLES)
        assert [module.training for module in teacher.modules()] == flags
        assert all(torch.equal(a, b) for a, b in zip(before, rng_states(device)))
        with EXPECT_ERROR(ValueError, "at least 2"):
            mc_dropout_sigma(teacher, tokens, 1)
        results[str(device)] = dict(sigma=sigma.tolist(), distinct_draws=torch.unique(draws, dim=0).shape[0],
                                    attention_only_stochastic=True, zero_dropout_sigma=zero.tolist())
    return results


def check_weights():
    pred = torch.tensor([0.25, -0.25, 2.0])
    y = torch.tensor([0.0, -0.75, 0.0])
    features = torch.tensor([[[1., 3.], [2., 0.], [3., 1.]], [[0., 0.], [1., 1.], [4., 4.]],
                             [[0., 2.], [1., 3.], [5., 5.]]])
    out = {"feats": {1: features, 2: features * 0.5}, "global_": features[:, :, 0]}
    sigma = torch.tensor([0., 0.125, 0.5], requires_grad=True)
    expected = torch.tensor([1., 0.5, 0.])
    error = [0.75, 0.5, 0.]
    mc = uag_weights(CONFIG, out, pred, y, (1, 2), sigma=sigma)
    changed = uag_weights(CONFIG, out, pred, pred, (1, 2), sigma=sigma)
    for key in WEIGHT_KEYS:
        assert torch.equal(mc[key], expected) and torch.equal(changed[key], expected)
    assert torch.equal(mc["logit"], torch.tensor(error)) and torch.equal(changed["logit"], torch.ones_like(y))
    assert all(torch.isfinite(v).all() and not v.requires_grad and ((0 <= v) & (v <= 1)).all() for v in mc.values())
    batch = p1b.cfg_with("uag.reduce=batch")
    averaged = uag_weights(batch, out, pred, y, (1, 2), sigma=sigma)
    assert all(torch.equal(averaged[k], v.mean().expand_as(v)) for k, v in mc.items())
    for scale in ("0", "-0.1", ".inf"):
        with EXPECT_ERROR(ValueError, "mc_scale.*positive"):
            uag_weights(p1b.cfg_with(f"uag.mc_scale={scale}"), out, pred, y, (1, 2), sigma=sigma)

    # Exact FP32 outputs captured before the P8 change on the fixed CPU fixture.
    global_entropy = [0.2423209547996521, 0.7503094673156738, 0.8825033903121948]
    snapshots = {
        "error": [error] * 5,
        "feature_entropy": [[0.07396167516708374, 0.7503094673156738, 0.7503094673156738],
                            [0.022067666053771973, 0.31551796197891235, 0.31551796197891235],
                            global_entropy, global_entropy, error],
        "none": [[1., 1., 1.]] * 5,
    }
    for mode, snapshot in snapshots.items():
        for reduce in ("per_sample", "batch"):
            cfg = dict(uag=dict(mode=mode, reduce=reduce, error_scale=1.0))
            got = uag_weights(cfg, out, pred, y, (1, 2))
            for key, values in zip((*WEIGHT_KEYS, "logit"), snapshot):
                expected = torch.tensor(values)
                if reduce == "batch":
                    expected = expected.mean().expand_as(expected)
                assert torch.equal(got[key], expected), (mode, reduce, key, got[key], expected)
    return dict(legacy_modes=list(snapshots), reductions=["per_sample", "batch"], label_free_feature_terms=True)


def check_student_only():
    seed_everything(0)
    cfg = p1b.cfg_with("kd.enabled=false")
    method = AGMDMethod(PROTOCOL, cfg)
    assert method.stages == ("student",) and not method.needs_dat("student", "train")
    student, _ = p1b.quiet(method.build, "student")
    student.to(DEVICE).train()
    batch = p1b.real_batch(TRAIN0, 2, load_dat=False)
    assert "dat" not in batch
    del batch["has_dat"]
    loss, terms = method.training_step("student", student, {}, batch)
    loss.backward()
    assert torch.isfinite(loss) and set(terms) == {"task"}
    assert student.head.weight.grad is not None and torch.isfinite(student.head.weight.grad).all()
    info = student.graph.layout.node_info()
    assert [(v["stage"], v["n_nodes"], v["n_key_present"], v["anat_edges"]) for v in info["layers"]] == [
        (1, 166, 10, 38), (2, 166, 10, 38)]
    return dict(parameters=p1b.nparams(student), task=float(loss.detach()), graph=info["layers"])


def check_moca_independence():
    method = AGMDMethod(PROTOCOL, CONFIG)
    student, _ = p1b.quiet(method.build, "student")
    teacher, _ = p1b.quiet(method.build, "teacher")
    student.to(DEVICE).eval()
    freeze(teacher.to(DEVICE))
    batch = p1b.real_batch(TRAIN0, 2)
    assert {"aux", "aux_mask"} <= batch.keys()
    modified = {key: value for key, value in batch.items() if key not in ("aux", "aux_mask")}
    with torch.no_grad():
        for stage, model, frozen in (("teacher", teacher, {}), ("student", student, {"teacher": teacher})):
            expected, expected_terms = method.training_step(stage, model, frozen, batch)
            got, got_terms = method.training_step(stage, model, frozen, modified)
            assert torch.equal(expected, got)
            assert all(torch.equal(v, got_terms[k]) for k, v in expected_terms.items())
    return "MoCA and its availability mask can be absent from the batch"


def check_vram():
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required to measure batch-8 PPMI VRAM")
    seed_everything(0)
    method = AGMDMethod(PROTOCOL, CONFIG)
    teacher, _ = p1b.quiet(method.build, "teacher")
    student, _ = p1b.quiet(method.build, "student")
    freeze(teacher.to(DEVICE))
    student.to(DEVICE).train()
    loader = make_loader(TRAIN0, PROTOCOL, True, shuffle=False, seed=0)
    scaler = TargetScaler.fit(TRAIN0)
    optimizer = torch.optim.Adam(student.parameters(), lr=float(PROTOCOL["training"]["lr"]))
    counts = Counter()
    modules = dict(mri_encoder=teacher.mri_encoder, dat_encoder=teacher.dat_encoder, graph=teacher.graph,
                   readout=teacher.readout, head=teacher.head, **{f"cmt_{k}": v for k, v in teacher.cmt.items()})
    steps = []
    with ExitStack() as stack:
        for name, module in modules.items():
            def count_forward(_module, _args, _output, key=name):
                counts.update((key,))
            stack.callback(module.register_forward_hook(count_forward).remove)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(loader, 1):
            counts.clear()
            batch = to_device(batch, DEVICE)
            batch["y_z"] = scaler.normalize(batch["y"])
            batch_size = len(batch["unit"])
            assert batch["mri"].shape == batch["dat"].shape == (batch_size, 1, *PROTOCOL["preprocessing"]["input_size"])
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, terms = method.training_step("student", student, {"teacher": teacher}, batch)
            assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values())
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            assert int(optimizer.state[student.head.weight]["step"]) == step
            assert counts == {name: 1 + SAMPLES if name in ("readout", "head") else 1 for name in modules}, counts
            steps.append(dict(step=step, batch_size=batch_size, seconds=round(time.perf_counter() - start, 3),
                              peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                              teacher_forwards=dict(counts)))
            torch.cuda.reset_peak_memory_stats()
    full, tail = divmod(len(TRAIN0), PROTOCOL["training"]["batch_size"])
    assert [step["batch_size"] for step in steps] == [8] * full + ([tail] if tail else [])
    assert len(steps) >= 2 and steps[-1]["batch_size"] == 1
    assert all(p.grad is None for p in teacher.parameters())
    assert not any(module.training for module in teacher.modules())
    return dict(batch=8, mc_samples=SAMPLES, oom=False, steps=steps)


def run_command(commands, *args):
    cmd = [sys.executable, *map(str, args)]
    commands.append(subprocess.list2cmdline(cmd))
    process = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8",
                             errors="replace")
    if process.returncode:
        raise RuntimeError(f"Exit {process.returncode} from {args[0]}\n{process.stdout[-7000:]}")
    return process.stdout


@torch.no_grad()
def train_fold_diagnostics(method, teacher, rows):
    paired = [row for row in rows if row.dat_path is not None]
    loader = make_loader(paired, method.protocol, True, shuffle=False, seed=0)
    values = []
    for batch in loader:
        batch = to_device(batch, DEVICE)
        tokens = teacher(batch["mri"], batch["dat"])["graph"]
        values.append(mc_dropout_sigma(teacher, tokens, SAMPLES).cpu())
    sigma = torch.cat(values)
    w = 1.0 - (sigma / SCALE).clamp(0., 1.)
    assert sigma.shape == (len(paired),) and torch.isfinite(sigma).all()
    mean = float(sigma.mean())
    cv = float(sigma.std(correction=1)) / mean if mean > 0 else None
    above = float((w > 0.99).float().mean())
    return dict(n=len(paired), sigma_min=float(sigma.min()), sigma_median=float(sigma.quantile(0.5)),
                sigma_max=float(sigma.max()), sigma_mean=mean, sigma_cv=cv, w_min=float(w.min()),
                w_median=float(w.quantile(0.5)), w_max=float(w.max()), w_mean=float(w.mean()),
                w_lt_0_5=float((w < 0.5).float().mean()), w_eq_0=float((w == 0).float().mean()), w_gt_0_99=above,
                degenerate=(cv is not None and cv < 0.05) or above == 1.0)


class TeacherOnly(AGMDMethod):
    """Use the shared trainer for a teacher-only acceptance fixture."""

    def __init__(self, protocol, cfg):
        super().__init__(protocol, cfg)
        self.stages = ("teacher",)


def smoke_teacher_student(tmp, commands):
    # Keep the same effective protocol for teacher reuse, including max_epochs.
    protocol = apply_overrides(PROTOCOL, ["training.max_epochs=2"])
    source = tmp / "teacher" / "fold0_seed0"
    med_state = {}
    rows = read_manifest(protocol, med_state)
    assert len({r.subject_id for r in rows}) == med_state["qc_approval"]["qc_counts"]["PASS"]
    fingerprint = data_fingerprint(protocol, rows)
    commands.append("OFF30 fixture: train.run_one(agmd, TeacherOnly, max_epochs=2, fold=0, seed=0, uag.mode=error)")
    train.run_one("agmd", TeacherOnly, protocol, p1b.cfg_with("uag.mode=error"), rows, fingerprint, 0, 0, source,
                  DEVICE, med_state=med_state)
    torch.cuda.empty_cache()
    run_command(commands, "train.py", *FIXTURE_ARGS, "--fold", "0", "--seed", "0", "--pset", "training.max_epochs=2",
                "--out", tmp, "--teacher_from", source.parent)
    run = tmp / "agmd" / "fold0_seed0"
    meta = read_json(run / "run.json")
    source_meta = read_json(source / "run.json")
    assert meta["status"] == source_meta["status"] == "complete"
    assert meta["protocol"] == source_meta["protocol"] == protocol
    assert source_meta["config"]["uag"]["mode"] == "error" and meta["config"]["uag"] == CONFIG["uag"]
    assert Path(meta["stages"]["teacher"]["reused_from"]).resolve() == (source / "teacher.pt").resolve()
    assert not (run / "teacher.pt").exists()
    for folder, role in ((source, "teacher"), (run, "student")):
        for name in (role, f"{role}_last"):
            assert (folder / f"{name}.pt").is_file()
    log = (run / "train.log").read_text(encoding="utf-8")
    assert "mode=mc_dropout reduce=per_sample mc_samples=8 mc_scale=0.25 logit_error_scale=1.0" in log
    assert log.count("graph node_info:") == 2
    history = read_csv(run / "history.csv")
    columns = ("train_uag_sigma_mean", "train_uag_sigma_batch_max", "train_uag_w")
    assert len(history) == 2 and all(row["stage"] == "student" for row in history)
    stats = [{key: float(row[key]) for key in columns} for row in history]
    assert all(all(math.isfinite(x) for x in row.values()) and
               0 <= row[columns[0]] <= row[columns[1]] and 0 <= row[columns[2]] <= 1 for row in stats)

    scaler = TargetScaler.fit(TRAIN0)
    for override in ("graph.nodes=atlas", "graph.atlas_path=another-atlas.nii.gz", "cmt.scale=none",
                     "transformer.dropout=0.2"):
        with EXPECT_ERROR(ValueError, "teacher_config"):
            train.reusable_teacher(source.parent, 0, 0, protocol, p1b.cfg_with(override), p1b.SPLIT0, scaler,
                                   AGMDMethod.teacher_sections, method_name="agmd", fingerprint=fingerprint)
    method = AGMDMethod(protocol, CONFIG)
    teacher, _ = p1b.quiet(method.build, "teacher")
    teacher.load_state_dict(torch.load(source / "teacher.pt", weights_only=True, map_location="cpu")["state_dict"])
    seed_everything(0)
    diagnostics = train_fold_diagnostics(method, freeze(teacher.to(DEVICE)), TRAIN0)
    del teacher
    torch.cuda.empty_cache()

    validation = {}
    command = "OFF30 fixture: evaluate.evaluate_run(agmd, fold0_seed0, val, best/last) and export_run + predict_mri"
    commands.append(command)
    validation_rows = rows_of(rows, p1b.SPLIT0.val)
    mri_rows = [dict(subject_id=r.subject_id, visit_id=r.visit_id, mri_path=r.mri_path) for r in validation_rows]
    for checkpoint in ("best", "last"):
        preds, _ = evaluate.evaluate_run("agmd", run, "val", ["all"], DEVICE, False, {}, checkpoint)
        assert Counter(p["role"] for p in preds) == {r: len(mri_rows) for r in ("teacher", "student", "train_mean")}
        assert all(math.isfinite(p["y_pred"]) for p in preds)
        bundle = export_run(run, checkpoint)
        layout, _ = p1b.quiet(GraphLayout, protocol, CONFIG)
        expected_layout = layout.deployment_payload()
        assert bundle["layout"]["metadata"] == expected_layout["metadata"]
        for key in ("pools", "anatomical"):
            assert all(torch.equal(x, y) for x, y in zip(bundle["layout"][key], expected_layout[key]))
        with patch("data.manifest.read_manifest", side_effect=AssertionError("training manifest accessed")), \
             patch("models.backbones.resnet.load_medicalnet", side_effect=AssertionError("pretrained accessed")), \
             patch("baselines.agmd.graph.load_atlas", side_effect=AssertionError("atlas accessed")):
            deployed = predict_mri(bundle, mri_rows, DEVICE, batch_size=len(mri_rows))
        expected = {p["unit"]: p["y_pred"] for p in preds if p["role"] == "student"}
        difference = max(abs(p["y_pred"] - expected[p["unit"]]) for p in deployed)
        assert difference <= 2e-5, difference
        validation[checkpoint] = dict(n=len(mri_rows), deployment_abs_delta=difference)
    return dict(teacher_epochs=2, student_epochs=2, fold=0, seed=0, history=stats,
                teacher_reuse=True, incompatible_teacher_rejected=True, val=validation, diagnostics=diagnostics)


def check_tuning(commands):
    stdout = run_command(commands, "scripts/tune.py", *PPMI_ARGS, "--dry_run")
    planned = [json.loads(line) for line in re.findall(r"(?m)^  cfg\d+: (.*)$", stdout)]
    combos = {(row["protocol.training.lr"], row["kd.w_shallow"], row["uag.mc_scale"]) for row in planned}
    assert len(planned) == 8 and combos == set(product((1e-4, 1e-3), (1.0, 0.1), (0.25, 0.1)))
    assert all(stdout.count(f"--fold {fold} --seed 0") == 8 for fold in PROTOCOL["data"]["split"]["folds"])
    return planned


def run_p1b_subset(commands):
    commands.append("OFF30 fixture: atlas, layout, shape, gradient, MRI-only and subject-local graph checks in process")
    for check in (p1b.atlas_test, p1b.layout_test, p1b.shape_test, p1b.gradient_test, p1b.mri_only_test, p1b.case_graph_test):
        check()
    assert all(row["passed"] for row in p1b.RESULTS), p1b.RESULTS
    return dict(checks=p1b.RESULTS, evidence=p1b.LINES)


def check_ntua(tmp, commands):
    run_command(commands, "tests/run_checks_agmd.py", "--report_dir", tmp)
    checks = read_json(tmp / "checks_report_agmd.json")
    assert len(checks) == 11 and all(row["passed"] and "SKIPPED" not in row["detail"] for row in checks), checks
    return checks


def check_deployment(commands):
    stdout = run_command(commands, "tests/run_checks_mri_inference.py")
    match = re.search(r"(?m)^\[\s*$", stdout)
    assert match is not None, stdout[-2000:]
    evidence, _ = json.JSONDecoder().raw_decode(stdout[match.start():])
    assert len({row["method"] for row in evidence if row["dimensions"] == 3}) == 7
    return dict(fixtures=evidence, unittest_summary=stdout.strip().splitlines()[-3:])


def save_reports(report):
    results = report["checks"]
    lines = ["# AGMD† · Kiểm tra kỹ thuật trên fixture OFF30", "", f"**{report['status']}** · {report['protocol']}, "
             f"{report['device']}, PyTorch {report['torch']}.",
             "", "| Kiểm tra | Kết quả | Giây |", "|---|---|---:|"]
    lines.extend(f"| {name} | {row['status']} | {row['seconds']:.2f} |" for name, row in results.items())
    if results["VRAM"]["status"] == "PASS":
        lines.extend(["", "## Student MC T=8, gồm batch cuối một ca", "",
                      "| Bước | Batch | Allocated (GiB) | Reserved (GiB) | Giây |", "|---|---:|---:|---:|---:|"])
        for row in results["VRAM"]["details"]["steps"]:
            lines.append(f"| {row['step']} | {row['batch_size']} | {row['peak_allocated_gib']:.2f} | {row['peak_reserved_gib']:.2f} | "
                         f"{row['seconds']:.3f} |")
        lines.append("Mỗi bước chạy encoder MRI/DaT, từng CMT và graph đúng một lần. Readout và head chạy 1 + 8 lần.")
    smoke = results["Smoke"]
    if smoke["status"] == "PASS":
        d = smoke["details"]["diagnostics"]
        lines.extend(["", "## Diagnostics chỉ trên tập train", "",
                      f"Fold 0, seed 0, {d['n']} ca có DaT. Teacher checkpoint best sau 2 epoch, student reuse 2 epoch.",
                      "", "| Đại lượng | Min | Median | Max | Mean |", "|---|---:|---:|---:|---:|"])
        for key in ("sigma", "w"):
            lines.append(f"| {key} | {d[key+'_min']:.6g} | {d[key+'_median']:.6g} | {d[key+'_max']:.6g} | "
                         f"{d[key+'_mean']:.6g} |")
        cv = "không xác định" if d["sigma_cv"] is None else f"{d['sigma_cv']:.4f}"
        lines.append(f"\nCV σ {cv}. Tỷ lệ w < 0.5 là {d['w_lt_0_5']:.1%}, w = 0 là {d['w_eq_0']:.1%}, "
                     f"w > 0.99 là {d['w_gt_0_99']:.1%}. Cờ thoái hóa D-039: {d['degenerate']}.")
        lines.append("History ghi trung bình σ, trung bình w và trung bình có trọng số theo batch của các cực đại σ. "
                     "Bảng trên ghi cực đại trực tiếp trên tập train.")
    for name, row in results.items():
        if row["status"] == "FAIL":
            lines.extend(["", f"### Lỗi {name}", "", "```text", row["error"], "```"])
    lines.extend(["", "## Giới hạn", "",
                  "- Smoke ngắn chỉ kiểm cơ chế và đường chạy. Chưa kiểm calibration sau hội tụ, full tune/train/infer, "
                  "tái lập bit-exact CUDA hoặc hiệu năng đa fold/seed.",
                  "- Smoke kiểm checkpoint best/last trên val tổng hợp. Diagnostics dùng training rows của fixture.",
                  "- Chỉ dry-run tuner dùng đường dẫn protocol chính. Kết luận readiness OFF30 cần thêm duyệt atlas và smoke thực.",
                  "", "## Lệnh đã chạy", "", "```text", *report["commands"], "```",
                  "", "## SHA-256", "", f"Source hash chung `{report['source_hash']}`.", "",
                  "| File | SHA-256 |", "|---|---|"])
    lines.extend(f"| `{path}` | `{digest}` |" for path, digest in report["hashes"].items())
    lines += ["", "Chi tiết từng kiểm tra, timing và parity nằm trong JSON đi kèm.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    write_json(REPORT.with_suffix(".json"), report)


def main():
    torch.set_num_threads(4)
    commands = [subprocess.list2cmdline([sys.executable, *sys.argv])]
    checks = {}
    with tempfile.TemporaryDirectory(prefix="agmd_p8_") as folder:
        tmp = Path(folder)
        steps = [("MC sampling", check_mc_sampling), ("Weights", check_weights),
                 ("PPMI 3D", lambda: run_p1b_subset(commands)), ("Student-only matched capacity", check_student_only),
                 ("MoCA independence", check_moca_independence),
                 ("VRAM", check_vram), ("Smoke", lambda: smoke_teacher_student(tmp, commands)),
                 ("Tuning dry-run", lambda: check_tuning(commands)), ("NTUA regression", lambda: check_ntua(tmp, commands)),
                 ("MRI inference parity", lambda: check_deployment(commands))]
        for name, check in steps:
            start = time.perf_counter()
            try:
                details = check()
                if name == "VRAM":
                    assert max(row["peak_allocated_gib"] for row in details["steps"]) <= 7.0, details
                result = dict(status="PASS", details=details)
            except Exception as error:
                result = dict(status="FAIL", error=traceback.format_exc())
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    result["oom"] = True
                    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                    result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
                print(result["error"], flush=True)
            finally:
                torch.cuda.empty_cache()
            result["seconds"] = round(time.perf_counter() - start, 3)
            checks[name] = result
            print(f"[{result['status']}] {name} ({result['seconds']:.2f}s)", flush=True)
    failed = any(row["status"] != "PASS" for row in checks.values())
    status = "FAIL (fixture)" if failed else "PASS (fixture)"
    paths = ["baselines/agmd/uag.py", "baselines/agmd/method.py", "baselines/agmd/model.py", "baselines/agmd/graph.py",
             p1b.ARGS.config, p1b.ARGS.protocol, "configs/baselines/agmd.yaml", "configs/tuning/agmd.yaml", "train.py",
             "evaluate.py", "scripts/tune.py", "utils/mri_inference.py", "tests/run_checks_agmd_uag_ppmi.py",
             "tests/run_checks_agmd_ppmi.py", "tests/run_checks_agmd.py", "tests/run_checks_mri_inference.py"]
    device = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else str(DEVICE)
    report = dict(status=status, protocol=PROTOCOL["name"], torch=torch.__version__, device=device, checks=checks,
                  commands=commands, source_hash=source_hash(), hashes={p: file_sha256(ROOT / p) for p in paths})
    save_reports(report)
    print(f"{status} · {REPORT}.md/.json", flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
