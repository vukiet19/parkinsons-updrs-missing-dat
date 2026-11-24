"""DFTD OFF30 acceptance with a synthetic fixture or the QC-approved cohort.

The default uses a synthetic cohort and includes missing-DaT and NTUA regression
checks. Pass --approved-data only after publication of the shared cohort.
Each invocation runs one single-epoch smoke on fold 0 / seed 0, with validation
only. Runs, predictions, bundles and legacy reports use a temporary directory.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.dftd.encoders import region_vectors
from baselines.dftd.method import DFTDMethod
from baselines.dftd.mi import PairDiscriminator, negative_index
from data.dataset import make_loader, to_device
from data.manifest import TargetScaler, data_fingerprint, make_split, read_manifest, rows_of, verify_approved_data
from ppmi_fixture import approved_fixture
from train import run_one
from utils.config import apply_overrides, config_hash, load_yaml, resolve_path
from utils.io import read_csv, read_json, write_csv, write_json
from utils.mri_inference import export_run, load_bundle, predict_mri, read_mri_manifest
from utils.repro import choose_device, file_sha256, seed_everything, source_hash

PROTOCOL = "configs/protocol_ppmi.yaml"
CONFIG = "configs/baselines/dftd.yaml"
GRID = "configs/tuning/dftd.yaml"
ROUTINGS = ("alg1", "joint")
LEARNING_RATES = (1e-4, 1e-3)
SMOKE_OVERRIDES = ("training.max_epochs=1", "training.save_last_checkpoint=false")


def run_cli(commands, *args):
    command = [sys.executable, *map(str, args)]
    print(subprocess.list2cmdline(command), flush=True)
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, encoding="utf-8", errors="replace", capture_output=True)
    commands.append(dict(argv=command, returncode=result.returncode, seconds=round(time.perf_counter() - start, 2)))
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}:\n{result.stdout[-2500:]}\n{result.stderr[-2500:]}")
    return result.stdout


def tune_plan(protocol_path, protocol, cfg, tmp, commands):
    out = tmp / "tuning"
    stdout = run_cli(commands, "scripts/tune.py", "--method", "dftd", "--protocol", protocol_path,
                     "--dry_run", "--out", out)
    plans = [json.loads(row) for row in re.findall(r"^  cfg\d+: (\{.*\})$", stdout, re.MULTILINE)]
    expected = set(itertools.product(LEARNING_RATES, (1.0, 0.5), ROUTINGS))
    axes = ("protocol.training.lr", "kd.alpha", "gradient_routing")
    assert all(set(plan) == set(axes) for plan in plans), plans
    assert len(plans) == protocol["training"]["tuning_budget"] == 8
    assert {tuple(plan[key] for key in axes) for plan in plans} == expected
    assert plans[0] == dict(zip(axes, (protocol["training"]["lr"], cfg["kd"]["alpha"], cfg["gradient_routing"])))
    assert cfg["kd"]["beta"] == 0.05
    folds = protocol["data"]["split"]["folds"]
    planned_trials = re.findall(r"(?m)^ +python train\.py .*?--fold (\d+) --seed 0 .*?--tag cfg(\d+)", stdout)
    assert sorted((int(f), int(i)) for f, i in planned_trials) == list(itertools.product(sorted(folds), range(8)))
    assert "primary role 'student'" in stdout and not out.exists()
    return dict(plans=plans, folds=folds, trials=len(planned_trials), effective_beta=0.05, output_created=False)


def approved_cohort(protocol):
    stats = {}
    rows = read_manifest(protocol, stats)
    approval = stats.get("qc_approval")
    if approval is None:
        raise ValueError("DFTD OFF30 requires a manifest with QC publication evidence.")
    fingerprint = data_fingerprint(protocol, rows)
    verify_approved_data(stats, fingerprint)
    n = approval["qc_counts"]["PASS"]
    subjects = {r.subject_id for r in rows}
    assert len(rows) == len(subjects) == n and all(r.dat_path is not None for r in rows)

    splits = {}
    outer_test = []
    batch_size = protocol["training"]["batch_size"]
    assert batch_size == 8
    for fold in protocol["data"]["split"]["folds"]:
        split = make_split(rows, fold, protocol)
        assert set(split.train) | set(split.val) | set(split.test) == subjects
        assert len(split.train) + len(split.val) + len(split.test) == n
        outer_test.extend(split.test)
        train_rows = rows_of(rows, split.train)
        loader = make_loader(train_rows, protocol, load_dat=True, shuffle=False, seed=0)
        batches = [len(indices) for indices in loader.batch_sampler]
        assert sum(batches) == len(train_rows)
        splits[fold] = dict(train=len(split.train), val=len(split.val), test=len(split.test), batches=batches)
    assert sorted(outer_test) == sorted(subjects)
    assert splits[0]["batches"][-1] == 1, "Fold 0 must exercise a one-case final batch."
    evidence = dict(qc_counts=approval["qc_counts"], qc_sha256=approval["qc_sha256"],
                    manifest_sha256=approval["manifest_sha256"], cohort_hash=approval["cohort_hash"],
                    data_fingerprint=fingerprint, splits=splits)
    return rows, evidence


def native_batch(rows, protocol, device, *, load_dat=True, n=8, scaler=None):
    batch = to_device(next(iter(make_loader(rows[:n], protocol, load_dat, shuffle=False, seed=0))), device)
    batch["y_z"] = (scaler or TargetScaler.fit(rows)).normalize(batch["y"])
    return batch


def manual_regions(grid):
    shape = (7, 8, 7)
    cuts = [(0, 2, 4, 6, length) if grid == 4 else (0, 4, length) for length in shape]
    coordinates = list(itertools.product(*(zip(axis, axis[1:]) for axis in cuts)))
    ramp = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    seen = torch.zeros_like(ramp, dtype=torch.int)
    means = []
    counts = {}
    for bounds in coordinates:
        block = tuple(slice(low, high) for low, high in bounds)
        seen[block] += 1
        means.append(ramp[block].mean())
        volume = ramp[block].numel()
        counts[volume] = counts.get(volume, 0) + 1
    assert torch.all(seen == 1)
    torch.testing.assert_close(region_vectors(ramp[None, None], grid)[0, :, 0], torch.stack(means))
    assert counts == ({8: 36, 4: 24, 2: 4} if grid == 4 else {64: 2, 48: 4, 36: 2})
    return counts


def group_of(name):
    parts = name.split(".")
    if name.startswith("iid.student."):
        return ".".join(parts[:3])
    return ".".join(parts[:2]) if name.startswith(("enc_", "iid.")) else parts[0]


def gradient_groups(terms, model, routing):
    named = list(model.named_parameters())
    a_trunk = {"enc_a.stem", "enc_a.trunk"}
    b_trunk = {"enc_b.stem", "enc_b.trunk"}
    both_trunks = a_trunk | b_trunk
    common = {"enc_a.head_c", "enc_b.head_c"}
    specific = {"enc_a.head_s", "enc_b.head_s"}
    student_hidden = {"iid.student.layers"}
    student = student_hidden | {"iid.student.head"}
    lit = {"iid.lit"}
    weights = {"w_a", "w_b"}
    teacher = {"iid.t_fuse0", "iid.t_layers", "iid.t_fuse", "iid.t_head"}
    mri = a_trunk | {"enc_a.head_c", "enc_a.head_s"}
    joint = routing == "joint"
    expected = {
        "rel": both_trunks | common | weights | {"S_c_a", "S_c_b"},
        "spe_mi": both_trunks | specific | weights | {"S_s_a", "S_s_b"} | (common if joint else set()),
        "adv_bce": both_trunks | specific | {"D_s_a", "D_s_b"} | (common if joint else set()),
        "lit": lit | (both_trunks | common if joint else set()),
        "task_s": student | (mri if joint else set()),
        "task_t": student_hidden | teacher | lit | (both_trunks | common | specific if joint else set()),
        "kd_output": student | (mri if joint else set()),
        "kd_feature": student_hidden | (mri if joint else set()),
    }
    matrix = {}
    parameters = [p for _, p in named]
    for key, wanted in expected.items():
        grads = torch.autograd.grad(terms[key], parameters, retain_graph=True, allow_unused=True)
        reached = {}
        nonzero = set()
        for (name, _), grad in zip(named, grads):
            group = group_of(name)
            assert (grad is not None) == (group in wanted), (routing, key, name)
            if grad is not None:
                assert torch.isfinite(grad).all(), (routing, key, name)
                reached[group] = reached.get(group, 0) + 1
                if bool(grad.any()):
                    nonzero.add(group)
        assert nonzero == wanted, (routing, key, sorted(nonzero), sorted(wanted))
        matrix[key] = dict(sorted(reached.items()))
    return matrix


def native_gradients(protocol, cfg, rows, routing, device):
    seed_everything(0)
    method = DFTDMethod(protocol, apply_overrides(cfg, [f"gradient_routing={routing}"]))
    model = method.build("complete").to(device)
    batch = native_batch(rows, protocol, device, n=2)
    assert tuple(batch["mri"].shape) == tuple(batch["dat"].shape) == (2, 1, 97, 115, 97)
    with torch.inference_mode():
        encoded = [encoder(batch[key]) for encoder, key in ((model.enc_a, "mri"), (model.enc_b, "dat"))]
        for ec in encoded:
            assert tuple(ec["m"].shape) == (2, 256, 7, 8, 7)
            assert tuple(ec["r"].shape) == (2, 64, 256)
            assert tuple(ec["c"].shape) == tuple(ec["s"].shape) == (2, 512)
        f_a, f_b = [torch.cat((ec["c"], ec["s"]), 1) for ec in encoded]
        out = model.iid(f_a, f_b, encoded[0]["c"])
        assert tuple(out["cb_hat"].shape) == (2, 512)
        for key in ("s_feats", "t_feats"):
            assert [tuple(h.shape) for h in out[key]] == [(2, 512), (2, 256), (2, 64)]
        assert tuple(out["pred_s"].shape) == tuple(out["pred_t"].shape) == (2,)
    loss, terms = method.training_step("complete", model, {}, batch)
    assert all(torch.isfinite(value) for value in (loss, *terms.values()))
    assert terms["rel"] >= 0 and terms["spe_mi"] >= 0
    result = gradient_groups(terms, model, routing)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return dict(input_shape=list(batch["mri"].shape), region_shape=[2, 64, 256], gradient_parameter_counts=result,
                parameters=sum(p.numel() for p in model.parameters()))


def edge_losses(protocol, cfg, rows, device):
    counts = {str(grid): manual_regions(grid) for grid in (4, 2)}
    seed_everything(0)
    method = DFTDMethod(protocol, apply_overrides(cfg, ["regions.grid_dat=2"]))
    model = method.build("complete").to(device)
    for n in (1, 2):
        batch = native_batch(rows, protocol, device, n=n)
        if n == 2:
            batch["subject_id"] = ["same", "same"]
        model.zero_grad(set_to_none=True)
        loss, terms = method.training_step("complete", model, {}, batch)
        assert torch.isfinite(loss) and terms["rel"] == terms["spe_mi"] == 0
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    valid_batch = native_batch(rows, protocol, device, n=2)
    with torch.inference_mode():
        assert tuple(model.enc_b(valid_batch["dat"])["r"].shape) == (2, 8, 256)
    model.zero_grad(set_to_none=True)
    loss, terms = method.training_step("complete", model, {}, valid_batch)
    assert torch.isfinite(loss) and terms["rel"] > 0 and terms["spe_mi"] > 0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return dict(regions_by_voxel_count=counts, invalid_negative_batches=[1, 2], dat2_regions=8)


def loss_gradients(protocol, cfg):
    seed_everything(0)
    ids = ["a", "b", "b", "a", "c", "c"]
    k, valid = negative_index(ids, "other_subject", "cpu")
    assert valid.all() and all(ids[i] != ids[j] for i, j in enumerate(k.tolist()))
    disc = PairDiscriminator(4, 4, [8, 4])
    c, s = [torch.rand(6, 4, requires_grad=True) for _ in range(2)]
    params = (c, s, *disc.parameters())
    j = disc.net(torch.cat([c, s], 1)).squeeze(1)
    p = disc.net(torch.cat([c, s[k]], 1)).squeeze(1)
    plain = F.binary_cross_entropy_with_logits(j, torch.zeros_like(j)) + F.binary_cross_entropy_with_logits(p, torch.ones_like(p))
    reference = torch.autograd.grad(plain, params)
    for reverse_c in (False, True):
        reversed_loss = disc(c, s, k, valid, reverse_c)
        torch.testing.assert_close(reversed_loss, plain)
        for i, (actual, wanted) in enumerate(zip(torch.autograd.grad(reversed_loss, params), reference)):
            torch.testing.assert_close(actual, -wanted if i == 1 or (i == 0 and reverse_c) else wanted)

    # Without a valid shuffled partner, only the joint BCE component is defined.
    same = ["same"] * len(ids)
    k, valid = negative_index(same, "other_subject", "cpu")
    assert not valid.any()
    torch.testing.assert_close(disc(c, s, k, valid, reverse_c=False),
                               F.binary_cross_entropy_with_logits(j, torch.zeros_like(j)))

    for detach in (True, False):
        method = DFTDMethod(protocol, apply_overrides(cfg, [f"kd.detach_teacher={str(detach).lower()}"]))
        pred_s, pred_t = [torch.rand(2, requires_grad=True) for _ in range(2)]
        s_hidden, t_hidden = [[torch.rand(2, 4, requires_grad=True) for _ in range(3)] for _ in range(2)]
        terms = method._iid_terms(dict(pred_s=pred_s, pred_t=pred_t, s_feats=s_hidden, t_feats=t_hidden),
                                  torch.zeros_like(pred_s))
        for name, student, teacher in (("kd_output", [pred_s], [pred_t]), ("kd_feature", s_hidden, t_hidden)):
            grads = torch.autograd.grad(terms[name], student + teacher, allow_unused=True)
            assert all(g is not None and g.any() for g in grads[:len(student)])
            assert all((g is None) if detach else (g is not None and g.any()) for g in grads[len(student):])
    return dict(grl="s reversed, c reversed only on request, discriminator unchanged",
                invalid_negative="product BCE masked, joint BCE retained",
                kd_targets="both output and feature targets detached by default; both get gradients in two-way mode")


def adam_probe(protocol, cfg, rows, routing, lr, device):
    seed_everything(0)
    method = DFTDMethod(protocol, apply_overrides(cfg, [f"gradient_routing={routing}"]))
    scaler = TargetScaler.fit(rows)
    loader = make_loader(method.train_rows("complete", rows), protocol, load_dat=True, shuffle=True, seed=0)
    evidence = dict(routing=routing, lr=lr, steps=[], oom=False)
    if device.type == "cuda":
        free, capacity = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
        evidence.update(device_free_start_bytes=free, device_total_bytes=capacity)
    try:
        model = method.build("complete").to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=protocol["training"]["weight_decay"])
        weights = (model.w_a, model.w_b)
        n_parameters = len(list(model.parameters()))
        for step, batch in enumerate(loader, start=1):
            batch = to_device(batch, device)
            size = batch["mri"].shape[0]
            assert size == batch["dat"].shape[0]
            batch["y_z"] = scaler.normalize(batch["y"])
            loss, terms = method.training_step("complete", model, {}, batch)
            assert all(torch.isfinite(value) for value in (loss, *terms.values()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            previous = [w.theta.detach().clone() for w in weights]
            grad_norms = [float(w.theta.grad.norm()) for w in weights]
            if size == 1:
                assert terms["rel"] == terms["spe_mi"] == 0
                assert grad_norms == [0.0, 0.0]
            else:
                assert all(value > 0 for value in grad_norms)
            optimizer.step()
            with torch.no_grad():
                delta = [float((w.theta - before).abs().max()) for w, before in zip(weights, previous)]
                if size > 1:
                    assert all(value > 0 for value in delta)
                omega = [[float(w().min()), float(w().max())] for w in weights]
            assert len(optimizer.state) == n_parameters
            evidence["steps"].append(dict(step=step, batch_size=size, loss=float(loss.detach()),
                                          terms={key: float(value.detach()) for key, value in terms.items()},
                                          theta_grad_l2=grad_norms, theta_update_max=delta, omega_min_max=omega,
                                          adam_parameters_with_state=len(optimizer.state)))
            del batch, loss, terms
        batch_sizes = [step["batch_size"] for step in evidence["steps"]]
        assert sum(batch_sizes) == len(rows) and batch_sizes.count(8) >= 2 and batch_sizes[-1] == 1
        evidence["batch_sizes"] = batch_sizes
    except torch.cuda.OutOfMemoryError as error:
        evidence.update(oom=True, error=str(error))
    if device.type == "cuda":
        evidence.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return evidence


def no_pair(protocol, cfg, rows, tmp):
    # No images are opened by run_one before the training-row guard.
    offline = apply_overrides(protocol, ["backbone.pretrained=none", "training.max_epochs=2"])
    missing = [replace(row, dat_path=None) for row in rows]
    results = []
    for enabled in (True, False):
        run = tmp / f"missing_all_stage2_{enabled}"
        override = apply_overrides(cfg, [f"stage2.enabled={str(enabled).lower()}"])
        try:
            run_one("dftd", DFTDMethod, offline, override, missing, "fixture", 0, 0, run, torch.device("cpu"))
        except ValueError as error:
            assert "requires at least one MRI-DaT pair" in str(error), str(error)
        else:
            raise AssertionError("DFTD accepted full training with no paired rows")
        meta = read_json(run / "run.json")
        assert meta["status"] == "failed" and not list(run.glob("*.pt"))
        results.append(dict(stage2_enabled=enabled, status=meta["status"], error=meta["error"], checkpoints=0))
    return results


def smoke(protocol_path, protocol, cfg, cohort, train_rows, val_rows, tmp, commands):
    out = tmp / "runs"
    run_cli(commands, "train.py", "--method", "dftd", "--protocol", protocol_path, "--fold", "0", "--seed", "0",
            "--out", out, "--pset", *SMOKE_OVERRIDES)
    run = out / "dftd" / "fold0_seed0"
    meta = read_json(run / "run.json")
    assert meta["status"] == "complete"
    assert meta["data_fingerprint"] == cohort["data_fingerprint"]
    assert meta["protocol_hash"] == config_hash(apply_overrides(protocol, SMOKE_OVERRIDES))
    assert meta["config_hash"] == config_hash(cfg)
    assert meta["stages"]["incomplete"] == dict(skipped=True, reason="no training rows for this stage")
    assert meta["stages"]["complete"]["best_epoch"] == 1
    assert meta["scaler"] == vars(TargetScaler.fit(train_rows))
    assert {path.name for path in run.glob("*.pt")} == {"complete.pt"}

    run_cli(commands, "evaluate.py", "--method", "dftd", "--run_dir", run, "--split", "val", "--checkpoint", "best")
    predictions = read_csv(run / "eval_val" / "predictions.csv")
    by_role = {role: [row for row in predictions if row["role"] == role]
               for role in ("student", "teacher", "teacher_imputed", "train_mean")}
    assert {row["role"] for row in predictions} == set(by_role)
    for group in by_role.values():
        assert {row["unit"] for row in group} == {row.unit for row in val_rows}
        assert all(math.isfinite(float(row["y_pred"])) for row in group)
    assert not list(run.glob("eval_test*"))
    history = read_csv(run / "history.csv")
    assert [(row["stage"], int(row["epoch"])) for row in history] == [("complete", 1)]
    return dict(fold=0, seed=0, epochs=1, stages={"complete": "trained", "incomplete": "skipped"},
                routing=meta["config"]["gradient_routing"], roles={key: len(group) for key, group in by_role.items()},
                validation_only=True,
                protocol_hash=meta["protocol_hash"], config_hash=meta["config_hash"], source_hash=meta["source_hash"],
                best_epoch=meta["stages"]["complete"]["best_epoch"],
                stage_seconds=meta["stages"]["complete"]["seconds"],
                checkpoint_bytes={path.name: path.stat().st_size for path in run.glob("*.pt")})


def warm_start(protocol, cfg, rows, tmp, device):
    method = DFTDMethod(protocol, cfg)
    paired = method.build("complete")
    path = tmp / "runs" / "dftd" / "fold0_seed0" / "complete.pt"
    if path.is_file():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        paired.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
    else:
        # Keep stage-2 coverage independent of a failed CLI run.
        with torch.no_grad():
            paired.iid.t_head.bias.add_(1)
    mixed = [replace(row, dat_path=None) if i < 3 else row for i, row in enumerate(rows)]
    paired_rows, missing = (method.train_rows(stage, mixed) for stage in method.stages)
    assert len(paired_rows) + len(missing) == len(rows) and len(missing) == 3
    assert all(row.dat_path is not None for row in paired_rows)
    model = method.build("incomplete").to(device)
    method.init_stage("incomplete", model, {"complete": paired})
    initial = paired.state_dict()
    assert all(torch.equal(t.cpu(), initial[k]) for k, t in model.state_dict().items())
    model.train()
    assert all(not module.training for rad in model.rad_modules() for module in rad.modules())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=protocol["training"]["lr"], weight_decay=protocol["training"]["weight_decay"])
    batch = native_batch(missing, protocol, device, load_dat=False, scaler=TargetScaler.fit(rows))
    assert "dat" not in batch
    loss, terms = method.training_step("incomplete", model, {}, batch)
    assert torch.isfinite(loss) and set(terms) == {"task_s", "task_t", "kd_output", "kd_feature"}
    loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer.step()
    changed_iid = []
    bn_buffers = 0
    for key, tensor in model.state_dict().items():
        same = torch.equal(tensor.cpu(), initial[key])
        if key.startswith("iid."):
            if not same:
                changed_iid.append(key)
        else:
            assert same, key
            bn_buffers += key.endswith(("running_mean", "running_var", "num_batches_tracked"))
    assert bn_buffers and any(k.startswith("iid.lit.") for k in changed_iid)
    model.eval()
    with torch.inference_mode(), patch.object(model.enc_b, "forward", side_effect=AssertionError("DaT encoder accessed")):
        for role in ("student", "teacher_imputed"):
            row = replace(missing[0], dat_path=tmp / "nonexistent_dat.nii.gz")
            mri = native_batch([row], protocol, device, load_dat=method.needs_dat(role, "eval"))
            assert "dat" not in mri
            clean = method.predict(role, model, mri)
            poisoned = method.predict(role, model, dict(mri, dat=torch.full_like(mri["mri"], float("nan"))))
            assert torch.isfinite(clean).all() and torch.equal(clean, poisoned)
    return dict(warm_start="CLI best checkpoint" if path.is_file() else "initialized fixture",
                missing_rows=len(missing), frozen_bn_buffers=bn_buffers, changed_iid_tensors=len(changed_iid),
                mri_only_roles=["student", "teacher_imputed"])


def export_parity(protocol, val_rows, tmp, device):
    run = tmp / "runs" / "dftd" / "fold0_seed0"
    meta = read_json(run / "run.json")
    if meta["status"] != "complete":
        raise RuntimeError("Export parity needs the completed smoke checkpoint.")
    complete = meta["stages"]["complete"]
    manifest = tmp / "mri_only.csv"
    write_csv(manifest, [dict(subject_id=row.subject_id, visit_id=row.visit_id, mri_path=row.mri_path)
                         for row in val_rows])
    rows = read_mri_manifest(manifest)
    with patch("data.manifest.read_manifest", side_effect=AssertionError("training manifest accessed")), \
         patch("models.backbones.load_medicalnet", side_effect=AssertionError("pretrained accessed")):
        bundle = export_run(run)
        path = tmp / "student_best.pt"
        torch.save(bundle, path)
        del bundle
        restored = torch.load(path, map_location="cpu", weights_only=True)
        assert set(restored["protocol"]["data"]) == {"spatial_dims"}
        assert restored["protocol"]["backbone"]["pretrained"] == "none"
        assert {"weights_path", "weights_sha256"}.isdisjoint(restored["protocol"]["backbone"])
        assert {key.split(".")[0] for key in restored["state_dict"]} == {"enc_a", "student"}
        pre = protocol["preprocessing"]
        assert restored["reference_grid"] == dict(shape=pre["input_size"], affine=pre["affine"])
        provenance = restored["provenance"]
        assert provenance["stage"] == "complete" and provenance["checkpoint"] == "best"
        assert provenance["checkpoint_sha256"] == complete["checkpoint_sha256"]
        predictions = predict_mri(restored, rows, device, batch_size=protocol["training"]["batch_size"])
        expected = {row["unit"]: float(row["y_pred"]) for row in read_csv(run / "eval_val" / "predictions.csv")
                    if row["role"] == "student"}
        assert expected.keys() == {row["unit"] for row in predictions}
        delta = max(abs(row["y_pred"] - expected[row["unit"]]) for row in predictions)
        assert delta <= 2e-5, delta
        count = sum(p.numel() for p in load_bundle(restored, "cpu").parameters())
    return dict(epoch=complete["best_epoch"], checkpoint_sha256=provenance["checkpoint_sha256"],
                max_abs_delta_updrs=delta, mri_parameters=count, bundle_bytes=path.stat().st_size,
                predictions=len(predictions))


def ntua_regression(tmp, commands):
    run_cli(commands, "tests/run_checks_dftd.py", "--report_dir", tmp)
    return [{"name": row["name"], "passed": row["passed"], "seconds": row["seconds"]}
            for row in read_json(tmp / "checks_report_dftd.json")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=PROTOCOL, help="Shared PPMI protocol YAML")
    parser.add_argument("--approved-data", action="store_true", help="Use the published cohort instead of the fixture")
    parser.add_argument("--report-json", type=Path, help="Optional path for the structured check evidence")
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol_path = resolve_path(args.protocol)
    if args.approved_data:
        return run_checks(args, protocol_path, load_yaml(protocol_path))
    with tempfile.TemporaryDirectory(prefix="off30_dftd_fixture_") as folder:
        fixture_protocol, protocol, _, _ = approved_fixture(Path(folder), protocol_path)
        return run_checks(args, fixture_protocol, protocol)


def run_checks(args, protocol_path, protocol):
    cfg = load_yaml(ROOT / CONFIG)
    device = choose_device(protocol["training"]["device"])
    rows, cohort = approved_cohort(protocol)
    split = make_split(rows, 0, protocol)
    train_rows, val_rows = rows_of(rows, split.train), rows_of(rows, split.val)

    report = dict(started=datetime.now().astimezone().isoformat(), invocation=[sys.executable, *sys.argv],
                  source_hash=source_hash(), protocol=protocol["name"], output_root=protocol["output_root"],
                  data_mode="approved" if args.approved_data else "synthetic", cohort=cohort,
                  config_hashes={file: file_sha256(resolve_path(file)) for file in (args.protocol, CONFIG, GRID)},
                  protocol_hash=config_hash(protocol), config_hash=config_hash(cfg),
                  test_sha256=file_sha256(Path(__file__)), medicalnet_sha256=protocol["backbone"]["weights_sha256"],
                  environment=dict(platform=platform.platform(), python=platform.python_version(),
                                   torch=torch.__version__, cuda=torch.version.cuda, device=str(device),
                                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None),
                  checks=[], commands=[])

    def check(name, fn):
        start = time.perf_counter()
        try:
            detail = fn()
            passed = not (isinstance(detail, dict) and detail.get("oom", False))
        except Exception:
            detail, passed = traceback.format_exc(), False
        report["checks"].append(dict(name=name, passed=passed, detail=detail, seconds=round(time.perf_counter() - start, 2)))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}", flush=True)
        if not passed:
            print(detail, flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    commands = report["commands"]
    with tempfile.TemporaryDirectory(prefix="off30_dftd_checks_") as temp:
        tmp = Path(temp)
        check("QC publication, fingerprint and five subject splits", lambda: cohort)
        check("D-039 outer-fold CLI dry-run", lambda: tune_plan(protocol_path, protocol, cfg, tmp, commands))
        check("PPMI CLI smoke and best-checkpoint validation roles",
              lambda: smoke(protocol_path, protocol, cfg, cohort, train_rows, val_rows, tmp, commands))
        check("MRI-only bundle roundtrip and best-checkpoint parity", lambda: export_parity(protocol, val_rows, tmp, device))
        if not args.approved_data:
            check("Full mode rejects all-unpaired fold", lambda: no_pair(protocol, cfg, rows, tmp))
            check("Region grid variants and invalid negatives", lambda: edge_losses(protocol, cfg, train_rows, device))
            check("Exact GRL sign and fixed teacher KD targets", lambda: loss_gradients(protocol, cfg))
        for routing in ROUTINGS:
            check(f"Native shapes and per-parameter gradient routing ({routing})",
                  lambda: native_gradients(protocol, cfg, train_rows, routing, device))
        for routing, lr in itertools.product(ROUTINGS, LEARNING_RATES):
            check(f"Batch 8 and singleton Adam, omega and VRAM ({routing}, lr={lr:g})",
                  lambda: adam_probe(protocol, cfg, train_rows, routing, lr, device))
        if not args.approved_data:
            check("Missing-DaT warm start, frozen M_RAD/BN and MRI-only roles",
                  lambda: warm_start(protocol, cfg, train_rows, tmp, device))
            check("NTUA regression suite", lambda: ntua_regression(tmp, commands))

    report["finished"] = datetime.now().astimezone().isoformat()
    report["source_hash_end"] = source_hash()
    report["passed"] = sum(row["passed"] for row in report["checks"])
    report["total"] = len(report["checks"])
    if args.report_json:
        write_json(args.report_json, report)
    print(f"{report['passed']}/{report['total']} OFF30 checks passed ({report['data_mode']}).", flush=True)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
