"""Vanilla KD fixtures and an optional validation-only OFF30 smoke.

    python tests/run_checks_kd_basic_off30.py
    python tests/run_checks_kd_basic_off30.py --smoke runs_ppmi/off30/smoke_kd_basic_20261122
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from baselines import get_method
from data.dataset import ImageRows, load_image
from data.manifest import Split, TargetScaler, read_manifest, rows_of
import evaluate
import train
from utils.config import load_yaml
from utils.inference import freeze
from utils.io import read_csv, read_json, write_json
from utils.mri_inference import export_run, predict_mri
from utils.repro import file_sha256

PROTOCOL = "configs/protocol_ppmi.yaml"
CONFIG = "configs/baselines/kd_basic.yaml"


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(memoryview(value.detach().cpu().numpy()).cast("B"))
    return digest.hexdigest()


class VanillaKDChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(30)
        self.protocol = load_yaml(ROOT / PROTOCOL)
        self.protocol["backbone"]["pretrained"] = "none"
        self.protocol["preprocessing"]["input_size"] = [32, 32, 32]
        self.cfg = load_yaml(ROOT / CONFIG)
        self.batch = dict(mri=torch.randn(3, 1, 32, 32, 32), dat=torch.randn(3, 1, 32, 32, 32),
                          y_z=torch.tensor([3., -2., 1.]), has_dat=torch.tensor([True, False, True]),
                          aux=torch.tensor([0., 0.5, 1.]))
        self.batch["dat"][1].fill_(float("nan"))

    def build(self):
        method = get_method("kd_basic")(self.protocol, self.cfg)
        return method, method.build("student"), freeze(method.build("teacher"))

    def test_output_and_hint_gradients_and_frozen_teacher(self):
        method, student, teacher = self.build()
        before = state_hash(teacher)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        regressor = student.regressors["2"]
        self.assertIsInstance(regressor, torch.nn.Conv3d)
        for term, weights in (("kd_output", (1., 0.)), ("kd_hint", (0., 1.))):
            with self.subTest(term=term):
                self.cfg["kd"].update(zip(("lambda_out", "lambda_hint"), weights))
                optimizer.zero_grad(set_to_none=True)
                loss, terms = method.training_step("student", student, {"teacher": teacher}, self.batch)
                self.assertGreater(terms[term].item(), 0.)
                (loss - terms["task"]).backward()
                stem_grad = student.encoder.stem[0].weight.grad
                self.assertTrue(torch.isfinite(stem_grad).all())
                self.assertGreater(torch.count_nonzero(stem_grad).item(), 0)
                if term == "kd_hint":
                    self.assertGreater(torch.count_nonzero(regressor.weight.grad).item(), 0)
                    self.assertEqual(torch.count_nonzero(student.head.weight.grad).item(), 0)
                else:
                    self.assertIsNone(regressor.weight.grad)
                    self.assertGreater(torch.count_nonzero(student.head.weight.grad).item(), 0)
                optimizer.step()
        self.assertEqual(state_hash(teacher), before)
        self.assertTrue(all(not mod.training for mod in teacher.modules()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))

    def test_masked_rows_task_only_and_moca_ignored(self):
        method, student, teacher = self.build()
        student.eval()
        pred = student(self.batch["mri"])["pred"]
        expected_task = (pred - self.batch["y_z"]).square().mean()
        mask = self.batch["has_dat"]
        with patch.object(teacher, "forward", wraps=teacher.forward) as forward:
            loss, terms = method.training_step("student", student, {"teacher": teacher}, self.batch)
        forward.assert_called_once()
        mri, dat = forward.call_args.args
        torch.testing.assert_close(mri, self.batch["mri"][mask])
        torch.testing.assert_close(dat, self.batch["dat"][mask])
        torch.testing.assert_close(terms["task"], expected_task)
        change = dict(self.batch, aux=torch.full((3,), float("nan")), y_z=self.batch["y_z"].clone())
        change["y_z"][1] += 20.
        changed, after = method.training_step("student", student, {"teacher": teacher}, change)
        for key in ("kd_output", "kd_hint"):
            torch.testing.assert_close(after[key], terms[key], atol=0., rtol=0.)
        torch.testing.assert_close(changed - loss, after["task"] - expected_task)
        no_dat = dict(mri=self.batch["mri"], y_z=self.batch["y_z"], has_dat=torch.zeros(3, dtype=torch.bool))
        loss, terms = method.training_step("student", student, {}, no_dat)
        torch.testing.assert_close(loss, expected_task, atol=0., rtol=0.)
        self.assertEqual((terms["kd_output"].item(), terms["kd_hint"].item()), (0., 0.))

    def check_anchor_parity(self, protocol, mri):
        self.cfg["kd"]["enabled"] = False
        kd = get_method("kd_basic")(protocol, self.cfg)
        anchor = get_method("mri_only")(protocol, {"method": "mri_only"})
        self.assertEqual(kd.stages, ("student",))
        self.assertFalse(kd.needs_dat("student", "train"))
        self.assertFalse(kd.needs_dat("student", "eval"))
        actual, expected = kd.build("student"), anchor.build("student")
        expected.load_state_dict(actual.state_dict(), strict=True)
        batch = dict(mri=mri, y_z=self.batch["y_z"])
        for method, model in ((kd, actual), (anchor, expected)):
            loss, _ = method.training_step("student", model, {}, batch)
            loss.backward()
        torch.testing.assert_close(kd.predict("student", actual.eval(), batch),
                                   anchor.predict("student", expected.eval(), batch), atol=0., rtol=0.)
        for x, y in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(x.grad, y.grad, atol=0., rtol=0.)

    def test_anchor_parity_in_3d(self):
        self.check_anchor_parity(self.protocol, self.batch["mri"])

    def test_anchor_parity_in_ntua_2d(self):
        protocol = load_yaml(ROOT / "configs/protocol.yaml")
        protocol["backbone"]["pretrained"] = "none"
        self.check_anchor_parity(protocol, torch.randn(3, 3, 32, 32))

    def test_teacher_reuse_guards(self):
        method = get_method("kd_basic")(self.protocol, self.cfg)
        split = Split(0, ("fixture_train",), ("fixture_val",), ("fixture_test",))
        scaler = TargetScaler(10., 2.)
        fingerprint = "synthetic-fingerprint"
        with TemporaryDirectory(prefix="kd_off30_reuse_") as tmp:
            root = Path(tmp)
            path = root / "fold0_seed0" / "teacher.pt"
            path.parent.mkdir()
            ckpt = dict(method="kd_basic", stage="teacher", fold=0, seed=0, data_fingerprint=fingerprint)
            meta = dict(method="kd_basic", status="complete", fold=0, seed=0, protocol=self.protocol,
                        scaler=vars(scaler), config=self.cfg, split=split.as_dict(), data_fingerprint=fingerprint)

            def reuse(run_meta, checkpoint):
                torch.save(checkpoint, path)
                stages = {"teacher": {"checkpoint_sha256": file_sha256(path)}}
                write_json(path.parent / "run.json", dict(run_meta, stages=stages))
                return train.reusable_teacher(root, 0, 0, self.protocol, self.cfg, split, scaler,
                                              method.teacher_sections, method_name="kd_basic",
                                              fingerprint=fingerprint)

            self.assertEqual(reuse(meta, ckpt)[0], path)
            changes = (("protocol", dict(self.protocol, name="other"), "protocol"),
                       ("fold", 1, "identity"), ("seed", 1, "identity"),
                       ("scaler", dict(mean=11., std=2.), "scaler"),
                       ("split", dict(split.as_dict(), val=["changed"]), "split"),
                       ("data_fingerprint", "other", "data_fingerprint"),
                       ("config", dict(self.cfg, teacher={}), "teacher_config"),
                       ("method", "mm_teacher", "identity"))
            for field, value, error in changes:
                with self.subTest(metadata=field), self.assertRaisesRegex(ValueError, error):
                    reuse(dict(meta, **{field: value}), ckpt)
            for field, value in (("stage", "student"), ("fold", 1), ("seed", 1),
                                 ("method", "mm_teacher"), ("data_fingerprint", "other")):
                with self.subTest(checkpoint=field), self.assertRaisesRegex(ValueError, "checkpoint identity"):
                    reuse(meta, dict(ckpt, **{field: value}))


def smoke(output):
    output = output.resolve()
    run_dir = output / "kd_basic" / "fold0_seed0"
    device = torch.device("cuda")
    fit_stage = train.fit_stage
    profile = {}

    def profile_stage(method, stage, frozen, train_rows, val_rows, scaler, protocol, device, seed, log,
                      keep_last=None):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        step = method.training_step
        sizes = []

        def record_step(role, model, frozen, batch):
            sizes.append(len(batch["mri"]))
            return step(role, model, frozen, batch)

        with patch.object(method, "training_step", side_effect=record_step):
            result = fit_stage(method, stage, frozen, train_rows, val_rows, scaler, protocol, device, seed, log,
                               keep_last=keep_last)
        torch.cuda.synchronize(device)
        profile[stage] = dict(train_batch_sizes=sizes, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        assert sizes == [8, 8, 1] * protocol["training"]["max_epochs"], (stage, sizes)
        print(f"[smoke] {stage}: {profile[stage]}", flush=True)
        return result

    with patch.object(train, "fit_stage", side_effect=profile_stage), patch.object(sys, "argv", [
        "train.py", "--method", "kd_basic", "--protocol", PROTOCOL, "--fold", "0", "--seed", "0",
        "--out", str(output), "--device", "cuda", "--pset", "training.max_epochs=2",
    ]):
        train.main()
    write_json(output / "cuda_profile.json", profile)

    opened = []
    read_image = ImageRows._image
    meta = read_json(run_dir / "run.json")
    cohort = read_manifest(meta["protocol"])
    val_rows = rows_of(cohort, meta["split"]["val"])
    val_mri = {r.mri_path for r in val_rows}

    def mri_only_read(dataset, path, channels):
        assert not dataset.load_dat and path in val_mri, "Student evaluator requested a non-validation MRI image"
        opened.append(path)
        return read_image(dataset, path, channels)

    with patch.object(ImageRows, "_image", mri_only_read), patch.object(sys, "argv", [
        "evaluate.py", "--method", "kd_basic", "--run_dir", str(run_dir), "--split", "val",
        "--roles", "student", "--device", "cuda",
    ]):
        evaluate.main()
    assert set(opened) == val_mri and len(opened) == len(val_rows), "Validation MRI access differs from split"

    bundle_path = output / "kd_basic_bundle.pt"
    bundle = export_run(run_dir)
    torch.save(bundle, bundle_path)
    bundle = torch.load(bundle_path, weights_only=True, map_location="cpu")
    assert bundle["reference_grid"] is not None and "manifest" not in bundle["protocol"]["data"]
    assert all(key.startswith(("encoder.", "head.")) for key in bundle["state_dict"])
    with patch("utils.mri_inference.load_image", wraps=load_image) as load:
        inferred = predict_mri(bundle, [dict(subject_id=r.subject_id, visit_id=r.visit_id, mri_path=r.mri_path)
                                       for r in val_rows], device, batch_size=8)
    assert {call.args[0] for call in load.call_args_list} == val_mri and load.call_count == len(val_rows)
    scored = {row["unit"]: float(row["y_pred"]) for row in read_csv(run_dir / "eval_val" / "predictions.csv")
              if row["role"] == "student"}
    deployed = {row["unit"]: row["y_pred"] for row in inferred}
    assert scored.keys() == deployed.keys() and len(scored) == len(val_rows)
    delta = max(abs(scored[key] - deployed[key]) for key in scored)
    assert math.isfinite(delta) and delta <= 2e-5, f"Bundle parity differs: {delta}"
    report = dict(n_val=len(scored), parity_max_abs=delta, tolerance=2e-5, evaluator_mri_reads=len(opened),
                  bundle_mri_reads=load.call_count, bundle_sha256=file_sha256(bundle_path),
                  data_fingerprint=meta["data_fingerprint"], protocol_hash=meta["protocol_hash"],
                  config_hash=meta["config_hash"], source_hash=meta["source_hash"])
    write_json(output / "bundle_parity.json", report)
    print(f"[smoke] parity: {report}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    if "--smoke" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--smoke", type=Path, required=True)
        smoke(parser.parse_args().smoke)
    else:
        unittest.main(verbosity=2)
