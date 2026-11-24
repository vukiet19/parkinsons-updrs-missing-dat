"""Deployment parity/independence tests; initialized checkpoint fixtures, no training.

python tests/run_checks_mri_inference.py
Temporary fixture checkpoints are removed; existing runs/reports are untouched.
"""

import copy
import gc
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np
import torch

from baselines import get_method
from data.dataset import load_image
from utils.config import load_yaml, resolve_path
from utils.io import read_csv, write_csv, write_json
from utils.repro import file_sha256
from utils.mri_inference import (METHODS, export_run, inference_protocol, load_bundle, predict_mri,
                                 read_mri_manifest, validate_bundle)

torch.set_num_threads(4)


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="pd_mri_deployment_test_")
        cls.root = Path(cls.temp.name)
        cls.protocol = load_yaml(ROOT / "configs/protocol_ppmi.yaml")
        pre = cls.protocol["preprocessing"]
        rng = np.random.default_rng(30)
        cls.image = cls.root / "mri.nii.gz"
        nib.save(nib.Nifti1Image(rng.standard_normal(pre["input_size"], dtype=np.float32), pre["affine"]), cls.image)
        cls.rows = [dict(subject_id="new_subject", visit_id="BL", mri_path=cls.image)]
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        print(json.dumps(cls.results, indent=2), flush=True)
        cls.temp.cleanup()

    def fixture(self, name, protocol=None, cfg=None):
        p = copy.deepcopy(protocol or self.protocol)
        cfg = cfg or load_yaml(ROOT / f"configs/baselines/{dict(crad='crad_ppmi', agmd='agmd_ppmi').get(name, name)}.yaml")
        if name == "agmd":
            cfg = copy.deepcopy(cfg)
            pre = p["preprocessing"]
            atlas = np.zeros(pre["input_size"], dtype=np.int16)
            for index, label in enumerate(cfg["graph"]["atlas_key_rois"]):
                start = 1 + 3 * index
                atlas[start:start + 2, 10:20, 10:20] = label
            atlas_path = self.root / "synthetic_atlas.nii.gz"
            nib.save(nib.Nifti1Image(atlas, pre["affine"]), atlas_path)
            cfg["graph"].update(atlas_path=str(atlas_path), atlas_sha256=file_sha256(atlas_path))
        # The fixture model has the real topology, initialized locally. Metadata
        # points to absent training data/pretrained files to prove independence.
        method = get_method(name)(p, cfg)
        stage = "complete" if name == "dftd" else "main" if name == "shaspec" else "student"
        offline = copy.deepcopy(p)
        offline["backbone"]["pretrained"] = "none"
        source = get_method(name)(offline, cfg).build(stage).to(self.device).eval()
        run = self.root / f"{name}_{p['data']['spatial_dims']}d"
        run.mkdir(exist_ok=True)
        p["data"]["manifest"] = str(self.root / "absent_training_manifest.csv")
        p["data"]["root"] = str(self.root / "absent_training_images")
        p["backbone"]["weights_path"] = str(self.root / "absent_pretrained.pth")
        meta = dict(method=name, status="complete", protocol=p, config=cfg, fold=0, seed=0,
                    scaler=dict(mean=20.0, std=2.5), stages={stage: dict(best_epoch=0)},
                    source_hash="initialized_fixture_not_a_trained_result", data_fingerprint="fixture")
        write_json(run / "run.json", meta)
        torch.save(dict(method=name, stage=stage, fold=0, seed=0, state_dict=source.state_dict()), run / f"{stage}.pt")
        return run, method, source

    def test_all_seven_3d_students_match_without_training_assets(self):
        for name in METHODS:
            with self.subTest(method=name):
                run, method, source = self.fixture(name)
                x = load_image(self.image, 1, [97, 115, 97], "none", 3)[None].to(self.device)
                with torch.inference_mode():
                    expected = method.predict("student", source, dict(mri=x, has_dat=torch.zeros(1, dtype=torch.bool, device=self.device)))
                    expected = float(expected[0]) * 2.5 + 20.0
                source_count = sum(v.numel() for v in source.parameters())
                bundle = export_run(run, reference_image=self.image)
                target = self.root / f"{name}_bundle.pt"
                torch.save(bundle, target)
                restored = torch.load(target, weights_only=True, map_location="cpu")
                # Refuse access to the original pretrained/atlas/data machinery.
                with patch("data.manifest.read_manifest", side_effect=AssertionError("training manifest accessed")), \
                     patch("models.backbones.resnet.load_medicalnet", side_effect=AssertionError("pretrained accessed")), \
                     patch("baselines.agmd.graph.load_atlas", side_effect=AssertionError("atlas accessed")):
                    got = predict_mri(restored, self.rows, self.device)[0]["y_pred"]
                self.assertAlmostEqual(expected, got, delta=2e-5)
                compact = load_bundle(restored, "cpu")
                compact_count = sum(v.numel() for v in compact.parameters())
                self.assertLessEqual(compact_count, source_count)
                if name in ("kd_basic", "cregkd", "dftd", "shaspec"):
                    self.assertLess(compact_count, source_count)
                self.assertNotIn("manifest", restored["protocol"]["data"])
                self.results.append(dict(method=name, dimensions=3, parity_abs=abs(expected-got),
                                         source_parameters=source_count, deployment_parameters=compact_count,
                                         checkpoint_bytes=target.stat().st_size, no_external_assets=True))
                if name == "mri_only":
                    self.base_bundle = restored
                del source, compact, bundle, restored, x
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

    def test_manifest_labels_dat_and_fold_are_unused(self):
        csv = self.root / "inputs.csv"
        write_csv(csv, [dict(subject_id="new_subject", mri_path="mri.nii.gz", dat_path="absent_DAT.nii.gz",
                             updrs3="not_a_label", aux_score="not_a_score", fold="not_a_fold")])
        rows = read_mri_manifest(csv)
        self.assertEqual(rows, self.rows)
        write_csv(csv, [dict(subject_id="s", mri_path="mri.nii.gz")] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_mri_manifest(csv)

    def test_shared_table_checks_effective_cohort_and_final_seeds(self):
        from data.manifest import make_split, read_manifest
        from ppmi_fixture import approved_fixture
        from scripts.summarize_pilot import collect

        _, protocol, _, approval = approved_fixture(self.root / "table_fixture")
        cohort = read_manifest(protocol)
        n = len(cohort)
        folds = protocol["data"]["split"]["folds"]
        seeds = protocol["training"]["seeds"]
        entries = []
        for method in ("mri_only", "agmd"):
            root = self.root / f"table_{method}"
            runs = []
            for fold in folds:
                for seed in seeds:
                    run = f"fold{fold}_seed{seed}"
                    runs.append(run)
                    write_json(root / run / "run.json", dict(method=method, status="complete", fold=fold, seed=seed,
                               protocol=protocol, split=make_split(cohort, fold, protocol).as_dict(),
                               data_fingerprint=approval["data_fingerprint"]))
            row = dict(role="student", scope="pooled_oof", n_units=n, n_seeds=len(seeds), mae_mean=1.0,
                       mae_sd=0.1, rmse_mean=2.0, rmse_sd=0.2, pearson_r_mean=None, r2_mean=0.0)
            write_json(root / "summary_test.json", dict(method=method, split="test", summary=[row], runs=runs))
            entries.append(dict(name=method, method=method, run_dir=str(root), role="student"))
        table = collect(entries, protocol=protocol)
        self.assertEqual([r["n_units"] for r in table], [n, n])
        path = Path(entries[1]["run_dir"]) / "summary_test.json"
        incomplete = json.loads(path.read_text())
        incomplete["runs"] = incomplete["runs"][:1]
        write_json(path, incomplete)
        with self.assertRaisesRegex(ValueError, "fold x seed"):
            collect(entries, protocol=protocol)

    def test_2d_slice_aggregation_and_cli(self):
        from PIL import Image
        p = load_yaml(ROOT / "configs/protocol.yaml")
        p["backbone"]["pretrained"] = "none"
        p["preprocessing"]["input_size"] = [64, 64]
        run, method, source = self.fixture("mri_only", p)
        a, b = self.root / "slice_a.png", self.root / "slice_b.png"
        Image.fromarray(np.full((64, 64), 70, dtype=np.uint8)).save(a)
        Image.fromarray(np.full((64, 64), 150, dtype=np.uint8)).save(b)
        manifest = self.root / "slices.csv"
        write_csv(manifest, [dict(subject_id="s", visit_id="v", mri_path=f.name) for f in (a, b)])
        bundle = export_run(run)
        rows = read_mri_manifest(manifest)
        x = torch.stack([load_image(f, 3, [64, 64], "imagenet", 2) for f in (a, b)]).to(self.device)
        with torch.inference_mode():
            expected = float(method.predict("student", source, dict(mri=x)).mean()) * 2.5 + 20
        got = predict_mri(bundle, rows, self.device, batch_size=2)
        self.assertEqual(got[0]["n_rows"], 2)
        self.assertAlmostEqual(got[0]["y_pred"], expected, delta=2e-5)
        exported = self.root / "cli.pt"
        result = subprocess.run([sys.executable, "export_inference.py", "--run_dir", str(run), "--output", str(exported)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.root / "cli_predictions.csv"
        result = subprocess.run([sys.executable, "infer.py", "--bundle", str(exported), "--manifest", str(manifest), "--output", str(output), "--device", "cpu", "--batch_size", "2"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(read_csv(output)), 1)
        self.assertFalse(json.loads(output.with_suffix(".json").read_text())["benchmark_eligible"])
        direct = self.root / "cli_direct.csv"
        result = subprocess.run([sys.executable, "infer.py", "--run_dir", str(run), "--manifest", str(manifest), "--output", str(direct), "--device", "cpu", "--batch_size", "2"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_csv(output), read_csv(direct))
        result = subprocess.run([sys.executable, "infer.py", "--bundle", str(exported), "--manifest", str(manifest), "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "Existing output was overwritten")

    def test_grid_scaler_identity_and_checkpoint_selection_guards(self):
        run, _, source = self.fixture("mri_only")
        bundle = export_run(run, reference_image=self.image)
        bad = copy.deepcopy(bundle)
        bad["scaler"]["std"] = 0
        with self.assertRaises(ValueError):
            validate_bundle(bad)
        image = nib.load(str(self.image))
        wrong = self.root / "wrong_affine.nii.gz"
        affine = image.affine.copy()
        affine[0, 3] += 2
        nib.save(nib.Nifti1Image(image.get_fdata(dtype=np.float32), affine), wrong)
        with self.assertRaisesRegex(ValueError, "affine"):
            predict_mri(bundle, [dict(subject_id="s", visit_id="BL", mri_path=wrong)], self.device)
        wrong_shape = self.root / "wrong_shape.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32), image.affine), wrong_shape)
        with self.assertRaisesRegex(ValueError, "no automatic resampling"):
            predict_mri(bundle, [dict(subject_id="s", visit_id="BL", mri_path=wrong_shape)], self.device)
        bad_voxels = self.root / "nonfinite.nii.gz"
        voxels = image.get_fdata(dtype=np.float32)
        voxels[0, 0, 0] = np.nan
        nib.save(nib.Nifti1Image(voxels, image.affine), bad_voxels)
        with self.assertRaisesRegex(FloatingPointError, "Non-finite voxels"):
            predict_mri(bundle, [dict(subject_id="s", visit_id="BL", mri_path=bad_voxels)], self.device)
        with self.assertRaises(FileNotFoundError):
            export_run(run, "last")
        state = {k: v.detach().cpu().clone() for k, v in source.state_dict().items()}
        state["head.bias"] += 1
        torch.save(dict(method="mri_only", stage="student", fold=0, seed=0, state_dict=state), run / "student_last.pt")
        last = export_run(run, "last", self.image)
        first_pred = predict_mri(bundle, self.rows, self.device)[0]["y_pred"]
        last_pred = predict_mri(last, self.rows, self.device)[0]["y_pred"]
        self.assertAlmostEqual(last_pred-first_pred, 2.5, delta=2e-5)
        bad_ckpt = dict(method="mri_only", stage="student", fold=0, seed=99, state_dict=state)
        torch.save(bad_ckpt, run / "student_last.pt")
        with self.assertRaisesRegex(ValueError, "identity"):
            export_run(run, "last")
        del source


if __name__ == "__main__":
    unittest.main(verbosity=2)
