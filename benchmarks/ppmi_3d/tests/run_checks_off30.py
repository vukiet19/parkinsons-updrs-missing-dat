"""OFF30 publication, checkpoint and tuning guards using synthetic inputs only."""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare
import evaluate
import train
from data.dataset import make_loader
from data.manifest import TargetScaler, data_fingerprint, make_split, read_manifest, rows_of
from ppmi_fixture import write_fixture
from scripts import make_ppmi_manifest as mpm, train_eval_prune, tune
from utils.config import config_hash
from utils.io import read_csv, read_json, write_csv, write_json


class Off30Tests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="off30_guard_")
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        _, self.protocol, self.qc = write_fixture(self.folder, size=(4, 5, 4), n=10)
        self.root = Path(self.protocol["data"]["root"])
        self.manifest = Path(self.protocol["data"]["manifest"])
        self.case = self.root / "FIX00/case.json"

    def build(self, **kwargs):
        return mpm.build_manifest(self.protocol, self.qc, log=lambda *_: None, **kwargs)

    def publish(self):
        rows, evidence = self.build()
        return mpm.publish_manifest(self.protocol, self.root, self.manifest, self.qc, rows, evidence)

    def test_qc_release_and_fingerprint(self):
        with self.assertRaisesRegex(FileNotFoundError, "QC-approved manifest"):
            read_manifest(self.protocol)
        qc = read_csv(self.qc)
        write_csv(self.qc, [dict(qc[0], visual_qc="FAIL", note="motion"), *qc[1:]])
        rows, evidence = self.build()
        self.assertEqual(evidence["exclusion_reasons"], {"FIX00": "motion"})
        targets = {r["subject_id"]: r["updrs3"] for r in rows}
        self.assertEqual({r["subject_id"]: r["fold"] for r in rows}, mpm.assign_folds(targets, 5, 0))
        record = self.publish()
        self.assertEqual(record, self.publish())
        cohort = read_manifest(self.protocol)
        self.assertEqual(len(cohort), 9)
        self.assertEqual(record["data_fingerprint"], data_fingerprint(self.protocol, cohort))
        write_csv(self.qc, qc)
        with self.assertRaisesRegex(ValueError, "QC approval differs"):
            read_manifest(self.protocol)
        with self.assertRaisesRegex(ValueError, "Use a new protocol version"):
            self.publish()

    def test_invalid_clinical_schema(self):
        original = read_json(self.case)
        for dotted, value, message in (
                ("subject", "other", "subject differs"),
                ("case_id", "other", "case_id differs"),
                ("clinical.visit", " ", "non-empty visit"),
                ("clinical.MoCA.visit", "V06", "clinical.visit"),
                ("clinical.MDS_UPDRS_III.visit", "V04", "clinical.visit"),
                ("clinical.MoCA.total", 31, "expected a score"),
                ("clinical.MDS_UPDRS_III.total", float("inf"), "finite numeric"),
                ("clinical.MDS_UPDRS_III.total", 1, "sum does not match"),
                ("clinical.MDS_UPDRS_III.items", [], "33 entries"),
                ("clinical.PDSTATE", "ON", "clinical.PDSTATE OFF"),
                ("clinical.PDSTATE", None, "missing case.json:clinical.PDSTATE"),
                ("med_state", "ON", "disagree"),
                ("MoCA", 30, "flat and nested MoCA"),
                ("label_visit", "V06", "clinical.visit disagree")):
            with self.subTest(field=dotted):
                case = copy.deepcopy(original)
                *parts, key = dotted.split(".")
                record = case
                for part in parts:
                    record = record[part]
                if value is None:
                    record.pop(key)
                else:
                    record[key] = value
                self.case.write_text(json.dumps(case), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    self.build()
        for key, value, message in (("code", "item01", "duplicate code"),
                                    ("score", 1.5, "integer score"), ("score", 5, "integer score")):
            case = copy.deepcopy(original)
            case["clinical"]["MDS_UPDRS_III"]["items"][0][key] = value
            write_json(self.case, case)
            with self.assertRaisesRegex(ValueError, message):
                self.build()
        write_json(self.case, original)
        with self.assertRaisesRegex(ValueError, "declared medication state is ON"):
            self.build(med_state_all="ON", med_state_source="external fixture")
        with self.assertRaisesRegex(ValueError, "split_seed"):
            self.build(fold_seed=9)

    def test_invalid_images_and_reviewer_evidence(self):
        path = self.root / "FIX00" / mpm.DAT_REL
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing"):
            self.build()
        affine = np.asarray(self.protocol["preprocessing"]["affine"], dtype=np.float64)
        wrong = affine.copy()
        wrong[0, 3] += 4
        for voxels, grid, message in (
                (np.zeros((4, 5, 4, 1), np.float32), affine, "shape"),
                (np.zeros((4, 4, 4), np.float32), affine, "shape"),
                (np.zeros((4, 5, 4), np.float32), wrong, "target grid"),
                (np.full((4, 5, 4), np.nan, np.float32), affine, "non-finite voxels")):
            nib.save(nib.Nifti1Image(voxels, grid), str(path))
            with self.assertRaisesRegex(ValueError, message):
                self.build()
        path.write_bytes(original)
        qc = read_csv(self.qc)
        for rows, message in ((qc[1:], "missing"), (qc + qc[:1], "duplicated"),
                              ([dict(qc[0], visual_qc="pending"), *qc[1:]], "PENDING"),
                              ([dict(qc[0], reviewer=""), *qc[1:]], "requires reviewer"),
                              ([dict(qc[0], visual_qc="FAIL"), *qc[1:]], "reason for FAIL")):
            write_csv(self.qc, rows)
            with self.assertRaisesRegex(ValueError, message):
                self.build()

    def test_teacher_reuse_identity_and_fingerprint(self):
        approval = self.publish()
        rows = read_manifest(self.protocol)
        split = make_split(rows, 0, self.protocol)
        scaler = TargetScaler.fit(rows_of(rows, split.train))
        fingerprint = approval["data_fingerprint"]
        folder = self.folder / "fold0_seed0"
        folder.mkdir()
        cfg = {"method": "crad"}
        path = folder / "teacher.pt"
        ckpt = dict(method="crad", stage="teacher", fold=0, seed=0, data_fingerprint=fingerprint)
        torch.save(ckpt, path)
        meta = dict(method="crad", fold=0, seed=0, status="complete", protocol=self.protocol, config=cfg,
                    split=split.as_dict(), scaler=dict(mean=scaler.mean, std=scaler.std),
                    data_fingerprint=fingerprint, stages={"teacher": {}})
        write_json(folder / "run.json", meta)
        reuse = lambda fp: train.reusable_teacher(self.folder, 0, 0, self.protocol, cfg, split, scaler, fingerprint=fp)
        self.assertEqual(reuse(fingerprint)[0], path)
        with self.assertRaisesRegex(ValueError, "data_fingerprint"):
            reuse("mismatch")
        torch.save(dict(ckpt, stage="student"), path)
        with self.assertRaisesRegex(ValueError, "checkpoint identity"):
            reuse(fingerprint)

    def test_evaluator_checks_approval_fingerprint_and_split(self):
        record = self.publish()
        stats = {}
        rows = read_manifest(self.protocol, stats)
        split = make_split(rows, 0, self.protocol)
        scaler = TargetScaler.fit(rows_of(rows, split.train))
        meta = dict(method="mri_only", fold=0, seed=0, status="complete", protocol=self.protocol, config={},
                    split=split.as_dict(), scaler=dict(mean=scaler.mean, std=scaler.std),
                    data_fingerprint=record["data_fingerprint"], med_state=stats)
        run = self.folder / "run"
        for changed, message in (
                (dict(med_state={}), "QC approval differs"),
                (dict(data_fingerprint="other"), "data/manifest differ"),
                (dict(split=dict(meta["split"], val=[])), "stored split differs"),
                (dict(scaler=dict(mean=scaler.mean + 1, std=scaler.std)), "target scaler")):
            write_json(run / "run.json", dict(meta, **changed))
            with self.assertRaisesRegex(ValueError, message):
                evaluate.evaluate_run("mri_only", run, "test", ["all"], torch.device("cpu"), False, {})

    def test_run_group_accepts_fold_configs_but_rejects_mixed_cohorts(self):
        approval = self.publish()
        fingerprint = approval["data_fingerprint"]
        runs = [self.folder / f"fold{fold}_seed0" for fold in (0, 1)]
        for fold, run in enumerate(runs):
            protocol = copy.deepcopy(self.protocol)
            protocol["training"]["lr"] = (fold + 1) * 1e-4
            meta = dict(method="mri_only", fold=fold, seed=0, status="complete", protocol=protocol,
                        data_fingerprint=fingerprint, med_state=dict(qc_approval=approval))
            write_json(run / "run.json", meta)
        self.assertEqual(evaluate.group_folds("mri_only", runs), self.protocol["data"]["split"]["folds"])
        for changed in (dict(data_fingerprint="other"), dict(fold=0), dict(med_state={})):
            write_json(runs[1] / "run.json", dict(meta, **changed))
            with self.assertRaisesRegex(ValueError, "inconsistent OFF30 cohort or fold/seed"):
                evaluate.group_folds("mri_only", runs)

    def test_off30_pair_comparison_checks_subjects_folds_seeds_and_targets(self):
        approval = self.publish()
        cohort = read_manifest(self.protocol)
        split = make_split(cohort, 0, self.protocol)
        run = "fold0_seed0"
        a, b = self.folder / "a", self.folder / "b"
        predictions = [dict(role="student", fold=0, seed=0, unit=r.unit, subject_id=r.subject_id,
                            y_true=r.target, y_pred=r.target + 1) for r in rows_of(cohort, split.test)]
        meta = dict(fold=0, seed=0, status="complete", protocol=self.protocol, split=split.as_dict(),
                    data_fingerprint=approval["data_fingerprint"], med_state=dict(qc_approval=approval))
        for folder in (a, b):
            write_json(folder / run / "run.json", meta)
            write_json(folder / "summary_test.json", dict(runs=[run]))
            write_csv(folder / "predictions_test.csv", predictions)
        compare_pair = lambda: compare.compare(a, "student", b, "student", "test", "hier_bootstrap", n_boot=100)
        self.assertEqual(compare_pair()["delta_mae"], 0.0)
        first, *rest = predictions
        for changed in ({"fold": 1}, {"seed": 1}, {"subject_id": "wrong-subject"}):
            write_csv(b / "predictions_test.csv", [dict(first, **changed), *rest])
            with self.assertRaisesRegex(ValueError, "predictions differ from the subjects"):
                compare_pair()
        write_csv(b / "predictions_test.csv", [dict(first, y_true=999), *rest])
        with self.assertRaisesRegex(ValueError, "same approval, data, subjects, folds, seeds and targets"):
            compare_pair()

    def test_tuner_selects_independently_and_locks_resume(self):
        args = argparse.Namespace(method="mri_only", pset=[], mode="outer_fold", protocol="fixture.yaml", limit=None,
                                  keep_ckpt=False)
        configs = [{"protocol.training.lr": lr} for lr in (0.001, 0.002)]
        model = self.folder / "model.yaml"
        model.write_text("method: mri_only\n", encoding="utf-8")
        self.protocol["data"]["split"]["folds"] = [0, 1]
        effective = [(dict(self.protocol, training=dict(self.protocol["training"], lr=lr)), {"method": "mri_only"})
                     for lr in (0.001, 0.002)]
        calls = []

        def simulate(cmd, **_kwargs):
            calls.append(cmd[1])
            if cmd[1] == "train.py":
                index = int(cmd[cmd.index("--tag") + 1][-1])
                root = Path(cmd[cmd.index("--out") + 1]) / f"mri_only_cfg{index}"
                trial = read_json(root / "trial.json")
                fold, = trial["folds"]
                meta = {k: trial[k] for k in ("method", "seed", "protocol_hash", "config_hash",
                                             "data_fingerprint", "source_hash")}
                self.assertEqual(int(cmd[cmd.index("--fold") + 1]), fold)
                write_json(root / f"fold{fold}_seed0/run.json", dict(meta, status="complete", fold=fold,
                           split=trial["splits"][str(fold)]))
            else:
                self.assertEqual(cmd[cmd.index("--split") + 1], "val")
                path = Path(cmd[cmd.index("--run_dir") + 1])
                val = float(path.parent.name[-1] != path.name[4])
                write_csv(path / "eval_val/metrics.csv", [dict(role="student", mae=val, rmse=val)])

        with patch("scripts.tune.subprocess.run", side_effect=simulate):
            for fold in (0, 1):
                group = str(fold)
                root = self.folder / f"fold{fold}"
                common = dict(method=args.method, mode=args.mode, folds=[fold], seed=0, splits={group: {"fold": fold}},
                              data_fingerprint="fixture-fingerprint", source_hash="fixture-source")
                base = ["train.py", "--method", args.method, "--protocol", args.protocol]
                run = lambda metadata: tune.select_group(args, base, group, root, metadata, "student", configs,
                                                          effective, self.folder / "grid.yaml", model)
                run(common)
                selected = yaml.safe_load((root / "selected.yaml").read_text(encoding="utf-8"))
                self.assertEqual(selected["selected_index"], fold)
                self.assertEqual(selected["protocol_hash"], config_hash(effective[fold][0]))
                self.assertIn(f"--fold {fold}", selected["final_command"])
                count = len(calls)
                run(common)
                self.assertEqual(len(calls), count)
                with self.assertRaisesRegex(RuntimeError, "Incomplete tuning group"):
                    run(dict(common, data_fingerprint="different-data"))
                self.assertEqual(len(calls), count)
        with patch("scripts.train_eval_prune.source_hash", return_value="fixture-source"):
            selections = train_eval_prune.load_fold_selections(self.folder, args.method, self.protocol,
                                                               "fixture-fingerprint")
            self.assertEqual([s[0]["training"]["lr"] for s in selections.values()], [0.001, 0.002])
            with self.assertRaisesRegex(ValueError, "does not match the requested benchmark"):
                train_eval_prune.load_fold_selections(self.folder, args.method, self.protocol, "other-dataset")
            model.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration differs from selection"):
                train_eval_prune.load_fold_selections(self.folder, args.method, self.protocol, "fixture-fingerprint")

    def test_shared_splits_train_only_scaler_and_singleton_batch(self):
        _, protocol, qc = write_fixture(self.folder / "full", size=(4, 5, 4))
        root = Path(protocol["data"]["root"])
        manifest = Path(protocol["data"]["manifest"])
        records, evidence = mpm.build_manifest(protocol, qc, log=lambda *_: None)
        mpm.publish_manifest(protocol, root, manifest, qc, records, evidence)
        seen = []
        rows = read_manifest(protocol)
        candidates = {r.subject_id for r in rows}
        for fold in protocol["data"]["split"]["folds"]:
            split = make_split(rows, fold, protocol)
            parts = [set(getattr(split, name)) for name in ("train", "val", "test")]
            self.assertEqual(list(map(len, parts)), [17, 7, 6])
            self.assertEqual(set.union(*parts), candidates)
            self.assertEqual(sum(map(len, parts)), len(candidates))
            seen.extend(split.test)
            train_rows = rows_of(rows, split.train)
            scaler = TargetScaler.fit(train_rows)
            self.assertAlmostEqual(scaler.mean, np.mean([r.target for r in train_rows]))
            self.assertAlmostEqual(scaler.std, np.std([r.target for r in train_rows]))
        self.assertCountEqual(seen, candidates)
        loader = make_loader(train_rows, protocol, load_dat=True, shuffle=False, seed=0)
        self.assertEqual([len(batch["unit"]) for batch in loader], [8, 8, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
