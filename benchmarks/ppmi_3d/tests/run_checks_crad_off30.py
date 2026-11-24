"""Check OFF30 CAFR and student behavior with an approved synthetic cohort."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.crad.losses import attention_kd
from baselines.crad.method import CRADMethod
from data.dataset import make_loader
from data.manifest import TargetScaler, read_manifest
from ppmi_fixture import write_fixture
from scripts import make_ppmi_manifest as manifest_builder
from utils.config import load_yaml
from utils.inference import freeze
from utils.io import read_json, write_json


class CRADOff30Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(0)
        scratch = tempfile.TemporaryDirectory(prefix="crad_off30_fixture_")
        cls.addClassCleanup(scratch.cleanup)
        folder = Path(scratch.name)
        _, cls.protocol, qc = write_fixture(folder, size=(33, 35, 33), n=10)
        case = folder / "images/FIX01/case.json"
        clinical_case = read_json(case)
        clinical_case["clinical"]["MoCA"]["total"] = 30
        write_json(case, clinical_case)
        rows, evidence = manifest_builder.build_manifest(cls.protocol, qc, log=lambda *_: None)
        manifest_builder.publish_manifest(cls.protocol, folder / "images", folder / "manifest.csv",
                                          qc, rows, evidence)
        ordered = sorted(read_manifest(cls.protocol), key=lambda row: row.aux)
        cls.rows = [ordered[0], ordered[-1]]
        cls.cfg = load_yaml(ROOT / "configs/baselines/crad_ppmi.yaml")
        cls.protocol["backbone"]["pretrained"] = "none"
        cls.method = CRADMethod(cls.protocol, cls.cfg)

    def test_moca_loss_layer_weights_and_frozen_teacher(self):
        self.assertEqual(self.method.report_name, "CRAD†")
        self.assertTrue(self.cfg["cafr"]["enabled"])
        self.assertEqual(self.cfg["sdu"]["attention_confidence"], "cafr")
        self.assertEqual([row.aux for row in self.rows], [0.0, 30.0])
        rows = self.method.train_rows("teacher", self.rows)
        batch = next(iter(make_loader(rows, self.protocol, load_dat=True, shuffle=False, seed=0)))
        torch.testing.assert_close(batch["aux"], torch.tensor([0.0, 1.0]))
        self.assertTrue(batch["aux_mask"].all())
        batch["y_z"] = TargetScaler.fit(rows).normalize(batch["y"])
        teacher = self.method.build("teacher").train()
        predictions = torch.arange(1, 5, dtype=torch.float32) / 10
        with torch.no_grad():
            for head, value in zip(teacher.cafr.heads, predictions):
                head.weight.zero_()
                head.bias.fill_(value)
        errors = predictions[None, :] - torch.tensor([[0.0], [1.0]])
        loss, teacher_terms = self.method.training_step("teacher", teacher, {}, batch)
        expected_cafr = (0.5 * errors.square()).sum(dim=1).mean()
        torch.testing.assert_close(teacher_terms["cafr"], expected_cafr)
        loss.backward()
        for head in teacher.cafr.heads:
            for param in head.parameters():
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(param.grad.abs().max().item(), 0.0)

        teacher.zero_grad(set_to_none=True)
        teacher = freeze(teacher)
        state = {name: value.clone() for name, value in teacher.state_dict().items()}
        self.assertTrue(any(name.endswith("running_mean") for name in state))
        student = self.method.build("student").train()
        optimizer = torch.optim.SGD(student.parameters(), lr=1e-4)
        observed_distances = []

        def capture_distances(*args, **kwargs):
            distances = attention_kd(*args, **kwargs)
            observed_distances.append(distances.detach())
            return distances

        with patch("baselines.crad.method.attention_kd", side_effect=capture_distances):
            student_loss, student_terms = self.method.training_step("student", student, {"teacher": teacher}, batch)
        weights = 1.0 - errors.square()
        torch.testing.assert_close(student_terms["conf_attention"], weights.mean())
        torch.testing.assert_close(student_terms["kd_attention"],
                                   (weights * observed_distances[0]).mean(dim=0).sum())
        self.assertTrue(torch.isfinite(student_loss))
        student_loss.backward()
        optimizer.step()
        self.assertTrue(all(not module.training for module in teacher.modules()))
        self.assertTrue(all(param.grad is None and not param.requires_grad for param in teacher.parameters()))
        for name, value in teacher.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]), name)
        print(json.dumps(dict(raw_moca=[0, 30], normalized=batch["aux"].tolist(),
                              cafr_loss=teacher_terms["cafr"].item(), layer_weights=weights.tolist(),
                              frozen_state_tensors=len(state))), flush=True)

    def test_student_loads_mri_only_and_predicts_independently_of_moca(self):
        student = self.method.build("student").eval()
        load_dat = self.method.needs_dat("student", "eval")
        self.assertFalse(load_dat)
        row = self.rows[0]
        alternate = replace(row, dat_path=row.dat_path.with_name("missing_dat.nii.gz"), aux=30.0)
        batches = [next(iter(make_loader([entry], self.protocol, load_dat, shuffle=False, seed=0)))
                   for entry in (row, alternate)]
        self.assertNotEqual(batches[0]["aux"].item(), batches[1]["aux"].item())
        with torch.inference_mode():
            self.assertTrue(all("dat" not in batch for batch in batches))
            first, second = [self.method.predict("student", student, batch) for batch in batches]
        torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
