"""Acceptance checks for the OFF30 MRI+DaT anchor using synthetic inputs."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import get_method
from baselines.anchors.model import MMTeacherNet
from baselines.kd_basic.model import MMTeacherNet as KDTeacherNet
from data.manifest import Row
from utils.config import load_yaml, resolve_path


class MMTeacherOff30Tests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_yaml(resolve_path("configs/protocol_ppmi.yaml"))
        self.cfg = load_yaml(resolve_path("configs/baselines/mm_teacher.yaml"))
        self.method = get_method("mm_teacher")(self.protocol, self.cfg)
        self.paired = Row("FIX01", "BL", Path("mri.nii.gz"), Path("dat.nii.gz"), 12., 25., 0)

    def test_teacher_requires_all_pairs_in_approved_cohort(self):
        self.assertEqual(self.method.stages, ("teacher",))
        self.assertEqual(self.method.primary_role, "teacher")
        for phase in ("train", "eval"):
            self.assertTrue(self.method.needs_dat("teacher", phase))
        rows = [self.paired, replace(self.paired, subject_id="FIX02")]
        self.assertEqual(self.method.train_rows("teacher", rows), rows)
        for missing in ("mri_path", "dat_path"):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "1 QC-approved row"):
                self.method.train_rows("teacher", [rows[0], replace(rows[1], **{missing: None})])

    def test_ntua_still_selects_paired_rows(self):
        protocol = load_yaml(resolve_path("configs/protocol.yaml"))
        method = get_method("mm_teacher")(protocol, self.cfg)
        unpaired = replace(self.paired, subject_id="FIX02", dat_path=None)
        self.assertEqual(method.train_rows("teacher", [self.paired, unpaired]), [self.paired])

    def test_singleton_3d_forward_backward_and_shared_model(self):
        self.assertIs(MMTeacherNet, KDTeacherNet)
        self.protocol["backbone"]["pretrained"] = "none"
        torch.manual_seed(0)
        model = self.method.build("teacher")
        pre = self.protocol["preprocessing"]
        size = tuple(pre["input_size"])
        batch = dict(mri=torch.randn(1, pre["mri_channels"], *size),
                     dat=torch.randn(1, pre["dat_channels"], *size), y_z=torch.tensor([1.0]))

        with torch.no_grad():
            pred = self.method.predict("teacher", model, batch)
        self.assertEqual(tuple(pred.shape), (1,))
        self.assertTrue(torch.isfinite(pred).all())

        loss, _ = self.method.training_step("teacher", model, {}, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in ("mri_encoder", "dat_encoder", "head"):
            with self.subTest(module=name):
                grads = [param.grad for param in getattr(model, name).parameters()]
                self.assertTrue(all(grad is not None and torch.isfinite(grad).all() for grad in grads))
                self.assertTrue(any(torch.count_nonzero(grad) > 0 for grad in grads))


if __name__ == "__main__":
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    unittest.main(verbosity=2)
