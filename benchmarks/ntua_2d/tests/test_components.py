import copy
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from common.data import Images, TargetScale, collate_subjects, read_subjects, subject_mean
from common.fit import freeze_teacher
from common.models import Student
from kd_0.loss import Loss as BasicLoss
from kd_1.components.attention import PFCMAD
from kd_1.components.pfac import PFAC
from kd_1.components.sdu import layer_confidence, output_confidence
from kd_1.loss import Loss as CradLoss
from kd_1.model import Teacher


class ComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_subject_pooling_does_not_overweight_extra_images(self):
        values = torch.tensor([2.0, 4.0, 10.0], requires_grad=True)
        pooled = subject_mean(values, torch.tensor([0, 0, 1]), 2)
        torch.testing.assert_close(pooled, torch.tensor([3.0, 10.0]))
        pooled.mean().backward()
        torch.testing.assert_close(values.grad, torch.tensor([0.25, 0.25, 0.5]))

    def test_pfac_matches_original_outer_product_diagonal(self):
        features = torch.randn(2, 5, 8, 8, requires_grad=True)
        attended, weights = PFAC()(features)
        normalized = torch.nn.functional.normalize(features, dim=1)
        q = (normalized.mean((-2, -1)) + normalized.amax((-2, -1))).relu()
        outer = torch.bmm(q.unsqueeze(-1), q.unsqueeze(1))
        expected = outer.diagonal(dim1=1, dim2=2).sigmoid()[:, :, None, None]
        torch.testing.assert_close(weights, expected)
        self.assertEqual(sum(p.numel() for p in PFAC().parameters()), 0)
        attended.square().mean().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_pfc_updates_all_student_stages_only(self):
        teacher = [torch.randn(3, c, 6, 6, requires_grad=True) for c in (2, 4, 8, 16)]
        student = [torch.randn_like(value, requires_grad=True) for value in teacher]
        loss = PFCMAD()(student, teacher, torch.tensor([0, 1, 1]), torch.ones(2, 4))
        loss.backward()
        self.assertTrue(all(value.grad is None for value in teacher))
        self.assertTrue(all(value.grad.abs().sum() > 0 for value in student))

    def test_confidence_is_detached_and_softens_bad_teacher(self):
        teacher = torch.tensor([0.1, 0.8], requires_grad=True)
        student = torch.tensor([0.5, 0.1], requires_grad=True)
        weights = output_confidence(teacher, student, torch.zeros(2))
        self.assertFalse(weights.requires_grad)
        self.assertGreater(weights[0].item(), weights[1].item())
        self.assertTrue(((weights >= 0) & (weights <= 1)).all())
        layer = layer_confidence(torch.full((2, 4), 0.5, requires_grad=True), torch.tensor([0.5, 1.0]))
        torch.testing.assert_close(layer[0], torch.ones(4))
        torch.testing.assert_close(layer[1], torch.full((4,), 0.75))
        self.assertFalse(layer.requires_grad)

    def test_target_scaler_fits_training_subjects_only(self):
        subjects = read_subjects()
        train = [s for s in subjects if s.outer_fold != 0]
        scale = TargetScale.fit(train)
        altered = [replace(s, updrs=99999) if s.outer_fold == 0 else s for s in subjects]
        self.assertEqual(scale, TargetScale.fit([s for s in altered if s.outer_fold != 0]))

    def test_student_dataset_does_not_open_dat(self):
        subject = replace(read_subjects()[0], dat=Path('deliberately_missing_dat.png'))
        data = Images([subject], include_dat=False)
        batch = collate_subjects([data[0]])
        self.assertNotIn('dat', batch)
        self.assertEqual(batch['mri'].shape[-2:], (224, 224))

    def test_folds_keep_subjects_and_images_together(self):
        subjects = read_subjects()
        self.assertEqual([sum(s.outer_fold == f for s in subjects) for f in range(3)], [5, 5, 5])
        for outer in range(3):
            held = {s.subject_id for s in subjects if s.outer_fold == outer}
            for inner in range(3):
                train = {s.subject_id for s in subjects if s.outer_fold != outer and s.inner_folds[outer] != inner}
                val = {s.subject_id for s in subjects if s.outer_fold != outer and s.inner_folds[outer] == inner}
                self.assertTrue(train.isdisjoint(val) and held.isdisjoint(train | val))
                self.assertEqual(len(train | val), 10)

    def test_cafr_reaches_teacher_encoder_and_kd_keeps_teacher_frozen(self):
        torch.manual_seed(11)
        batch = dict(mri=torch.randn(3, 3, 64, 64), dat=torch.randn(2, 3, 64, 64),
                     owners=torch.tensor([0, 1, 1]), subject_id=['a', 'b'],
                     mmse=torch.tensor([0.8, 1.0]))
        teacher = Teacher('none').train()
        output = teacher(batch)
        self.assertEqual(output['mmse'].shape, (2, 4))
        cafr_loss = (output['mmse'] - batch['mmse'][:, None]).square().mean()
        cafr_loss.backward()
        for stage in teacher.mri_encoder.stages:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.parameters()))
        teacher.zero_grad(set_to_none=True)
        freeze_teacher(teacher)
        before = {key: value.clone() for key, value in teacher.state_dict().items()}
        student = Student('none').train()
        with torch.no_grad():
            target = teacher(batch)
        output = student(batch)
        config = dict(output_weight=1, cafr_weight=1, attention_weight=1)
        loss, _ = CradLoss(config).student(output, target, batch, torch.tensor([0.0, 0.5]))
        loss.backward()
        for stage in student.mri_encoder.stages:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_student_prediction_never_reads_clinical_or_dat(self):
        model = Student('none').eval()
        batch = dict(mri=torch.randn(2, 3, 64, 64), owners=torch.tensor([0, 1]), subject_id=['a', 'b'])
        with torch.no_grad():
            first = model(batch)['prediction']
            batch.update(mmse=torch.tensor([999., -999.]), target=torch.tensor([999., -999.]),
                         dat=torch.full((2, 3, 64, 64), float('nan')))
            second = model(batch)['prediction']
        torch.testing.assert_close(first, second)


if __name__ == '__main__':
    unittest.main()
