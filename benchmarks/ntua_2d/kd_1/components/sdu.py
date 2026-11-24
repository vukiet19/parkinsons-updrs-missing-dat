"""Detached confidence weights; UPDRS confidence is adapted for regression."""

import torch


@torch.no_grad()
def layer_confidence(teacher_mmse, true_mmse):
    # CRAD Eq. 14: MMSE is normalized to [0, 1] before computing squared error.
    error = (teacher_mmse - true_mmse[:, None]).square()
    return 1.0 - error.clamp(0.0, 1.0)


@torch.no_grad()
def output_confidence(teacher_prediction, student_prediction, target):
    # Use absolute errors in train-standardized UPDRS units, per patient.
    teacher_error = (teacher_prediction - target).abs()
    student_error = (student_prediction - target).abs()
    confidence = 1.0 - teacher_error.clamp(0.0, 1.0)
    relative_penalty = (student_error / (teacher_error + student_error + 1e-8)).square()
    # Soften supervision when the teacher is less accurate than the student.
    penalty = torch.where(student_error < teacher_error, relative_penalty, 1.0)
    return confidence * penalty
