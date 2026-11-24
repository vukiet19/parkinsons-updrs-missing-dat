"""PF-CMAD: align attended MRI features at all four corresponding stages."""

import torch
from torch import nn

from common.data import subject_mean
from .pfac import PFAC


class PFCMAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfac = PFAC()

    def forward(self, student_features, teacher_features, owners, confidence):
        if len(student_features) != 4 or len(teacher_features) != 4:
            raise ValueError('PF-CMAD requires all four corresponding encoder stages.')
        losses = []
        for stage, (student, teacher) in enumerate(zip(student_features, teacher_features)):
            if student.shape != teacher.shape:
                raise ValueError(f'Feature shapes differ at stage {stage + 1}.')
            attended_student, _ = self.pfac(student)
            attended_teacher, _ = self.pfac(teacher.detach())
            per_image = (attended_student - attended_teacher).square().mean(dim=(1, 2, 3))
            per_subject = subject_mean(per_image, owners, confidence.shape[0])
            losses.append(per_subject * confidence[:, stage])
        return torch.stack(losses, dim=1).mean()
