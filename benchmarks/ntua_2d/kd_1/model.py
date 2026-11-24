"""Add teacher CAFR heads while preserving the shared prediction architecture."""

from common.models import Student, Teacher as BasicTeacher
from .components.cafr import CAFR


class Teacher(BasicTeacher):
    def __init__(self, weights='imagenet'):
        super().__init__(weights)
        self.cafr = CAFR(self.mri_encoder.channels)

    def forward(self, batch):
        output = super().forward(batch)
        output['mmse'] = self.cafr(output['features'], batch['owners'], len(batch['subject_id']))
        return output


__all__ = ['Teacher', 'Student']
