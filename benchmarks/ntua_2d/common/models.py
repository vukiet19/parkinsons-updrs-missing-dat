"""The same fusion, projection, and regression heads in both baselines."""

import torch
from torch import nn

from .data import subject_mean
from .encoder import Encoder


def regressor():
    return nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))


# The teacher sees both modalities and emits exactly one score per patient.
class Teacher(nn.Module):
    def __init__(self, weights='imagenet'):
        super().__init__()
        self.mri_encoder = Encoder(weights)
        self.dat_encoder = Encoder(weights)
        self.fusion = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.regressor = regressor()

    def forward(self, batch):
        features = self.mri_encoder(batch['mri'])
        mri = subject_mean(features[-1].mean(dim=(-2, -1)), batch['owners'], len(batch['subject_id']))
        dat = self.dat_encoder(batch['dat'])[-1].mean(dim=(-2, -1))
        fused = self.fusion(torch.cat([mri, dat], dim=1))
        return dict(prediction=self.regressor(fused).squeeze(-1), features=features)


# The student never reads DaT or clinical scores in its forward pass.
class Student(nn.Module):
    def __init__(self, weights='imagenet'):
        super().__init__()
        self.mri_encoder = Encoder(weights)
        self.projection = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.regressor = regressor()

    def forward(self, batch):
        features = self.mri_encoder(batch['mri'])
        mri = subject_mean(features[-1].mean(dim=(-2, -1)), batch['owners'], len(batch['subject_id']))
        return dict(prediction=self.regressor(self.projection(mri)).squeeze(-1), features=features)
