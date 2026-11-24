"""CAFR: image-only MMSE supervision at every MRI encoder stage."""

import torch
from torch import nn

from common.data import subject_mean


class CAFR(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(count, 32), nn.ReLU(), nn.Linear(32, 1))
            for count in channels
        ])

    def forward(self, features, owners, subject_count):
        predictions = []
        for feature, head in zip(features, self.heads):
            # Pool spatially, predict per image, then average within the patient.
            image_scores = head(feature.mean(dim=(-2, -1))).squeeze(-1)
            predictions.append(subject_mean(image_scores, owners, subject_count))
        return torch.stack(predictions, dim=1)  # [patients, four stages], MMSE / 30
