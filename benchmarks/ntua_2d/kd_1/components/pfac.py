"""2D adaptation of CRAD's default ChannelAttention_wn (PFAC_wn)."""

import torch
from torch import nn
from torch.nn import functional as F


class PFAC(nn.Module):
    """Create one channel weight, then apply it to the original feature map."""

    def forward(self, features):
        normalized = F.normalize(features, p=2, dim=1)
        average = normalized.mean(dim=(-2, -1), keepdim=True)
        maximum = normalized.amax(dim=(-2, -1), keepdim=True)
        descriptor = F.relu(average + maximum)
        # diag(q @ q.T) equals q squared, so no C-by-C matrix is needed.
        weights = torch.sigmoid(descriptor.square())
        return features * weights, weights
