"""A trainable 2D ResNet18 exposing all four residual stages."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class Encoder(nn.Module):
    channels = (64, 128, 256, 512)

    def __init__(self, weights: str = 'imagenet'):
        super().__init__()
        pretrained = ResNet18_Weights.IMAGENET1K_V1 if weights == 'imagenet' else None
        base = resnet18(weights=pretrained)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stages = nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4])
        # No adapters and no frozen stages: the entire encoder can learn.

    def forward(self, image):
        value = self.stem(image)
        features = []
        for stage in self.stages:
            value = stage(value)
            features.append(value)
        return features
