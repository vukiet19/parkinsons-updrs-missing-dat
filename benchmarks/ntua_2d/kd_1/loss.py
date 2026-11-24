"""Combine supervision, CAFR, confidence-weighted output KD, and PF-CMAD."""

from torch import nn

from .components.attention import PFCMAD
from .components.sdu import layer_confidence, output_confidence


class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_weight = config['output_weight']
        self.cafr_weight = config['cafr_weight']
        self.attention_weight = config['attention_weight']
        self.attention = PFCMAD()

    def teacher(self, output, batch, target):
        supervised = (output['prediction'] - target).square().mean()
        cafr = (output['mmse'] - batch['mmse'][:, None]).square().mean()
        total = supervised + self.cafr_weight * cafr
        return total, dict(supervised=supervised, cafr=cafr)

    def student(self, output, teacher, batch, target):
        supervised = (output['prediction'] - target).square().mean()
        confidence = output_confidence(teacher['prediction'], output['prediction'], target)
        distilled = (confidence * (output['prediction'] - teacher['prediction'].detach()).square()).mean()
        stages = layer_confidence(teacher['mmse'], batch['mmse'])
        attention = self.attention(output['features'], teacher['features'], batch['owners'], stages)
        total = supervised + self.output_weight * distilled + self.attention_weight * attention
        terms = dict(supervised=supervised, output_kd=distilled, attention_kd=attention,
                     output_confidence=confidence.mean(), layer_confidence=stages.mean())
        for k in range(4):
            terms[f'confidence_s{k + 1}'] = stages[:, k].mean()
        return total, terms
