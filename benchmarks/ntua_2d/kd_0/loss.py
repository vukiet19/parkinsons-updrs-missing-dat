"""True-score supervision plus ordinary scalar output distillation."""

from torch import nn


class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_weight = config['output_weight']

    def teacher(self, output, batch, target):
        supervised = (output['prediction'] - target).square().mean()
        return supervised, dict(supervised=supervised)

    def student(self, output, teacher, batch, target):
        supervised = (output['prediction'] - target).square().mean()
        # Detach the target so this loss can only teach the student.
        distilled = (output['prediction'] - teacher['prediction'].detach()).square().mean()
        total = supervised + self.output_weight * distilled
        return total, dict(supervised=supervised, output_kd=distilled)
