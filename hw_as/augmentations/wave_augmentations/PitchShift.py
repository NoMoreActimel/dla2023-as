import numpy as np
import torch_audiomentations
from torch import Tensor

from hw_as.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, prob=0.5, sample_rate=16000, *args, **kwargs):
        self.prob = prob
        self._aug = torch_audiomentations.PitchShift(sample_rate=sample_rate)

    def __call__(self, data: Tensor):
        if np.random.uniform() < self.prob:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
