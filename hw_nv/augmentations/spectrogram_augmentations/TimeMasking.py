import numpy as np
import torchaudio
from torch import Tensor
from torch import nn

from hw_nv.augmentations.base import AugmentationBase

class TimeMaskingSpecAug(AugmentationBase):
    def __init__(self, prob=0.5, max_time_mask=100, *args, **kwargs):
        self.prob = prob
        self._aug = torchaudio.transforms.TimeMasking(max_time_mask)

    def __call__(self, data: Tensor):
        if np.random.uniform() < self.prob:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        return data
