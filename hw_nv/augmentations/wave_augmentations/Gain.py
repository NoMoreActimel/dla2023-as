import numpy as np
import torch_audiomentations
from torch import Tensor

from hw_nv.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, prob=0.5, *args, **kwargs):
        self.prob = prob
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        if np.random.uniform() < self.prob:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
