import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch import nn

from hw_nv.augmentations.base import AugmentationBase

class TimeStretchSpecAug(AugmentationBase):
    def __init__(
            self,
            prob=0.5,
            stretch_min=0.8,
            stretch_max=1.2,
            n_freq=128,
            *args,
            **kwargs
        ):
        self.prob = prob
        self.stretch_min = stretch_min
        self.stretch_max = stretch_max
        self._aug = torchaudio.transforms.TimeStretch(n_freq=n_freq)

    def __call__(self, data: Tensor):
        if np.random.uniform() < self.prob:
            alpha = np.random.uniform()            
            stretch = self.stretch_min * alpha + self.stretch_max * (1 - alpha)
            x = data.unsqueeze(1)
            x = self._aug(x, stretch).squeeze(1)
            return x.abs()  # drop the phase
        return data
