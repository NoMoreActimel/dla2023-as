import numpy as np
import torchaudio
from torch import Tensor
from torch import nn

from hw_nv.augmentations.base import AugmentationBase

class FreqMaskingSpecAug(AugmentationBase):
    def __init__(self, prob=0.5, max_freq_mask=20, *args, **kwargs):
        self.prob = prob
        self._aug = torchaudio.transforms.FrequencyMasking(max_freq_mask)

    def __call__(self, data: Tensor):
        if np.random.uniform() < self.prob:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        return data