import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from hw_nv.base.base_metric import BaseMetric

class SiSDRMetricWrapper(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=True)        

    def __call__(self, predicts, target, **kwargs):
        """
        predicts: dict of model predicts by L1, L2 and L3 filters 
        """        
        if self.si_sdr.device != target.device:
            self.si_sdr = self.si_sdr.to(target.device)

        si_sdrs = {}
        for filter, predict in predicts.items():
            si_sdrs[filter] = self.si_sdr(predict, target)
        
        si_sdr = 0.8 * si_sdrs["L1"] + 0.1 * si_sdrs["L2"] + 0.1 * si_sdrs["L3"]
        return si_sdr
