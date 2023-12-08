import torch
from torch import Tensor
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from hw_nv.base.base_metric import BaseMetric

class PESQMetricWrapper(BaseMetric):
    def __init__(self, fs, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs, mode)        

    def __call__(self, predicts, target, **kwargs):
        """
        predicts: dict of model predicts by L1, L2 and L3 filters 
        """
        if self.pesq.device != target.device:
            self.pesq = self.pesq.to(target.device)

        try:
            pesqs = {}
            for filter, predict in predicts.items():
                pesqs[filter] = self.pesq(predict, target)
        except:
            return torch.tensor(-1.)
        
        pesq = 0.8 * pesqs["L1"] + 0.1 * pesqs["L2"] + 0.1 * pesqs["L3"]
        return pesq
