import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from typing import Tuple


class SpExPlusLoss(_Loss):
    def __init__(self, gamma=1):
        """
        Implementation of SpExPlus loss from
        https://www.isca-speech.org/archive/pdfs/interspeech_2020/ge20_interspeech.pdf
        Includes both SI-SDR Loss on predicted denoised audio of the target speaker
        And Cross-Entropy Loss on predicted speaker logits from speaker encoder part of the model
        params:
            gamma: coefficient of Cross-Entropy Loss for speaker encoder
        """
        super().__init__()
        self.EPS = 1e-6
        self.CELoss = nn.CrossEntropyLoss()
        self.gamma = gamma

    def si_sdr(self, predict, target):
        predict = predict - torch.mean(predict, dim=-1)
        target = target - torch.mean(target, dim=-1)

        alpha = torch.sum(target * predict, dim=-1) / torch.linalg.norm(target, dim=-1) ** 2

        numerator = torch.linalg.norm(alpha * target, dim=-1)
        denominator = torch.linalg.norm(alpha * target - predict, dim=-1) + self.EPS

        return 20 * torch.log10(numerator / denominator + self.EPS)


    def forward(self, predicts, target, audio_length, speaker_logits=None, speaker_id=None, 
                eval_mode=False, **batch) -> Tuple[Tensor, Tensor]:
        """
        predicts: dict with predicts by filters "L1", "L2", "L3" 
        target: target audio
        speaker_logits: speaker encoder logits for Cross-Entropy Loss
        speaker_id: target speaker id for Cross-Entropy Loss
        audio_length: mixed audio length from dataset, used for masking

        in train mode returns -SiSDR + self.gamma * CELoss, -SiSDR, CELoss
        in eval mode returns -SiSDR, -SiSDR, 0.

        such strange behaviour is implemented for the case of different train-val datasets
        so there might be speaker_ids from validation part that are absent 
        in the speaker_logits from model
        """
        n = target.shape[0]

        sisdr_losses = {}
        mask = torch.arange(target.shape[1], device=target.device)[None, :] < audio_length[:, None]
        target = target[mask]
        
        for filter, filter_predicts in predicts.items():
            filter_predicts = filter_predicts[mask]
            sisdr_losses[filter] = self.si_sdr(filter_predicts, target) / n
            
        sisdr_loss = 0.8 * sisdr_losses["L1"] + 0.1 * sisdr_losses["L2"] + 0.1 * sisdr_losses["L3"]

        if eval_mode:
            ce_loss = 0.
        else:
            ce_loss = self.CELoss(speaker_logits, speaker_id.long())

        return -sisdr_loss + self.gamma * ce_loss, -sisdr_loss, ce_loss
