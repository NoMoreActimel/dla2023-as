from typing import List

import torch
from torch import Tensor

import numpy as np

from hw_nv.base.base_metric import BaseMetric
from hw_nv.base.base_text_encoder import BaseTextEncoder
from hw_nv.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamsearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, compute_on_train=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.compute_on_train = compute_on_train
        assert hasattr(self.text_encoder, "ctc_beam_search"), "Incompatible text encoder, ctc_beam_search is needed"

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []

        log_probs = log_probs.detach().cpu()
        log_probs_length = log_probs_length.detach().cpu().numpy()

        if self.text_encoder.use_lm:
            predicted_texts = self.text_encoder.ctc_lm_beam_search(log_probs, log_probs_length)
        else:
            predicted_texts = np.array([
                self.text_encoder.ctc_beam_search(log_prob, length)[0].text
                for log_prob, length in zip(log_probs, log_probs_length)
            ])
            
        for pred_text, target_text in zip(predicted_texts, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred_text))

        return sum(wers) / len(wers)
