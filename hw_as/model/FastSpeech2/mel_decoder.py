import torch

from torch import nn

from .encoder_decoder import FFTBlock


class MelDecoder(nn.Module):
    def __init__(self, model_config):
        raise NotImplementedError()
        model_config["mel_size"]

        self.fft_block = FFTBlock(
            d_model=,
            d_inner=,
            n_head=,
            d_k=,
            d_v=,
            dropout = model_config["dropout"]
        )