import numpy as np
import torch

from torch import nn

from hw_nv.model.HiFiGAN.generator import Generator
from hw_nv.model.HiFiGAN.period_discriminator import MultiPeriodDiscriminator
from hw_nv.model.HiFiGAN.scale_discriminator import MultiScaleDiscriminator


class HiFiGANModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config["model"]["args"]

        self.generator = Generator(self.model_config)
        self.MPD = MultiPeriodDiscriminator(self.model_config)
        self.MSD = MultiScaleDiscriminator(self.model_config)
    
    def forward(self, **batch):
        return self.generator(batch["mel"])

    def discriminate(self, wav, wav_gen):
        result = {"true": {}, "gen": {}}

        for input_type, input in zip(["true", "gen"], [wav, wav_gen]):
            for D_name, D in zip(["MPD", "MSD"], [self.MPD, self.MSD]):
                outputs, layer_outputs = D(input)
                result[input_type][f"{D_name}_outputs"] = outputs
                result[input_type][f"{D_name}_layer_outputs"] = layer_outputs

        return result

    def get_number_of_parameters(self):
        return {
            "generator": self.get_number_of_module_parameters(self.generator),
            "MPD": self.get_number_of_module_parameters(self.MPD),
            "MSD": self.get_number_of_module_parameters(self.MSD)
        }

    @staticmethod
    def get_number_of_module_parameters(module):
        module_parameters = filter(lambda p: p.requires_grad, module.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        return params
