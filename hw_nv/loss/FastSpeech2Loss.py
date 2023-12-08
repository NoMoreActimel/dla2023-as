import torch

from torch import nn


class FastSpeech2Loss(nn.Module):
    def __init__(self, pitch_feature_level=None, energy_feature_level=None):
        super().__init__()
        self.pitch_feature_level = pitch_feature_level
        self.energy_feature_level = energy_feature_level

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self,
                mel_predict,
                log_duration_predict,
                pitch_predict,
                energy_predict,
                mel_target,
                duration_target,
                pitch_target,
                energy_target,
                **kwargs):
        
        mel_loss = self.mae_loss(mel_predict, mel_target)
        
        log_duration_target = torch.log(1 + duration_target)
        duration_loss = self.mse_loss(log_duration_predict, log_duration_target)

        pitch_loss = self.mse_loss(pitch_predict, pitch_target)
        energy_loss = self.mse_loss(energy_predict, energy_target)

        loss = mel_loss + duration_loss + pitch_loss + energy_loss
        return loss, mel_loss, duration_loss, pitch_loss, energy_loss
