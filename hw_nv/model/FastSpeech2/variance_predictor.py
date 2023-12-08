import torch
import torch.nn.functional as F

from torch import nn


class VariancePredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config):
        super().__init__()

        self.input_size = model_config["encoder_decoder"]["encoder_dim"]
        predictor_config = model_config["variance_predictor"]
        self.filter_size = predictor_config["filter_size"]
        self.kernel = predictor_config["kernel_size"]
        self.conv_output_size = predictor_config["filter_size"]
        self.dropout = predictor_config["dropout"]

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.relu = nn.ReLU()
        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()

        if not self.training:
            out = out.unsqueeze(0)
        
        return out


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)
