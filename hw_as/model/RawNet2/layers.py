import torch

from torch import nn

from hw_as.model.RawNet2.sincconv import SincConv_fast


class SincFilter(nn.Module):
    def __init__(self, sinc_channels, sinc_filter_length, min_low_hz,
                 min_band_hz, maxpool_kernel_size=3, leakyrelu_slope=0.3):
        super().__init__()

        self.sincconv = SincConv_fast(
            out_channels=sinc_channels,
            kernel_size=sinc_filter_length,
            min_low_hz=min_low_hz,
            min_band_hz=min_band_hz
        )
        self.layers = nn.Sequential(
            nn.MaxPool1d(maxpool_kernel_size),
            nn.BatchNorm1d(sinc_channels),
            nn.LeakyReLU(leakyrelu_slope)
        )

    def forward(self, input):
        output = input.unsqueeze(1)
        output = self.sincconv(output)
        output = torch.abs(output)
        return self.layers(output)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 maxpool_kernel_size=3, leakyrelu_slope=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(leakyrelu_slope),
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(leakyrelu_slope),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2
            ),
            nn.MaxPool1d(maxpool_kernel_size),
            FMSBlock(out_channels)
        )

    def forward(self, input):
        return self.layers(input)

class FMSBlock(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.fc_attention = nn.Linear(n_features, n_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        batch, filter = input.shape[:2]

        output = self.avgpool(input)
        output = output.view(batch, filter)

        output = self.fc_attention(output)
        output = self.sigmoid(output)
        output = output.view(batch, filter, -1)

        output = input * output + output
        return output
