from torch import nn

from hw_as.model.RawNet2.layers import SincFilter, ResBlock

class RawNet2Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.sinc_channels = model_config["sinc_channels"]
        self.sinc_filter_length = model_config["sinc_filter_length"]
        self.sinc_filter_type = model_config.get("sinc_filter_type", "mel")
        self.sinc_filter_min_low_hz = model_config.get("sinc_filter_min_low_hz", 0)
        self.sinc_filter_min_band_hz = model_config.get("sinc_filter_min_band_hz", 0)

        self.channels = model_config["channels"] # must be 2-element tuple

        self.gru_hidden_size = model_config["GRU_hidden_size"]
        self.gru_num_layers = model_config["GRU_num_layers"]

        self.leakyrelu_slope = model_config.get("leakyrelu_slope", 0.3)

        self.sinc_filter = SincFilter(
            sinc_channels=self.sinc_channels,
            sinc_filter_length=self.sinc_filter_length,
            sinc_filter_type=self.sinc_filter_type,
            min_low_hz=self.sinc_filter_min_low_hz,
            min_band_hz=self.sinc_filter_min_band_hz,
            maxpool_kernel_size=3,
            leakyrelu_slope=self.leakyrelu_slope
        )

        self.resblocks = nn.Sequential(
            ResBlock(self.sinc_channels, self.channels[0], kernel_size=3, leakyrelu_slope=self.leakyrelu_slope),
            ResBlock(self.channels[0], self.channels[1], kernel_size=3, leakyrelu_slope=self.leakyrelu_slope),
            ResBlock(self.channels[1], self.channels[1], kernel_size=3, leakyrelu_slope=self.leakyrelu_slope),
            ResBlock(self.channels[1], self.channels[1], kernel_size=3, leakyrelu_slope=self.leakyrelu_slope),
            ResBlock(self.channels[1], self.channels[1], kernel_size=3, leakyrelu_slope=self.leakyrelu_slope),
            ResBlock(self.channels[1], self.channels[1], kernel_size=3, leakyrelu_slope=self.leakyrelu_slope),
        )

        self.gru_layers = nn.Sequential(
            nn.BatchNorm1d(self.channels[1]),
            nn.LeakyReLU(self.leakyrelu_slope),
            TransposeForGRU(),
            nn.GRU(
                input_size=self.channels[1],
                hidden_size=self.gru_hidden_size,
                num_layers=self.gru_num_layers,
                batch_first=True
            )
        )
        self.fc = nn.Linear(self.gru_hidden_size, 2)
    
    def forward(self, wav, **kwargs):
        output = self.sinc_filter(wav)
        output = self.resblocks(output)
        gru_output, gru_hidden_state = self.gru_layers(output)
        predict = self.fc(gru_hidden_state[-1])
        return predict

class TransposeForGRU(nn.Module):
    def forward(self, input):
        return input.transpose(1, 2)
    