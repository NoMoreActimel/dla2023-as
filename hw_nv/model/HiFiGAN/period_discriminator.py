import torch
import torch.nn.functional as F

from torch import nn

from hw_as.model.HiFiGAN.utils import get_padding


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, model_config):
        """
        MultiPeriodDiscriminator of HiFiGAN model. Implementation is written with initialization
        from model_config file for the sake of convenience, probably I will change it later on.
        not likely though ( ⚆ _ ⚆ ) 

        params to specify in model_config["period_discriminator"]:
            periods: list of periods for each PeriodDiscriminator, defaults to [2, 3, 5, 7, 11]

            in_channels: input channels for discriminators, default is 1
            out_channels: output channels for discriminators, default is 1
            hidden_channels: list of hidden channels for intermediate Conv1d layers
                in discriminators, defaults to [32, 128, 512, 1024, 1024]
            kernel_size: kernel size for all Conv2d layer in discriminators,
                except the last one, defaults to 5
            strides: list of strides for each Conv2d layer in discriminators,
                except the last one, defaults to [3, 3, 3, 3, 1]
            out_kernel_size: kernel size of the output Conv2d layer, 
                which maps back to output_channels, defaults to 3
            apply_weight_norm: whether to apply weight normalization on all convolutional layers
        """
        super().__init__()
        self.model_config = model_config
        self.discriminator_config = model_config.get("period_discriminator", {})

        self.periods = [2, 3, 5, 7, 11]
        if self.discriminator_config:
            self.periods = self.discriminator_config.get("periods", self.periods)
    
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(
                period=period,
                **model_config.get("period_discriminator", {})
            )
            for period in self.periods
        ])

    def forward(self, input):
        """ Returns list of (output, layer_outputs) for each discriminator """
        outputs = []
        layer_outputs = []

        for discriminator in self.discriminators:
            output, layer_output = discriminator(input)
            outputs.append(output)
            layer_outputs += layer_output
        
        return outputs, layer_outputs


class PeriodDiscriminator(nn.Module):
    def __init__(
            self,
            period,
            in_channels=1,
            hidden_channels=[32, 128, 512, 1024, 1024],
            out_channels=1,
            kernel_size=5,
            strides=[3, 3, 3, 3, 1],
            out_kernel_size=3,
            apply_weight_norm=True,
            **kwargs
            ):
        super().__init__()

        self.period = period

        assert len(hidden_channels) == len(strides), \
            f"Lengths of hidden_channels: {len(hidden_channels)} and " \
            f"strides: {len(strides)} must match!"
        
        hidden_channels = [in_channels] + hidden_channels
        
        self.layers = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=hidden_channels[ind + 1],
                        kernel_size=[kernel_size, 1],
                        stride=[stride, 1],
                        padding=[get_padding(kernel_size, 1), 0]
                    ),
                    nn.LeakyReLU(0.1)
                )
                for ind, (channels, stride) in enumerate(zip(hidden_channels, strides))
            ],
            nn.Conv2d(
                in_channels=hidden_channels[-1],
                out_channels=out_channels,
                kernel_size=[out_kernel_size, 1],
                padding=[get_padding(out_kernel_size, 1), 0]
            )
        )

        if apply_weight_norm:
            self.weight_norm()
    
    def weight_norm(self):
        def _weight_norm(module):
            if isinstance(module, nn.Conv2d):
                nn.utils.weight_norm(module)
        self.apply(_weight_norm)

    def transform_to_2d(self, input):
        B, C, T = input.shape

        if T % self.period:
            input = F.pad(input, (0, self.period - (T % self.period)), 'reflect')
            T = input.shape[-1]
        
        return input.view(B, C, T // self.period, self.period)

    def forward(self, input):
        input = self.transform_to_2d(input)
        
        output = input
        layer_outputs = []

        for layer in self.layers:
            output = layer(output)
            layer_outputs.append(output)
    
        output = output.view(output.shape[0], 1, -1)
        return output, layer_outputs
