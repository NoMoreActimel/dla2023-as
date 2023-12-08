import torch

from torch import nn

from hw_as.model.HiFiGAN.utils import get_padding


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, model_config):
        """
        MultiScaleDiscriminator of HiFiGAN model. Implementation is written with initialization
        from model_config file for the sake of convenience, probably I will change it later on.
        not likely though ( ⚆ _ ⚆ ) 

        params to specify in model_config["scale_discriminator"]:
            in_channels: input channels for discriminators, default is 1
            out_channels: output channels for discriminators, default is 1
            hidden_channels: list of hidden channels for intermediate Conv1d layers in discriminators,
                defaults to [128, 128, 256, 512, 1024, 1024, 1024]
            kernel_sizes: list of kernel sizes for intermediate Conv1d layers in discriminators,
                defaults to [15, 41, 41, 41, 41, 41, 5]
            strides: list of strides for intermediate Conv1d layers in discriminators,
                defaults to [1, 2, 2, 4, 4, 1, 1]
            groups: list of groups for intermediate Conv1d layers in discriminators,
                defaults to [1, 4, 16, 16, 16, 16, 1]
            out_kernel_size: kernel size of the output Conv1d layer, 
                which maps back to output_channels, defaults to 3
        
        You may also provide pooling config in model_config["scale_discriminator"]["pooling"]:
            kernel_size: window size for AvgPool1d, defaults to 4
            stride: stride for AvgPool1d, defaults to 2
            padding: padding for AvgPool1d, defaults to 2
        
        This implementation applies spectral norm in first discriminator,
        switching to weight norm in others.
        """
        super().__init__()
        self.model_config = model_config
        self.discriminator_config = model_config.get("scale_discriminator", {})
        self.pooling_config = self.discriminator_config.get(
            "pooling", {"kernel_size": 4, "stride": 2, "padding": 2}
        )

        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(
                apply_weight_norm=False,
                apply_spectral_norm=True,
                **self.discriminator_config
            ),
            ScaleDiscriminator(
                apply_weight_norm=True,
                apply_spectral_norm=False,
                **self.discriminator_config
            ),
            ScaleDiscriminator(
                apply_weight_norm=True,
                apply_spectral_norm=False,
                **self.discriminator_config
            )
        ])
        self.pooling = nn.AvgPool1d(**self.pooling_config)
    
    def forward(self, input):
        outputs = []
        layer_outputs = []
        output = input

        for discriminator in self.discriminators:
            output, layer_output = discriminator(output)
            outputs.append(output)
            layer_outputs += layer_output
            output = self.pooling(output)
        
        return outputs, layer_outputs


class ScaleDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels=1,
            hidden_channels=[128, 128, 256, 512, 1024, 1024, 1024],
            kernel_sizes=[15, 41, 41, 41, 41, 41, 5],
            strides=[1, 2, 2, 4, 4, 1, 1],
            groups=[1, 4, 16, 16, 16, 16, 1],
            out_channels=1,
            out_kernel_size=3,
            apply_weight_norm=True,
            apply_spectral_norm=False,
            **kwargs
            ):
        super().__init__()

        layer_items = [hidden_channels, kernel_sizes, strides, groups]
        assert len(set(len(item) for item in layer_items)) == 1, \
            f"Lengths of hidden_channels: {len(hidden_channels)}, " \
            f"kernel_sizes: {len(kernel_sizes)}, strides: {len(strides)}" \
            f" and groups: {len(groups)} must match!"
        
        hidden_channels = [in_channels] + hidden_channels
        layer_items[0] = hidden_channels
        
        self.layers = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=hidden_channels[ind + 1],
                        kernel_size=kernel,
                        stride=stride,
                        padding=get_padding(kernel, 1),
                        groups=group
                    ),
                    nn.LeakyReLU(0.1)
                )
                for ind, (channels, kernel, stride, group) in enumerate(zip(*layer_items))
            ],
            nn.Conv1d(
                in_channels=hidden_channels[-1],
                out_channels=out_channels,
                kernel_size=out_kernel_size,
                padding=get_padding(out_kernel_size, 1)
            )
        )

        if apply_weight_norm:
            self.weight_norm()
        if apply_spectral_norm:
            self.spectral_norm()
    
    def weight_norm(self):
        def _weight_norm(module):
            if isinstance(module, nn.Conv1d):
                nn.utils.weight_norm(module)
        self.apply(_weight_norm)
    
    def spectral_norm(self):
        def _spectral_norm(module):
            if isinstance(module, nn.Conv1d):
                nn.utils.spectral_norm(module)
        self.apply(_spectral_norm)

    def forward(self, input):
        output = input
        layer_outputs = []

        for layer in self.layers:
            output = layer(output)
            layer_outputs.append(output)

        return output, layer_outputs
