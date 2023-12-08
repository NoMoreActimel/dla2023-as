import torch

from torch import nn

from hw_nv.model.HiFiGAN.utils import get_padding
from hw_nv.model.HiFiGAN.utils import get_transposed_padding
from hw_nv.model.HiFiGAN.utils import get_transposed_output_padding


class Generator(nn.Module):
    def __init__(self, model_config):
        """
        Generator of HiFiGAN model. Implementation is written with initialization
        from model_config file for the sake of convenience, probably I will change it later on.
        not likely though ( ⚆ _ ⚆ ) 
        
        params to mention in model_config["generator"]:
            in_channels: in_channels of the input convolution, defaults to 80
            hidden_channels: number of hidden channels at the start of MRFs, defaults to 512
            out_channels: out_channels in output convolution, defaults to 1
            binding_conv_kernel_size: kernel size of input and output convolutions, defaults to 7

            apply_weight_norm: whether to apply weight normalization 
                at each Conv1d and ConvTranspose1d layer, default is True
            
            upsample_strides: list of strides for upsampling ConvTranspose1d blocks,
                defaults to [8, 8, 2, 2]
            kernel sizes for ConvTranpose1d blocks would be taken as 2 * upsample_strides

            in model_config["generator"]["MRF"]:
            kenrels: list of kernel sizes for MRF blocks,
                1 kernel size for each block,
                defaults to [3, 7, 11]
            dilations: list of dilations lists for MRF blocks, 
                1 list of dilations for each block,
                defauls to [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        """
        super().__init__()
        self.model_config = model_config
        self.generator_config = model_config["generator"]

        self.in_channels = self.generator_config.get("in_channels", 80)
        # must be the same as n_mels in mel_spec_config

        self.out_channels = self.generator_config.get("out_channels", 1)
        self.hidden_channels = self.generator_config.get("hidden_channels", 512)
        self.binding_conv_kernel_size = self.generator_config.get(
            "binding_conv_kernel_size", 7
        )

        self.mrf_kernels = self.generator_config["MRF"].get("kernels", [3, 7, 11])
        self.mrf_dilations = self.generator_config["MRF"].get(
            "dilations", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        )
        assert len(self.mrf_kernels) == len(self.mrf_dilations), \
            f"MRF kernels and dilations lengths' mismatch"

        self.upsample_strides = self.generator_config.get("upsample_strides", [8, 8, 2, 2])
        
        self.input_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.binding_conv_kernel_size,
            padding=get_padding(self.binding_conv_kernel_size, 1)
        )

        self.layers = nn.Sequential(*[
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=self.hidden_channels // (2 ** ind),
                        out_channels=self.hidden_channels // (2 ** (ind + 1)),
                        kernel_size=stride * 2,
                        stride=stride,
                        padding=get_transposed_padding(stride * 2, stride),
                        output_padding=get_transposed_output_padding(stride * 2, stride)
                    ),
                    MRF(
                        channels=self.hidden_channels // (2 ** (ind + 1)),
                        kernels=self.mrf_kernels,
                        dilations=self.mrf_dilations
                    )
                )
                for ind, stride in enumerate(self.upsample_strides)
            ],
            nn.Conv1d(
                in_channels=self.hidden_channels // (2 ** len(self.upsample_strides)),
                out_channels=self.out_channels,
                kernel_size=self.binding_conv_kernel_size,
                padding=get_padding(self.binding_conv_kernel_size, 1)
            )
        )

        self.apply_weight_norm = self.generator_config.get("apply_weight_norm", True)
        if self.apply_weight_norm:
            self.weight_norm()
        
        self.reset_parameters()
    
    def weight_norm(self):
        def _weight_norm(module):
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.utils.weight_norm(module)
                print(f"Applied weight_norm to {module}")

        self.apply(_weight_norm)
    
    def remove_weight_norm(self):
        def _remove_weight_norm(module):
            try: nn.utils.remove_weight_norm(module)
            except ValueError: return
        
        self.apply(_remove_weight_norm)
        
    def reset_parameters(self):
        def _reset_parameters(module):
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                module.weight.data.normal_(0.0, 0.01)
                print(f"Reset parameters in {module} to normal(0.0, 0.01)")

        self.apply(_reset_parameters)

    def forward(self, input):
        output = self.input_conv(input)
        output = self.layers(output)
        return output


class MRF(nn.Module):
    def __init__(self, channels, kernels, dilations):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            MRFResidualBlock(channels, kernel_size, dilation)
            for kernel_size, dilation in zip(kernels, dilations)
        ])

    def forward(self, input):
        output = 0.
        for block in self.res_blocks:
            output = block(input) + output
        return output
        

class MRFResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=get_padding(kernel_size, dilation)
                ),
                nn.LeakyReLU(0.1),
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=1,
                    padding=get_padding(kernel_size, 1)
                )
            )
            for dilation in dilations
        ])
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            layer_output = layer(output)
            output = layer_output + output
        return output
