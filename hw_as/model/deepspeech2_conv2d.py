from torch import nn


class DeepSpeech2Conv2d(nn.Module):
    def __init__(
            self,
            input_channels=1,
            output_channels=32,
            stride=(2, 2),
            kernel_size=(41, 11),
            relu_clipping_threshold=20
    ):
        """
            Implementation of first-layers convolutions of DeepSpeech2 model
            In particular: 1-layer Conv2d on Frequency x Time axes accordingly
            Default parameters are taken as they are declared in the paper
            Shape of the input tensor: Batch x Time x Freq
            :params:
            input_channels: number of input channels
            output_channels: number of output channels
            stride: stride of Conv2d
            kernel_size: kernel_size of Conv2d on Freq x Time, defaulted to 41 x 11
            relu_clipping_threshold: 
                activation function is defined as min(ReLU(x), threshold) = min(max(0, x), threshold)
                in the original paper threshold = 20
                we make use of the pytorch nn.Hardtanh with the similar behavior
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.relu_clipping_threshold = relu_clipping_threshold
        self.padding = int((kernel_size[0] - 1) / stride[0]), int((kernel_size[1] - 1) / stride[1])
        self.stride = stride
        self.kernel_size = kernel_size

        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.Hardtanh(0, self.relu_clipping_threshold, inplace=True),
        )
        
    def forward(self, input, input_lengths):
        # input: Batch x Freq x Time -> Batch x InputChannels x Freq x Time
        output = input.unsqueeze(1)
        output_lengths = input_lengths

        for layer in self.layers:
            output = layer(output)

            # we need to apply masking by Time axis 
            if isinstance(layer, nn.Conv2d):
                output_lengths = (output_lengths.float() + (2 * self.padding[1] - self.kernel_size[1] - 2)) / self.stride[1]
                output_lengths = output_lengths.int() + 1
                for i, length in enumerate(output_lengths):
                    output[i, length:, :] = 0.
        
        # output: Batch x OutputChannels x TransformedFreq x Time -> Batch x Freq x Time
        # Freq = OutputChannels * TransformedFreq
        B, C, F, T = output.shape
        output = output.view(B, C * F, T)
        return output, output_lengths