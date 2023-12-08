import numpy as np
from torch import nn

from hw_nv.base import BaseModel
from hw_nv.model.deepspeech2_conv2d import DeepSpeech2Conv2d
from hw_nv.model.deepspeech2_rnn_layer import DeepSpeech2RNNLayer


class DeepSpeech2Model(BaseModel):
    def __init__(
            self,
            n_feats,
            n_class,
            mel_spectrogram=True,
            conv2d_input_channels=1,
            conv2d_output_channels=32,
            conv2d_stride=(2, 2),
            conv2d_kernel_size=(41, 11),
            conv2d_relu_clipping_threshold=20,
            rnn_type='LSTM',
            n_rnn_layers=5,
            rnn_hidden_size=512,
            rnn_dropout_prob=0.1,
            verbose=False,
            **batch
    ):
        """
            Implementation of DeepSpeech2 model from https://arxiv.org/pdf/1512.02595.pdf
            This model classifies audio samples by log-spectrogram
            Expected input shape is Batch x Time x Freq
            :params:
            n_feats: number of frequencies - length by Freq axis
            n_class: number of classes
            mel_spectrogram: whether the input is mel-spectrogram or not, bool
            conv2d_input_channels: number of input channels
            conv2d_output_channels: number of output channels
            conv2d_stride: stride of Conv2d
            conv2d_kernel_size: kernel_size of Conv2d on Freq x Time, defaulted to 41 x 11
            conv2d_relu_clipping_threshold: 
                activation function is defined as min(ReLU(x), threshold) = min(max(0, x), threshold)
                in the original paper threshold = 20
                we make use of the pytorch nn.Hardtanh with the similar behavior
            rnn_type: type of RNN to use, must be either 'RNN', 'LSTM' or 'GRU'
            n_rnn_layers: number of RNN layers in final model
            rnn_hidden_size: size of hidden (freq) dimension in RNN layers
            rnn_dropout_prob: dropout probability in RNN layers
            verbose: debug parameter, whether to print shapes after each layer or no
        """
        super().__init__()
        # input -> 
        # 2 InvariantConv2d layers by time and freq axis
        # 7 RNN layers
        # Lookahead Convolutions
        # Linear + Softmax ? 
        self.conv2d = DeepSpeech2Conv2d(
            input_channels=conv2d_input_channels,
            output_channels=conv2d_output_channels,
            stride=conv2d_stride,
            kernel_size=conv2d_kernel_size,
            relu_clipping_threshold=conv2d_relu_clipping_threshold
        )

        self.input_size = n_feats
        self.mel_spectrogram = mel_spectrogram

        rnn_input_size = (self.input_size + 2 * self.conv2d.padding[0] - self.conv2d.kernel_size[0]) / self.conv2d.stride[0]
        rnn_input_size = int(np.floor(rnn_input_size)) + 1
        rnn_input_size = rnn_input_size * self.conv2d.output_channels

        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size

        print(f'rnn_input_size: {rnn_input_size}, rnn_hidden_size={rnn_hidden_size}')
        self.rnns = nn.ModuleList([
            DeepSpeech2RNNLayer(
                input_size=rnn_input_size if i == 0 else 2 * rnn_hidden_size,  # bidirectional => 2 * hidden_dim
                hidden_size=rnn_hidden_size,
                rnn_type=rnn_type,
                dropout_prob=rnn_dropout_prob
            ) for i in range(n_rnn_layers)
        ])

        self.fc = nn.Linear(2 * rnn_hidden_size, n_class)
        self.verbose = verbose


    def forward(self, spectrogram, **batch):
        input = torch.log(spectrogram) if not self.mel_spectrogram else spectrogram
        input_lengths = batch["spectrogram_length"]
        if self.verbose: print(f'\nInitial input shape : Batch x Freq x Time : {input.shape}')

        output, output_lengths = self.conv2d(input, input_lengths)
        if self.verbose: print(f'Shape after Conv2d : Batch x Freq x Time : {output.shape}')

        for i, rnn in enumerate(self.rnns):
            output, output_lengths = rnn(output, output_lengths)
            if self.verbose: print(f'Shape after {i+1}-th RNN layer : Batch x Freq x Time : {output.shape}')

        output = self.fc(output.transpose(1, 2))
        if self.verbose: print(f'Shape after FC Linear layer : Batch x Time x Classes {output.shape}')

        return {'logits': output}


    def transform_input_lengths(self, input_lengths):
        output_lengths = input_lengths

        for layer in self.conv2d.layers:
            if isinstance(layer, nn.Conv2d):    
                output_lengths = (
                    output_lengths + 2 * self.conv2d.padding[1] - self.conv2d.kernel_size[1]
                ) // self.conv2d.stride[1] + 1

        return output_lengths