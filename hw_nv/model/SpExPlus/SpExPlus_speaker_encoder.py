import torch
from torch import nn

from hw_nv.base import BaseModel

class SpExPlusSpeakerEncoder(BaseModel):
    def __init__(
            self,
            resblocks_in_dim,
            resblocks_out_dim,
            speaker_embedding_dim,
            num_speakers,
            n_res_blocks=3,
            res_maxpool_kernel_size=3,
            **batch
    ):
        """ 
        Speaker Encoder - this part encodes speaker embedding
        from clean target audio, in order to help the main part of the model.
        Learning is performed both on the Cross-Entropy Loss for speaker classification
        and SI-SDR Loss backpropagated from the main backbone.

        params:
            resblocks_in_dim: output dimension of twin-speech-encoder
            resblocks_out_dim: hidden dimension of resnet blocks in speaker encoder
            speaker_embedding_dim: dimension of speaker embedding from speaker encoder
            num_speakers: number of speakers, to train speaker encoder on CE Loss
        """
        super().__init__()

        self.res_maxpool_kernel_size = res_maxpool_kernel_size
        self.n_res_blocks = n_res_blocks

        self.res_blocks = nn.Sequential(
            *[
                ResBlock(resblocks_in_dim, resblocks_in_dim, res_maxpool_kernel_size)
                for _ in range(n_res_blocks // 2)
            ],
            ResBlock(resblocks_in_dim, resblocks_out_dim, res_maxpool_kernel_size),
            *[
                ResBlock(resblocks_out_dim, resblocks_out_dim, res_maxpool_kernel_size)
                for _ in range((n_res_blocks - 1) // 2)
            ],
        )

        self.conv1d = nn.Conv1d(
            in_channels=resblocks_out_dim,
            out_channels=speaker_embedding_dim,
            kernel_size=1
        )
        self.fc = nn.Linear(speaker_embedding_dim, num_speakers)


    def forward(self, input, input_time_length):
        output = self.res_blocks(input)
        output = self.conv1d(output)
        
        # count new TimeDim, changed in MaxPool1d in each ResBlock
        time_reduction_coeff = (self.res_maxpool_kernel_size) ** self.n_res_blocks
        output_time_length = input_time_length // time_reduction_coeff

        output_norm_factor = output_time_length.float().view(-1, 1)
        speaker_embed = output.sum(-1) / output_norm_factor
        
        speaker_logits = self.fc(speaker_embed)

        return speaker_embed, speaker_logits


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, maxpool_kernel_size=3):
        """
        ResNet block for speaker encoder with PReLU activations
            https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
        """
        super(ResBlock, self).__init__()

        self.conv1d_1 = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            bias=False
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=1,
            bias=False
        )

        self.batch_norm1 = nn.BatchNorm1d(out_dim)
        self.batch_norm2 = nn.BatchNorm1d(out_dim)

        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()

        self.maxpool1d = nn.MaxPool1d(kernel_size=maxpool_kernel_size)

        self.downsample = in_dim != out_dim
        if self.downsample:
            self.conv1d_downsample = nn.Conv1d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=1,
                bias=False
            )
    
    def forward(self, input):
        output = self.conv1d_1(input)
        output = self.batch_norm1(output)
        output = self.activation1(output)

        output = self.conv1d_2(output)
        output = self.batch_norm2(output)

        if self.downsample:
            input = self.conv1d_downsample(input)

        output += input
        output = self.activation2(output)

        return self.maxpool1d(output)