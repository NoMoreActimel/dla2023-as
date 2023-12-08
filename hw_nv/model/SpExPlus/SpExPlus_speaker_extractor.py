import torch
from torch import nn

from hw_nv.base import BaseModel


class SpExPlusSpeakerExtractor(BaseModel):
    def __init__(
            self,
            in_channels,
            out_channels,
            tcn_conv_channels,
            tcn_kernel_size,
            tcn_dilation,
            speaker_embedding_dim,
            n_tcn_stacks=4,
            n_tcn_blocks=8,
            causal=False,
            **batch
    ):
        """
        Speaker Extractor - this part encodes mixed audio, adding 
        speaker-embedding from Speaker Encoder part to its Stacked TCN Blocks.
        This part predicts mask on mixed encoded audio.

        It consists of n_tcn_stacks stacks of n_tcn_blocks
        and 3 masks for 3 filter-lengths of the SpEx+ model.

        Speaker embedding is added to the first TCNBlock in each stack.

        params:
            in_channels: in_channels of encoder, also the output dimension of TCNBlocks
            out_channels: out_channels of mask convolutions, corresponds to n_filters in encoder
            tcn_conv_channels: number of channels in hidden dilated convolutions in TCNBlocks
            tcn_kernel_size: kernel_size in hidden dilated convolutions in TCNBlocks
            tcn_dilation: dilation in hidden dilated convolutions in TCNBlocks
            speaker_embedding_dim: dimension of speaker embedding from Speaker Encoder
            n_tcn_stacks: number of TCNBlock stacks in the extractor, default = 4
            n_tcn_blocks: number of TCNBlocks in each stack, default = 8
            causal=False: whether the model is causal
        """
        super().__init__()

        self.n_tcn_blocks = n_tcn_blocks
        self.n_tcn_stacks = n_tcn_stacks

        self.TCN_stacks = nn.ModuleList([
            TCNBlock(
                in_channels=in_channels,
                conv_channels=tcn_conv_channels,
                kernel_size=tcn_kernel_size,
                dilation=tcn_dilation,
                use_speaker_embedding=(block_idx == 0),
                speaker_embedding_dim=speaker_embedding_dim,
                causal=causal
            )
            for block_idx in range(n_tcn_blocks)
            for stack_idx in range(n_tcn_stacks)
        ])

        self.masks = nn.ModuleDict({
            filter: nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1
            )
            for filter in ["L1", "L2", "L3"]
        })
        self.activation = nn.ReLU()


    def forward(self, input, speaker_embed):
        prev_output = input
        output = input

        for block_idx, TCN_block in enumerate(self.TCN_stacks):
            output = TCN_block(output, speaker_embed)

            # add skip-connections after each TCN stack
            if (block_idx + 1) % self.n_tcn_blocks == 0:
                output += prev_output
                prev_output = output
        
        masks = {
            filter: self.activation(mask(output))
            for filter, mask in self.masks.items()
        }
        return masks


class TCNBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            conv_channels,
            kernel_size,
            dilation,
            use_speaker_embedding=False,
            speaker_embedding_dim=None,
            causal=False
    ):
        """
        Temporal Convolution Block, consists of:
        - possible concatination with speaker embedding
        - Conv1d + PReLU + LayerNorm
        - Dilated depth-wise Conv1d + PReLU + LayerNorm + Conv1d

        Computes GlobalLayerNorm if causal == False, 
        otherwise LayerNorm is taken in the channelwise manner.

        Optionally: takes speaker embedding as an additional input.

        params:
            in_channels: in_channels for first Conv1d
            conv_channels: number of channels in hidden dilated convolution
            kernel_size: kernel_size of dilated convolion
            dilation: dilation of dilated convolution
            use_speaker_embedding: whether to add speaker embedding to input
            speaker_embedding_dim: dimension of speaker embedding, if needed
            causal: if the model is not causal, then we make use of 
            Global Layer Normalization ???
        """
        super().__init__()

        self.use_speaker_embedding = use_speaker_embedding
        self.speaker_embedding_dim = speaker_embedding_dim
        self.causal = causal

        self.conv1d_first = nn.Conv1d(
            in_channels=in_channels + speaker_embedding_dim if use_speaker_embedding else in_channels,
            out_channels=conv_channels,
            kernel_size=1
        )
        self.activation_1 = nn.PReLU()
        self.layer_norm_1 = LayerNorm(causal, conv_channels)

        self.dilated_conv_padding = dilation * (kernel_size - 1) // 2
        if causal:
            self.dilated_conv_padding //= 2
        
        self.dilated_conv1d = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=conv_channels,
            padding=self.dilated_conv_padding
        )
        self.activation_2 = nn.PReLU()
        self.layer_norm_2 = LayerNorm(causal, conv_channels)

        self.conv1d_last = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=in_channels,
            kernel_size=1
        )


    def forward(self, input, speaker_embed=None):
        output = input
        if self.use_speaker_embedding:
            speaker_embed = torch.unsqueeze(speaker_embed, -1).repeat(1, 1, input.shape[-1])
            output = torch.cat([input, speaker_embed], dim=1)
        
        output = self.activation_1(self.conv1d_first(output))
        output = self.layer_norm_1(output)

        output = self.dilated_conv1d(output)
        if self.causal:
            output = output[:, :, -self.dilated_conv_padding]
        
        output = self.activation_2(output)
        output = self.layer_norm_2(output)

        output = self.conv1d_last(output)
        output += input
        return output
    

class LayerNorm(nn.Module):
    def __init__(self, causal, norm_shape):
        super().__init__()
        self.causal = causal
        self.EPS = 1e-6

        if causal:
            self.layer_norm = nn.LayerNorm(norm_shape)
        else:
            self.gamma = nn.Parameter(torch.ones(norm_shape, 1))
            self.beta = nn.Parameter(torch.zeros(norm_shape, 1))
    
    def forward(self, input):
        if self.causal:
            return self.layer_norm(input.transpose(1, 2)).transpose(1, 2)
        
        mean = torch.mean(input, dim=(1, 2) ,keepdim=True)
        var = torch.mean((input - mean) ** 2, dim=(1, 2), keepdim=True)

        output = self.gamma * (input - mean) / torch.sqrt(var + self.EPS) + self.beta
        return output
