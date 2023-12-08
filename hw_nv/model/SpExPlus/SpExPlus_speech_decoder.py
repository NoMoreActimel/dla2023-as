import torch
from torch import nn

from hw_nv.base import BaseModel

class SpExPlusSpeechDecoder(BaseModel):
    def __init__(
            self,
            L1, L2, L3,
            n_filters,
            **batch
    ):
        """
        Speech Decoder decodes masked hidden states back to audio
        by applying 1d-Transposed-Convolutions with 3 different filter lengths.
        Receives same params as Speech Encoder.

        params:
            L1, L2, L3: filter lengths of the Decoder ConvTranspose1D's
            n_filters: number of filters for each of L1, L2, L3
        """
        super().__init__()

        self.L1, self.L2, self.L3 = L1, L2, L3

        self.decoder_convs = nn.ModuleDict({
            filter: nn.ConvTranspose1d(
                in_channels=n_filters,
                out_channels=1,
                kernel_size=getattr(self, filter),
                stride=getattr(self, filter) // 2
            )
            for filter in ["L1", "L2", "L3"]
        })

    def forward(self, input_by_filter):        
        outputs = {
            filter: decoder_conv(input_by_filter[filter]).squeeze(1)
            for filter, decoder_conv in self.decoder_convs.items()
        }

        return outputs
