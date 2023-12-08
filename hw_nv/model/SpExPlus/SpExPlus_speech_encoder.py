import torch
from torch import nn

from hw_nv.base import BaseModel

class SpExPlusTwinSpeechEncoder(BaseModel):
    def __init__(
            self,
            L1, L2, L3,
            n_filters,
            out_channels,
            **batch
    ):
        """
        Twin Speech Encoders, which encode initial audios with 
        1d-Convolutions with 3 different filter lengths. 
        These layers provide model backbone with 
        multiple scales of temporal resolutions.

        params:
            L1, L2, L3: filter lengths of the Encoder Conv1D's
            n_filters: number of filters for each of L1, L2, L3
            out_channels: number of output channels after the final Conv1D projection
        """
        super().__init__()

        self.L1, self.L2, self.L3 = L1, L2, L3

        self.encoder_convs = nn.ModuleDict({
            filter: nn.Conv1d(
                in_channels=1,
                out_channels=n_filters,
                kernel_size=getattr(self, filter),
                stride=L1 // 2
            )
            for filter in ["L1", "L2", "L3"]
        })

        # layer norms and projections do not share weights
        self.encoder_layer_norms = nn.ModuleDict({
            key: nn.LayerNorm(3 * n_filters)
            for key in ["mixed", "ref"]
        })
        self.activation = nn.ReLU()
        self.projections = nn.ModuleDict({
            key: nn.Conv1d(
                in_channels=3 * n_filters,
                out_channels=out_channels,
                kernel_size=1
            )
            for key in ["mixed", "ref"]
        })

    def forward(self, input, input_type):
        """
        input: input audio tensor
        input_type: either "mixed" or "ref"

        returns:
            outputs_cat: concatenated and normalized encoded inputs by all filters
            outputs: dict of encoded inputs by L1, L2, L3 as keys
        """
        outputs = {}
        
        # Batch x 1 x Time -> Batch x N_filters x Dim
        outputs["L1"] = self.activation(self.encoder_convs["L1"](input))

        input_time_length = input.shape[-1]
        dim_length = outputs["L1"].shape[-1]

        L2_padding = ((dim_length - 1) * (self.L1 // 2) + self.L2) - input_time_length
        L3_padding = ((dim_length - 1) * (self.L1 // 2) + self.L3) - input_time_length

        # add L2_padding and L3_padding zeroes to the end of last dim
        input_l2 = nn.functional.pad(input, pad=(0, L2_padding))
        input_l3 = nn.functional.pad(input, pad=(0, L3_padding))

        outputs["L2"] = self.activation(self.encoder_convs["L2"](input_l2))
        outputs["L3"] = self.activation(self.encoder_convs["L3"](input_l3))
        
        # Cat to Batch x (3 * N_filters) x Dim
        outputs_cat = torch.cat([outputs["L1"], outputs["L2"], outputs["L3"]], dim=1)

        # use specific layer norm for mixed and ref audios
        outputs_cat = outputs_cat.transpose(1, 2)
        outputs_cat = self.encoder_layer_norms[input_type](outputs_cat)
        outputs_cat = outputs_cat.transpose(1, 2)

        # Batch x (3 * N_filters) x Dim -> Batch x Out_channels x Dim
        outputs_cat = self.projections[input_type](outputs_cat)

        return outputs_cat, outputs
