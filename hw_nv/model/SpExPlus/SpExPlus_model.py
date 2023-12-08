import torch
from torch import nn

from hw_nv.base import BaseModel
from hw_nv.model.SpExPlus.SpExPlus_speech_encoder import SpExPlusTwinSpeechEncoder
from hw_nv.model.SpExPlus.SpExPlus_speaker_encoder import SpExPlusSpeakerEncoder
from hw_nv.model.SpExPlus.SpExPlus_speaker_extractor import SpExPlusSpeakerExtractor
from hw_nv.model.SpExPlus.SpExPlus_speech_decoder import SpExPlusSpeechDecoder

class SpExPlusModel(BaseModel):
    def __init__(
            self,
            L1=20, L2=80, L3=160,
            n_filters=256,
            encoder_out_dim=256,
            speaker_encoder_hidden_dim=512,
            speaker_embedding_dim=256,
            num_speakers=101,
            tcn_conv_channels=512,
            tcn_kernel_size=3,
            tcn_dilation=1,
            n_tcn_stacks=4,
            n_tcn_blocks=8,
            causal=False,
            **batch
    ):
        """
        Implementation of SpEx Plus Model for Speech Separation Task
        from the "SpEx+: A Complete Time Domain Speaker Extraction Network" paper.
        https://www.isca-speech.org/archive/pdfs/interspeech_2020/ge20_interspeech.pdf

        The model itself encodes mixed and target-speaker waveforms into 
        the shared latent space and then uses target-speaker embedding 
        to predict the mask for the mixed audio. 

        Overall, architecture consists of:
        1) Twin Speech Encoders, which encode initial audios with 
            1d-Convolutions with 3 different filter lengths. 
            These layers provide model backbone with multiple scales of temporal resolutions.
        2) Speaker Encoder - this part encodes speaker embedding
            from clean target audio, in order to help the main part of the model.
            Learning is performed both on the Cross-Entropy Loss for speaker classification
            and SI-SDR Loss backpropagated from the main backbone.
        3) Speaker Extractor - this part encodes mixed audio, 
            adding speaker-embedding from Speaker Encoder part to its Stacked TCN Blocks.
            This part predicts mask on mixed encoded audio
        4) Decoders on 3 different scales - decode masked mixed audio back,
            predicting denoised audio of target speaker.
            Its outputs are fed to the SI-SDR Loss, that trains the main model.

        To get the further understanding of model architecture, refer to the paper itself.

        params:
            L1, L2, L3: filter lengths of the Encoder Conv1D's
            n_filters: number of filters for each of L1, L2, L3
            encoder_out_dim: output dimension of twin-speech-encoder
            speaker_encoder_hidden_dim: hidden dimension of resnet blocks in speaker encoder
            speaker_embedding_dim: dimension of speaker embedding from speaker encoder
            num_speakers: number of speakers, to train speaker encoder on CE Loss
            tcn_kernel_size: kernel_size in hidden dilated convolutions in TCNBlocks
            tcn_dilation: dilation in hidden dilated convolutions in TCNBlocks
            speaker_embedding_dim: dimension of speaker embedding from Speaker Encoder
            n_tcn_stacks: number of TCNBlock stacks in the extractor, default = 4
            n_tcn_blocks: number of TCNBlocks in each stack, default = 8
            causal=False: whether the model is causal
        """
        super().__init__()

        self.speech_encoder = SpExPlusTwinSpeechEncoder(
            L1, L2, L3, n_filters,
            out_channels=encoder_out_dim
        )
        self.speaker_encoder = SpExPlusSpeakerEncoder(
            resblocks_in_dim=encoder_out_dim,
            resblocks_out_dim=speaker_encoder_hidden_dim,
            speaker_embedding_dim=speaker_embedding_dim,
            num_speakers=num_speakers
        )
        self.speaker_extractor = SpExPlusSpeakerExtractor(
            in_channels=encoder_out_dim,
            out_channels=n_filters,
            tcn_conv_channels=tcn_conv_channels,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilation=tcn_dilation,
            speaker_embedding_dim=speaker_embedding_dim,
            n_tcn_stacks=n_tcn_stacks,
            n_tcn_blocks=n_tcn_blocks,
            causal=causal
        )
        self.speech_decoder = SpExPlusSpeechDecoder(
            L1, L2, L3, n_filters
        )
    
    def _unsqueeze_inputs(self, input, ref):
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
        if len(input.shape) == 2:
            input = input.unsqueeze(1)
        if len(ref.shape) == 1:
            ref = ref.unsqueeze(0)
        if len(ref.shape) == 2:
            ref = ref.unsqueeze(1)

        return input, ref

    
    def forward(self, input, ref, ref_length, **batch):
        input, ref = self._unsqueeze_inputs(input, ref)
        
        input_length = input.shape[-1]
        
        input_encoded, input_encoded_by_filters = self.speech_encoder(input, input_type="mixed")
        ref_encoded, ref_encoded_by_filters = self.speech_encoder(ref, input_type="ref")

        ref_time_length = (ref_length - self.speech_encoder.L1) // (self.speech_encoder.L1 // 2) + 1
        speaker_embed, speaker_logits = self.speaker_encoder(ref_encoded, ref_time_length)

        masks = self.speaker_extractor(input_encoded, speaker_embed)
        outputs = {
            filter: input_encoded_by_filters[filter] * mask
            for filter, mask in masks.items()
        }

        for filter, output in outputs.items():
            if len(output.shape) == 2:
                outputs[filter] = output.unsqueeze(1)

        outputs = self.speech_decoder.forward(outputs)
        outputs = {
            filter: output[:, :input_length]
            for filter, output in outputs.items()
        }

        return outputs, speaker_logits