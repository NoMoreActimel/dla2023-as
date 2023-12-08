""" from https://github.com/NVIDIA/tacotron2 """

import torch
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write

import hw_nv.utils.audio.stft as stft
from .audio_processing import griffin_lim


class AudioTools:
    def __init__(self, max_wav_value, stft_params):
        self.stft = stft.TacotronSTFT(**stft_params)
        self.max_wav_value = max_wav_value
        self.sampling_rate = stft_params["sampling_rate"]

    def load_wav_to_torch(self, full_path):
        sampling_rate, data = read(full_path)
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate

    def get_mel(self, filename):
        audio, sampling_rate = self.load_wav_to_torch(filename)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        # melspec = torch.from_numpy(_normalize(melspec.numpy()))

        return melspec

    def get_mel_from_wav(self, audio):
        sampling_rate = self.sampling_rate
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec, magnitudes = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec, magnitudes

    def inv_mel_spec(self, mel, out_filename, griffin_iters=60):
        mel = torch.stack([mel])
        # mel = torch.stack([torch.from_numpy(_denormalize(mel.numpy()))])
        mel_decompress = self.stft.spectral_de_normalize(mel)
        mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.mm(mel_decompress[0], self.stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling

        audio = griffin_lim(torch.autograd.Variable(
            spec_from_mel[:, :, :-1]), self.stft.stft_fn, griffin_iters)

        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio_path = out_filename
        write(audio_path, self.sampling_rate, audio)