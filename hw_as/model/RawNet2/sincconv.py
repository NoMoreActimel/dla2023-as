import math
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn


# code provided at seminar: https://github.com/XuMuK1/dla2023/blob/2023/week10/antispoofing_seminar.ipynb

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        # In the future we will set high hz as band_hz + low + min_band_hz + min_low_hz
        # Where band_hz is (high_hz - low_hz). Therefore, it is reasonable to
        # do diff and do not set high_hz as sr/2

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)


        # filter lower frequency (out_channels, 1)
        # fixed f1 from the paper
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1), requires_grad=False)

        # filter frequency band (out_channels, 1)
        # fixed f2 (f2 = f1+diff) from the paper
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1), requires_grad=False)

        # len(g) = kernel_size
        # It is symmetric, therefore we will do computations only with left part, while creating g.

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);

        # self.window is eq. (8)


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

        # self.n_ = 2 * pi * n / sr

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        self.low_hz_ = self.low_hz_.to(waveforms.device)
        self.band_hz_ = self.band_hz_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_) # eq. (5) + make sure low >= min_low_hz

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2) # eq. (6) + make sure band has length >= min_band_hz
        band=(high-low)[:,0] # g[0] / 2

        f_times_t_low = torch.matmul(low, self.n_) # 2 * pi * n * freq / sr
        f_times_t_high = torch.matmul(high, self.n_)

        # 2*f2*sinc(2*pi*f2*n) - 2*f1*sinc(2*pi*f1*n)
        # 2*f2*sin(2*pi*f2*n) / (2 * pi * f2 * n) - 2*f1*sin(2*pi*f1*n) / (2 * pi * f1 * n)
        # sin(2*pi*f2*n) / (pi n) - sin(2*pi*f1*n) / (pi n)

        # (2 / sr) * sin(f_times_t_high) / self.n_ -  (2 / sr) * sin(f_times_t_low) / self.n_
        # (1/ sr) * (sin(f_times_t_high) - sin(f_times_t_low)) / (self.n_ / 2)

        # sr * correct eq. (4)

        # because self.n_ = 2 * pi * n / sr

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2*band.view(-1,1) # g[0] = 2 * (f2 - f1) = 2 * band, w[0] = 1
        band_pass_right= torch.flip(band_pass_left,dims=[1]) # g[n] = g[-n]
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1) # create full g[n]
        band_pass = band_pass / (2*band[:,None]) # normalize so the max is 1

        # band_pass_left = sr * correct (4)
        # center = freq (not scaled via division) = sr * scaled_freq
        # thus, after normalization we will divide all by sr and get normalized correct(4) + normalized center

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) # x[n] * g[n]
