#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Du Xingjian
# E-Mail:diggerdu97@gmail.com

from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np

from contrastive_model import constants


def softmask(X, X_ref, power=1, split_zeros=False):
    if X.shape != X_ref.shape:
        raise ValueError(f"Shape mismatch: {X.shape}!={X_ref.shape}")
    if torch.any(X < 0) or torch.any(X_ref < 0):
        raise ValueError("X and X_ref must be non-negative")

    if power <= 0:
        raise ValueError("power must be strictly positive")

    dtype = X.dtype
    if dtype not in [torch.float16, torch.float32, torch.float64]:
        raise ValueError("data type error")

    Z = torch.max(X, X_ref)
    bad_idx = (Z < torch.finfo(dtype).tiny)
    if bad_idx.sum() > 0:
        Z[bad_idx] = 1

    if np.isfinite(power):
        mask = (X / Z)**power
        ref_mask = (X_ref / Z)**power
        mask /= (mask + ref_mask)
    else:
        mask = X > X_ref
    return mask


def get_binary_kernel2d(window_size):
    window_range = window_size[0] * window_size[1]
    kernel = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def _compute_zero_padding(kernel_size):
    return (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2


class MedianBlur(nn.Module):
    def __init__(self, kernel_size, channel, reduce_method='median'):
        super(MedianBlur, self).__init__()
        tmp_kernel = get_binary_kernel2d(kernel_size).float()
        kernel = tmp_kernel.repeat(channel, 1, 1, 1)
        self.register_buffer("kernel", kernel.contiguous())
        self.padding = _compute_zero_padding(kernel_size)
        self.reduce_method = reduce_method

    def forward(self, input: torch.Tensor):
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
        if not len(input.shape) == 4:
            raise ValueError(f"Invalid input shape, expected BxCxHxW. Got: {input.shape}")

        b, c, h, w = input.shape
        features = F.conv2d(input, self.kernel, padding=self.padding, stride=1, groups=c)
        features = features.view(b, c, -1, h, w)

        if self.reduce_method == 'median':
            res = torch.median(features, dim=2)[0]
        else:
            res = torch.mean(features, dim=2)
        return res


class HPSS(nn.Module):
    def __init__(self, kernel_size, channel=2, power=2.0, mask=False, margin=1.0, reduce_method='median'):
        super(HPSS, self).__init__()
        self.harm_median_filter = MedianBlur(kernel_size=(1, kernel_size), channel=channel, reduce_method=reduce_method)
        self.perc_median_filter = MedianBlur(kernel_size=(kernel_size, 1), channel=channel, reduce_method=reduce_method)
        self.power = power
        self.mask = mask
        self.margin = margin

    def forward(self, S):
        if np.isscalar(self.margin):
            margin_harm = self.margin
            margin_perc = self.margin
        else:
            margin_harm = self.margin[0]
            margin_perc = self.margin[1]

        if margin_harm < 1 or margin_perc < 1:
            raise ValueError("Margins must be >= 1.0")

        harm = self.harm_median_filter(S)
        perc = self.perc_median_filter(S)

        split_zeros = (margin_harm == 1 and margin_perc == 1)
        mask_harm = softmask(harm, perc * margin_harm, power=self.power, split_zeros=split_zeros)
        mask_perc = softmask(perc, harm * margin_perc, power=self.power, split_zeros=split_zeros)

        if self.mask:
            return mask_harm, mask_perc
        return {"harm_spec": S * mask_harm, "perc_spec": S * mask_perc}


class HPSSFeatureExtractor(nn.Module):
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 win_length: int = 1024,
                 hop_length: int = 160,
                 f_min: float = 60.0,
                 f_max: float = 7800.0,
                 n_mels: int = 64,
                 kernel_size: int = 31,
                 reduce_method: str = 'mean') -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels

        self.spectrogram = T.Spectrogram(n_fft=self.n_fft)
        
        self.hpss = HPSS(kernel_size=kernel_size, channel=1, reduce_method=reduce_method)
        self.melscale = T.MelScale(n_mels=self.n_mels,
                                   sample_rate=self.sample_rate,
                                   f_min=self.f_min,
                                   f_max=self.f_max)
        self.amplitude_to_db = T.AmplitudeToDB()

    def forward(self, x: torch.Tensor):
        # Expecting x of shape (B, 1, S) or (1, S)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        features = []

        for i in range(batch_size):
            waveform = x[i]
            spec = self.spectrogram(waveform)  # shape (1, freq, time), complex
            mag = spec.abs().unsqueeze(0)      # shape (1, 1, freq, time)
            res = self.hpss(mag)
            harm_spec = res['harm_spec']       # (1, 1, freq, time)
            perc_spec = res['perc_spec']       # (1, 1, freq, time)

            # Convert to power
            harm_power = harm_spec**2
            perc_power = perc_spec**2

            # Convert to mel
            harm_mel = self.melscale(harm_power.squeeze(0))
            perc_mel = self.melscale(perc_power.squeeze(0))

            harm_db = self.amplitude_to_db(harm_mel)
            perc_db = self.amplitude_to_db(perc_mel)

            hp_mel_db = torch.stack((harm_db, perc_db), dim=0)
            features.append(hp_mel_db)

        features = torch.stack(features, dim=0)
        if batch_size == 1:
            features = features.squeeze(0)

        return features


class CoColaFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_extractor_type: constants.ModelFeatureExtractorType = constants.ModelFeatureExtractorType.HPSS,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 win_length: int = None,
                 hop_length: int = None,
                 f_min: float = 60.0,
                 f_max: float = 7800.0,
                 n_mels: int = 64,
                 kernel_size: int = 31,
                 reduce_method: str = 'mean') -> None:
        super().__init__()
        self.feature_extractor_type = feature_extractor_type
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels

        if self.feature_extractor_type == constants.ModelFeatureExtractorType.HPSS:
            self.feature_extractor = HPSSFeatureExtractor(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.n_mels,
                kernel_size=kernel_size,
                reduce_method=reduce_method
            )
        elif self.feature_extractor_type == constants.ModelFeatureExtractorType.MEL_SPECTROGRAM:
            self.feature_extractor = nn.Sequential(
                T.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    f_min=self.f_min,
                    f_max=self.f_max,
                    n_mels=self.n_mels,
                ),
                T.AmplitudeToDB()
            )

    def forward(self, x: Union[dict, torch.Tensor]):
        if isinstance(x, dict):
            x['anchor'] = self.feature_extractor(x['anchor'])
            x['positive'] = self.feature_extractor(x['positive'])
            return x
        else:
            return self.feature_extractor(x)

    def to_dict(self):
        return {
            "feature_extractor_type": self.feature_extractor_type.value,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "n_mels": self.n_mels
        }
