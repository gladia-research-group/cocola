"""Audio preprocessing utility classes."""

from typing import Union

import numpy as np
import torch
import torchaudio.transforms as T
from torch import nn
import librosa

from contrastive_model import constants


class HPSS(nn.Module):
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 win_length: int = 400,
                 hop_length: int = 160,
                 fmin: float = 60.0,
                 fmax: float = 7800.0,
                 n_mels: int = 64) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels

    def forward(self, x: torch.Tensor):
        x = x.squeeze(0).cpu().numpy()
        stft_x = librosa.stft(x,
                              n_fft=self.n_fft,
                              win_length=self.win_length,
                              hop_length=self.hop_length)
        harmonic_stft_x, percussive_stft_x = librosa.decompose.hpss(stft_x)
        mel_harmonic_x = librosa.feature.melspectrogram(S=np.abs(harmonic_stft_x)**2,
                                                        sr=self.sample_rate,
                                                        fmin=self.fmin,
                                                        fmax=self.fmax,
                                                        n_mels=self.n_mels)
        mel_percussive_x = librosa.feature.melspectrogram(S=np.abs(percussive_stft_x)**2,
                                                          sr=self.sample_rate,
                                                          fmin=self.fmin,
                                                          fmax=self.fmax,
                                                          n_mels=self.n_mels)
        mel_db_harmonic_x = librosa.power_to_db(mel_harmonic_x, ref=np.max)
        mel_db_percussive_x = librosa.power_to_db(mel_percussive_x, ref=np.max)
        processed_x = np.stack(
            (mel_db_harmonic_x, mel_db_percussive_x), axis=0)
        processed_x = torch.from_numpy(processed_x)
        return processed_x


class CoColaFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_extractor_type: constants.ModelFeatureExtractorType = constants.ModelFeatureExtractorType.HPSS,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 win_length: int = 400,
                 hop_length: int = 160,
                 f_min: float = 60.0,
                 f_max: float = 7800.0,
                 n_mels: int = 64) -> None:
        super().__init__()
        self.feature_extractor_type = feature_extractor_type
        if self.feature_extractor_type == constants.ModelFeatureExtractorType.HPSS:
            self.feature_extractor = HPSS()
        elif self.feature_extractor_type == constants.ModelFeatureExtractorType.MEL_SPECTROGRAM:
            self.feature_extractor = torch.nn.Sequential(
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
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels

    def forward(self, x: Union[dict, torch.Tensor]):
        """Performs feature extraction.

        Args:
            x (Union[dict, torch.Tensor]): the waveform or anchor positive dictionary on which feature extraction is applied.

        Returns:
            Union[dict, torch.Tensor]: features tensor or dictionary.
        """
        if isinstance(x, dict):
            x['anchor'] = self.feature_extractor(x['anchor'])
            x['positive'] = self.feature_extractor(x['positive'])
            return x
        else:
            return self.feature_extractor(x)

    def to_dict(self):
        """Used for serialization."""
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
