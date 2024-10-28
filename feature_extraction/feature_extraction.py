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
                 f_min: float = 60.0,
                 f_max: float = 7800.0,
                 n_mels: int = 64) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels

    def forward(self, x: torch.Tensor):
        """Extract HPSS feature tensor(s) from input audio tensor(s).

        Args:
            x (torch.Tensor): The audio tensor(s) of shape (B, 1, S) or (1, S).

        Returns:
            torch.Tensor: The HPSS features tensor(s) of shape (B, 2, H, W) or (2, H, W).
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        features = []
        for i in range(batch_size):
            audio = x[i].squeeze(0).cpu().numpy()
            stft = librosa.stft(audio,
                                n_fft=self.n_fft,
                                win_length=self.win_length,
                                hop_length=self.hop_length)
            harmonic_stft, percussive_stft = librosa.decompose.hpss(stft)
            mel_harmonic = librosa.feature.melspectrogram(S=np.abs(harmonic_stft)**2,
                                                            sr=self.sample_rate,
                                                            fmin=self.f_min,
                                                            fmax=self.f_max,
                                                            n_mels=self.n_mels)
            mel_percussive = librosa.feature.melspectrogram(S=np.abs(percussive_stft)**2,
                                                            sr=self.sample_rate,
                                                            fmin=self.f_min,
                                                            fmax=self.f_max,
                                                            n_mels=self.n_mels)
            mel_db_harmonic = librosa.power_to_db(mel_harmonic, ref=np.max)
            mel_db_percussive = librosa.power_to_db(mel_percussive, ref=np.max)
            hp_mel_db = np.stack(
                (mel_db_harmonic, mel_db_percussive), axis=0)
            hp_mel_db = torch.from_numpy(hp_mel_db)
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
                 win_length: int = 400,
                 hop_length: int = 160,
                 f_min: float = 60.0,
                 f_max: float = 7800.0,
                 n_mels: int = 64) -> None:
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
            self.feature_extractor = HPSS(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.n_mels
            )
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
