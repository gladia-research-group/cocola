"""
Lightning DataModules.
"""

import os
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch
import librosa
import numpy as np

from contrastive_model import constants
from data.coco_chorales_contrastive_preprocessed import CocoChoralesContrastivePreprocessed
from data.moisesdb_contrastive_preprocessed import MoisesdbContrastivePreprocessed
from data.slakh2100_contrastive_preprocessed import Slakh2100ContrastivePreprocessed


def hpss(waveform, sample_rate=16000):
    x = waveform.squeeze(0).cpu().numpy()
    stft_x = librosa.stft(x,
                          n_fft=1024,
                          win_length=400,
                          hop_length=160)
    harmonic_stft_x, percussive_stft_x = librosa.decompose.hpss(stft_x)
    mel_harmonic_x = librosa.feature.melspectrogram(S=np.abs(harmonic_stft_x)**2,
                                                    sr=sample_rate,
                                                    fmin=60.0,
                                                    fmax=7800.0,
                                                    n_mels=64)
    mel_percussive_x = librosa.feature.melspectrogram(S=np.abs(percussive_stft_x)**2,
                                                      sr=sample_rate,
                                                      fmin=60.0,
                                                      fmax=7800.0,
                                                      n_mels=64)
    mel_db_harmonic_x = librosa.power_to_db(mel_harmonic_x, ref=np.max)
    mel_db_percussive_x = librosa.power_to_db(mel_percussive_x, ref=np.max)
    processed_x = np.stack((mel_db_harmonic_x, mel_db_percussive_x), axis=0)
    processed_x = torch.from_numpy(processed_x)
    return processed_x


class CoColaDataModule(L.LightningDataModule):
    def __init__(self,
                 root_dir: str = "~",
                 dataset: constants.Dataset = constants.Dataset.CCS,
                 batch_size: int = 32,
                 chunk_duration: int = 5,
                 target_sample_rate: int = 16000,
                 generate_submixtures: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = Path(root_dir)
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_duration = chunk_duration
        self.target_sample_rate = target_sample_rate
        self.generate_submixtures = generate_submixtures
        self.transform = hpss

    def setup(self, stage: str):
        if self.dataset in {constants.Dataset.CCS,
                            constants.Dataset.CCS_RANDOM,
                            constants.Dataset.CCS_STRING,
                            constants.Dataset.CCS_BRASS,
                            constants.Dataset.CCS_WOODWIND}:
            ensemble = self.dataset.value.split("/")[1]
            self.train_dataset, self.val_dataset, self.test_dataset = self._get_cocochorales_splits(ensemble=ensemble)

        elif self.dataset == constants.Dataset.SLAKH2100:
            self.train_dataset, self.val_dataset, self.test_dataset = self._get_slakh2100_splits()

        elif self.dataset == constants.Dataset.MOISESDB:
            self.train_dataset, self.val_dataset, self.test_dataset = self._get_moisesdb_splits()

        elif self.dataset == constants.Dataset.MIXED:
            coco_train_dataset, coco_val_dataset, coco_test_dataset = self._get_cocochorales_splits("random")
            moisesdb_train_dataset, moisesdb_val_dataset, moisesdb_test_dataset = self._get_moisesdb_splits()
            slakh_train_dataset, slakh_val_dataset, slakh_test_dataset = self._get_slakh2100_splits()

            self.train_dataset = ConcatDataset(
                [coco_train_dataset, moisesdb_train_dataset, slakh_train_dataset])
            self.val_dataset = ConcatDataset(
                [coco_val_dataset, moisesdb_val_dataset, slakh_val_dataset])
            self.test_dataset = ConcatDataset(
                [coco_test_dataset, moisesdb_test_dataset, slakh_test_dataset])
            
    def _get_cocochorales_splits(self, ensemble: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "coco_chorales_contrastive"

        train_dataset = CocoChoralesContrastivePreprocessed(
            root_dir=root_dir,
            download=True,
            preprocess=True,
            split="train",
            ensemble=ensemble,
            chunk_duration=self.chunk_duration,
            target_sample_rate=self.target_sample_rate,
            generate_submixtures=self.generate_submixtures,
            device=device,
            preprocess_transform=self.transform)
        
        val_dataset = CocoChoralesContrastivePreprocessed(
            root_dir=root_dir,
            download=True,
            preprocess=True,
            split="valid",
            ensemble=ensemble,
            chunk_duration=self.chunk_duration,
            target_sample_rate=self.target_sample_rate,
            generate_submixtures=self.generate_submixtures,
            device=device,
            preprocess_transform=self.transform)
        
        test_dataset = CocoChoralesContrastivePreprocessed(
            root_dir=root_dir,
            download=True,
            preprocess=True,
            split="test",
            ensemble=ensemble,
            chunk_duration=self.chunk_duration,
            target_sample_rate=self.target_sample_rate,
            generate_submixtures=self.generate_submixtures,
            device=device,
            preprocess_transform=self.transform)
        return train_dataset, val_dataset, test_dataset
    
    def _get_slakh2100_splits(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "slakh2100_contrastive"

        train_dataset = Slakh2100ContrastivePreprocessed(
            root_dir=root_dir,
            download=True,
            preprocess=True,
            split="train",
            chunk_duration=self.chunk_duration,
            target_sample_rate=self.target_sample_rate,
            generate_submixtures=self.generate_submixtures,
            device=device,
            preprocess_transform=self.transform)
        
        val_dataset = Slakh2100ContrastivePreprocessed(
            root_dir=root_dir,
            download=True,
            preprocess=True,
            split="validation",
            chunk_duration=self.chunk_duration,
            target_sample_rate=self.target_sample_rate,
            generate_submixtures=self.generate_submixtures,
            device=device,
            preprocess_transform=self.transform)
        
        test_dataset = Slakh2100ContrastivePreprocessed(
            root_dir=root_dir,
            download=True,
            preprocess=True,
            split="test",
            chunk_duration=self.chunk_duration,
            target_sample_rate=self.target_sample_rate,
            generate_submixtures=self.generate_submixtures,
            device=device,
            preprocess_transform=self.transform)

        return train_dataset, val_dataset, test_dataset
    
    def _get_moisesdb_splits(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        root_dir = self.root_dir / "moisesdb_contrastive"

        train_dataset = MoisesdbContrastivePreprocessed(
            root_dir=root_dir,
            split="train",
            preprocess=True,
            chunk_duration=self.chunk_duration,
            target_sample_rate=self.target_sample_rate,
            generate_submixtures=self.generate_submixtures,
            device=device,
            preprocess_transform=self.transform
        )

        val_dataset = MoisesdbContrastivePreprocessed(
            root_dir=root_dir,
            split="valid",
            preprocess=True,
            chunk_duration=self.chunk_duration,
            target_sample_rate=self.target_sample_rate,
            generate_submixtures=self.generate_submixtures,
            device=device,
            preprocess_transform=self.transform
        )

        test_dataset = MoisesdbContrastivePreprocessed(
            root_dir=root_dir,
            split="test",
            preprocess=True,
            chunk_duration=self.chunk_duration,
            target_sample_rate=self.target_sample_rate,
            generate_submixtures=self.generate_submixtures,
            device=device,
            preprocess_transform=self.transform
        )
        
        return train_dataset, val_dataset, test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=os.cpu_count(),
            persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=os.cpu_count(),
            persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=os.cpu_count(),
            persistent_workers=True)
