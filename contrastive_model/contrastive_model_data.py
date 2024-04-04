import os

import lightning as L
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch
import torchaudio.transforms as T

from contrastive_model import constants
from data import (
    coco_chorales_contrastive_preprocessed,
    moisesdb_contrastive_preprocessed,
    slakh2100_contrastive_preprocessed
)

class CoColaDataModule(L.LightningDataModule):
    def __init__(self,
                 dataset: constants.Dataset = constants.Dataset.MIXED,
                 batch_size: int = 32,
                 chunk_duration: int = 5,
                 positive_noise: float = 0.001,
                 generate_submixtures: bool = True):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_duration = chunk_duration
        self.positive_noise = positive_noise
        self.generate_submixtures = generate_submixtures
        self.transform = torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                win_length=400,
                hop_length=160,
                f_min=60.0,
                f_max=7800.0,
                n_mels=64,
            ),
            T.AmplitudeToDB()
        )

    def setup(self, stage: str):
        if self.dataset in {constants.Dataset.CCS,
                             constants.Dataset.CCS_RANDOM,
                             constants.Dataset.CCS_STRING,
                             constants.Dataset.CCS_BRASS,
                             constants.Dataset.CCS_WOODWIND}:
            ensemble = self.dataset.value.split("/")[1]
            self.sample_rate = coco_chorales_contrastive_preprocessed.CocoChoralesContrastivePreprocessed.SAMPLE_RATE
            self.train_dataset = coco_chorales_contrastive_preprocessed.get_dataset(split="train", ensemble=ensemble,
                                                                               chunk_duration=self.chunk_duration, positive_noise=self.positive_noise,
                                                                               generate_submixtures=self.generate_submixtures, transform=self.transform)
            self.val_dataset = coco_chorales_contrastive_preprocessed.get_dataset(split="valid", ensemble=ensemble,
                                                                               chunk_duration=self.chunk_duration, positive_noise=self.positive_noise,
                                                                               generate_submixtures=self.generate_submixtures, transform=self.transform)
            self.test_dataset = coco_chorales_contrastive_preprocessed.get_dataset(split="test", ensemble=ensemble,
                                                                               chunk_duration=self.chunk_duration, positive_noise=self.positive_noise,
                                                                               generate_submixtures=self.generate_submixtures, transform=self.transform)

            assert self.train_dataset.target_sample_rate == self.val_dataset.target_sample_rate == self.test_dataset.target_sample_rate
            self.sample_rate = self.train_dataset.target_sample_rate

        elif self.dataset == constants.Dataset.SLAKH2100:
            self.train_dataset = slakh2100_contrastive_preprocessed.get_dataset(
                split="train", chunk_duration=self.chunk_duration, positive_noise=self.positive_noise, generate_submixtures=self.generate_submixtures, transform=self.transform)
            self.val_dataset = slakh2100_contrastive_preprocessed.get_dataset(
                split="validation", chunk_duration=self.chunk_duration, positive_noise=self.positive_noise, generate_submixtures=self.generate_submixtures, transform=self.transform)
            self.test_dataset = slakh2100_contrastive_preprocessed.get_dataset(
                split="test", chunk_duration=self.chunk_duration, positive_noise=self.positive_noise, generate_submixtures=self.generate_submixtures, transform=self.transform)
            
            assert self.train_dataset.target_sample_rate == self.val_dataset.target_sample_rate == self.test_dataset.target_sample_rate
            self.sample_rate = self.train_dataset.target_sample_rate

        elif self.dataset == constants.Dataset.MOISESDB:
            dataset = moisesdb_contrastive_preprocessed.get_dataset(chunk_duration=self.chunk_duration,
                                                                    positive_noise=self.positive_noise,
                                                                    generate_submixtures=self.generate_submixtures,
                                                                    transform=self.transform)

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset=dataset, lengths=[0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

            self.sample_rate = dataset.target_sample_rate

        elif self.dataset == constants.Dataset.MIXED:
            coco_train_dataset = coco_chorales_contrastive_preprocessed.get_dataset(split="train", ensemble="*",
                                                                                    chunk_duration=self.chunk_duration,
                                                                                    positive_noise=self.positive_noise,
                                                                                    generate_submixtures=self.generate_submixtures,
                                                                                    transform=self.transform)
            coco_val_dataset = coco_chorales_contrastive_preprocessed.get_dataset(split="valid", ensemble="*",
                                                                                    chunk_duration=self.chunk_duration,
                                                                                    positive_noise=self.positive_noise,
                                                                                    generate_submixtures=self.generate_submixtures,
                                                                                    transform=self.transform)
            coco_test_dataset = coco_chorales_contrastive_preprocessed.get_dataset(split="test", ensemble="*",
                                                                                    chunk_duration=self.chunk_duration,
                                                                                    positive_noise=self.positive_noise,
                                                                                    generate_submixtures=self.generate_submixtures,
                                                                                    transform=self.transform)
            

            moisesdb_dataset = moisesdb_contrastive_preprocessed.get_dataset(chunk_duration=self.chunk_duration,
                                                                             positive_noise=self.positive_noise,
                                                                             generate_submixtures=self.generate_submixtures,
                                                                             transform=self.transform)
            moisesdb_train_dataset, moisesdb_val_dataset, moisesdb_test_dataset = random_split(
                dataset=moisesdb_dataset, lengths=[0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

            slakh_train_dataset = slakh2100_contrastive_preprocessed.get_dataset(
                split="train", chunk_duration=self.chunk_duration, positive_noise=self.positive_noise,
                generate_submixtures=self.generate_submixtures, transform=self.transform)
            slakh_val_dataset = slakh2100_contrastive_preprocessed.get_dataset(
                split="validation", chunk_duration=self.chunk_duration, positive_noise=self.positive_noise,
                generate_submixtures=self.generate_submixtures, transform=self.transform)
            slakh_test_dataset = slakh2100_contrastive_preprocessed.get_dataset(
                split="test", chunk_duration=self.chunk_duration, positive_noise=self.positive_noise,
                generate_submixtures=self.generate_submixtures, transform=self.transform)

            assert coco_train_dataset.target_sample_rate == coco_val_dataset.target_sample_rate == coco_test_dataset.target_sample_rate ==\
                moisesdb_dataset.target_sample_rate == \
                slakh_train_dataset.target_sample_rate == slakh_val_dataset.target_sample_rate == slakh_test_dataset.target_sample_rate
            self.sample_rate = coco_train_dataset.target_sample_rate

            self.train_dataset = ConcatDataset(
                [coco_train_dataset, moisesdb_train_dataset, slakh_train_dataset])
            self.val_dataset = ConcatDataset(
                [coco_val_dataset, moisesdb_val_dataset, slakh_val_dataset])
            self.test_dataset = ConcatDataset(
                [coco_test_dataset, moisesdb_test_dataset, slakh_test_dataset])
        

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=32,
                shuffle=True,
                drop_last=True,
                num_workers=os.cpu_count(),
                persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
                self.val_dataset,
                batch_size=32,
                shuffle=False,
                drop_last=True,
                num_workers=os.cpu_count(),
                persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
                self.test_dataset,
                batch_size=32,
                shuffle=False,
                drop_last=True,
                num_workers=os.cpu_count(),
                persistent_workers=True)