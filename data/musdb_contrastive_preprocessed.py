"""Musdb Contrastive Torch Dataset."""

from typing import Dict, Literal
from pathlib import Path
import random
import os
import logging
import shutil
import json

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets.utils import download_and_extract_archive
import torchaudio
import torchaudio.transforms as T

random.seed(14703)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class MusdbContrastivePreprocessed(Dataset):
    """
    Musdb Dataset (adapted for contrastive learning): https://sigsep.github.io/datasets/musdb.html
    """
    VERSION = "1.0.0"
    URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip"
    SAMPLE_RATE = 44100
    ORIGINAL_DIR_NAME = "original"
    PREPROCESSED_DIR_NAME = "preprocessed"
    PREPROCESSING_INFO_FILE_NAME = "preprocessing_info.json"

    def __init__(
            self,
            root_dir="~/musdb_contrastive",
            download="false",
            preprocess="false",
            split="train",
            chunk_duration=5,
            positive_noise=0.001,
            target_sample_rate=16000,
            generate_submixtures=True,
            device="cpu",
            transform=None) -> None:

        self.root_dir = Path(root_dir)
        self.download = download
        self.preprocess = preprocess
        self.split = split
        self.chunk_duration = chunk_duration
        self.positive_noise = positive_noise
        self.target_sample_rate = target_sample_rate
        self.generate_submixtures = generate_submixtures
        self.transform = transform

        if self.split not in ["train", "test"]:
            raise ValueError(
                "`split` must be one of ['train', 'test'].")

        if self.download and not self._is_downloaded_and_extracted():
            self._download_and_extract()
        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found. Please use `download=True` to download it.")
        logging.info(
            f"Found original dataset split {self.split} at {(self.root_dir / self.ORIGINAL_DIR_NAME / self.split).expanduser()}.")

        if self.preprocess and not self._is_preprocessed():
            self.device = device
            self.resample_transform = T.Resample(
                self.SAMPLE_RATE, self.target_sample_rate).to(self.device)
            self._preprocess_and_save()
        if not self._is_preprocessed():
            raise RuntimeError(
                f"Preprocessed dataset split {self.split} not found. Please use `preprocess=True` to preprocess it.")
        logging.info(
            f"Found preprocessed dataset split {self.split} at {(self.root_dir / self.PREPROCESSED_DIR_NAME / self.split).expanduser()}.")

        self.file_paths_df = self._load_file_paths()

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = (self.root_dir / self.ORIGINAL_DIR_NAME /
                     self.split).expanduser()
        return split_dir.exists() and any(split_dir.iterdir())

    def _download_and_extract(self) -> None:
        download_and_extract_archive(
            self.URL, self.root_dir / self.ORIGINAL_DIR_NAME, remove_finished=True)

    def _right_pad(self, waveform):
        num_frames = waveform.shape[1]
        expected_num_frames = self.chunk_duration * self.target_sample_rate
        if num_frames < expected_num_frames:
            num_missing_frames = expected_num_frames - num_frames
            last_dim_padding = (0, num_missing_frames)
            waveform = torch.nn.functional.pad(waveform, last_dim_padding)
        return waveform

    def _mix_down(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _mix_stems(self, stems):
        return 1/len(stems) * sum(stems)

    def _is_preprocessed(self) -> bool:
        preprocessed_dir = (
            self.root_dir / self.PREPROCESSED_DIR_NAME / self.split).expanduser()

        if not preprocessed_dir.exists():
            return False

        with open(preprocessed_dir / self.PREPROCESSING_INFO_FILE_NAME, "r") as preprocessing_info_file:
            preprocessing_info = json.load(preprocessing_info_file)
            if preprocessing_info["chunk_duration"] != self.chunk_duration or \
               preprocessing_info["target_sample_rate"] != self.target_sample_rate or \
               preprocessing_info["generate_submixtures"] != self.generate_submixtures:
                logging.info(
                    "Found preprocessed dataset with different preprocessing parameters. Overwriting with new preprocessed dataset.")
                shutil.rmtree(preprocessed_dir)

        return preprocessed_dir.exists() and any(preprocessed_dir.iterdir())

    def _preprocess_and_save(self) -> bool:
        preprocessed_dir = (
            self.root_dir / self.PREPROCESSED_DIR_NAME / self.split).expanduser()
        preprocessed_dir.mkdir(parents=True)

        preprocessing_info = {
            "chunk_duration": self.chunk_duration,
            "target_sample_rate": self.target_sample_rate,
            "generate_submixtures": self.generate_submixtures
        }

        with open(preprocessed_dir / self.PREPROCESSING_INFO_FILE_NAME, "w") as preprocessing_info_file:
            json.dump(preprocessing_info, preprocessing_info_file)

        original_dir = (self.root_dir / self.ORIGINAL_DIR_NAME /
                        self.split).expanduser()

        tracks = original_dir.glob("*/")

        for track in tqdm(tracks, desc="Preprocessing tracks"):
            stems_paths = list(track.glob("*.wav"))
            original_track_name = track.name
            chunk_num_frames = self.chunk_duration * self.target_sample_rate
            frame_offset = 0
            for _ in range(2):
                stems = [torch.split(
                    self.resample_transform(self._mix_down(
                        torchaudio.load(stem_path)[0].to(self.device))),
                    split_size_or_sections=chunk_num_frames,
                    dim=1)
                    for stem_path in stems_paths]

                for i in range(len(stems[0])):
                    stems_idxs = range(len(stems))
                    anchor_mix_size = random.randint(
                        1, len(stems_idxs) // 2) if self.generate_submixtures else 1
                    positive_mix_size = random.randint(
                        1, len(stems_idxs) // 2) if self.generate_submixtures else 1
                    anchor_mix_idxs = random.sample(stems_idxs, anchor_mix_size)
                    positive_mix_idxs = random.sample(
                        [stem_idx for stem_idx in stems_idxs if stem_idx not in anchor_mix_idxs], positive_mix_size)
                    anchor_mix_id = ''.join(str(idx)
                                            for idx in anchor_mix_idxs)
                    positive_mix_id = ''.join(str(idx)
                                            for idx in positive_mix_idxs)

                    example_id = f"{original_track_name}_chunk{i}_comb{anchor_mix_id}_{positive_mix_id}_shift{_}"
                    example_path = preprocessed_dir / example_id
                    example_path.mkdir()

                    anchor_waveform = self._mix_stems(
                        [self._right_pad(stems[j][i]) for j in anchor_mix_idxs])
                    positive_waveform = self._mix_stems(
                        [self._right_pad(stems[j][i]) for j in positive_mix_idxs])

                    torch.save(anchor_waveform, example_path / "anchor.pt")
                    torch.save(positive_waveform, example_path / "positive.pt")
            frame_offset += chunk_num_frames // 2

    def _load_file_paths(self) -> None:
        split_dir = (self.root_dir / self.PREPROCESSED_DIR_NAME /
                     self.split).expanduser()
        tracks = split_dir.glob("*/")
        file_paths_df = pd.concat(map(lambda track: pd.DataFrame(
            [[track / "anchor.pt", track / "positive.pt"]], columns=["anchor", "positive"]), tracks), ignore_index=True)
        return file_paths_df

    def __len__(self) -> int:
        return len(self.file_paths_df)

    def __getitem__(self, idx) -> Dict[Literal["anchor", "positive"], torch.Tensor]:
        track_row = self.file_paths_df.iloc[idx]
        anchor_path, positive_path = track_row["anchor"], track_row["positive"]

        anchor, positive = torch.load(anchor_path, map_location="cpu"), torch.load(
            positive_path, map_location="cpu")
        
        noise_tensor = (self.positive_noise * torch.randn(
                1, self.target_sample_rate * self.chunk_duration))

        positive = positive + noise_tensor
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)

        return {"anchor": anchor, "positive": positive}


def get_dataset(
        split: str,
        chunk_duration: int,
        positive_noise: float,
        generate_submixtures: bool,
        transform=None) -> MusdbContrastivePreprocessed:
    """
    Provides a dataset for the MusdbContrastive Dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MusdbContrastivePreprocessed(
        download=True,
        preprocess=True,
        split=split,
        chunk_duration=chunk_duration,
        positive_noise=positive_noise,
        target_sample_rate=16000,
        generate_submixtures=generate_submixtures,
        transform=transform,
        device=device)

    return dataset


if __name__ == "__main__":
    transform = torch.nn.Sequential(
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
    dataset = get_dataset(split="train", chunk_duration=5,
                          positive_noise=0.001, generate_submixtures=True, transform=transform)

    train_dataset, valid_dataset = random_split(
        dataset=dataset, lengths=[0.9, 0.1])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
        persistent_workers=True)

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
        persistent_workers=True)
