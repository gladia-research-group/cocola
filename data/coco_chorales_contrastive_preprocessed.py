"""CocoChorales Contrastive Torch Dataset."""

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
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive
import torchaudio
import torchaudio.transforms as T

random.seed(14703)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class CocoChoralesContrastivePreprocessed(Dataset):
    """
    CocoChorales Dataset (adapted for contrastive learning): https://magenta.tensorflow.org/datasets/cocochorales
    """

    VERSION = "1.0.0"
    URLS = {
        "train": [f"https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/train/{i}.tar.bz2" for i in [1, 2, 3, 25, 26, 27, 49, 50, 51, 73, 74, 75]],
        "test": [f"https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/test/{i}.tar.bz2" for i in [1, 4, 7, 10]],
        "valid": [f"https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/valid/{i}.tar.bz2" for i in [1, 4, 7, 10]]
    }
    MD5S = {
        "train": [
            "999ba8284b0646a6c7f3ef357e15fd59",
            "f1b6ae484940d66ec006c0599d8b0f48",
            "b2237240c49d3537872d35d98199fdc6",
            "e540dc37fcb47f75995544df3720af3f",
            "7490eb20468f421313bab7882f59c9cf",
            "200eb27e786d27d04347129d10a7731b",
            "358817b12ee126e697f14ef6805cdc48",
            "96e81212eeb8b65619103dd16094a08f",
            "32799d360b9b9764b511d399327509e0",
            "0fa937613c947d0cc18d2d4682504fa0",
            "e5c50a10b0b2af5ee26867c108a94a92",
            "f78dfe2f212e4991a78be7e8e4e98fc5",
        ],
        "test": [
            "2c9e617b9f3ec622e0da35734036af49",
            "461fc00182c5e312ac379d97df4bceb6",
            "f808fc2502059e9a994cea85ccd4d3a0",
            "afebac996cd3d643b7c99d575a3ad048"
        ],
        "valid": [
            "697766f8e53ffc9f64708b8bf4acedb1",
            "4edd6803d082dc090f08823cc003cc94",
            "502128334a38e682ac0a06682207d13b",
            "ded860cabdf005eafde1095ebab7787e",
        ]
    }
    SAMPLE_RATE = 16000
    ORIGINAL_DIR_NAME = "original"
    PREPROCESSED_DIR_NAME = "preprocessed"
    PREPROCESSING_INFO_FILE_NAME = "preprocessing_info.json"

    def __init__(
            self,
            root_dir="~/coco_chorales_contrastive",
            download="false",
            preprocess="false",
            split="train",
            ensemble="random",
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
        self.ensemble = ensemble
        self.chunk_duration = chunk_duration
        self.positive_noise = positive_noise
        self.target_sample_rate = target_sample_rate
        self.generate_submixtures = generate_submixtures
        self.transform = transform

        if self.split not in ["train", "valid", "test"]:
            raise ValueError(
                "`split` must be one of ['train', 'valid', 'test'].")

        if self.ensemble not in ["random", "brass", "string", "woodwind", "*"]:
            raise ValueError(
                "`ensemble` must be one of ['random', 'brass', 'string', 'woodwind', '*'].")

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
        for i, url in enumerate(self.URLS[self.split]):
            download_and_extract_archive(
                url, self.root_dir / self.ORIGINAL_DIR_NAME / self.split, md5=self.MD5S[self.split][i], remove_finished=True)

    def _right_pad(self, waveform):
        num_frames = waveform.shape[1]
        expected_num_frames = self.chunk_duration * self.target_sample_rate
        if num_frames < expected_num_frames:
            num_missing_frames = expected_num_frames - num_frames
            last_dim_padding = (0, num_missing_frames)
            waveform = torch.nn.functional.pad(waveform, last_dim_padding)
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

        tracks = original_dir.glob(f"{self.ensemble}_track*/")

        for track in tqdm(tracks, desc="Preprocessing tracks"):
            stems_paths = list(track.glob("stems_audio/*.wav"))
            original_track_name = track.name
            chunk_num_frames = self.chunk_duration * self.target_sample_rate
            frame_offset = 0
            for _ in range(2):
                stems = [torch.split(
                    self.resample_transform(torchaudio.load(
                        stem_path)[0].to(self.device)),
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
        tracks = split_dir.glob(f"{self.ensemble}_track*")
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
        ensemble: str,
        chunk_duration: int,
        positive_noise: float,
        generate_submixtures: bool,
        transform=None) -> CocoChoralesContrastivePreprocessed:
    """
    Provides a dataloader for the CocoChoralesContrastive Dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CocoChoralesContrastivePreprocessed(
        download=True,
        preprocess=True,
        split=split,
        ensemble=ensemble,
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
    train_dataset = get_dataset(split="train", ensemble="*",
                                chunk_duration=5, positive_noise=0.001, generate_submixtures=True, transform=transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
        persistent_workers=True)

    valid_dataset = get_dataset(split="valid", ensemble="*",
                                chunk_duration=5, positive_noise=0.001, generate_submixtures=True, transform=transform)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
        persistent_workers=True)
