"""Moisesdb Contrastive Torch Dataset."""

from typing import Dict, Literal
from pathlib import Path
import random
import logging
import shutil
import json

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import torchaudio.transforms as T

from data.utils import right_pad, mix_down, mix_stems

random.seed(14703)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class MoisesdbContrastivePreprocessed(Dataset):
    """
    Moisesdb Dataset (adapted for contrastive learning): https://github.com/moises-ai/moises-db
    Download Instructions: download the dataset from https://music.ai/research/
    """
    VERSION = "1.0.0"
    SAMPLE_RATE = 44100
    ORIGINAL_DIR_NAME = "moisesdb_v0.1"
    PREPROCESSED_DIR_NAME = "preprocessed_hpss"
    PREPROCESSING_INFO_FILE_NAME = "preprocessing_info.json"

    def __init__(
            self,
            root_dir="~/moisesdb_contrastive",
            preprocess=True,
            chunk_duration=5,
            target_sample_rate=16000,
            generate_submixtures=True,
            device="cpu",
            preprocess_transform=None,
            runtime_transform=None) -> None:

        self.root_dir = Path(root_dir) if isinstance(
            root_dir, str) else root_dir
        self.preprocess = preprocess
        self.chunk_duration = chunk_duration
        self.target_sample_rate = target_sample_rate
        self.generate_submixtures = generate_submixtures
        self.preprocess_transform = preprocess_transform
        self.runtime_transform = runtime_transform

        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset not found. Please download Moisesdb and extract it inside {self.root_dir / self.ORIGINAL_DIR_NAME}.")
        logging.info(
            f"Found original dataset at {(self.root_dir / self.ORIGINAL_DIR_NAME).expanduser()}.")

        if self.preprocess and not self._is_preprocessed():
            self.device = device
            self.resample_transform = T.Resample(
                self.SAMPLE_RATE, self.target_sample_rate).to(self.device)
            self._preprocess_and_save()
        if not self._is_preprocessed():
            raise RuntimeError(
                f"Preprocessed dataset not found. Please use `preprocess=True` to preprocess it.")
        logging.info(
            f"Found preprocessed dataset at {(self.root_dir / self.PREPROCESSED_DIR_NAME).expanduser()}.")

        self.file_paths_df = self._load_file_paths()

    def _is_downloaded_and_extracted(self) -> bool:
        original_dir = (self.root_dir / self.ORIGINAL_DIR_NAME).expanduser()
        return original_dir.exists() and any(original_dir.iterdir())

    def _is_preprocessed(self) -> bool:
        preprocessed_dir = (
            self.root_dir / self.PREPROCESSED_DIR_NAME).expanduser()

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
            self.root_dir / self.PREPROCESSED_DIR_NAME).expanduser()
        preprocessed_dir.mkdir(parents=True)

        preprocessing_info = {
            "chunk_duration": self.chunk_duration,
            "target_sample_rate": self.target_sample_rate,
            "generate_submixtures": self.generate_submixtures
        }

        with open(preprocessed_dir / self.PREPROCESSING_INFO_FILE_NAME, "w") as preprocessing_info_file:
            json.dump(preprocessing_info, preprocessing_info_file)

        original_dir = (self.root_dir / self.ORIGINAL_DIR_NAME).expanduser()

        tracks = original_dir.glob("*/")

        for track in tqdm(tracks, desc="Preprocessing tracks"):
            stems_paths = list(track.glob("*/*.wav"))
            original_track_name = track.name
            chunk_num_frames = self.chunk_duration * self.target_sample_rate

            frame_offset = 0
            for _ in range(2):
                stems = [torch.split(
                    self.resample_transform(mix_down(
                        torchaudio.load(str(stem_path), frame_offset=frame_offset)[0].to(self.device))),
                    split_size_or_sections=chunk_num_frames,
                    dim=1)
                    for stem_path in stems_paths]

                stems_idxs = range(len(stems))
                # Take the min as sometimes stems are not exactly the same size
                for i in range(min(len(stem) for stem in stems)):
                    anchor_mix_size = random.randint(
                        1, len(stems_idxs) - 1) if self.generate_submixtures else 1
                    anchor_mix_idxs = random.sample(
                        stems_idxs, anchor_mix_size)

                    positive_mix_size = random.randint(
                        1, len(stems_idxs) - len(anchor_mix_idxs)) if self.generate_submixtures else 1
                    positive_mix_idxs = random.sample(
                        [stem_idx for stem_idx in stems_idxs if stem_idx not in anchor_mix_idxs], positive_mix_size)

                    anchor_mix_id = '-'.join(str(idx)
                                             for idx in anchor_mix_idxs)
                    positive_mix_id = '-'.join(str(idx)
                                               for idx in positive_mix_idxs)

                    example_id = f"{original_track_name}_chunk{i}_comb{anchor_mix_id}_{positive_mix_id}_shift{frame_offset}"
                    example_path = preprocessed_dir / example_id
                    example_path.mkdir()

                    anchor = mix_stems(
                        [right_pad(stems[j][i], self.chunk_duration * self.target_sample_rate) for j in anchor_mix_idxs])
                    positive = mix_stems(
                        [right_pad(stems[j][i], self.chunk_duration * self.target_sample_rate) for j in positive_mix_idxs])

                    if self.preprocess_transform:
                        anchor = self.preprocess_transform(anchor)
                        positive = self.preprocess_transform(positive)

                    torch.save(anchor, example_path / "anchor.pt")
                    torch.save(positive, example_path / "positive.pt")

                frame_offset += chunk_num_frames // 2

    def _load_file_paths(self) -> None:
        preprocessed_dir = (
            self.root_dir / self.PREPROCESSED_DIR_NAME).expanduser()
        tracks = preprocessed_dir.glob("*/")
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
        
        if self.runtime_transform:
            anchor = self.runtime_transform(anchor)
            positive = self.runtime_transform(positive)

        return {"anchor": anchor, "positive": positive}
