"""Slakh2100 Contrastive Torch Dataset."""

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


class Slakh2100ContrastivePreprocessed(Dataset):
    """
    Slakh2100 Dataset (adapted for contrastive learning): http://www.slakh.com
    """
    VERSION = "1.0.0"
    URL = "https://zenodo.org/records/7708270/files/slakh2100_redux_16k.tar.gz"
    SAMPLE_RATE = 44100
    ORIGINAL_DIR_NAME = "slakh2100_wav"
    PREPROCESSED_DIR_NAME = "slakh2100_wav/preprocessed"
    PREPROCESSING_INFO_FILE_NAME = "preprocessing_info.json"

    def __init__(
            self,
            root_dir="/speech/dbwork/mul/spielwiese4/students/demancum/",
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

        if self.split not in ["train", "test", "validation"]:
            raise ValueError(
                "`split` must be one of ['train', 'test', 'validation].")

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

        #self.file_paths_df = self._load_file_paths()

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = (self.root_dir / self.ORIGINAL_DIR_NAME/
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

    def _mix_stems(self, stems):
        return sum(stems)

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
            #stems_paths = list(track.glob("*.wav"))
            instrument_dirs = list(track.glob("*"))
            filtered_instrument_dirs = [path for path in instrument_dirs if path.stem != "mixture"]

            original_track_name = track.name
            chunk_num_frames = self.chunk_duration * self.target_sample_rate
            frame_offset = 0
            for _ in range(2):
                stems = [torch.split(
                    self.resample_transform(
                        torchaudio.load(str(stem_path), frame_offset=frame_offset)[0].to(self.device)),
                    split_size_or_sections=chunk_num_frames,
                    dim=1)
                    for stem_path in filtered_instrument_dirs]

                stems_idxs = range(len(stems))
                for i in range(min(len(stem) for stem in stems)):
                    anchor_mix_size = random.randint(
                        1, len(stems_idxs)-1) if self.generate_submixtures else 1
                    #positive_mix_size = random.randint(
                    #    1, len(stems_idxs) // 2) if self.generate_submixtures else 1
                    anchor_mix_idxs = random.sample(
                        stems_idxs, anchor_mix_size)
                    #positive_mix_idxs = random.sample(
                    #    [stem_idx for stem_idx in stems_idxs if stem_idx not in anchor_mix_idxs], positive_mix_size)
                    
                    anchor_mix_id = '-'.join(Path(filtered_instrument_dirs[idx].name).stem for idx in anchor_mix_idxs)
                    #positive_mix_id = '-'.join(instrument_dirs[idx].name for idx in positive_mix_idxs)
                    
                    others_mix_idxs = [stem_idx for stem_idx in stems_idxs if stem_idx not in anchor_mix_idxs]

                    #others_mix_id = [instrument_dirs[idx].name for idx in [stem_idx for stem_idx in stems_idxs if stem_idx not in anchor_mix_idxs]]

                    others_mix_id = [Path(filtered_instrument_dirs[idx].name).stem for idx in others_mix_idxs]

                    

                    example_id = f"{original_track_name}_chunk{i}_shift{frame_offset}"
                    example_path = preprocessed_dir / example_id
                    example_path.mkdir()

                    anchor_waveform = self._mix_stems(
                        [self._right_pad(stems[j][i]) for j in anchor_mix_idxs])
                    #positive_waveform = self._mix_stems(
                    #    [self._right_pad(stems[j][i]) for j in positive_mix_idxs])

                    
                    with open(example_path / "remaining_stems.json", "w") as final: json.dump(others_mix_id, final)
                    
                    for idx, j in enumerate(others_mix_idxs):
                        other_waveform = self._right_pad(stems[j][i])
                        torchaudio.save(str(example_path / f"{others_mix_id[idx]}.wav"), other_waveform.to("cpu"), self.target_sample_rate)

                    torchaudio.save(str(example_path / f"mix_{anchor_mix_id}.wav"), anchor_waveform.to("cpu"), self.target_sample_rate)
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
        transform=None) -> Slakh2100ContrastivePreprocessed:
    """
    Provides a dataset for the Slakh2100Contrastive Dataset.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Slakh2100ContrastivePreprocessed(
        download=False,
        preprocess=True,
        split=split,
        chunk_duration=chunk_duration,
        positive_noise=positive_noise,
        target_sample_rate=44100, #CHECK SAMPLE RATE
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
    train_dataset = get_dataset(
        split="test", chunk_duration=20, positive_noise=0.0, generate_submixtures=True, transform=None) #CHECK CHUNK DURATION
    

