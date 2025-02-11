"""Slakh2100 Contrastive Torch Dataset (Online Version)."""

from typing import Dict, Literal
from pathlib import Path
import random
import logging

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

from data.utils import right_pad, mix_stems, mix_down

random.seed(14703)

class Slakh2100ContrastivePreprocessed(Dataset):
    """
    Slakh2100 Dataset (adapted for contrastive learning, online loading):
    http://www.slakh.com
    """

    VERSION = "1.0.0"
    URL = "https://zenodo.org/records/7708270/files/slakh2100_redux_16k.tar.gz"
    SAMPLE_RATE = 16000
    ORIGINAL_DIR_NAME = "original"

    def __init__(
            self,
            root_dir="~/slakh2100_contrastive",
            split="train",
            chunk_duration=5,
            target_sample_rate=16000,
            generate_submixtures=True,
            device="cpu",
            preprocess_transform=None,
            runtime_transform=None,
            samples_per_epoch_train=10000,
            samples_per_epoch_val=320,
            seed_val=42
        ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.chunk_duration = chunk_duration
        self.target_sample_rate = target_sample_rate
        self.generate_submixtures = generate_submixtures
        self.preprocess_transform = preprocess_transform
        self.runtime_transform = runtime_transform
        self.device = device

        self.samples_per_epoch_train = samples_per_epoch_train
        self.samples_per_epoch_val = samples_per_epoch_val
        self.seed_val = seed_val

        if self.split not in ["train", "test", "validation"]:
            raise ValueError("`split` must be one of ['train', 'test', 'validation'].")

        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found.")
        logging.info(
            f"Found original dataset split {self.split} at {(self.root_dir / self.ORIGINAL_DIR_NAME / 'slakh2100_redux_16k' / self.split)}.")

        self.resample_transform = T.Resample(
            self.SAMPLE_RATE, self.target_sample_rate)

        self._build_index()
        
        # Create a separate RNG for validation if needed
        if self.split == "validation":
            self.rng = random.Random(self.seed_val)
        else:
            self.rng = random

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = (self.root_dir / self.ORIGINAL_DIR_NAME / "slakh2100_redux_16k" /self.split)
        return split_dir.exists() and any(split_dir.iterdir())

    def _build_index(self):
        original_dir = (self.root_dir / self.ORIGINAL_DIR_NAME /"slakh2100_redux_16k" / self.split)
        tracks = list(original_dir.glob("*/"))
        if not tracks:
            raise RuntimeError(f"No tracks found in split {self.split}.")

        self.track_index = []
        for track in tqdm(tracks, desc="Building track index"):

            stems_paths = list(track.glob("stems/S*.wav"))
            if not stems_paths:
                continue

            # Get total number of frames (assuming all stems have the same duration)
            info = torchaudio.info(str(stems_paths[0]))
            num_frames = info.num_frames
            sample_rate = info.sample_rate

            self.track_index.append({
                'track_name': track.name,
                'stems_paths': stems_paths,
                'num_frames': num_frames,
                'sample_rate': sample_rate
            })

        if not self.track_index:
            raise RuntimeError(f"No valid tracks found in split {self.split}.")

    def __len__(self) -> int:
        # For training, use samples_per_epoch_train
        # For validation, use samples_per_epoch_val
        # (or do something else for test)
        if self.split == "train":
            return self.samples_per_epoch_train
        elif self.split == "validation":
            return self.samples_per_epoch_val
        else:
            # You could define a custom length for test or just return all tracks
            return len(self.track_index)

    def __getitem__(self, idx):
        """
        For training:
          - pick a random track from self.track_index
          - pick a random chunk within that track
        For validation (deterministic):
          - use idx to pick which track
          - use a seeded random or a simple formula for chunk offset
        """
        if self.split == "train":
            # Random track each time
            track_info = self.rng.choice(self.track_index)
            # Then get item from track with a random offset
            return self._get_item_from_track(track_info, random_offset=True)
        elif self.split == "validation":
            # Option 1: cycle through tracks in a deterministic way
            # track_info = self.track_index[idx % len(self.track_index)]
            # Option 2: choose a random track but in a deterministic manner
            #           by seeding rng with idx + self.seed_val
            # We'll do Option 1 for a simpler approach
            track_info = self.track_index[idx % len(self.track_index)]
            return self._get_item_from_track(track_info, random_offset=True, deterministic=True, idx=idx)

    def _get_item_from_track(self, track_info, random_offset=False, deterministic=False, idx=None):

        stems_paths = track_info['stems_paths']

        sample_rate = track_info['sample_rate']
        num_frames = track_info['num_frames']
        chunk_num_frames = int(self.chunk_duration * sample_rate)


        if random_offset:
            max_start_frame = max(num_frames - chunk_num_frames, 0)
            if deterministic and idx is not None:
                # Example: a repeatable “random” offset from idx
                # (so it's the same each epoch)
                offset_rng = random.Random(self.seed_val + idx)
                frame_offset = offset_rng.randint(0, max_start_frame)
            else:
                frame_offset = self.rng.randint(0, max_start_frame)
        stems = []
        for stem_path in stems_paths:
            try:
                waveform, sr = torchaudio.load(
                    str(stem_path),
                    frame_offset=frame_offset,
                    num_frames=chunk_num_frames
                )
            except Exception as e:
                # If a stem fails to load, raise an error to trigger the retry logic
                raise RuntimeError(f"Error loading {stem_path}: {e}")

            if sr != self.target_sample_rate:
                waveform = self.resample_transform(waveform)
                chunk_num_frames = int(self.chunk_duration * self.target_sample_rate)
            
            # Mix down to mono
            #waveform = mix_down(waveform) #TODO uncomment if you want to train w/o HPSS
            if self.split == "train":
                stems.append(waveform)
                stems.append(waveform)
            else:
                stems.append(waveform)


        stems_idxs = list(range(len(stems)))

        if self.generate_submixtures and len(stems_idxs) > 1:
            anchor_mix_size = random.randint(1, len(stems_idxs) - 1)
            anchor_mix_idxs = random.sample(stems_idxs, anchor_mix_size)

            positive_mix_size = random.randint(1, len(stems_idxs) - len(anchor_mix_idxs))
            positive_mix_idxs = random.sample(
                [idx for idx in stems_idxs if idx not in anchor_mix_idxs], positive_mix_size)
        else:
            anchor_mix_idxs = random.sample(stems_idxs, 1)
            positive_mix_idxs = random.sample(
                [idx for idx in stems_idxs if idx not in anchor_mix_idxs], 1)

        anchor = mix_stems(
            [right_pad(stems[j], chunk_num_frames) for j in anchor_mix_idxs])
        positive = mix_stems(
            [right_pad(stems[j], chunk_num_frames) for j in positive_mix_idxs])

        if self.preprocess_transform:
            anchor = self.preprocess_transform(anchor)
            positive = self.preprocess_transform(positive)

        item = {"anchor": anchor.cpu(), "positive": positive.cpu()}

        if self.runtime_transform:
            item = self.runtime_transform(item)

        
        return item
