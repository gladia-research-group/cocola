import random
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from typing import Dict, Literal
from pathlib import Path
import logging
import time

from data.utils import right_pad, mix_down, mix_stems

class MoisesdbContrastivePreprocessed(Dataset):
    """
    Moisesdb Dataset with random overlap between chunks for contrastive learning.
    """

    VERSION = "1.0.0"
    SAMPLE_RATE = 44100
    ORIGINAL_DIR_NAME = "moisesdb_v0.1"

    def __init__(
        self,
        root_dir="~/moisesdb_contrastive",
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

        if self.split not in ["train", "valid", "test"]:
            raise ValueError("`split` must be one of ['train', 'valid', 'test'].")

        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found.")
        logging.info(
            f"Found original dataset split {self.split} at {(self.root_dir / self.ORIGINAL_DIR_NAME / self.split)}.")

        self.resample_transform = T.Resample(
            self.SAMPLE_RATE, self.target_sample_rate)

        self._build_index()

        # Create a separate RNG for validation if needed
        if self.split == "valid":
            self.rng = random.Random(self.seed_val)
        else:
            self.rng = random

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = self.root_dir / self.ORIGINAL_DIR_NAME / self.split
        return split_dir.exists() and any(split_dir.iterdir())

    def _build_index(self):
        original_dir = self.root_dir / self.ORIGINAL_DIR_NAME / self.split
        tracks = list(original_dir.glob("*"))
        if not tracks:
            raise RuntimeError(f"No tracks found in split {self.split}.")

        self.track_index = []
        for track in tqdm(tracks, desc="Building track index"):

            stems_paths = list(track.glob("*/*.wav")) 
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
        elif self.split == "valid":
            return self.samples_per_epoch_val
        else:
            # You could define a custom length for test or just return all tracks
            return len(self.track_index)

    def __getitem__(self, idx):
        if self.split == "train":
            track_info = self.rng.choice(self.track_index)
            return self._get_item_from_track(track_info)
        elif self.split == "valid":
            track_info = self.track_index[idx % len(self.track_index)]
            return self._get_item_from_track(track_info, idx=idx)
        else:
            track_info = self.track_index[idx % len(self.track_index)]
            return self._get_item_from_track(track_info, idx=idx)


    def _get_item_from_track(self, track_info, idx=None):
        stems_paths = track_info['stems_paths']
        sample_rate = track_info['sample_rate']
        num_frames = track_info['num_frames']
        chunk_num_frames = int(self.chunk_duration * sample_rate)

        max_start_frame = max(num_frames - chunk_num_frames, 0)
        if self.split == "valid" and idx is not None:
            # Usa un generatore "seedato" per ottenere lo stesso offset ogni volta
            rng_offset = random.Random(self.seed_val + idx)
            frame_offset = rng_offset.randint(0, max_start_frame)
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
                raise RuntimeError(f"Error loading {stem_path}: {e}")

            if sr != self.target_sample_rate:
                waveform = self.resample_transform(waveform)
                chunk_num_frames = int(self.chunk_duration * self.target_sample_rate)

            stems.append(waveform)
            # Aggiungi il waveform due volte solo in fase di training
            if self.split == "train":
                stems.append(waveform)

        # Prosegui con la logica per generare mix anchor e positive...
        if self.generate_submixtures and len(stems) > 1:
            stems_idxs = list(range(len(stems)))
            anchor_mix_size = random.randint(1, len(stems_idxs) - 1)
            anchor_mix_idxs = random.sample(stems_idxs, anchor_mix_size)
            positive_mix_size = random.randint(1, len(stems_idxs) - len(anchor_mix_idxs))
            positive_mix_idxs = random.sample(
                [i for i in stems_idxs if i not in anchor_mix_idxs],
                positive_mix_size
            )
        else:
            stems_idxs = list(range(len(stems)))
            anchor_mix_idxs = random.sample(stems_idxs, 1)
            positive_mix_idxs = random.sample(
                [i for i in stems_idxs if i not in anchor_mix_idxs], 1
            )

        anchor = mix_stems([right_pad(stems[j], chunk_num_frames) for j in anchor_mix_idxs])
        positive = mix_stems([right_pad(stems[j], chunk_num_frames) for j in positive_mix_idxs])

        if self.preprocess_transform:
            anchor = self.preprocess_transform(anchor)
            positive = self.preprocess_transform(positive)

        item = {"anchor": anchor.cpu(), "positive": positive.cpu()}

        if self.runtime_transform:
            item = self.runtime_transform(item)
        return item
