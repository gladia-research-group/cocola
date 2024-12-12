import random
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from typing import Dict, Literal
from pathlib import Path
import logging


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
            samples_per_epoch=10000) -> None:

        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.chunk_duration = chunk_duration
        self.target_sample_rate = target_sample_rate
        self.generate_submixtures = generate_submixtures
        self.preprocess_transform = preprocess_transform
        self.runtime_transform = runtime_transform
        self.device = device
        self.samples_per_epoch = samples_per_epoch  # Total number of samples per epoch

        if self.split not in ["train", "valid", "test"]:
            raise ValueError(
                "`split` must be one of ['train', 'valid', 'test'].")

        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found.")
        logging.info(
            f"Found original dataset split {self.split} at {(self.root_dir / self.ORIGINAL_DIR_NAME / self.split)}.")

        self.resample_transform = T.Resample(
            self.SAMPLE_RATE, self.target_sample_rate)

        self._build_index()

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
                'sample_rate': sample_rate,
            })

        if not self.track_index:
            raise RuntimeError(f"No valid tracks found in split {self.split}.")

    def __len__(self) -> int:
        return self.samples_per_epoch  # Define the number of samples per epoch

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        max_retries = 5
        for _ in range(max_retries):
            track_info = random.choice(self.track_index)
            try:
                return self._get_item_from_track(track_info)
            except Exception as e:
                print(f"Error loading track: {e}")
                # Retry with another track
        raise RuntimeError("Could not find a valid sample after multiple retries.")

    def _get_item_from_track(self, track_info):
        stems_paths = track_info['stems_paths']
        #print(stems_paths)
        sample_rate = track_info['sample_rate']
        num_frames = track_info['num_frames']
        chunk_num_frames = int(self.chunk_duration * sample_rate)

        # Randomly select a starting frame
        max_start_frame = num_frames - chunk_num_frames
        if max_start_frame <= 0:
            frame_offset = 0
        else:
            frame_offset = random.randint(0, max_start_frame)

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
            waveform = mix_down(waveform)
            stems.append(waveform)

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
