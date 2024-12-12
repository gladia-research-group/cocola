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
            download=True,
            split="train",
            chunk_duration=5,
            target_sample_rate=16000,
            generate_submixtures=True,
            device="cpu",
            preprocess_transform=None,
            runtime_transform=None,
            samples_per_epoch=10000) -> None:

        self.root_dir = Path(root_dir).expanduser()
        self.download = download
        self.split = split
        self.chunk_duration = chunk_duration
        self.target_sample_rate = target_sample_rate
        self.generate_submixtures = generate_submixtures
        self.preprocess_transform = preprocess_transform
        self.runtime_transform = runtime_transform
        self.samples_per_epoch = samples_per_epoch
        self.device = device

        if self.split not in ["train", "test", "validation"]:
            raise ValueError("`split` must be one of ['train', 'test', 'validation'].")

        # Scarica il dataset se non presente
        if self.download and not self._is_downloaded_and_extracted():
            self._download_and_extract()

        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found. Please set `download=True` or place the data properly.")
        logging.info(
            f"Found original dataset split {self.split} at {(self.root_dir / self.ORIGINAL_DIR_NAME / 'slakh2100_redux_16k' / self.split)}.")

        self.resample_transform = T.Resample(
            self.SAMPLE_RATE, self.target_sample_rate)

        # Costruisce l'indice dei brani (track e stems)
        self._build_index()

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = (self.root_dir / self.ORIGINAL_DIR_NAME / "slakh2100_redux_16k" /
                     self.split)
        return split_dir.exists() and any(split_dir.iterdir())

    def _download_and_extract(self) -> None:
        download_and_extract_archive(
            self.URL, self.root_dir / self.ORIGINAL_DIR_NAME, remove_finished=True)

    def _build_index(self):
        original_dir = (self.root_dir / self.ORIGINAL_DIR_NAME /
                        "slakh2100_redux_16k" / self.split)
        tracks = list(original_dir.glob("*/"))
        if not tracks:
            raise RuntimeError(f"No tracks found in split {self.split}.")

        self.track_index = []
        for track in tqdm(tracks, desc="Building track index"):
            # Carica tutti gli stems .flac nella cartella stems
            stems_paths = list(track.glob("stems/S*.flac"))
            if not stems_paths:
                continue

            # Assume che tutti gli stems abbiano la stessa durata
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
            raise RuntimeError("No valid tracks found in the given split.")

    def __len__(self) -> int:
        return self.samples_per_epoch

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
        chunk_num_frames = int(self.chunk_duration * sample_rate)
        num_frames = track_info['num_frames']

        # Offset casuale del chunk
        max_start_frame = max(0, num_frames - chunk_num_frames)
        frame_offset = random.randint(0, max_start_frame) if max_start_frame > 0 else 0

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

        # Scelta casuale degli stems per anchor e positive
        if self.generate_submixtures and len(stems_idxs) > 1:
            anchor_mix_size = random.randint(1, len(stems_idxs) - 1)
            anchor_mix_idxs = random.sample(stems_idxs, anchor_mix_size)
            positive_mix_size = random.randint(1, len(stems_idxs) - len(anchor_mix_idxs))
            positive_mix_idxs = random.sample([idx for idx in stems_idxs if idx not in anchor_mix_idxs],
                                              positive_mix_size)
        else:
            # Caso semplice: se non si generano submixtures, si prende 1 stem per anchor e 1 per positive (se disponibile)
            anchor_mix_idxs = random.sample(stems_idxs, 1)
            remaining = [idx for idx in stems_idxs if idx not in anchor_mix_idxs]
            if remaining:
                positive_mix_idxs = random.sample(remaining, 1)
            else:
                positive_mix_idxs = anchor_mix_idxs  # Se c'è un solo stem, usa quello sia per anchor che positive

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
