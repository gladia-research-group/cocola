"""CocoChorales Contrastive Torch Dataset (Online Version)."""

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

from data.utils import right_pad, mix_stems,mix_down

random.seed(14703)


class CocoChoralesContrastivePreprocessed(Dataset):
    """
    CocoChorales Dataset (adapted for online contrastive learning without precomputation).
    Reference: https://magenta.tensorflow.org/datasets/cocochorales
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

    def __init__(
            self,
            root_dir="~/coco_chorales_contrastive",
            download=True,
            split="train",
            ensemble="random",
            chunk_duration=5,
            target_sample_rate=16000,
            generate_submixtures=True,
            device="cpu",
            preprocess_transform=None,
            runtime_transform=None,
            samples_per_epoch=100000) -> None:

        self.root_dir = Path(root_dir).expanduser()
        self.download = download
        self.split = split
        self.ensemble = ensemble
        self.chunk_duration = chunk_duration
        self.target_sample_rate = target_sample_rate
        self.generate_submixtures = generate_submixtures
        self.preprocess_transform = preprocess_transform
        self.runtime_transform = runtime_transform
        self.samples_per_epoch = samples_per_epoch
        self.device = device

        if self.split not in ["train", "valid", "test"]:
            raise ValueError("`split` must be one of ['train', 'valid', 'test'].")

        if self.ensemble not in ["random", "brass", "string", "woodwind", "*"]:
            raise ValueError("`ensemble` must be one of ['random', 'brass', 'string', 'woodwind', '*'].")

        # Download if not present
        if self.download and not self._is_downloaded_and_extracted():
            self._download_and_extract()

        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found. Please use `download=True` to download it.")
        logging.info(
            f"Found original dataset split {self.split} at {(self.root_dir / self.ORIGINAL_DIR_NAME / self.split)}.")

        self.resample_transform = T.Resample(
            self.SAMPLE_RATE, self.target_sample_rate)

        # Costruisce l’indice dei brani e degli stems
        self._build_index()

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = self.root_dir / self.ORIGINAL_DIR_NAME / self.split
        return split_dir.exists() and any(split_dir.iterdir())

    def _download_and_extract(self) -> None:
        for i, url in enumerate(self.URLS[self.split]):
            download_and_extract_archive(
                url, self.root_dir / self.ORIGINAL_DIR_NAME / self.split, md5=self.MD5S[self.split][i], remove_finished=True)

    def _build_index(self):
        original_dir = self.root_dir / self.ORIGINAL_DIR_NAME / self.split
        pattern = f"{self.ensemble}_track*" if self.ensemble != "random" else "*_track*"
        tracks = list(original_dir.glob(pattern))
        if not tracks:
            raise RuntimeError(f"No tracks found for ensemble {self.ensemble} in split {self.split}.")

        self.track_index = []
        for track in tqdm(tracks, desc="Building track index"):
            stems_paths = list(track.glob("stems_audio/*.flac")) #TODO before it was .wav
            if not stems_paths:
                continue

            # Supponiamo che tutti gli stems abbiano la stessa durata
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
            raise RuntimeError("No valid tracks found.")

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

        # Calcola un offset casuale
        max_start = max(0, num_frames - chunk_num_frames)
        frame_offset = random.randint(0, max_start) if max_start > 0 else 0

        # Carica i chunk dei singoli stems
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
            #waveform = mix_down(waveform) #TODO before it was uncommented
            stems.append(waveform)

            #stems.append(waveform)

        stems_idxs = list(range(len(stems)))

        # Selezione casuale degli stems per anchor e positive
        if self.generate_submixtures and len(stems_idxs) > 1:
            anchor_mix_size = random.randint(1, len(stems_idxs) - 1)
            anchor_mix_idxs = random.sample(stems_idxs, anchor_mix_size)
            positive_mix_size = random.randint(1, len(stems_idxs) - len(anchor_mix_idxs))
            positive_mix_idxs = random.sample([idx for idx in stems_idxs if idx not in anchor_mix_idxs], positive_mix_size)
        else:
            # Caso base: 1 stem per anchor e uno diverso per positive (se possibile)
            anchor_mix_idxs = random.sample(stems_idxs, 1)
            remaining = [idx for idx in stems_idxs if idx not in anchor_mix_idxs]
            if remaining:
                positive_mix_idxs = random.sample(remaining, 1)
            else:
                positive_mix_idxs = anchor_mix_idxs  # Se c'è un solo stem

        anchor = mix_stems([right_pad(stems[j], chunk_num_frames) for j in anchor_mix_idxs])
        positive = mix_stems([right_pad(stems[j], chunk_num_frames) for j in positive_mix_idxs])

        if self.preprocess_transform:
            anchor = self.preprocess_transform(anchor)
            positive = self.preprocess_transform(positive)

        item = {"anchor": anchor.cpu(), "positive": positive.cpu()}

        if self.runtime_transform:
            item = self.runtime_transform(item)

        return item
