import os
import torch
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Convert audio dataset')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
parser.add_argument('--root_dir', type=str, default="/disk1/demancum", help='Root directory')
parser.add_argument('--target_sr', type=int, default=16000, help='Target sampling rate')
args = parser.parse_args()

dataset = args.dataset
root_dir = args.root_dir
target_sr = args.target_sr

output_dir = os.path.join("/speech/dbwork/mul/spielwiese3/demancum/hp_datasets", dataset)
input_dir = os.path.join(root_dir, dataset)

file_paths = []
for root, dirs, files in os.walk(input_dir):
    # Filter out directories starting with '._'
    dirs[:] = [d for d in dirs if not d.startswith('._')]

    for f in files:
        # Skip files starting with '._'
        if f.startswith('._'):
            continue
        if f.lower().endswith(('.wav', '.flac', '.mp3')):
            file_paths.append(os.path.join(root, f))

for input_path in tqdm(file_paths, desc="Processing files"):
    rel_path = os.path.relpath(input_path, input_dir)
    
    # Changed: always save output as .wav instead of .flac
    output_path = os.path.join(output_dir, rel_path)
    output_wav_path = output_path.replace(os.path.splitext(output_path)[1], '.wav')  # Changed

    # Check if output already exists
    if os.path.exists(output_wav_path):
        # Skip processing
        continue

    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

    waveform, sr = torchaudio.load(input_path)

    if sr != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = transform(waveform)
        sr = target_sr

    # Downmix to mono if more than one channel exists
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    y = waveform.squeeze().numpy()

    # Perform harmonic-percussive source separation
    stft = librosa.stft(y)
    harmonic_stft, percussive_stft = librosa.decompose.hpss(stft)

    # Inverse STFT for harmonic and percussive components
    y_harm = librosa.istft(harmonic_stft)
    y_perc = librosa.istft(percussive_stft)

    # Create a 2-channel array: [harmonic, percussive]
    stereo = np.vstack((y_harm, y_perc))

    # Changed: save as WAV (2 channels) instead of FLAC
    torchaudio.save(
        output_wav_path,
        torch.tensor(stereo),
        sample_rate=sr,
        format="wav"  # Changed
    )
