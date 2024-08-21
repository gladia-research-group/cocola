import os
import torch
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import Trainer
import numpy as np
from contrastive_model.contrastive_model import CoCola
from data.coco_chorales_contrastive_preprocessed import CocoChoralesContrastivePreprocessed
from data.moisesdb_contrastive_preprocessed import MoisesdbContrastivePreprocessed
from data.slakh2100_contrastive_preprocessed import Slakh2100ContrastivePreprocessed
from data.musdb_contrastive_preprocessed import MusdbContrastivePreprocessed
from contrastive_model import constants
import librosa 
import torchaudio
import random

random.seed(42)

def hpss(waveform, sample_rate=16000):
    x = waveform.squeeze(0).cpu().numpy()
    stft_x = librosa.stft(x,
                          n_fft=1024,
                          win_length=400,
                          hop_length=160)
    harmonic_stft_x, percussive_stft_x = librosa.decompose.hpss(stft_x)
    mel_harmonic_x = librosa.feature.melspectrogram(S=np.abs(harmonic_stft_x)**2,
                                                    sr=sample_rate,
                                                    fmin=60.0,
                                                    fmax=7800.0,
                                                    n_mels=64)
    mel_percussive_x = librosa.feature.melspectrogram(S=np.abs(percussive_stft_x)**2,
                                                      sr=sample_rate,
                                                      fmin=60.0,
                                                      fmax=7800.0,
                                                      n_mels=64)
    mel_db_harmonic_x = librosa.power_to_db(mel_harmonic_x, ref=np.max)
    mel_db_percussive_x = librosa.power_to_db(mel_percussive_x, ref=np.max)
    processed_x = np.stack((mel_db_harmonic_x, mel_db_percussive_x), axis=0)
    processed_x = torch.from_numpy(processed_x)
    return processed_x
 
def pitch_shift_hpss(data, sample_rate=16000):
    x, y = data["anchor"], data["positive"]
    n_steps = random.choice([1, 6, 8, 10, 11, -1, -6, -8, -10, -11])
    y = torchaudio.functional.pitch_shift(y, sample_rate=sample_rate, n_steps=n_steps)
    processed_x = hpss(x)
    processed_y = hpss(y)
    processed = {
        "anchor": processed_x,
        "positive": processed_y
    }
    return processed


CHECKPOINT = '/speech/dbwork/mul/spielwiese3/demancum/cocola_hpss/ciflwfwc/checkpoints/epoch=59-step=377880.ckpt' #astral-valley-21 more_negative_RAND_MASK_DOUBLE_CHANNEL

model = CoCola.load_from_checkpoint(CHECKPOINT)
trainer = Trainer()


# Using train split since it is not used from training
test_dataset_shifted = MusdbContrastivePreprocessed(
    root_dir='/disk1/demancum/musdb_contrastive',
    download=False,
    preprocess=True, # Need to preprocess now because it is not used at training time
    split="train",
    chunk_duration=5,
    target_sample_rate=16000,
    generate_submixtures=True,
    device="cuda",
    #transform=hpss
    runtime_transform=pitch_shift_hpss
)

test_dataloader = DataLoader(
    test_dataset_shifted,
    batch_size=2,
    shuffle=False,
    drop_last=True,
    num_workers=os.cpu_count(),
    persistent_workers=True
    )

print("PERCUSSIVE RESULTS")

model.set_embedding_mode(constants.EmbeddingMode.PERCUSSIVE)
trainer.test(model=model, dataloaders=test_dataloader)
