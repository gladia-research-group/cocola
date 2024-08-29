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
 
def speed_change_hpss(data, sample_rate=16000, target_chunk_duration=5):
    x, y = data["anchor"], data["positive"]

    factor = random.choice([0.25, 0.50, 0.75, 1.25, 1.50, 1.75]) #factor = random.choice([0.25, 0.50, 0.75, 1.25, 1.50, 1.75]) #factor = random.choice([1.50])
    y, _ = torchaudio.functional.speed(y, sample_rate, factor)

    processed_x = hpss(x[:, :sample_rate*target_chunk_duration])
    processed_y = hpss(y[:, :sample_rate*target_chunk_duration])
    processed = {
        "anchor": processed_x,
        "positive": processed_y
    }
    return processed


CHECKPOINT = '/speech/dbwork/mul/spielwiese3/demancum/cocola_hpss/10izn5cw/checkpoints/checkpoint-epoch=119-val_loss=0.00.ckpt' #young-sun-27 No_double_neg_HPSS

model = CoCola.load_from_checkpoint(CHECKPOINT)
trainer = Trainer()


# Using train split since it is not used from training
test_dataset_shifted = MusdbContrastivePreprocessed(
    root_dir='/disk1/demancum/musdb_contrastive',
    download=False,
    preprocess=True, # Need to preprocess now because it is not used at training time
    split="train",
    test="speed_change",
    chunk_duration=15,
    target_sample_rate=16000,
    generate_submixtures=True,
    device="cuda",
    #transform=hpss
    runtime_transform=speed_change_hpss
)

test_dataloader = DataLoader(
    test_dataset_shifted,
    batch_size=2,
    shuffle=False,
    drop_last=True,
    num_workers=os.cpu_count(),
    persistent_workers=True
    )

print("HARMONIC RESULTS")

model.set_embedding_mode(constants.EmbeddingMode.HARMONIC)
trainer.test(model=model, dataloaders=test_dataloader)
