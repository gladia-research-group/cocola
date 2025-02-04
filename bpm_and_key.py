import os
import csv
import torchaudio
from deeprhythm import DeepRhythmPredictor
from tqdm import tqdm
import torch

root_dir = "/speech/dbwork/mul/spielwiese4/students/demancum/moisesdb/moisesdb/moisesdb_v0.1"
output_csv = "/speech/dbwork/mul/spielwiese3/demancum/dataset_stats/moisesdb/bpm_confidence.csv"

model = DeepRhythmPredictor()

song_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
             if os.path.isdir(os.path.join(root_dir, d))]

with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["song", "bpm", "confidence"])
    
    for song_dir in tqdm(song_dirs, desc="Processing songs"):
        instrument_dirs = [os.path.join(song_dir, d) for d in os.listdir(song_dir) 
                           if os.path.isdir(os.path.join(song_dir, d))]
        
        waveforms = []
        sr = None
        
        # Carica tutti i file wav della canzone
        for instr_dir in instrument_dirs:
            wav_files = [os.path.join(instr_dir, w) for w in os.listdir(instr_dir) 
                         if w.endswith(".wav")]
            
            for wav_file in wav_files:
                waveform, current_sr = torchaudio.load(wav_file)
                if sr is None:
                    sr = current_sr
                else:
                    if current_sr != sr:
                        raise ValueError(f"Sample rate differente per il file {wav_file}")
                waveforms.append(waveform)
        
        if len(waveforms) == 0:
            # Nessun waveform trovato per questa canzone
            continue
        
        # Trova la lunghezza massima
        max_length = max(w.shape[1] for w in waveforms)
        
        # Esegui padding dei waveform più corti
        padded_waveforms = []
        for w in waveforms:
            length = w.shape[1]
            if length < max_length:
                pad_amount = max_length - length
                w_padded = torch.cat([w, torch.zeros((w.shape[0], pad_amount))], dim=1)
            else:
                w_padded = w
            padded_waveforms.append(w_padded)
        
        # Somma tutti i waveform
        mixture_waveform = torch.zeros_like(padded_waveforms[0])
        for w in padded_waveforms:
            mixture_waveform += w

        if mixture_waveform.size(0) > 1:
            mixture_waveform = mixture_waveform.mean(dim=0)
        
        mixture_waveform = mixture_waveform.view(-1)

        bpm, confidence = model.predict_from_audio(mixture_waveform, sr, include_confidence=True)
        song_name = os.path.basename(song_dir)
        writer.writerow([song_name, bpm, confidence])
