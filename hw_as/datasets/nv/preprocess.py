from scipy.io import wavfile
from tqdm import tqdm

import librosa
import numpy as np
import os
import torch

from hw_as.utils.text import _clean_text


def preprocess_wavs_and_texts(raw_data_dir, data_dir, sample_rate, max_wav_length):
    print(f'Processing wav and text data...')
    with open(raw_data_dir / 'metadata.csv', encoding='utf-8') as f:
        (data_dir / 'wavs').mkdir(exist_ok=True, parents=True)
        for index, line in enumerate(f.readlines()):
            if (index + 1) % 1000 == 0:
                print(f"{index+1} Done")
            
            parts = line.strip().split('|')
            name, text = parts[0], parts[2]

            wav_path = raw_data_dir / 'wavs' / f'{name}.wav'
            assert wav_path.exists(), \
                "Error during wav and text processing: {wav_path} does not exist"

            wav, _ = librosa.load(wav_path, sr=sample_rate)
            if wav.shape[0] > max_wav_length:
                left = torch.randint(0, wav.shape[-1] - max_wav_length, (1,))
                wav = wav[left : left + max_wav_length]

            wavfile.write(
                data_dir / 'wavs' / f'{name}.wav',
                sample_rate,
                wav
            )

def preprocess_mels(data_dir, mel_generator, mel_config):
    print(f'Generating mel-spectograms...')

    wav_dir = data_dir / 'wavs'
    spec_dir = data_dir / 'specs'

    assert wav_dir.exists(), f'Wav dir does not exist: {wav_dir}'
    if not spec_dir.exists():
        spec_dir.mkdir(exist_ok=True, parents=True)

    STOP_MAX_ITER = -1
    for i, wav_filename in enumerate(tqdm(os.listdir(wav_dir))):
        if i == STOP_MAX_ITER:
            break

        if wav_filename[-4:] != ".wav":
            continue

        name = wav_filename.split('.')[0]
        wav_path = wav_dir / wav_filename
        wav, _ = librosa.load(wav_path, sr=mel_config.sr)
        wav = torch.FloatTensor(wav).unsqueeze(0)

        spec = mel_generator(wav)
        np.save(spec_dir / f"{name}_spec.npy", spec)
