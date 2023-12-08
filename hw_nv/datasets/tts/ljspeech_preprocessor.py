import json
import librosa
import numpy as np
import os
import pyworld
import pywt
import tgt
import torch
import zipfile

from pathlib import Path
from scipy.interpolate import interp1d
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_nv.utils.audio.tools import AudioTools
from hw_nv.utils.text import _clean_text

URL_LINKS = {
    "mfa_alignments": "https://drive.google.com/file/d/1ukb8o-SnqhXCxq7drI3zye3tZdrGvQDA/view?usp=drive_link", 
}

class LJSpeechPreprocessor:
    def __init__(
            self,
            raw_data_dir,
            data_dir,
            config,
            perform_mfa_alignment=False,
            download_mfa_alignment=False,
            mfa_bin_path=None,
            mfa_pretrained_model_path=None
    ):
        """
        This class processes wavs from raw_data_dir,
        acquiring mel-spectrograms, durations, pitches and energies.
        raw_data_dir and data_dir must already exist

        Pipeline steps:
        1) Preprocessing of texts and wavs of original dataset
            This step normalizes wavs by max_wav_value 
            and cleans dataset's texts as follows:
            - converting to ascii
            - lowercase
            - expanding numbers to letters
            - expanding abbreviations
            - collapsing whitespaces
        2) Applying MFA alignment between utterances and phonemes
            MFA must be installed in mfa_bin_path
            the corresponding pretrained model is taken
            from mfa_pretrained_model_path
        3) Processing wavs as described in the paper, this step gets:
            - mel-spectrograms
            - durations
            - pitch-spectrograms:
                - computing pitch f0 contours with PyWorld
                - applying linear interpolation on the frame-level
                - normalizing f0s to zero mean and unit variance
                - applying Continuous Wavelet Transform from PyWavelets
            - energies:
                - compute energies as L2 norm of STFT magnitudes
                - normalize them to zero mean and unit variance
                - quantize energies to 256 possible values on frame-level
        
        params:
            raw_data_dir: directory of original LJ data
            data_dir: directory to write processed data to
            config: general config file
            perform_mfa_alignment: bool, should be turned to False,
                if MFA alignment was already performed
            mfa_bin_path: path to MFA to run the script
            mfa_pretrained_model: path to pretrained MFA model for the alignment
        """
        self.raw_data_dir = raw_data_dir
        self.data_dir = data_dir
        self.config = config
        self.prep_config = config["preprocessing"]

        self.preprocess_wavs = self.prep_config["preprocess_wavs"]

        self.use_pitch_spectrogram = self.prep_config["use_pitch_spectrogram"]
        self.pitch_energy_normalization = self.prep_config["pitch_energy_normalization"]

        self.max_wav_value = self.prep_config["max_wav_value"]
        self.sample_rate = self.prep_config["sr"]
        self.hop_size = self.prep_config["hop_size"]

        self.val_size = self.prep_config.get("val_size", 0.2)
        self.random_state = self.prep_config.get("random_state", None)

        # MFA flags
        self.perform_mfa_alignment = self.prep_config["mfa"].get(
            "perform", perform_mfa_alignment
        )
        self.download_mfa_alignment = self.prep_config["mfa"].get(
            "download", download_mfa_alignment
        )
        self.mfa_bin_path = self.prep_config["mfa"].get(
            "bin_path", mfa_bin_path
        )
        self.mfa_pretrained_model_path = self.prep_config["mfa"].get(
            "pretrained_model_path", mfa_pretrained_model_path
        )
        self.mfa_alignments_path = Path(self.prep_config["mfa"].get(
            "mfa_alignments_path", str(self.data_dir / "textgrid")
        ))

        # subdirectories for mel-spectrograms, duration, pitch and energy
        self.spec_path = self.data_dir / "spectrogram"
        self.spec_path.mkdir(exist_ok=True, parents=True)
        self.duration_path = self.data_dir / "duration"
        self.duration_path.mkdir(exist_ok=True, parents=True)
        self.pitch_path = self.data_dir / "pitch"
        self.pitch_path.mkdir(exist_ok=True, parents=True)
        self.energy_path = self.data_dir / "energy"
        self.energy_path.mkdir(exist_ok=True, parents=True)

        self.train_metadata_filename = "train.txt"
        self.val_metadata_filename = "val.txt"

        assert "STFT" in config["preprocessing"], \
            "STFT params must be provided in preprocessing config"
        
        self.audio_tools = AudioTools(
            max_wav_value=self.max_wav_value,
            stft_params=config["preprocessing"]["STFT"]
        )
    
    def process(self):
        if self.preprocess_wavs:
            self.preprocess_wavs_and_texts()
        else:
            print("Wavs and texts already preprocessed")

        if self.perform_mfa_alignment:
            self.align_with_mfa()
        elif self.download_mfa_alignment:
            self.download_mfa_alignments()
        else:
            print("MFA alignments are already performed")

        n_samples = 0
        n_success_samples = 0

        names = []
        text_data = []

        pitches = []
        energies = []

        print("Processing wavs and texts to get pitch and energy...")

        STOP_MAX_ITER = -1

        for i, wav_filename in enumerate(tqdm(os.listdir(self.raw_data_dir))):
            if i == STOP_MAX_ITER:
                break

            if wav_filename[-4:] != ".wav":
                continue

            name = wav_filename.split('.')[0]
            text, raw_text, pitch, energy = self.process_utterance(name)

            n_samples += 1
            if text is None:
                continue
            n_success_samples += 1

            names.append(name)
            text_data.append("\t".join([name, text, raw_text]))
            pitches.append(pitch)
            energies.append(energy)

        stats = {"pitch": {}, "energy": {}}

        # normalize and quantize pitches and energies
        pitches = [np.array(pitch) for pitch in pitches]
        stats["pitch"]["mean"] = float(np.mean([np.mean(pitch) for pitch in pitches]))
        stats["pitch"]["std"] = float(np.mean([np.std(pitch) for pitch in pitches]))
        pitches = [
            (pitch - stats["pitch"]["mean"]) / stats["pitch"]["std"]
            for pitch in pitches
        ]
        stats["pitch"]["min"] = float(np.min([np.min(pitch) for pitch in pitches]))
        stats["pitch"]["max"] = float(np.max([np.max(pitch) for pitch in pitches]))

        energies = [np.array(energy) for energy in energies]
        stats["energy"]["mean"] = float(np.mean([np.mean(energy) for energy in energies]))
        stats["energy"]["std"] = float(np.mean([np.std(energy) for energy in energies]))
        energies = [
            (energy - stats["energy"]["mean"]) / stats["energy"]["std"]
            for energy in energies
        ]
        stats["energy"]["min"] = float(np.min([np.min(energy) for energy in energies]))
        stats["energy"]["max"] = float(np.max([np.max(energy) for energy in energies]))

        """
        We will apply Continuous Wavelet Transform to normalized f0 contours,
        to get pitch spectrograms

        code example to get scale-to-frequency-mapping using pywt:

            import pywt

            scale = 2
            sampling_period = 22000
            wavelet = 'mexh'

            f = pywt.scale2frequency(wavelet, scale)/sampling_period
            print(f * sampling_period * 4)
            # 0.5
            # scale = 1 / x -> freq = x
        
        maximum frequency after STFT is sample_rate / 2
        We have applied log-scaling to both spectrograms and pitches,
        so log(max_freq) should be alright
        """

        if self.use_pitch_spectrogram:
            scales = 1 / np.arange(int(np.log(self.sample_rate / 2)))
            pitch_spectrograms = pywt.cwt(pitch, scales, 'mexh')
        
            for pitch_spec in pitch_spectrograms:
                np.save(self.pitch_path / f"{name}_pitch_spec.npy", pitch_spec)


        with open(self.data_dir / "pitch_energy_stats.json", "w") as f:
            f.write(json.dumps(stats))

        for name, pitch in zip(names, pitches):
            np.save(self.pitch_path / f"{name}_pitch.npy", pitch)
        for name, energy in zip(names, energies):
            np.save(self.energy_path / f"{name}_energy.npy", energy)
        
        train_data, val_data = train_test_split(
            text_data, test_size=self.val_size,
            random_state=self.random_state
        )

        self.write_metadata(train_data, filename=self.train_metadata_filename)
        self.write_metadata(val_data, filename=self.val_metadata_filename)

    def preprocess_wavs_and_texts(self):
        print(f'Processing wav and text data...')
        with open(self.raw_data_dir / 'metadata.csv', encoding='utf-8') as f:
            for index, line in enumerate(f.readlines()):
                if (index + 1) % 1000 == 0:
                    print(f"{index+1} Done")
                
                parts = line.strip().split('|')
                name, text = parts[0], parts[2]

                wav_path = self.raw_data_dir / 'wavs' / f'{name}.wav'
                assert wav_path.exists(), \
                    "Error during wav and text processing: {wav_path} does not exist"

                wav, _ = librosa.load(wav_path, sr=self.sample_rate)
                wav = wav / np.abs(wav).max() * self.max_wav_value
                wavfile.write(
                    self.raw_data_dir / f'{name}.wav',
                    self.sample_rate,
                    wav
                )

                text = _clean_text(text, cleaner_names=["english_cleaners"])
                with open(self.raw_data_dir / f'{name}.lab', 'w') as text_file:
                    text_file.write(text)

    def align_with_mfa(self):
        print("Launching MFA alignment between utterances and phonemes...")
        print(f"Alignments will be written to {self.mfa_alignments_path}")
        mfa_command = (
            f"{self.mfa_bin_path} {self.raw_data_dir} "
            f"{self.mfa_pretrained_model_path} "
            f"english {self.mfa_alignments_path}"
        )
        os.system(mfa_command)
        print("MFA alignments are ready")
    
    def download_mfa_alignments(self):
        zip_alignments_path = self.raw_data_dir / "TextGrid.zip"
        print(f"Downloading MFA alignments to {zip_alignments_path}...")
        download_file(URL_LINKS["mfa_alignments"], zip_alignments_path)

        print(f"Unzipping MFA alignments to {self.mfa_alignments_path}...")
        with zipfile.ZipFile(zip_alignments_path, 'r') as zip_ref: 
            zip_ref.extractall(self.mfa_alignments_path)
        print("MFA alignments are ready")

    def process_utterance(self, name):
        wav_path = self.raw_data_dir / f"{name}.wav"
        text_path = self.raw_data_dir / f"{name}.lab"
        textgrid_path = self.mfa_alignments_path / f"{name}.TextGrid"
        assert textgrid_path.exists(), \
            f"Error loading TextGrid path from MFA for {name}, path: {textgrid_path}"
    
        phones, durations, start, end = self.get_alignment(textgrid_path)
        total_duration = sum(durations)
        if start >= end:
            return None, None, None, None
        
        with open(text_path, "r") as text_file:
            raw_text = text_file.readline().strip("\n")
        text = "{%s}" % " ".join(phones)

        wav, _ = librosa.load(wav_path, sr=self.sample_rate)
        wav_start = int(self.sample_rate * start)
        wav_end = int(self.sample_rate * end)
        wav = wav[wav_start:wav_end]

        pitch_f0 = self.get_pitch_f0(wav)
        pitch_f0 = pitch_f0[:total_duration]
        if np.sum(pitch_f0 != 0) <= 1:
            return None, None, None, None

        mel_spectrogram, magnitudes = self.audio_tools.get_mel_from_wav(torch.FloatTensor(wav))
        mel_spectrogram = mel_spectrogram[:, :total_duration]

        energy = torch.norm(magnitudes, dim=1)
        energy = energy[:total_duration]

        np.save(self.duration_path / f"{name}_duration.npy", durations)
        np.save(self.spec_path / f"{name}_spec.npy", mel_spectrogram.transpose(1, 2))
        # np.save(self.pitch_path / f"{name}_pitch.npy", pitch_f0)
        # np.save(self.energy_path / f"{name}_energy.npy", energy)

        return text, raw_text, pitch_f0, energy


    def get_alignment(self, textgrid_path):
        textgrid = tgt.io.read_textgrid(textgrid_path)
        tier = textgrid.get_tier_by_name("phones")

        silent_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time, end_time, end_idx = 0, 0, 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if len(phones) == 0 and p not in silent_phones:
                start_time = s

            phones.append(p)
            if p not in silent_phones:
                # For ordinary phones only
                end_time = e
                end_idx = len(phones)
        
            wav_s = int(s * self.sample_rate / self.hop_size)
            wav_e = int(e * self.sample_rate / self.hop_size)
            durations.append(wav_e - wav_s)

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time


    def get_pitch_f0(self, wav):        
        frame_period = self.hop_size / self.sample_rate * 1000
        
        pitch_f0, t = pyworld.dio(
            wav.astype(np.float64), self.sample_rate,
            frame_period=frame_period,
        )
        pitch_f0 = pyworld.stonemask(
            wav.astype(np.float64), pitch_f0, t, self.sample_rate
        ).astype(np.float32)

        nonzero_indices = np.where(pitch_f0 != 0)[0]
        fill_values = (pitch_f0[nonzero_indices][0], pitch_f0[nonzero_indices][-1])

        interp_fn = interp1d(
            nonzero_indices,
            pitch_f0[nonzero_indices],
            fill_value=fill_values,
            bounds_error=False,
            kind="linear"
        )
        pitch_f0 = interp_fn(np.arange(pitch_f0.shape[0]))

        return pitch_f0
        
    def write_metadata(self, metadata, filename="train.txt"):
        with open(os.path.join(self.data_dir, filename), 'w', encoding='utf-8') as f:
            for m in metadata:
                f.write(m + '\n')
