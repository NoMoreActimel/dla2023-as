import json
import librosa
import logging
import numpy as np
import os
import shutil
from pathlib import Path
import torch
import torchaudio

from speechbrain.utils.data_utils import download_file
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from hw_nv.base.base_dataset import BaseDataset
from hw_nv.datasets.tts.ljspeech_preprocessor import LJSpeechPreprocessor
from hw_nv.utils import ROOT_PATH
from hw_nv.utils.text import text_to_sequence

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}

class LJspeechFastSpeech2Dataset(BaseDataset):
    def __init__(self, config=None, raw_data_dir=None, data_dir=None, train=True, *args, **kwargs):
        if config is None:
            config = kwargs["config_parser"]
        self.config = config

        self.train = train

        if data_dir is None:
            data_dir = config["preprocessing"].get("data_dir", None)
            if data_dir is None:
                data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech_processed"
        if raw_data_dir is None:
            raw_data_dir = config["preprocessing"].get("raw_data_dir", None)
            if raw_data_dir is None:
                raw_data_dir = ROOT_PATH / "data" / "ljspeech"

        self._raw_data_dir = Path(raw_data_dir)
        self._raw_data_dir.mkdir(exist_ok=True, parents=True)

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(exist_ok=True, parents=True)

        self.max_wav_value = self.config["preprocessing"]["max_wav_value"]
        self.sample_rate = self.config["preprocessing"]["sr"]

        if self.config["preprocessing"].get("load_dataset", True) and \
                len(os.listdir(self._raw_data_dir)) == 0:
            self._load_dataset()

        self.data_processor = LJSpeechPreprocessor(self._raw_data_dir, self._data_dir, self.config)
        if self.train and self.config["preprocessing"].get("perform_preprocessing", None):
            self.data_processor.process()
            
        self.spec_dir = self.data_processor.spec_path
        self.duration_dir = self.data_processor.duration_path
        self.pitch_dir = self.data_processor.pitch_path
        self.energy_dir = self.data_processor.energy_path
    
        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._raw_data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._raw_data_dir)
        for fpath in (self._raw_data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._raw_data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._raw_data_dir / "LJSpeech-1.1"))
        print(f"Unpacked LJSpeech to {self._raw_data_dir}")

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        name = data_dict["name"]
        raw_text = data_dict["raw_text"]
        text = np.array(text_to_sequence(data_dict["text"], ["english_cleaners"]))
        spectrogram = np.load(self.spec_dir / f"{name}_spec.npy")
        duration = np.load(self.duration_dir / f"{name}_duration.npy")
        pitch = np.load(self.pitch_dir / f"{name}_pitch.npy")
        energy = np.load(self.energy_dir / f"{name}_energy.npy")

        text = torch.from_numpy(text).long()

        spectrogram = torch.from_numpy(spectrogram).float()
        spec_length = spectrogram.shape[0]

        duration = torch.from_numpy(duration).long()
        pitch = torch.from_numpy(pitch).float()[:spec_length]
        energy = torch.from_numpy(energy).float().T.squeeze(-1)[:spec_length]

        return {
            "name": name,
            "raw_text": raw_text,
            "text": text,
            "spectrogram": spectrogram,
            "duration": duration,
            "pitch": pitch,
            "energy": energy
        }

    def _get_or_load_index(self):
        index_path = self._data_dir / f"index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []

        if self.train:
            metadata_filename = self.data_processor.train_metadata_filename
        else:
            metadata_filename = self.data_processor.val_metadata_filename

        metadata_path = self._data_dir / metadata_filename
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                name, text, raw_text = line.strip('\n').split('\t')
                index.append({"name": name, "text": text,"raw_text": raw_text})
        
        return index
    
    def collate_fn(self, batch_items):
        # src_seq,
        # src_pos,
        # mel_pos,
        # max_mel_length,
        # duration_target=None,
        # pitch_target=None,
        # energy_target=None,

        batch = {}
        batch["src_seq"] = [item["text"] for item in batch_items]
        batch["mel_target"] = [item["spectrogram"] for item in batch_items]
        batch["duration_target"] = [item["duration"] for item in batch_items]
        batch["pitch_target"] = [item["pitch"] for item in batch_items]
        batch["energy_target"] = [item["energy"] for item in batch_items]

        for key in ["src_seq", "mel_target", "duration_target", "pitch_target", "energy_target"]:
            batch[key] = pad_sequence(batch[key], batch_first=True)

        text_lengths = [item["text"].shape[0] for item in batch_items]
        mel_lengths = [item["spectrogram"].shape[0] for item in batch_items]

        src_pos = list()
        max_len = int(max(text_lengths))
        for length_src_row in text_lengths:
            src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                                (0, max_len-int(length_src_row)), 'constant'))
        batch["src_pos"] = torch.from_numpy(np.array(src_pos))

        mel_pos = list()
        batch["max_mel_length"] = int(max(mel_lengths))
        for length_mel_row in mel_lengths:
            mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                                (0, batch["max_mel_length"]-int(length_mel_row)), 'constant'))
        batch["mel_pos"] = torch.from_numpy(np.array(mel_pos))
                
        return batch
