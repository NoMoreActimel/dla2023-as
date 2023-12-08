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
from hw_nv.datasets.nv.mel_generation import MelSpectrogram, MelSpectrogramConfig
from hw_nv.datasets.nv.preprocess import preprocess_wavs_and_texts, preprocess_mels
from hw_nv.utils.text import text_to_sequence
from hw_nv.utils import ROOT_PATH

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}

class LJspeechMelDataset(BaseDataset):
    def __init__(self, config=None, raw_data_dir=None, data_dir=None, train=True, *args, **kwargs):
        if config is None:
            config = kwargs["config_parser"]
        self.config = config
        self.prep_config = config["preprocessing"]

        if data_dir is None:
            data_dir = self.prep_config.get("data_dir", None)
            if data_dir is None:
                data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech_processed"
        if raw_data_dir is None:
            raw_data_dir = self.prep_config.get("raw_data_dir", None)
            if raw_data_dir is None:
                raw_data_dir = ROOT_PATH / "data" / "ljspeech"

        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(exist_ok=True, parents=True)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        self.melspec_config = MelSpectrogramConfig(
            **self.prep_config["mel_spec_config"]
        )
        self.melspec_generator = MelSpectrogram(self.melspec_config)

        if self.prep_config.get("load_dataset", True) and \
                len(os.listdir(self.raw_data_dir)) == 0:
            self._load_dataset()
        
        if self.prep_config.get("preprocess_wavs", True):
            preprocess_wavs_and_texts(
                raw_data_dir=self.raw_data_dir,
                data_dir=self.data_dir,
                sample_rate=self.melspec_config.sr,
                max_wav_length=self.prep_config["max_wav_length"]
            )
        if self.prep_config.get("generate_mels", True):
            preprocess_mels(
                data_dir=self.data_dir,
                mel_generator=self.melspec_generator,
                mel_config=self.melspec_config
            )
        
        self.limit = self.prep_config.get("limit", None)

        self.wav_dir = self.data_dir / "wavs"
        self.spec_dir = self.data_dir / "specs"
        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self.raw_data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self.raw_data_dir)
        for fpath in (self.raw_data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self.raw_data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self.raw_data_dir / "LJSpeech-1.1"))
        # files = [file_name for file_name in (self.raw_data_dir / "wavs").iterdir()]
        # train_length = int(0.85 * len(files)) # hand split, test ~ 15% 
        # (self.raw_data_dir / "train").mkdir(exist_ok=True, parents=True)
        # (self.raw_data_dir / "test").mkdir(exist_ok=True, parents=True)
        # for i, fpath in enumerate((self.raw_data_dir / "wavs").iterdir()):
        #     if i < train_length:
        #         shutil.move(str(fpath), str(self.raw_data_dir / "train" / fpath.name))
        #     else:
        #         shutil.move(str(fpath), str(self.raw_data_dir / "test" / fpath.name))
        # shutil.rmtree(str(self.raw_data_dir / "wavs"))
        print(f"Unpacked LJSpeech to {self.raw_data_dir}")

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        name = data_dict["name"]

        wav, _ = librosa.load(self.wav_dir / f"{name}.wav", sr=self.melspec_config.sr)
        spectrogram = np.load(self.spec_dir / f"{name}_spec.npy")

        wav = torch.from_numpy(wav).float()
        spectrogram = torch.from_numpy(spectrogram).float()
        spec_length = spectrogram.shape[-1]

        # print(f"wav length: {data_dict['audio_length']}, spec length: {spec_length}")

        return {
            "name": name,
            "wav": wav,
            "spectrogram": spectrogram,
            "length": spec_length
        }

    def _get_or_load_index(self):
        index_path = self.data_dir / f"index.json"
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
        if not self.raw_data_dir.exists():
            self._load_dataset()        
        print("Preparing LJSpeech index")
        for i, wav_filename in enumerate(tqdm(os.listdir(self.wav_dir))):
            if wav_filename[-4:] != ".wav":
                continue
            name = wav_filename.split('.')[0]
            t_info = torchaudio.info(str(self.wav_dir / wav_filename))
            length = t_info.num_frames / t_info.sample_rate
            index.append({"name": name, "audio_length": length})
            if i + 1 == self.limit:
                break
        return index

    def collate_fn(self, batch_items):
        batch = {}
        batch["wav"] = [item["wav"] for item in batch_items]
        batch["mel"] = [item["spectrogram"].transpose(1, 2).squeeze(0) for item in batch_items]
        batch["length"] = [item["length"] for item in batch_items]

        batch["wav"] = pad_sequence(batch["wav"], batch_first=True).unsqueeze(1)
        batch["mel"] = pad_sequence(batch["mel"], batch_first=True).transpose(1, 2)
        batch["length"] = torch.LongTensor(batch["length"])

        return batch
