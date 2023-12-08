import librosa
import numpy as np
import os
import shutil

from scipy.io import wavfile
from speechbrain.utils.data_utils import download_file

from hw_nv.utils.text import _clean_text
from hw_nv.utils import ROOT_PATH

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}

class LJSpeechPreprocessor:
    def __init__(self, config=None, data_dir=None, raw_data_dir=None, *args, **kwargs):
        if config is None:
            config = kwargs["config_parser"]

        if data_dir is None:
            data_dir = config["preprocessing"].get("data_dir", None)
            if data_dir is None:
                data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech_processed"
            data_dir.mkdir(exist_ok=True, parents=True)
        if raw_data_dir is None:
            raw_data_dir = config["preprocessing"].get("raw_data_dir", None)
            if raw_data_dir is None:
                raw_data_dir = ROOT_PATH / "data" / "ljspeech"
            raw_data_dir.mkdir(exist_ok=True, parents=True)

        self._raw_data_dir = raw_data_dir
        self._data_dir = data_dir
        self.config = config
        
        self.max_wav_value = config["preprocessing"]["max_wav_value"]
        self.sample_rate = config["preprocessing"]["sr"]

    def load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

    def _preprocess_wavs_and_texts(self):
        print(f'Processing wav and text data...')
        with open(self._raw_data_dir / 'metadata.csv', encoding='utf-8') as f:
            for index, line in enumerate(f.readlines()):
                if (index + 1) % 100 == 0:
                    print("{:d} Done".format(index))
                
                parts = line.strip().split('|')
                name, text = parts[0], parts[2]

                wav_path = self._raw_data_dir / 'wavs' / f'{name}.wav'
                assert wav_path.exists(), \
                    "Error during wav and text processing: {wav_path} does not exist"

                wav, _ = librosa.load(wav_path, self.sample_rate)
                wav *= np.abs(wav).max() / self.max_wav_value
                wavfile.write(
                    self._data_dir / f'{name}.wav',
                    self.sample_rate,
                    wav.astype(np.int16)
                )

                text = _clean_text(text, cleaner_names=["english_cleaners"])
                with open(self._data_dir / f'{name}.lab', 'w') as text_file:
                    text_file.write(text)
