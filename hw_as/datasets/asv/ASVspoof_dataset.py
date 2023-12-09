import json
import torch
import torchaudio

from pathlib import Path
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from hw_as.base.base_dataset import BaseDataset
from hw_as.utils import ROOT_PATH


class ASVspoofDataset(BaseDataset):
    def __init__(self, data_dir, part, max_audio_length=None, max_index_length=None, *args, **kwargs):
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists(), \
            f"Please, download ASVspoof2019 dataset to {data_dir} from kaggle, " \
            f"follow the instructions in README"
        
        self.part = part
        self.max_audio_length = max_audio_length
        self.max_index_length = max_index_length
        self.sample_rate = 16000

        index = self.create_index()
        super().__init__(index, *args, **kwargs)

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
        print("Preparing ASVspoof2019 index")
        protocols_dir = self.data_dir / f"LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{self.part}.txt"

        index = []
        with open(protocols_dir, 'r') as f:
            lines = f.readlines()
            if self.max_index_length:
                lines = lines[:self.max_index_length]
            
            for line in lines:
                item = line.split()
                index.append({
                    "speaker_id": item[0],
                    "utterance_id": item[1],
                    "utterance_type": item[4] 
                })

        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        audio_path = self.data_dir / data_dict["utterance_id"] + ".flac"
        wav, _ = torchaudio.load(audio_path)  # default sample-rate for ASVspoof2019 is 16000
        if self.max_audio_length and wav.shape[0] > self.max_audio_length:
                left = torch.randint(0, wav.shape[-1] - self.max_audio_length, (1,))
                wav = wav[left : left + self.max_audio_length]

        return {
            "speaker_id": data_dict["speaker_id"],
            "utterance_id": data_dict["utterance_id"],
            "wav": wav,
            "length": wav.shape[0],
            "target": data_dict["utterance_type"] == "bonafide"
        }

    def collate_fn(self, batch_items):
        batch = {}
        batch["wav"] = [item["wav"] for item in batch_items]
        batch["length"] = [item["length"] for item in batch_items]
        batch["target"] = [item["target"] for item in batch_items]

        batch["wav"] = pad_sequence(batch["wav"], batch_first=True).unsqueeze(1)
        batch["length"] = torch.LongTensor(batch["length"])
        batch["target"] = torch.LongTensor(batch["target"])

        return batch
