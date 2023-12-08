import logging
import torch

from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    raise NotImplementedError()
    """
    Collate and pad fields in mixed dataset items
    """
    # input fields: [
    #   "ref_audio", "mix_audio", "target_audio",
    #   "ref_spectrogram", "mix_spectrogram", "target_spectrogram",
    #   "ref_path", "mix_path", "target_path",
    #   "ref_duration", "mix_duration", "target_duration"
    # ]

    audios = {"ref": [], "input": [], "target": []}
    lengths = {"ref": [], "input": [], "target": []}
    speaker_ids = []

    for item in dataset_items:
        speaker_ids.append(item["speaker_id"])

        audios["ref"].append(item["ref_audio"].squeeze(0))
        audios["input"].append(item["mix_audio"].squeeze(0))
        audios["target"].append(item["target_audio"].squeeze(0))
        
        lengths["ref"].append(item["ref_audio"].shape[-1])
        lengths["input"].append(item["mix_audio"].shape[-1])
        lengths["target"].append(item["target_audio"].shape[-1])    

    audio_length = dataset_items[0]["audio_length"]
    audio_lengths = torch.tensor([audio_length], dtype=torch.int32)
    audio_lengths = audio_lengths.repeat(len(dataset_items))
    
    return {
        "input": pad_sequence(audios["input"], batch_first=True),
        "ref": pad_sequence(audios["ref"], batch_first=True),
        "target": pad_sequence(audios["target"], batch_first=True),
        "input_length": torch.tensor(lengths["input"], dtype=torch.int32),
        "ref_length": torch.tensor(lengths["ref"], dtype=torch.int32),
        "target_length": torch.tensor(lengths["target"], dtype=torch.int32), 
        "speaker_id": torch.tensor(speaker_ids, dtype=torch.int32),
        "audio_length": audio_lengths
    }