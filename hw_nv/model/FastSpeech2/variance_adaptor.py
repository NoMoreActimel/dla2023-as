import json
import torch
import torch.nn.functional as F

from pathlib import Path
from torch import nn

from .length_regulator import LengthRegulator
from .variance_predictor import VariancePredictor
from hw_nv.utils import ROOT_PATH


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config
        self.adaptor_config = model_config["variance_adaptor"]

        self.duration_predictor = VariancePredictor(model_config)
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.length_regulator = LengthRegulator()

        data_dir = Path(model_config["data_dir"])

        pitch_energy_stats = self.adaptor_config.get("pitch_energy_stats", None)

        if pitch_energy_stats is None:
            with open(data_dir / "pitch_energy_stats.json", "r") as f:
                pitch_energy_stats = json.load(f)

        self.pitch_stats = pitch_energy_stats["pitch"]
        self.energy_stats = pitch_energy_stats["energy"]
        
        self.num_bins = self.adaptor_config["num_bins"]
        self.quantization_log_scaling = self.adaptor_config["quantization_log_scaling"]
        
        for feature in ["pitch", "energy"]:
            # bad vibes, I was a clown..
            setattr(
                self, f"{feature}_embedding",
                nn.Embedding(self.num_bins, model_config["encoder_decoder"]["encoder_dim"])
            )

            setattr(
                self, f"{feature}_quantization",
                self.adaptor_config[f"{feature}_quantization"]
            )

            feature_min = torch.tensor(getattr(self, f"{feature}_stats")["min"])
            feature_max = torch.tensor(getattr(self, f"{feature}_stats")["max"])

            if self.quantization_log_scaling:
                feature_scale = nn.Parameter(
                    torch.exp(torch.linspace(
                        torch.log(1 + feature_min), torch.log(1 + feature_max), self.num_bins
                    )),
                    requires_grad=False
                )
            else:
                feature_scale = nn.Parameter(
                    torch.linspace(
                        feature_min, feature_max, self.num_bins
                    ),
                    requires_grad=False
                )
        
            setattr(
                self, f"{feature}_scale",
                feature_scale
            )

    def forward(
            self,
            src_seq,
            max_mel_length=None,
            duration_target=None,
            pitch_target=None,
            energy_target=None,
            duration_coeff=1.0,
            pitch_coeff=1.0,
            energy_coeff=1.0
    ):
        # Duration
        log_duration = self.duration_predictor(src_seq)

        if duration_target is not None:
            output, mel_length = self.length_regulator(
                src_seq, durations=None, target=duration_target, mel_max_length=max_mel_length
            )
            duration = duration_target
        else:
            duration = torch.round((torch.exp(log_duration) - 1) * duration_coeff).long()
            duration[duration < 0] = 0
            output, mel_length = self.length_regulator(
                src_seq, durations=duration, target=None, mel_max_length=None
            )
                
        # Pitch
        pitch = self.pitch_predictor(output)

        if pitch_target is not None:
            pitch_bucketized = torch.bucketize(pitch_target, self.pitch_scale)
        else:
            pitch = pitch * pitch_coeff
            pitch_bucketized = torch.bucketize(pitch, self.pitch_scale)
        
        pitch_embed = self.pitch_embedding(pitch_bucketized)
        output = output + pitch_embed

        # Energy
        energy = self.energy_predictor(output)

        if energy_target is not None:
            energy_bucketized = torch.bucketize(energy_target, self.energy_scale)
        else:
            energy = energy * energy_coeff
            energy_bucketized = torch.bucketize(energy, self.energy_scale)
        
        energy_embed = self.energy_embedding(energy_bucketized)
        output = output + energy_embed

        return {
            "mel-spectrogram": output,
            "duration_true": duration_target,
            "log_duration": log_duration,
            "pitch": pitch,
            "energy": energy,
            "mel-length": mel_length
        }
