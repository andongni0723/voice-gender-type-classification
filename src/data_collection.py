from typing import Literal
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.spec_process import SpectrogramProcessor


@dataclass(frozen=True)
class VoiceSample:
    path: str
    label: Literal['male', 'female']
    sr: float
    waveform: np.array


class VoiceDataset(Dataset):
    def __init__(self, samples: list[VoiceSample], processor: SpectrogramProcessor,
                 label_dict: dict[str, int] = {'male': 0, 'female': 1},
                 cache_specs: bool = True):
        self.samples = samples
        self.processor = processor
        self.label_to_id = label_dict
        self._specs_cache: dict[int, tuple[Tensor, Tensor]] = {} if cache_specs else None
        """Save the sample and label cache, reduce transform tensor times."""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[idx]
        if self._specs_cache and idx in self._specs_cache: # Check data exist
            spec, label = self._specs_cache[idx]
        else:
            spec = self.processor.transform_to_tensor(sample.waveform)
            label = torch.tensor(self.label_to_id[sample.label], dtype=torch.long)
            if self._specs_cache:
                self._specs_cache[idx] = (spec, label)
        return spec, label