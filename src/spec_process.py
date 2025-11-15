import librosa
import librosa.feature
import numpy as np
import torch
from torch import Tensor


class SpectrogramProcessor:
    def __init__(self, n_mels: int = 128, f_max: int = 1024,
                 sr: int = 22050, hop_length: int = 512, target_frames: int = 128):
        self.n_mels = n_mels
        self.f_max = f_max
        self.sr = sr
        self.hop_length = 512
        self.target_frames = 128

    def waveform_to_mel(self, y: np.array) -> np.array:
        spec = librosa.feature.melspectrogram(y=y, sr=self.sr, fmax=self.f_max, n_mels=self.n_mels)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        return spec_db

    def pad_or_truncate(self, spec: np.array) -> np.array:
        """pad or truncate the spectrogram (填充或截斷聲譜圖)
        :param spec: mel spectrogram.
        :return: result after process.
        """
        _, n_frames = spec.shape
        if n_frames == self.target_frames:
            return spec
        if n_frames < self.target_frames:
            # pad on time axis (axis=1)
            return librosa.util.fix_length(spec, size=self.target_frames, axis=1)
        else:
            # truncate right side
            return spec[:, :self.target_frames]

    def transform_to_tensor(self, y: np.array) -> Tensor:
        spec = self.waveform_to_mel(y)
        spec = self.pad_or_truncate(spec)
        return torch.from_numpy(spec).float().unsqueeze(0)  # shape (1, H, W)
