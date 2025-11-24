from typing import Literal

import librosa
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_collection import VoiceSample, VoiceDataset
from src.spec_process import SpectrogramProcessor


class DataManager:
    def __init__(self, data_dir: str, processor: SpectrogramProcessor, total_count: int | None = None):
        self.data_dir = data_dir
        self.processor = processor
        self.total_count = total_count

    def _load_file_list(self) -> list[tuple[str, Literal['male', 'female']]]:
        from src.data_io import import_voice_with_filename_and_label
        return import_voice_with_filename_and_label(self.data_dir, self.total_count)

    def build_samples(self) -> list[VoiceSample]:
        """ Convert voice file to np.array.
        :return: A list with VoiceSample
        """
        samples = []
        file_list = self._load_file_list()
        for path, label in tqdm(file_list, total=len(file_list), desc='Build Samples'):
            y, sr = librosa.load(path, sr=self.processor.sr)
            y, _ = librosa.effects.trim(y)
            samples.append(VoiceSample(path, label, self.processor.sr, y))
        return samples

    def train_val_split(self,
        samples: list[VoiceSample],
        valid_ratio: float = 0.2,
        shuffle: bool = True
    ) -> tuple[list[VoiceSample], list[VoiceSample]]:
        """ split sample list to pair with different ratio.
        :return: Back two list, (train list, valid list)
        """
        if shuffle:
            from random import shuffle
            shuffle(samples)
        n_val = int(len(samples) * valid_ratio)
        return samples[n_val:], samples[:n_val]

    def get_dataloaders(self,
        batch_size: int = 8,
        valid_ratio: float = 0.2,
        cache_specs: bool = True
    ) -> tuple[DataLoader, DataLoader]:
        """
        :return: Back a train data loader and valid data loader.
        """
        samples = self.build_samples()
        train_samples, valid_samples = self.train_val_split(samples, valid_ratio)
        train_ds = VoiceDataset(train_samples, self.processor, cache_specs=cache_specs)
        valid_ds = VoiceDataset(valid_samples, self.processor, cache_specs=cache_specs)
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
        valid_dl = DataLoader(valid_ds, batch_size, shuffle=False, num_workers=4)
        return train_dl, valid_dl

if __name__ == '__main__':
    processor = SpectrogramProcessor()
    train_path = input('Input train data path : ')
    test_path = input('Input test data path  : ')
    try:
        train, valid = DataManager(train_path, processor, 10).get_dataloaders(batch_size=32)
        test, _ = DataManager(test_path, processor, 2).get_dataloaders(batch_size=32, valid_ratio=0)
        print(len(train.dataset), len(valid.dataset))
        print(len(test.dataset))
    except ValueError:
        print('FileNotFoundError: Data direction does not exist or contains not voice files.')