"""
Dataset class for loading and preprocessing phoneme audio files.
"""


import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable


class PhonemeDataset(Dataset):
    """
    Dataset for phoneme classsification and resonstruction
    """
    
    def __init__(
        self,
        file_paths: list[Path],
        labels: list[int],
        transform: Callable[[Tensor], Tensor],
        augment: bool = False,
        augmentation: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        sample_rate: int = 16000,
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.augmentation = augmentation
        self.sample_rate = sample_rate
        
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int, str]:
        """
        Return a tuple of (augmented_features, clean_features, label_index, filename)
        """
        file_path = self.file_paths[index]
        label_idx = self.labels[index]

        waveform, sr = torchaudio.load(str(file_path))  # waveform: [1, T] or [channels, T]

        if waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)

        if self.sample_rate and sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        clean_features = self.transform(waveform)

        if self.augment and self.augmentation:
            _, augmented_features = self.augmentation(waveform, clean_features)
        else:
            augmented_features = clean_features.clone()

        return augmented_features, clean_features, label_idx, str(file_path)
    
if __name__ == "__main__":
    
    from data_utils.loader import parse_dataset

    torchaudio.set_audio_backend("soundfile")

    # Manually set path to test data
    data_path = Path("data/New Stimuli 9-8-2024")

    file_paths, labels, label_map = parse_dataset(data_path)

    dataset = PhonemeDataset(file_paths, labels, augment=False)

    print(f"Testing dataset with {len(dataset)} samples...")

    # Grab a sample
    x_aug, x_clean, y = dataset[0]

    print(f"Augmented shape: {x_aug.shape}")
    print(f"Clean shape: {x_clean.shape}")
    print(f"Label index: {y}")