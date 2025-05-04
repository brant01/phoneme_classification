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
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        """
        Return an augmented sample from the dataset.

        Returns:
            Tuple:
                - augmented features (Tensor)
                - clean features (Tensor)
                - integer phoneme label (int)
        """
        file_path: Path = self.file_paths[idx]
        label: int = self.labels[idx]

        # Load waveform
        waveform, sr = torchaudio.load(str(file_path))
        waveform = waveform.mean(dim=0) if waveform.ndim > 1 else waveform  # convert to mono
        waveform = waveform.squeeze(0) if waveform.ndim > 1 else waveform

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        # normalize waveform
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

        # Get clean features (always used as target)
        clean_features = self.transform(waveform)

        # Apply augmentation to get training input
        if self.augment and self.augmentation:
            augmented_features = self.augmentation(waveform, clean_features, sample_rate=self.sample_rate)
        else:
            augmented_features = clean_features

        return augmented_features, clean_features, label
    
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