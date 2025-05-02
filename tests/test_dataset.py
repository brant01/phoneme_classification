from pathlib import Path
from data_utils.loader import parse_dataset
from data_utils.dataset import PhonemeDataset
from data_utils.transform import WaveletHilbertTransform
import torch

def test_dataset_getitem():
    data_path = Path("data/New Stimuli 9-8-2024")
    file_paths, labels, label_map, lengths = parse_dataset(data_path)

    output_len = int(max(lengths) * 1.1)
    transform = WaveletHilbertTransform(output_len=output_len)

    dataset = PhonemeDataset(file_paths, labels, transform=transform, augment=False)

    x_aug, x_clean, label = dataset[0]

    assert isinstance(x_aug, torch.Tensor)
    assert isinstance(x_clean, torch.Tensor)
    assert isinstance(label, int)

    assert x_aug.shape == x_clean.shape
    assert x_aug.ndim == 3   # [2, num_scales, T]
    assert label in labels