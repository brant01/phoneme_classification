from pathlib import Path
from data_utils.loader import parse_dataset
from data_utils.dataset import PhonemeDataset
from data_utils.transform import WaveletHilbertTransform


def test_dataset_getitem():
    data_path = Path("data/New Stimuli 9-8-2024")
    file_paths, labels, label_map, lengths = parse_dataset(data_path)

    output_len = int(max(lengths) * 1.1)
    transform = WaveletHilbertTransform(output_len=output_len)

    dataset = PhonemeDataset(file_paths, labels, transform=transform, augment=False)

    x_aug, x_clean, label, filename = dataset[0]  # UPDATED to unpack 4 values

    assert x_aug.shape == x_clean.shape
    assert isinstance(label, int)
    assert isinstance(filename, str)

    assert x_aug.shape == x_clean.shape
    assert x_aug.ndim == 3   # [2, num_scales, T]
    assert label in labels