from pathlib import Path
from data_utils.loader import parse_dataset

def test_parse_dataset_basic():
    data_path = Path("data/New Stimuli 9-8-2024")
    file_paths, labels, label_map, lengths = parse_dataset(data_path)

    assert len(file_paths) > 0
    assert len(file_paths) == len(labels)
    assert len(file_paths) == len(lengths)

    assert all(isinstance(p, Path) for p in file_paths)
    assert all(isinstance(label, int) for label in labels)
    assert all(label > 0 for label in lengths)
    assert isinstance(label_map, dict)
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in label_map.items())

    # Sanity: first label matches its mapping
    label_name = next(k for k, v in label_map.items() if v == labels[0])
    assert isinstance(label_name, str)