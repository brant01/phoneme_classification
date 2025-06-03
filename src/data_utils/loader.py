"""
Functions for loading and parsing the phoneme dataset from a directory of .wav files.
"""

from pathlib import Path
import torchaudio
from typing import List, Dict, Tuple
import re

def extract_label(file_path: Path) -> str:
    """
    Extract the phoneme label from the start of a filename.

    Rules:
    - Label is the first 1–4 lowercase letters at the start of the filename
    - Stops at the first digit, space, parenthesis, or other separator

    Args:
        file_path (Path): Path to a .wav file

    Returns:
        str: Parsed phoneme label in lowercase

    Raises:
        ValueError: If the label cannot be extracted
    """
    name = file_path.stem.lower()
    match = re.match(r"[a-z]{1,4}", name)
    if not match:
        raise ValueError(f"Cannot extract label from: {file_path.name}")
    return match.group(0)


def parse_dataset(data_dir: Path) -> Tuple[List[Path], List[int], Dict[str, int], List[int]]:
    """
    Parse a directory tree of .wav files and return file paths with integer labels.

    Args:
        data_dir (Path): Path to the root dataset directory.

    Returns:
        Tuple containing:
            - List of file paths
            - Corresponding list of integer labels
            - Dictionary mapping string labels to integer IDs
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    file_paths: List[Path] = []
    string_labels: List[str] = []
    lengths: List[int] = []

    for wav_file in data_dir.rglob("*.wav"):
        try:
            label = extract_label(wav_file)
            info = torchaudio.info(str(wav_file))
            num_samples = info.num_frames
        except Exception as e:
            print(f"Skipping file: {wav_file.name} — {e}")
            continue
        file_paths.append(wav_file)
        string_labels.append(label)
        lengths.append(num_samples)

    # Build string-to-int label mapping
    unique_labels = sorted(set(string_labels))
    label_map: Dict[str, int] = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels: List[int] = [label_map[label] for label in string_labels]

    return file_paths, int_labels, label_map, lengths

if __name__ == "__main__":
    
    data_dir = Path("data/New Stimuli 9-8-2024")

    print(f"Parsing dataset in: {data_dir}")
    file_paths, int_labels, label_map, lengths = parse_dataset(data_dir)

    print(f"Total WAV files found: {len(file_paths)}")
    print(f"Unique phoneme labels: {len(label_map)}")
    print("Label mapping:")
    for label, idx in label_map.items():
        print(f"  {idx}: {label}")
    print("Lengths of audio files:")
    for file_path, length in zip(file_paths, lengths):
        print(f"  {file_path.name}: {length} samples")