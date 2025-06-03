from pathlib import Path
from typing import Dict
import polars as pl
import torch
from models.vae import VAE
from data_utils.dataset import PhonemeDataset


def extract_latents(
    model: VAE,
    dataset: PhonemeDataset,
    device: torch.device,
    label_map: Dict,  # can be str→int or int→str
    run_dir: Path,
) -> None:
    """
    Run the encoder on all samples in the dataset and save latent vectors to a CSV.
    """
    # Normalize label map direction
    if all(isinstance(k, str) and isinstance(v, int) for k, v in label_map.items()):
        index_to_label = {v: k for k, v in label_map.items()}
    elif all(isinstance(k, int) and isinstance(v, str) for k, v in label_map.items()):
        index_to_label = label_map
    else:
        raise ValueError("label_map must be either Dict[str, int] or Dict[int, str]")

    model.eval()
    rows = []

    with torch.no_grad():
        for i in range(len(dataset)):
            _, features_clean, label_idx, filename = dataset[i]  # skip augmented waveform
            features_clean = features_clean.unsqueeze(0).to(device)  # [1, C, F, T]
            mu, _ = model.encoder(features_clean)  # Use mean only

            mu = mu.squeeze().cpu().numpy()

            if label_idx not in index_to_label:
                raise KeyError(f"Label index {label_idx} not found in label_map.")

            label_str = index_to_label[label_idx]
            row = {
                "file": str(filename),
                "label": label_str,
            }
            row.update({f"z{j+1}": mu[j] for j in range(len(mu))})
            rows.append(row)

    df = pl.DataFrame(rows)
    df.write_csv(run_dir / "latents.csv")