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
    label_map: Dict[int, str],
    run_dir: Path,
) -> None:
    """
    Run the encoder on all samples in the dataset and save latent vectors to a CSV.
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for i in range(len(dataset)):
            _, features_clean, label_idx, filename = dataset[i] # skip augmented waveform
            features_clean = features_clean.unsqueeze(0).to(device)  # [1, C, F, T]
            mu, _ = model.encoder(features_clean)  # Only use mean

            mu = mu.squeeze().cpu().numpy()
            label_str = label_map[label_idx]

            row = {"label": label_str}
            row.update({f"z{j}": mu[j] for j in range(len(mu))})
            rows.append({
                "file": str(filename),
                "label": label_map[label_idx],
                "z1": mu[0],
                "z2": mu[1],
                "z3": mu[2],
            })

    df = pl.DataFrame(rows)
    df.write_csv(run_dir / "latents.csv")