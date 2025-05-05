import torch
import shutil
from pathlib import Path
from models.vae import VAE
from utils.extract_latents import extract_latents
import polars as pl

class DummyDataset:
    def __init__(self, length: int = 10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        x = torch.randn(2, 32, 100)
        return x, x, idx, f"dummy_file_{idx}.wav"  # Include filename string

def test_extract_latents_outputs_correct_csv(tmp_path: Path = Path("tests/tmp_latents")):
    tmp_path.mkdir(parents=True, exist_ok=True)

    dataset = DummyDataset(length=10)
    model = VAE(input_shape=(32, 100), in_channels=2, latent_dim=3)
    label_map = {i: f"label_{i}" for i in range(10)}

    extract_latents(model, dataset, device=torch.device("cpu"), label_map=label_map, run_dir=tmp_path)

    csv_path = tmp_path / "latents.csv"
    assert csv_path.exists(), "Latents CSV was not created"

    df = pl.read_csv(csv_path)
    assert df.shape[0] == 10, "Unexpected number of rows in CSV"
    assert all(col in df.columns for col in ["file", "label", "z1", "z2", "z3"]), "Missing expected columns"
    assert df.null_count().sum_horizontal().item() == 0, "Found null values in output"

    shutil.rmtree(tmp_path)