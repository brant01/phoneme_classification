import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from models.vae import VAE
from models.losses import vae_loss
from tqdm import tqdm

from torch.optim import Adam
from pathlib import Path

from data_utils.loader import parse_dataset
from data_utils.dataset import PhonemeDataset
from data_utils.transform import WaveletHilbertTransform
from data_utils.augmentations import AugmentationPipeline

from utils.device import get_best_device

def train_one_epoch(
    model: VAE,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    beta: float = 0.1
) -> float:
    """
    Run one training epoch for a VAE.

    Args:
        model (VAE): The variational autoencoder model
        dataloader (DataLoader): Training data
        optimizer (Optimizer): Optimizer
        device (torch.device): Device to run on
        beta (float): Weight for KL divergence

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for x_aug, x_clean, _ in tqdm(dataloader, desc="Training"):
        x_aug = x_aug.to(device)
        x_clean = x_clean.to(device)

        optimizer.zero_grad()
        x_hat, mu, logvar = model(x_aug)
        loss = vae_loss(x_hat, x_clean, mu, logvar, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(
    data_path: Path,
    input_shape: tuple[int, int],
    latent_dim: int = 3,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: torch.device = torch.device("cpu"),
    output_dir: Path = Path("outputs")
    ) -> None:
    """
    Full training loop for the VAE model.
    """
    device = get_best_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output directory: {output_dir.resolve()}")

    # Load and parse data
    file_paths, labels, _, lengths = parse_dataset(data_path)
    print(f"[INFO] Found {len(file_paths)} files")
    print(f"[INFO] Found {len(set(labels))} unique labels")
    print(f"[INFO] Longest file length: {max(lengths)} samples")
    output_len = int(max(lengths) * 1.1)
    input_shape = (32, output_len) 
    print(f"[INFO] Computed output_len: {output_len}")
    print(f"[INFO] Input shape to VAE: {input_shape}")

    transform = WaveletHilbertTransform(output_len=output_len)
    augment = AugmentationPipeline(pitch_shift=True, partial_dropout=True, time_mask=True, freq_mask=True)

    dataset = PhonemeDataset(file_paths, labels, transform=transform, augmentation=augment, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model and optimizer
    model = VAE(input_shape=input_shape, latent_dim=latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, beta=beta)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = output_dir / f"vae_epoch{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        
    
def evaluate(
    model: VAE,
    dataloader: DataLoader,
    device: torch.device,
    beta: float = 1.0
) -> float:
    """
    Evaluate the VAE model on a validation set.

    Args:
        model (VAE): Trained VAE
        dataloader (DataLoader): Validation dataloader
        device (torch.device): CUDA / MPS / CPU
        beta (float): KL divergence weight

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_aug, x_clean, _ in dataloader:
            x_aug = x_aug.to(device)
            x_clean = x_clean.to(device)

            x_hat, mu, logvar = model(x_aug)
            loss = vae_loss(x_hat, x_clean, mu, logvar, beta=beta)
            total_loss += loss.item()

    return total_loss / len(dataloader)

if __name__ == "__main__":

    train(
        data_path=Path("data/New Stimuli 9-8-2024"),  # adjust if needed
        input_shape=(32, 16000),
        epochs=5,
        batch_size=4,
        latent_dim=3,
        lr=1e-3
    )