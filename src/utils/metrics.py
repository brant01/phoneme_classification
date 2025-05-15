from models.losses import vae_loss
import torch
from torch.utils.data import DataLoader
from typing import Tuple
import logging

def compute_validation_loss(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger
) -> Tuple[float, float, float]:
    """
    Compute average validation loss components.

    Returns:
        (avg_total_loss, avg_recon_loss, avg_kl_loss)
    """
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_batches = len(val_loader)

    with torch.no_grad():
        for x_aug, x_clean, _, _ in val_loader:
            x_aug = x_aug.to(device)
            x_clean = x_clean.to(device)
            x_hat, mu, logvar = model(x_aug)

            recon_loss, kl_loss = vae_loss(x_hat, x_clean, mu, logvar)
            loss = recon_loss + kl_loss  # KL weight applied during training

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    avg_loss = total_loss / n_batches
    avg_recon = total_recon / n_batches
    avg_kl = total_kl / n_batches

    logger.info(
        f"[VAL] Validation — Total: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}"
    )

    return avg_loss, avg_recon, avg_kl