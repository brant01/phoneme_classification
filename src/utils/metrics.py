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
    total_l1 = 0.0  # Add L1 tracking
    n_batches = len(val_loader)

    with torch.no_grad():
        for x_aug, x_clean, _, _ in val_loader:
            x_aug = x_aug.to(device)
            x_clean = x_clean.to(device)
            x_hat, mu, logvar = model(x_aug)

            # Updated call with 3 components
            recon_loss, kl_loss, latent_l1 = vae_loss(x_hat, x_clean, mu, logvar)
            
            # Add L1 weight - should match training
            l1_weight = 0.01
            
            # Update loss calculation to include L1
            loss = recon_loss + kl_loss + l1_weight * latent_l1

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_l1 += latent_l1.item()

    avg_loss = total_loss / n_batches
    avg_recon = total_recon / n_batches
    avg_kl = total_kl / n_batches
    # The L1 component isn't returned currently, but tracked for possible future use

    # The function's return signature remains the same for compatibility
    return avg_loss, avg_recon, avg_kl