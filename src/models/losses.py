import torch
import torch.nn.functional as F
from typing import Literal


def vae_loss(
    x_hat: torch.Tensor,
    x_target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    recon_loss_type: Literal["mse", "l1"] = "mse",
    reduction: Literal["mean", "sum"] = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss: reconstruction + KL divergence.

    Args:
        x_hat (Tensor): Reconstructed output [B, C, F, T]
        x_target (Tensor): Ground truth input [B, C, F, T]
        mu (Tensor): Latent mean [B, latent_dim]
        logvar (Tensor): Latent log-variance [B, latent_dim]
        recon_loss_type (str): Type of reconstruction loss ('mse' or 'l1').
        reduction (str): Reduction method for reconstruction loss ('mean' or 'sum').

    Returns:
        tuple:
            - recon_loss (Tensor): Reconstruction loss scalar
            - kl_div (Tensor): KL divergence scalar
    """
    if recon_loss_type == "mse":
        recon_loss = F.mse_loss(x_hat, x_target, reduction=reduction)
    elif recon_loss_type == "l1":
        recon_loss = F.l1_loss(x_hat, x_target, reduction=reduction)
    else:
        raise ValueError(f"Unsupported reconstruction loss type: {recon_loss_type}")

    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_div = kl_div / mu.size(0)  # Average over batch

    return recon_loss, kl_div