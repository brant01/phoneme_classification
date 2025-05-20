import torch
import torch.nn.functional as F


def vae_loss(
    x_hat: torch.Tensor, 
    x_target: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor,
    free_bits_threshold: float = 0.1,
    max_kl: float = 1000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    VAE loss components: reconstruction loss and KL divergence with Free Bits.

    Args:
        x_hat (Tensor): Reconstructed output [B, 2, F, T]
        x_target (Tensor): Clean input [B, 2, F, T]
        mu (Tensor): Latent mean [B, D]
        logvar (Tensor): Latent log-variance [B, D]
        free_bits_threshold (float): Minimum KL per dimension
        max_kl (float): Optional max cap for KL divergence to prevent explosion

    Returns:
        tuple:
            - recon_loss (Tensor): Scalar reconstruction loss
            - kl_div (Tensor): Scalar KL divergence with free bits
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(x_hat, x_target, reduction="mean")

    # Clamp logvar to avoid instability
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)

    # KL per-dimension, per-sample: shape [B, D]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Apply free bits threshold
    free_bits = torch.full_like(kl_per_dim, free_bits_threshold)
    kl_clamped = torch.maximum(kl_per_dim, free_bits)  # shape [B, D]

    # Sum over D, average over batch
    kl_div = kl_clamped.sum(dim=1).mean()

    # Clamp final KL to prevent runaway loss
    kl_div = torch.clamp(kl_div, max=max_kl)

    return recon_loss, kl_div