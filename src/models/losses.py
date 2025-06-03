import torch
import torch.nn.functional as F


def vae_loss(
    x_hat: torch.Tensor, 
    x_target: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor,
    free_bits_threshold: float = 0.1,
    max_kl: float = 1000.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Updated return type to include latent_l1
    """
    VAE loss components: reconstruction loss, KL divergence with Free Bits, and latent regularization.

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
            - latent_l1 (Tensor): L1 regularization on latent means
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(x_hat, x_target, reduction="mean")

    # Clamp logvar to avoid instability
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)

    # KL per-dimension, per-sample: shape [B, D]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Sum over dimensions first
    kl_per_sample = kl_per_dim.sum(dim=1)  # [B]
    
    # Apply free bits threshold to the total KL per sample
    free_bits_total = free_bits_threshold * mu.shape[1]  # Total free bits scaled by latent dimensions
    kl_clamped = torch.maximum(kl_per_sample, 
                              torch.tensor(free_bits_total, device=kl_per_sample.device))
    
    # Average over batch
    kl_div = kl_clamped.mean()

    # Clamp final KL to prevent runaway loss
    kl_div = torch.clamp(kl_div, max=max_kl)
    
    # Add L1 regularization on latent means to encourage sparsity
    latent_l1 = mu.abs().mean()
    
    return recon_loss, kl_div, latent_l1