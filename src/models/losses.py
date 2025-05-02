import torch
import torch.nn.functional as F

def vae_loss(x_hat: torch.Tensor, 
             x_target: torch.Tensor, 
             mu: torch.Tensor, 
             logvar: torch.Tensor, 
             beta: float = 1.0) -> torch.Tensor:
    """
    VAE loss = reconstruction + beta * KL divergence.

    Args:
        x_hat (Tensor): Reconstructed output [B, 2, F, T]
        x_target (Tensor): Clean input [B, 2, F, T]
        mu (Tensor): Latent mean [B, latent_dim]
        logvar (Tensor): Latent log-variance [B, latent_dim]
        beta (float): Weight for KL divergence (default = 1.0 for standard VAE)

    Returns:
        Tensor: Scalar total loss
    """
    recon_loss = F.mse_loss(x_hat, x_target)

    # KL divergence: D_KL(q(z|x) || p(z)) for N(μ, σ²) vs N(0,1)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    return recon_loss + beta * kl_loss