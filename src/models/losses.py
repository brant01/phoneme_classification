import torch
import torch.nn.functional as F

def vae_loss(
    x_hat: torch.Tensor, 
    x_target: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    VAE loss components: reconstruction loss and KL divergence.

    Args:
        x_hat (Tensor): Reconstructed output [B, 2, F, T]
        x_target (Tensor): Clean input [B, 2, F, T]
        mu (Tensor): Latent mean [B, latent_dim]
        logvar (Tensor): Latent log-variance [B, latent_dim]

    Returns:
        tuple:
            - recon_loss (Tensor): Scalar reconstruction loss
            - kl_div (Tensor): Scalar KL divergence loss
    """
    # Mean squared error over the whole spectrogram
    recon_loss = F.mse_loss(x_hat, x_target, reduction="mean")

    # Clamp logvar to avoid instability
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)

    # KL divergence term
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalize KL divergence by batch size × latent dimension
    kl_div = kl_div / (mu.shape[0] * mu.shape[1])

    return recon_loss, kl_div