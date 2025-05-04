
import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder


class VAE(nn.Module):
    """
    Variational Autoencoder composed of encoder, reparameterization, and decoder.
    """

    def __init__(self, input_shape: tuple[int, int], in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(input_shape=input_shape, in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(input_shape=input_shape, in_channels=in_channels, latent_dim=latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample z using the reparameterization trick: z = mu + σ * ε

        Args:
            mu (Tensor): Mean of latent distribution
            logvar (Tensor): Log-variance of latent distribution

        Returns:
            Tensor: Sampled latent vector z
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x (Tensor): Input tensor of shape [B, 2, F, T]

        Returns:
            Tuple: (reconstructed x̂, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        # Crop or pad x_hat to match x
        x_hat = self._match_output_length(x_hat, x)
        if torch.isnan(mu).any() or torch.isnan(logvar).any():
            print("[WARN] NaNs in latent parameters")
        if torch.isnan(x_hat).any():
            print("[WARN] NaNs in reconstruction")
        return x_hat, mu, logvar
    
    def _match_output_length(self, x_hat: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Ensure x_hat matches x_target length along time axis (dim=3).
        If mismatch is small, crop or pad x_hat.
        """
        T_out = x_hat.shape[-1]
        T_target = x_target.shape[-1]

        if T_out > T_target:
            return x_hat[..., :T_target]
        elif T_out < T_target:
            pad_amount = T_target - T_out
            return torch.nn.functional.pad(x_hat, (0, pad_amount), mode='constant', value=0.0)
        else:
            return x_hat