
import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder


class VAE(nn.Module):
    """
    Variational Autoencoder composed of encoder, reparameterization, and decoder.
    """

    def __init__(self, input_shape: tuple[int, int], latent_dim: int):
        """
        Args:
            input_shape (tuple): (num_scales, output_len)
            latent_dim (int): Dimension of latent space
        """
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(in_channels=2, input_shape=input_shape, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_shape=input_shape)

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
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar