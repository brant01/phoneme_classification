import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder


class VAE(nn.Module):
    """
    Variational Autoencoder composed of encoder, reparameterization, and decoder.
    """

    def __init__(self, input_shape: tuple[int, int], in_channels: int, latent_dim: int, num_groups: int = 8) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_shape=input_shape, 
            in_channels=in_channels, 
            latent_dim=latent_dim, 
            num_groups=num_groups
        )
        self.decoder = Decoder(
            input_shape=input_shape, 
            in_channels=in_channels, 
            latent_dim=latent_dim, 
            num_groups=num_groups
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        z = torch.clamp(z, min=-10.0, max=10.0)
        x_hat = self.decoder(z)
        x_hat = self._match_output_length(x_hat, x)
        if torch.isnan(mu).any() or torch.isnan(logvar).any():
            print("[WARN] NaNs in latent parameters")
        if torch.isnan(x_hat).any():
            print("[WARN] NaNs in reconstruction")
        return x_hat, mu, logvar

    def _match_output_length(self, x_hat: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        T_out = x_hat.shape[-1]
        T_target = x_target.shape[-1]
        if T_out > T_target:
            return x_hat[..., :T_target]
        elif T_out < T_target:
            pad_amount = T_target - T_out
            return torch.nn.functional.pad(x_hat, (0, pad_amount), mode='constant', value=0.0)
        else:
            return x_hat