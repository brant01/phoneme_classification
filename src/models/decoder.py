
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Convolutional decoder for a VAE.
    Input: [B, latent_dim]
    Output: [B, 2, F, T]
    """

    def __init__(self, 
                 latent_dim: int, 
                 output_shape: tuple[int, int], 
                 ) -> None:
        """
        Args:
            latent_dim (int): Latent dimension
            output_shape (tuple): Target output shape (num_scales, time_steps)
        """
        super().__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim

        # Reverse of encoder flattening size
        self.projected_channels = 256
        self.projected_shape = (self.projected_channels, output_shape[0] // 16, output_shape[1] // 16)
        flattened_size = self.projected_channels * self.projected_shape[1] * self.projected_shape[2]

        self.fc = nn.Linear(latent_dim, flattened_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # output in range [-1, 1]; adjust if needed
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z (Tensor): Latent vector [B, latent_dim]

        Returns:
            Tensor: Reconstructed features [B, 2, F, T]
        """
        x = self.fc(z)  # [B, flattened_size]
        x = x.view(-1, *self.projected_shape)
        x = self.decoder(x)
        return x