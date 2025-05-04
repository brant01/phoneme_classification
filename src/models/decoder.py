import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Convolutional decoder for a VAE.
    Input: [B, latent_dim]
    Output: [B, 2, F, T]
    """

    def __init__(self,
                 input_shape: tuple[int, int],
                 in_channels: int,
                 latent_dim: int) -> None:
        """
        Args:
            input_shape (tuple): (F, T) shape of output features
            in_channels (int): Number of output channels (usually 2 for real + phase)
            latent_dim (int): Dimension of latent space
        """
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Projected spatial size after deconv (assume 4x downsampling per Conv2D layer in encoder)
        self.projected_channels = 256
        F, T = input_shape
        self.projected_shape = (
            self.projected_channels,
            F // 16,
            T // 16,
        )
        flattened_size = self.projected_channels * self.projected_shape[1] * self.projected_shape[2]

        # Fully connected to go from latent space to projected feature map
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

            nn.ConvTranspose2d(32, self.in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Optional: adjust depending on output normalization
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (Tensor): Latent code [B, latent_dim]

        Returns:
            Tensor: Decoded features [B, in_channels, F, T]
        """
        x = self.fc(z)
        x = x.view(-1, *self.projected_shape)
        x = self.decoder(x)
        return x