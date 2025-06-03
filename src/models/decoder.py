import torch
import torch.nn as nn

from src.models.initialize import initialize_weights

class Decoder(nn.Module):
    """
    Convolutional decoder for a VAE.
    Input: [B, latent_dim]
    Output: [B, 2, F, T]
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        in_channels: int,
        latent_dim: int,
        num_groups: int = 8,
        dropout_prob: float = 0.3 
    ) -> None:
        """
        Args:
            input_shape (tuple): (F, T) shape of output features
            in_channels (int): Number of output channels (usually 2 for real + phase)
            latent_dim (int): Dimension of latent space
            num_groups (int): Number of groups for GroupNorm
            dropout_prob (float): Dropout probability for decoder layers
        """
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_groups = num_groups
        self.dropout_prob = dropout_prob

        self.projected_channels = 64 # reduced from 256

        F, T = input_shape
        self.projected_shape = (
            self.projected_channels,
            F // 16,
            T // 16,
        )

        flattened_size = self.projected_channels * self.projected_shape[1] * self.projected_shape[2]

        self.fc = nn.Linear(latent_dim, flattened_size)

        def deconv_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(self.num_groups, out_c), num_channels=out_c),
                nn.LeakyReLU(0.01),
                nn.Dropout2d(p=self.dropout_prob)  # <-- Dropout after activation
            )

        self.decoder = nn.Sequential(
            deconv_block(64, 48),  
            deconv_block(48, 32),  
            deconv_block(32, 16),  
            nn.ConvTranspose2d(16, self.in_channels, kernel_size=4, stride=2, padding=1),  
            nn.Tanh()
        )
        
        self.apply(initialize_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, *self.projected_shape)
        x = self.decoder(x)
        return x