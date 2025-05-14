
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Convolutional encoder for a VAE.
    Input: [B, C, F, T]  (e.g., [batch, 2, 32, 16000])
    Output: mu, logvar vectors of shape [B, latent_dim]
    """

    def __init__(self, 
                 in_channels: int, 
                 input_shape: tuple[int, int], 
                 latent_dim: int) -> None:
        """
        Args:
            in_channels (int): Input channels (e.g., 2 for wavelet + hilbert)
            input_shape (tuple): (num_scales, output_len)
            latent_dim (int): Dimension of latent space
        """
        super().__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # [B, C, F, T] → [B, 32, F/2, T/2]
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            # [B, 32, F/2, T/2] → [B, 64, F/4, T/4]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # [B, 64, F/4, T/4] → [B, 128, F/8, T/8]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            # [B, 128, F/8, T/8] → [B, 256, F/16, T/16]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Compute the flattened output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape).to(next(self.encoder.parameters()).device)
            conv_out = self.encoder(dummy)
            self.flattened_size = conv_out.view(1, -1).shape[1]


        # Fully connected layers to produce mean and log-variance
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder network.

        Args:
            x (Tensor): Input of shape [B, C, F, T]

        Returns:
            tuple: mu and logvar of shape [B, latent_dim]
        """
        x = self.encoder(x)                  # apply conv blocks
        x = torch.flatten(x, start_dim=1)    # flatten to [B, N]
        mu = self.fc_mu(x)                   # latent mean
        logvar = self.fc_logvar(x)           # latent log-variance
        return mu, logvar