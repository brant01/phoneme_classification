import torch
import torch.nn as nn

from src.models.initialize import initialize_weights


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_shape: tuple[int, int],
        latent_dim: int,
        num_groups: int = 8
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_groups = num_groups

        # Add an initial aggressive downsampling layer
        self.initial_downsample = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=4, padding=2),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(self.num_groups, out_c), num_channels=out_c),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.1)
            )

        self.encoder = nn.Sequential(
            conv_block(16, 32),    # Changed input from in_channels to 16
            conv_block(32, 64),
            conv_block(64, 128),
            # Remove the last layer to reduce dimensions
            # conv_block(128, 256)  # Removed
        )
        
        # Add global pooling to get a fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape)
            x = self.initial_downsample(dummy)
            x = self.encoder(x)
            x = self.global_pool(x)
            self.flattened_size = x.view(1, -1).shape[1]
            print(f"DEBUG: Flattened size after pooling: {self.flattened_size}")
            
        self.apply(initialize_weights)

        # Simplified bottleneck with fewer parameters
        self.bottleneck = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.01),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_downsample(x)  # Add this line
        x = self.encoder(x)
        x = self.global_pool(x)         # Add this line
        x = torch.flatten(x, start_dim=1)
        x = self.bottleneck(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-5.0, max=5.0)
        return mu, logvar