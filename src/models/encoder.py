import torch
import torch.nn as nn

from src.models.initialize import initialize_weights


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_shape: tuple[int, int],
        latent_dim: int,
        num_groups: int = 8  # Default for GroupNorm
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_groups = num_groups

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(self.num_groups, out_c), num_channels=out_c),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.1)
            )

        self.encoder = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape).to(next(self.encoder.parameters()).device)
            conv_out = self.encoder(dummy)
            self.flattened_size = conv_out.view(1, -1).shape[1]
            
        self.apply(initialize_weights)


        # Fully connected layers to produce mean and log-variance
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-5.0, max=5.0)
        return mu, logvar