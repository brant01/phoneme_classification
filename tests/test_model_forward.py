import torch
from models.vae import VAE

def test_vae_forward_pass():
    # Synthetic input
    batch_size = 4
    input_shape = (32, 16000)     # (num_scales, output_len)
    latent_dim = 3
    x = torch.randn(batch_size, 2, *input_shape)  # shape: [B, 2, F, T]

    # Instantiate model
    model = VAE(input_shape=input_shape, in_channels=2, latent_dim=latent_dim)

    # Forward pass
    x_hat, mu, logvar = model(x)

    # Assertions
    assert x_hat.shape == x.shape
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)

    # Optional: check numerical properties
    assert not torch.isnan(x_hat).any()
    assert not torch.isnan(mu).any()
    assert not torch.isnan(logvar).any()