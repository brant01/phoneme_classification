import torch
from torch import optim
from torch.utils.data import DataLoader

from src.experiment.exp_params import ExpParams
from models.vae import VAE
from models.losses import vae_loss
from data_utils.dataset import PhonemeDataset
from data_utils.transform import WaveletHilbertTransform
from data_utils.augmentations import AugmentationPipeline
from utils.extract_latents import extract_latents
from utils.run_manager import create_run_dir, save_config, save_loss_history

from tqdm import tqdm


def train(params: ExpParams, 
          device: torch.device, 
          parsed_data: tuple) -> None:
    """
    Full training loop based on experiment parameters.
    """
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output directory: {params.output_dir.resolve()}")
    print(f"[INFO] Found data at: {params.data_path.resolve()}")

    # Run manager for saving metadata
    run_dir = create_run_dir(params.output_dir)
    params.run_dir = run_dir  # Store for downstream access
    save_config(params, run_dir / "config.json")

    # --------------------------
    # Load dataset
    # --------------------------
    file_paths, labels, label_map, lengths = parsed_data
    print(f"[INFO] Found {len(file_paths)} files")
    print(f"[INFO] Found {len(label_map)} unique labels")

    output_len = int(max(lengths) * 1.2)
    print(f"[INFO] Longest file length: {max(lengths)} samples")
    print(f"[INFO] Computed output_len: {output_len}")

    transform = WaveletHilbertTransform(output_len=output_len)

    augment_fn = AugmentationPipeline(
        pitch_shift=params.use_pitch_shift,
        partial_dropout=params.use_partial_dropout,
        time_mask=params.use_time_mask,
        freq_mask=params.use_freq_mask,
        prob=1.0
    )

    dataset = PhonemeDataset(
        file_paths,
        labels,
        transform=transform,
        augment=True,
        augmentation=augment_fn,
        sample_rate=16000,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=0
    )

    # --------------------------
    # Build model
    # --------------------------
    C, F, T = dataset[0][1].shape
    input_shape = (F, T)
    in_channels = C
    print(f"[INFO] Input shape to VAE: {input_shape}")

    model = VAE(
        input_shape=input_shape,
        in_channels=in_channels,
        latent_dim=params.latent_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # --------------------------
    # Training loop
    # --------------------------
    torch.autograd.set_detect_anomaly(True)

    train_losses = []
    val_losses = []  # for future use

    for epoch in range(1, params.epochs + 1):
        model.train()
        total_loss = 0.0

        for x_aug, x_clean, _, _ in tqdm(dataloader, desc=f"Epoch {epoch}/{params.epochs}", leave=False):
            x_aug, x_clean = x_aug.to(device), x_clean.to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x_aug)

            if torch.isnan(mu).any() or torch.isnan(logvar).any():
                print("[WARN] NaNs in latent parameters")

            loss = vae_loss(x_hat, x_clean, mu, logvar, beta=params.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f"Epoch {epoch}/{params.epochs} — Loss: {avg_loss:.6f}")

        # Save checkpoint
        torch.save(
            model.state_dict(),
            run_dir / f"vae_epoch{epoch}.pt"
        )

    # --------------------------
    # Save loss history, model
    # --------------------------
    loss_dict = {
        "train_loss": train_losses,
        "val_loss": val_losses  # safe placeholder
    }
    save_loss_history(loss_dict, run_dir / "loss.csv")
    
    # Save final model state dict
    torch.save(model.state_dict(), run_dir / "model_final.pth")
    
    # Extract and save latent vectors
    dataset.augment = False  # disable aug for latent extraction
    extract_latents(model, dataset, device, label_map, run_dir)