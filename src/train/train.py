import multiprocessing
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
from utils.logger import get_logger
from utils.schedules import get_beta

from tqdm import tqdm

def train(params: ExpParams, 
          device: torch.device, 
          parsed_data: tuple) -> None:
    """
    Full training loop based on experiment parameters.
    """

    # Run manager for saving metadata
    run_dir = create_run_dir(params.output_dir)
    params.run_dir = run_dir  # Store for downstream access
    save_config(params, run_dir / "config.json")

    # Logger setup
    logger = get_logger("train", log_dir=str(params.log_dir))
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {params.output_dir.resolve()}")
    logger.info(f"Found data at: {params.data_path.resolve()}")

    # --------------------------
    # Load dataset
    # --------------------------
    file_paths, labels, label_map, lengths = parsed_data
    logger.info(f"Found {len(file_paths)} files")
    logger.info(f"Found {len(label_map)} unique labels")

    output_len = int(max(lengths) * 1.2)
    logger.info(f"Longest file length: {max(lengths)} samples")
    logger.info(f"Computed output_len: {output_len}")

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

    cpu_count = multiprocessing.cpu_count()
    num_workers = max(1, min(8, cpu_count // 2))
    logger.info(f"System has {cpu_count} CPUs, using {num_workers} DataLoader workers")
    
    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers = num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    # --------------------------
    # Build model
    # --------------------------
    C, F, T = dataset[0][1].shape
    input_shape = (F, T)
    in_channels = C
    logger.info(f"Input shape to VAE: {input_shape}")

    model = VAE(
        input_shape=input_shape,
        in_channels=in_channels,
        latent_dim=params.latent_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # --------------------------
    # Training loop
    # --------------------------
    #torch.autograd.set_detect_anomaly(True)

    train_losses = []
    val_losses = []  # for future use
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, params.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for x_aug, x_clean, _, _ in tqdm(dataloader, desc=f"Epoch {epoch}/{params.epochs}", leave=False):
            x_aug, x_clean = x_aug.to(device), x_clean.to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x_aug)

            if torch.isnan(mu).any() or torch.isnan(logvar).any():
                logger.warning("NaNs detected in latent parameters (mu or logvar).")

            recon_loss, kl_loss = vae_loss(x_hat, x_clean, mu, logvar)

            kl_weight = get_beta(
                epoch=epoch,
                schedule=params.kl_schedule,
                beta_start=params.kl_beta_start,
                beta_end=params.kl_beta_end,
                anneal_epochs=params.kl_anneal_epochs
            )

            loss = recon_loss + params.beta * kl_weight * kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        train_losses.append(avg_loss)

        logger.info(
            f"Epoch {epoch}/{params.epochs} — Total Loss: {avg_loss:.4f}, "
            f"Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, KL Weight: {kl_weight:.4f}"
        )

        # Save checkpoint each epoch
        torch.save(model.state_dict(), run_dir / f"vae_epoch{epoch}.pt")

        # Save best model
        if avg_loss < best_loss - params.early_stopping_delta:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "model_best.pth")
            logger.info("[INFO] Best model updated.")
        else:
            patience_counter += 1
            logger.info(f"[INFO] No improvement. Patience: {patience_counter}/{params.early_stopping_patience}")

        # Early stopping check
        if patience_counter >= params.early_stopping_patience:
            logger.info("[INFO] Early stopping triggered.")
            break

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