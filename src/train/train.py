import multiprocessing
import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

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
from utils.metrics import compute_validation_loss
from utils.evaluate_latent_classification import evaluate_latent_classification

from tqdm import tqdm

def train(params: ExpParams, 
          device: torch.device, 
          parsed_data: tuple) -> None:
    run_dir = create_run_dir(params.output_dir)
    params.run_dir = run_dir
    save_config(params, run_dir / "config.json")

    logger = get_logger("train", log_dir=str(params.log_dir))
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {params.output_dir.resolve()}")
    logger.info(f"Found data at: {params.data_path.resolve()}")

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

    cpu_count = multiprocessing.cpu_count()
    num_workers = max(1, min(10, cpu_count - 2))
    logger.info(f"System has {cpu_count} CPUs, using {num_workers} DataLoader workers")

    if params.use_kfold:
        kf = KFold(n_splits=params.n_splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(file_paths)):
            logger.info(f"Running fold {fold + 1}/{params.n_splits}")
            train_files = [file_paths[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_files = [file_paths[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            train_dataset = PhonemeDataset(
                train_files,
                train_labels,
                transform=transform,
                augment=True,
                augmentation=augment_fn,
                sample_rate=params.target_sr,
                n_augment=params.n_augment,
            )
            
            logger.info(f"Training dataset length (augmented): {len(train_dataset)}")

            val_dataset = PhonemeDataset(
                val_files,
                val_labels,
                transform=transform,
                augment=False,
                sample_rate=params.target_sr,
            )

            _run_training_loop(train_dataset, val_dataset, label_map, params, device, logger, run_dir / f"fold{fold + 1}", num_workers)
    else:
        dataset = PhonemeDataset(
            file_paths,
            labels,
            transform=transform,
            augment=True,
            augmentation=augment_fn,
            sample_rate=params.target_sr,
            n_augment=params.n_augment,
        )
        
        logger.info(f"Training dataset length (augmented): {len(dataset)}")

        _run_training_loop(dataset, None, label_map, params, device, logger, run_dir, num_workers)

def _run_training_loop(dataset, val_dataset, label_map, params, device, logger, run_dir, num_workers):
    run_dir.mkdir(parents=True, exist_ok=True)  # ensure fold subdir exists
    pin_memory = device.type == "cuda"

    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    C, F, T = dataset[0][1].shape
    input_shape = (F, T)
    in_channels = C
    logger.info(f"Input shape to VAE: {input_shape}")

    model = VAE(
        input_shape=input_shape,
        in_channels=in_channels,
        latent_dim=params.latent_dim,
        num_groups=params.num_groups
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    train_losses = []
    val_losses = []
    val_recon_list = []
    val_kl_list = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, params.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_l1 = 0.0
        total_class = 0.0

        for i, (x_aug, x_clean, y_labels, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{params.epochs}", leave=False)):
            x_aug, x_clean = x_aug.to(device), x_clean.to(device)
            y_labels = y_labels.to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x_aug)

            if torch.isnan(mu).any() or torch.isnan(logvar).any():
                logger.warning("NaNs detected in latent parameters (mu or logvar).")
                
            # Compute KL divergence per dimension
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
            kl_dim_mean = kl_per_dim.mean(dim=0)  # shape: [latent_dim]

            # Log every N steps or first batch of each epoch to avoid spam
            if epoch % params.log_latent_every == 0 and i == 0:
                logger.debug(f"[EPOCH {epoch}] KL per dimension: {kl_dim_mean.detach().cpu().numpy()}")

            recon_loss, kl_loss, latent_l1 = vae_loss(
                x_hat, x_clean, mu, logvar,
                free_bits_threshold=params.free_bits_threshold  
            )

            kl_weight = get_beta(
                epoch=epoch,
                schedule=params.kl_schedule,
                beta_start=params.kl_beta_start,
                beta_end=params.kl_beta_end,
                anneal_epochs=params.kl_anneal_epochs,
                cycle_length=params.kl_cycle_length
            )

            l1_weight = 0.01

            loss = recon_loss + params.beta * kl_weight * kl_loss + l1_weight * latent_l1

            if not hasattr(model, 'classifier'):
                # Create a more powerful classifier with multiple layers using LayerNorm instead of BatchNorm
                num_classes = len(label_map)
                model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(params.latent_dim, 32),
                    torch.nn.LayerNorm(32),  # LayerNorm works with any batch size
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(32, num_classes)
                ).to(device)
                # Add the classifier parameters to the optimizer
                optimizer.add_param_group({'params': model.classifier.parameters()})
                logger.info(f"Added MLP classification head: {params.latent_dim} -> 32 -> {num_classes}")
                        
            
            # Compute classification loss
            class_preds = model.classifier(mu)  # Use means directly
            class_loss = torch.nn.functional.cross_entropy(class_preds, y_labels)
            
            # Add classification loss to total (10% weighting)
            class_weight = 0.5
            loss += class_weight * class_loss
            
            # Track classification loss
            if not 'total_class' in locals():
                total_class = 0.0
            total_class += class_loss.item()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_l1 += latent_l1.item()

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        avg_l1 = total_l1 / len(dataloader)
        avg_class = total_class / len(dataloader)
        train_losses.append(avg_loss)

        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=params.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0)
            )
            val_loss, val_recon, val_kl = compute_validation_loss(model, val_loader, device, logger)
            
            latent_acc = None
            if params.log_latent_every and epoch % params.log_latent_every == 0:
                latent_acc = evaluate_latent_classification(
                    model=model,
                    train_dataset=dataset,
                    val_dataset=val_dataset,
                    device=device,
                    label_map=label_map,
                    run_dir=run_dir,
                )
                logger.info(f"[EVAL] Epoch {epoch} — Latent classification accuracy: {latent_acc * 100:.2f}%")
            
            val_losses.append(val_loss)
            val_recon_list.append(val_recon)
            val_kl_list.append(val_kl)
        else:
            val_loss, val_recon, val_kl = None, None, None

        log_msg = (
                f"Epoch {epoch}/{params.epochs} — "
                f"Train Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, "
                f"L1: {avg_l1:.4f}, Class: {avg_class:.4f}, KL Weight: {kl_weight:.4f})"
            )
        
        logger.info(f"[DEBUG] Epoch {epoch} — KL/Reconstruction Ratio: {avg_kl / (avg_recon + 1e-8):.2f}")
        
        if val_loss is not None:
            log_msg += f" | Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})"
        logger.info(log_msg)

        torch.save(model.state_dict(), run_dir / f"vae_epoch{epoch}.pt")

        current_loss = val_loss if val_loss is not None else avg_loss
        if current_loss < best_loss - params.early_stopping_delta:
            best_loss = current_loss
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "model_best.pth")
            logger.info("[INFO] Best model updated.")
        else:
            patience_counter += 1
            logger.info(f"[INFO] No improvement. Patience: {patience_counter}/{params.early_stopping_patience}")

        if patience_counter >= params.early_stopping_patience:
            logger.info("[INFO] Early stopping triggered.")
            break

    loss_dict = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_recon": val_recon_list,
        "val_kl": val_kl_list
    }
    save_loss_history(loss_dict, run_dir / "loss.csv")

    torch.save(model.state_dict(), run_dir / "model_final.pth")

    dataset.augment = False
    extract_latents(model, dataset, device, label_map, run_dir)
    
    if val_dataset is not None:
        acc = evaluate_latent_classification(
            model=model,
            train_dataset=dataset,
            val_dataset=val_dataset,
            device=device,
            label_map=label_map,
            run_dir=run_dir
        )
        logger.info(f"[SUMMARY] Latent classification accuracy: {acc * 100:.2f}%")
