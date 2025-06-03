import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import logging

from models.vae import VAE  # Adjust if your import path differs

def evaluate_latent_classification(
    model: VAE,
    train_dataset,
    val_dataset,
    device: torch.device,
    label_map: dict,
    run_dir
) -> float:
    """
    Train a classifier on VAE latent means and evaluate on validation set.

    Args:
        model (VAE): Trained VAE model
        train_dataset: Dataset to extract training latents
        val_dataset: Dataset to extract validation latents
        device (torch.device): Device to run computations
        label_map (dict): Mapping of label indices to phoneme names (not used here)
        run_dir: Path to save predictions and targets

    Returns:
        float: Validation classification accuracy
    """
    logger = logging.getLogger("train")
    model.eval()

    def extract_latents(dataset):
        latents = []
        targets = []
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        with torch.no_grad():
            for x_aug, _, y_idx, _ in loader:
                x_aug = x_aug.to(device)
                mu, _ = model.encoder(x_aug)
                latents.append(mu.cpu().numpy())
                targets.extend(y_idx.cpu().numpy())
        return np.concatenate(latents, axis=0), np.array(targets)

    X_train, y_train = extract_latents(train_dataset)
    X_val, y_val = extract_latents(val_dataset)

    # Log latent stats
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    logger.info(f"[LATENT STATS] Train latents mean (first 5 dims): {train_mean[:5]}")
    logger.info(f"[LATENT STATS] Train latents std  (first 5 dims): {train_std[:5]}")

    # Standardize latent features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train classifier (use RandomForest or LogisticRegression)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    logger.info(f"[LATENT EVAL] Classifier Accuracy: {acc * 100:.2f}%")

    # Save results
    np.save(run_dir / "latent_val_preds.npy", y_pred)
    np.save(run_dir / "latent_val_targets.npy", y_val)

    return acc