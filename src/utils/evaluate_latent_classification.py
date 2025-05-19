import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import logging

from models.vae import VAE  # or update if your VAE import path is different


def evaluate_latent_classification(
    model: VAE,
    train_dataset,
    val_dataset,
    device: torch.device,
    label_map: dict,
    run_dir
) -> float:
    """
    Train a simple logistic regression classifier on the VAE latents and evaluate accuracy.
    """

    logger = logging.getLogger("train")

    model.eval()
    with torch.no_grad():
        def extract_latents(dataset):
            latents = []
            targets = []
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            for x_aug, _, y_idx, _ in loader:
                x_aug = x_aug.to(device)
                _, mu, _ = model(x_aug)
                latents.append(mu.squeeze(0).cpu().numpy())
                targets.append(y_idx.item())

            return np.array(latents), np.array(targets)

        # Train features and labels
        X_train, y_train = extract_latents(train_dataset)
        X_val, y_val = extract_latents(val_dataset)

    # Fit simple classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict on validation
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    logger.info(f"[LATENT EVAL] Logistic Regression Accuracy: {acc * 100:.2f}%")

    # Save if needed
    np.save(run_dir / "latent_val_preds.npy", y_pred)
    np.save(run_dir / "latent_val_targets.npy", y_val)

    return acc