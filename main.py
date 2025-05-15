"""
Main entry point for phoneme classification experiment.
Sets up experiment parameters and dispatches training.
"""

from pathlib import Path
import torch
import random
import numpy as np
import torch.multiprocessing as mp

from experiment.experiment import Experiment
from experiment.exp_params import ExpParams


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set in this process

    set_random_seeds(42)

    params = ExpParams(
        data_path=Path("data/New Stimuli 9-8-2024"),
        output_dir=Path("outputs"),
        log_dir=Path("logs"),

        # Training settings
        epochs=300,
        batch_size=4,
        learning_rate=3e-4,
        beta=1.0,

        # KL Annealing
        kl_schedule="sigmoid",
        kl_beta_start=0.0,
        kl_beta_end=0.05,
        kl_anneal_epochs=1000,

        # Augmentations
        use_pitch_shift=True,
        use_partial_dropout=True,
        use_time_mask=True,
        use_freq_mask=True,
        device="auto",

        # Early stopping
        early_stopping_patience=100,
        early_stopping_delta=2.0,
    )

    exp = Experiment(params)
    exp.run()
