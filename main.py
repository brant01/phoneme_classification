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
        #data_path=Path("data/New Stimuli 9-8-2024/CV/Female"),
        output_dir=Path("outputs"),
        log_dir=Path("logs"),

        # Model
        input_shape=(32, 16000),
        latent_dim=8,
        num_groups=16,

        # Training
        epochs=200,
        batch_size=2,
        learning_rate=3e-4,
        beta=1.0,

        # KL Annealing
        kl_schedule="cyclical",
        kl_beta_start=0.0,
        kl_beta_end=1.0,
        kl_anneal_epochs=200,
        kl_cycle_length=5,

        # Augmentations
        use_pitch_shift=False,
        use_partial_dropout=True,
        use_time_mask=True,
        use_freq_mask=True,
        n_augment=5,

        # Device and early stopping
        device="auto",
        early_stopping_patience=1000,
        early_stopping_delta=2.0,

        # Cross-validation
        use_kfold=True,
        n_splits=5,
        
        # logging latent values
        log_latent_every = 5,  # Evaluate latent classification every N epochs
        
        # Free bits
        free_bits_threshold=0.1,
    )

    exp = Experiment(params)
    exp.run()
