"""
Main entry point for phoneme classification experiment.
Sets up experiment parameters and dispatches training.
"""

from pathlib import Path


from experiment.experiment import Experiment
from experiment.exp_params import ExpParams

if __name__ == "__main__":
        
    params = ExpParams(
        data_path=Path("data/New Stimuli 9-8-2024"),
        output_dir=Path("outputs"),
        log_dir=Path("logs"),

        # Training settings
        epochs=2,                 # Enough to converge (early stopping coming soon)
        batch_size=32,             # Reduce to 16 if you hit memory limits
        learning_rate=3e-4,        # Lower LR for stability
        beta=1.0,                  # Multiplier for KL term

        # KL Annealing
        kl_schedule="sigmoid",     # More gradual ramp-up than linear
        kl_beta_start=0.0,         # Start with no KL regularization
        kl_beta_end=1.0,           # Full KL regularization by anneal_epochs
        kl_anneal_epochs=40,       # Over first 40 epochs (20% of training)

        # Augmentations
        use_pitch_shift=True,
        use_partial_dropout=True,
        use_time_mask=True,
        use_freq_mask=True,

        # Execution
        device="auto",
        
        # Early stopping
        early_stopping_patience = 10,
        early_stopping_delta = 1.0  # minimum change to qualify as improvement
        
    )

    exp = Experiment(params)
    exp.run()