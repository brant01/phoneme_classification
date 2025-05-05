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
        epochs=1,
        batch_size=4,
        latent_dim=2,
        learning_rate=1e-3,
        beta=1.0,
        use_pitch_shift=True,
        use_partial_dropout=True,
        use_time_mask=True,
        use_freq_mask=True,
        device="auto",
        #verbose=False,
        #seed=42
    )

    exp = Experiment(params)
    exp.run()