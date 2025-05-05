from dataclasses import dataclass
from pathlib import Path
from typing import Literal

KLAnnealSchedule = Literal["none", "linear", "sigmoid"]

@dataclass
class ExpParams:
    """
    All experiment-level configuration and hyperparameters.
    """

    # Paths
    data_path: Path 
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")
    
    # Model
    input_shape: tuple[int, int] = (32, 16000)  # (channels, time)
    latent_dim: int = 3
    
    # Training
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    beta: float = 1.0  # global KL multiplier
    
    # KL Annealing
    kl_schedule: KLAnnealSchedule = "linear"
    kl_beta_start: float = 0.0
    kl_beta_end: float = 1.0
    kl_anneal_epochs: int = 10

    # Augmentation Flags
    use_pitch_shift: bool = True
    use_partial_dropout: bool = True
    use_time_mask: bool = True
    use_freq_mask: bool = True
    
    # Execution
    device: str = "auto"  # "cpu", "cuda", "mps", or "auto"
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1.0  # minimum change to qualify as improvement

    def to_dict(self) -> dict:
        """
        Convert the dataclass to a dictionary, converting Paths to strings.
        """
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }