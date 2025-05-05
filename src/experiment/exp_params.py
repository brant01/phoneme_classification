from dataclasses import dataclass
from pathlib import Path

@dataclass
class ExpParams:
    """
    ALl experiment level configuation and hyperparameters
    """
    # Paths
    data_path: Path 
    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")
    
    # Model
    input_shape: tuple[int, int] = (32, 16000) # channels, time
    latent_dim: int = 3
    
    # Training
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    beta: float = 1.0
    
    # Augmentation Flags
    use_pitch_shift: bool = True
    use_partial_dropout: bool = True
    use_time_mask: bool = True
    use_freq_mask: bool = True
    
    # Execution 
    device: str = "auto" # "cpu", "cuda", "mps", or "auto"
    #verbose: bool = False
    #seed: int = 42
    
    def to_dict(self) -> dict:
        """
        Convert the dataclass to a dictionary.
        """
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }