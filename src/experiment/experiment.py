"""
Experiment management class.
"""

import torch

from data_utils.loader import parse_dataset
from experiment.exp_params import ExpParams
from train.train import train
from utils.logger import get_logger


class Experiment:
    """
    Class to manage the full experiment setup and execution.
    """
    def __init__(self,
                 params: ExpParams) -> None:
        """
        Initialize the experiment with given parameters.
        Sets up logging, checks data, and prepares output structure.
        """
        
        self.params = params
        self.logger = get_logger('experiment', log_dir=params.log_dir)
        
        self.logger.info("Initializing experiment...")
        self.logger.info(f"Parameters:\n{params.to_dict()}")
        
        # Make output directory if needed
        self.params.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._check_data()
        self.device = self._get_device()
        
    def _check_data(self) -> None:
        """ Verify the data path exists and contains .wav files. """
        if not self.params.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.params.data_path}")
        
        wav_files = list(self.params.data_path.rglob("*.wav"))
        if not wav_files:
            raise RuntimeError(f"No .wav files found in {self.params.data_path}")
        
        self.logger.info(f"Found {len(wav_files)} .wav files in {self.params.data_path}")
        
    def _get_device(self) -> torch.device:
        """ Resolve best device from settings."""
        if self.params.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self.params.device)
    
    def run(self) -> None:
        """
        Runs the experiment - calls train() with parsed config.
        """
        
        file_paths, labels, label_map, lengths = parse_dataset(self.params.data_path)
        output_len = int(max(lengths) * 1.1)
        self.logger.info(f"Longest file length: {max(lengths)} samples")
        self.logger.info(f"Computed output_len: {output_len}")

        # Update input shape in params
        self.params.input_shape = (2, 32, output_len)

        # Store all for reuse
        self._parsed_data = (file_paths, labels, label_map, lengths)

        # Pass to training
        train(self.params, device=self.device, parsed_data=self._parsed_data)
        
        
        self.logger.info("Starting training...")
        train(self.params, device=self.device)  
        self.logger.info("Training completed.")