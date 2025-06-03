"""
Experiment management class.
"""

import torch
import torchaudio

from data_utils.loader import parse_dataset
from experiment.exp_params import ExpParams
from train.train import train
from utils.device import get_best_device
from utils.logger import get_logger

class Experiment:
    """
    Class to manage the full experiment setup and execution.
    """
    def __init__(self, params: ExpParams) -> None:
        self.params = params
        self.logger = get_logger('experiment', log_dir=params.log_dir)

        self.logger.info("Initializing experiment...")
        self.logger.info(f"Parameters:\n{params.to_dict()}")

        self.params.output_dir.mkdir(parents=True, exist_ok=True)


        # --- Only check the path you're actually using ---
        self._check_data()
        self.device = self._get_device()

    def _check_data(self) -> None:
        """Verify that the dataset is valid and summarize basic audio properties."""
        if not self.params.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.params.data_path}")

        wav_files = list(self.params.data_path.rglob("*.wav"))
        if not wav_files:
            raise RuntimeError(f"No .wav files found in {self.params.data_path}")

        self.logger.info(f"Found {len(wav_files)} .wav files in {self.params.data_path}")

        lengths = []
        sample_rates = set()

        for f in wav_files:
            waveform, sr = torchaudio.load(f)
            lengths.append(waveform.shape[1])
            sample_rates.add(sr)

        if len(sample_rates) > 1:
            raise ValueError(f"Inconsistent sample rates found: {sample_rates}")
        else:
            sr = sample_rates.pop()
            self.logger.info(f"Confirmed sample rate: {sr} Hz")

        min_len = min(lengths)
        max_len = max(lengths)
        avg_len = sum(lengths) / len(lengths)

        self.logger.info(f"Min length: {min_len} samples ({min_len/sr:.2f} s)")
        self.logger.info(f"Max length: {max_len} samples ({max_len/sr:.2f} s)")
        self.logger.info(f"Avg length: {avg_len:.0f} samples ({avg_len/sr:.2f} s)")

        
    def _get_device(self) -> torch.device:
        """ Resolve best device from settings."""
        if self.params.device == "auto":
            device = get_best_device()
            self.logger.info(f"Auto-selected device: {device}")
            
            # Log GPU details if applicable
            if device.type == "cuda":
                gpu_name = torch.cuda.get_device_name(device.index)
                mem_total = torch.cuda.get_device_properties(device.index).total_memory / 1e9
                self.logger.info(f"Using GPU: {gpu_name} with {mem_total:.1f} GB memory")
            elif device.type == "mps":
                self.logger.info("Using Apple Silicon GPU with MPS")
                
            return device
        else:
            # User specified a device
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
        file_paths, labels, label_map, lengths = parse_dataset(self.params.data_path)
        self._parsed_data = (file_paths, labels, label_map, lengths)
        
        self.logger.info("Starting training...")
        train(self.params, device=self.device, parsed_data=self._parsed_data)
        self.logger.info("Training completed.")