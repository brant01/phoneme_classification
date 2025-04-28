"""
Experiment management class.
"""

from pathlib import Path
from utils.env_loader import load_environment, get_env_variable
from utils.logger import get_logger

class Experiment:
    """
    Class to manage the full experiment setup and execution.
    """
    def __init__(self) -> None:
        """
        Initialize the experiment
        Loads environment variables, sets up logging, verifies data availability.
        """
        
        # Load environment variables
        load_environment()
        
        # Set up logger
        self.logger = get_logger('experiment')
        
        # Load important environment variables
        self.data_path = Path(get_env_variable('DATA_PATH', required=True))
        self.log_dir = Path(get_env_variable('LOG_DIR', default='logs'))
        self.output_dir = Path(get_env_variable('OUTPUT_DIR', default='outputs'))
        
        # Basic checks
        self.check_data_path()
        
    def check_data_path(self) -> None:
        """
        Verify that the data path exists and contains audio files.
        """
        if not self.data_path.exists():
            self.logger.error(f"Data path not found: {self.data_path}")
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        num_files = len(list(self.data_path.rglob('*.wav')))
        self.logger.info(f"Data path verified: {self.data_path}.")

        if num_files == 0:
            self.logger.error("No .wav files found in the data path. Cannot proceed.")
            raise RuntimeError("Data path contains no .wav files. Aborting experiment.")
        else:
            self.logger.info(f"Data path contains {num_files} .wav files.")
                
    def run(self) -> None:
        """
        Run the experiment.
        """
        self.logger.info("Experiment setup complete. Ready to start training.")