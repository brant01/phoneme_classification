"""
Utility to set up logging across the project.
"""

import logging
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = "logs"

def get_logger(
        name: str, 
        log_dir: Optional[str] = DEFAULT_LOG_DIR, 
        log_level: int = logging.INFO) -> logging.Logger:
    
    """
    Create and configure a logger.

    Args:
        name (str): Name of the logger.
        log_dir (Optional[str]): Directory where log files are saved. Defaults to "logs/".
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler (always active)
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)  # Create if missing
            log_file_path = log_dir_path / f"{name}.log"
            file_handler = logging.FileHandler(str(log_file_path))
            file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger