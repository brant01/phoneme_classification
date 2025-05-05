"""
Run manager utilities for creating output directories and saving experiment metadata
"""
import csv
import json
from datetime import datetime
from pathlib import Path

from experiment.exp_params import ExpParams

def create_run_dir(base_dir:Path) -> Path:
    """
    Create a timestamped directory for the current run.
    
    Args:
        base_dir (Path): Base directory for the run.
        
    Returns:
        Path: Path to the created run directory.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def save_config(params: ExpParams,
                path: Path) -> None:
    """
    Save experiment configuration to a JSON file.
    """
    
    with open(path, "w") as f:
        json.dump(params.to_dict(), f, indent=2)
        
def save_loss_history(losses: dict[str, list], save_path: Path) -> None:
    keys = list(losses.keys())
    max_len = max(len(v) for v in losses.values())
    
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch"] + keys)
        for i in range(max_len):
            row = [i + 1]
            for key in keys:
                if i < len(losses[key]):
                    row.append(losses[key][i])
                else:
                    row.append(0)  # or None or 0.0 depending on preference
            writer.writerow(row)
