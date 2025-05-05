
from typing import Literal
import math

def get_beta(
    epoch: int,
    schedule: Literal["linear", "sigmoid"],
    beta_start: float,
    beta_end: float,
    anneal_epochs: int,
) -> float:
    """
    Returns the beta value for KL annealing at a given epoch.

    Args:
        epoch (int): Current training epoch (starting from 1).
        schedule (str): Type of annealing schedule: 'linear' or 'sigmoid'.
        beta_start (float): Initial beta value.
        beta_end (float): Final beta value.
        anneal_epochs (int): Number of epochs over which to anneal.

    Returns:
        float: Current beta value.
    """
    if epoch > anneal_epochs:
        return beta_end

    progress = epoch / anneal_epochs

    if schedule == "linear":
        return beta_start + (beta_end - beta_start) * progress

    elif schedule == "sigmoid":
        # Scale progress from -6 to +6 for smoother sigmoid curve
        x = 12 * progress - 6
        beta = beta_start + (beta_end - beta_start) * (1 / (1 + math.exp(-x)))
        return beta

    else:
        raise ValueError(f"Unknown KL schedule: {schedule}")
