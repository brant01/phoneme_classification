
# This file contains the implementation of various KL schedules for training diffusion models.

def get_beta(
    epoch: int,
    schedule: str,
    beta_start: float,
    beta_end: float,
    anneal_epochs: int,
    cycle_length: int = 100  # default if not passed
) -> float:
    if schedule == "none":
        return beta_end
    elif schedule == "linear":
        return min(beta_end, beta_start + (beta_end - beta_start) * (epoch / anneal_epochs))
    elif schedule == "sigmoid":
        import math
        progress = min(epoch / anneal_epochs, 1.0)
        return beta_start + (beta_end - beta_start) / (1 + math.exp(-12 * (progress - 0.5)))
    elif schedule == "cyclical":
        import math
        # Cycle progress is between 0 and 1
        cycle_progress = (epoch % cycle_length) / cycle_length
        return beta_start + (beta_end - beta_start) * (0.5 * (1 - math.cos(2 * math.pi * cycle_progress)))
    else:
        raise ValueError(f"Unknown KL schedule: {schedule}")