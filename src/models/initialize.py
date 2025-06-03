import torch.nn as nn

def initialize_weights(module: nn.Module) -> None:
    """
    Apply weight initialization to a given module.
    Uses:
    - Kaiming Normal for Conv/ConvTranspose layers
    - Xavier Normal for Linear layers
    - Constant for biases (zero)
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)