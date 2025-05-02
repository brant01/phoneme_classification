import torch
from utils.device import get_best_device

def test_get_best_device_returns_device():
    device = get_best_device()
    assert isinstance(device, torch.device)

def test_get_best_device_logic():
    device = get_best_device()
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"