
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np
import pywt
from scipy.signal import hilbert

class WaveletHilbertTransform:
    """
    Transform waveform into multichannel representation:
    - Wavelet power features
    - Hilbert amplitude envelope
    """

    def __init__(self, output_len: int = 16000, wavelet: str = "morl", num_scales: int = 32) -> None:
        """
        Args:
            output_len (int): Length of output waveform (after padding).
            wavelet (str): Wavelet to use for CWT.
            num_scales (int): Number of wavelet scales.
        """
        self.output_len = output_len
        self.wavelet = wavelet
        self.num_scales = num_scales

    def __call__(self, waveform: Tensor) -> Tensor:
        """
        Convert input waveform into a [2, num_scales, output_len] tensor with:
        - [0]: wavelet transform (power)
        - [1]: Hilbert transform (amplitude envelope repeated across scales)
        """
        orig_device = waveform.device
        waveform = self._pad(waveform)

        if waveform.ndim == 1:
            x = waveform.cpu().numpy()
        elif waveform.ndim == 2:
            x = waveform[0].cpu().numpy()
        else:
            raise ValueError(f"Expected 1D or 2D input, got shape {waveform.shape}")

        scales = np.arange(1, self.num_scales + 1)
        coeffs, _ = pywt.cwt(x, scales, self.wavelet)
        power = np.abs(coeffs)

        envelope = np.abs(hilbert(x))

        wavelet_tensor = torch.from_numpy(power).float()
        hilbert_tensor = torch.from_numpy(envelope).float().unsqueeze(0)

        wavelet_tensor = (wavelet_tensor - wavelet_tensor.mean()) / (wavelet_tensor.std() + 1e-6)
        hilbert_tensor = (hilbert_tensor - hilbert_tensor.mean()) / (hilbert_tensor.std() + 1e-6)

        hilbert_repeated = hilbert_tensor.repeat(self.num_scales, 1)

        current_len = wavelet_tensor.shape[-1]
        if current_len < self.output_len:
            pad = self.output_len - current_len
            wavelet_tensor = F.pad(wavelet_tensor, (0, pad))
            hilbert_repeated = F.pad(hilbert_repeated, (0, pad))
        elif current_len > self.output_len:
            wavelet_tensor = wavelet_tensor[:, :self.output_len]
            hilbert_repeated = hilbert_repeated[:, :self.output_len]

        combined = torch.stack([wavelet_tensor, hilbert_repeated], dim=0)
        return torch.tanh(combined).to(orig_device)
    
    def _pad(self, x: Tensor) -> Tensor:
        T = x.shape[0]
        if T == self.output_len:
            return x
        elif T < self.output_len:
            pad = self.output_len - T
            left = pad // 2
            right = pad - left
            return F.pad(x, (left, right))
        else:
            # Center crop
            start = (T - self.output_len) // 2
            end = start + self.output_len
            return x[start:end]

