
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
        waveform = self._pad(waveform)

        # Handle 1D (assume mono) and 2D (assume [channels, T]) inputs
        if waveform.ndim == 1:
            x = waveform.numpy()
        elif waveform.ndim == 2:
            x = waveform[0].numpy()  # Use first channel
        else:
            raise ValueError(f"Expected 1D or 2D input, got shape {waveform.shape}")

        # Wavelet transform
        scales = np.arange(1, self.num_scales + 1)
        coeffs, _ = pywt.cwt(x, scales, self.wavelet)
        power = np.abs(coeffs)  # [num_scales, T_wavelet]

        # Hilbert envelope
        envelope = np.abs(hilbert(x))  # [T_wavelet]

        # Convert to torch tensors
        wavelet_tensor = torch.from_numpy(power).float()
        hilbert_tensor = torch.from_numpy(envelope).float().unsqueeze(0)  # [1, T]

        # Normalize
        wavelet_tensor = (wavelet_tensor - wavelet_tensor.mean()) / (wavelet_tensor.std() + 1e-6)
        hilbert_tensor = (hilbert_tensor - hilbert_tensor.mean()) / (hilbert_tensor.std() + 1e-6)

        # Expand envelope across frequency axis
        hilbert_repeated = hilbert_tensor.repeat(self.num_scales, 1)  # [num_scales, T]

        # Ensure fixed length output
        current_len = wavelet_tensor.shape[-1]
        if current_len < self.output_len:
            pad = self.output_len - current_len
            wavelet_tensor = F.pad(wavelet_tensor, (0, pad))
            hilbert_repeated = F.pad(hilbert_repeated, (0, pad))
        elif current_len > self.output_len:
            wavelet_tensor = wavelet_tensor[:, :self.output_len]
            hilbert_repeated = hilbert_repeated[:, :self.output_len]

        combined = torch.stack([wavelet_tensor, hilbert_repeated], dim=0)  # [2, num_scales, output_len]
        return torch.tanh(combined)

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