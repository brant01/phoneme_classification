
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
        waveform = self._pad(waveform)
        np_waveform = waveform.numpy()

        # Wavelet Transform (CWT power)
        scales = np.arange(1, self.num_scales + 1)
        coeffs, _ = pywt.cwt(np_waveform, scales, self.wavelet)
        power = np.abs(coeffs)  # shape: [num_scales, T]

        # Hilbert Transform (amplitude envelope)
        analytic = hilbert(np_waveform)
        envelope = np.abs(analytic)  # shape: [T]

        # Normalize and convert to torch
        wavelet_tensor = torch.from_numpy(power).float()
        hilbert_tensor = torch.from_numpy(envelope).float().unsqueeze(0)  # [1, T]

        # Optionally normalize (mean/std)
        wavelet_tensor = (wavelet_tensor - wavelet_tensor.mean()) / (wavelet_tensor.std() + 1e-6)
        hilbert_tensor = (hilbert_tensor - hilbert_tensor.mean()) / (hilbert_tensor.std() + 1e-6)

        # Final shape: [2, num_scales, T] if we expand hilbert to match wavelet shape
        hilbert_repeated = hilbert_tensor.repeat(self.num_scales, 1)

        combined = torch.stack([wavelet_tensor, hilbert_repeated], dim=0)  # [2, num_scales, T]

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
            raise ValueError("Input waveform longer than expected output_len.")