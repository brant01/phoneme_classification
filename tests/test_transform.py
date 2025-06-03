import torch
from data_utils.transform import WaveletHilbertTransform

def test_wavelet_hilbert_output_shape():
    transform = WaveletHilbertTransform(output_len=16000, num_scales=32)
    waveform = torch.randn(14000)
    result = transform(waveform)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 32, 16000)