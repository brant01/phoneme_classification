import torch
import numpy as np
from data_utils.augmentations import AugmentationPipeline
from data_utils.transform import WaveletHilbertTransform

def dummy_waveform(length=16000) -> torch.Tensor:
    t = torch.linspace(0, 1, steps=length)
    return 0.5 * torch.sin(2 * np.pi * 440 * t)

def test_pipeline_no_aug():
    waveform = dummy_waveform()
    transform = WaveletHilbertTransform(output_len=len(waveform))
    clean = transform(waveform)

    aug = AugmentationPipeline()
    result = aug(waveform, clean)

    assert result.shape == clean.shape
    assert torch.allclose(result, clean, atol=1e-5)

def test_pipeline_with_pitch():
    waveform = dummy_waveform()
    transform = WaveletHilbertTransform(output_len=len(waveform))
    clean = transform(waveform)

    aug = AugmentationPipeline(pitch_shift=True)
    result = aug(waveform, clean)

    assert result.shape == clean.shape
    assert not torch.allclose(result, clean)

def test_pipeline_with_masks():
    waveform = dummy_waveform()
    transform = WaveletHilbertTransform(output_len=len(waveform))
    clean = transform(waveform)

    aug = AugmentationPipeline(time_mask=True, freq_mask=True)
    result = aug(waveform, clean)

    assert result.shape == clean.shape
    assert (result == 0.0).any()

def test_pipeline_prob_zero():
    waveform = dummy_waveform()
    transform = WaveletHilbertTransform(output_len=len(waveform))
    clean = transform(waveform)

    aug = AugmentationPipeline(pitch_shift=True, partial_dropout=True, time_mask=True, freq_mask=True, prob=0.0)
    result = aug(waveform, clean)

    assert result.shape == clean.shape
    assert torch.allclose(result, clean, atol=1e-5)