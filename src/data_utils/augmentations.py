"""
In-memory data augmentation pipeline for waveform and feature tensors.
"""

from torch import Tensor
import torchaudio
import random

# --- Internal waveform-level augmentations ---
class WaveformAugmenter:
    def __init__(self, pitch_shift: bool = False, partial_dropout: bool = False, dropout_prob: float = 0.2):
        self.pitch_shift = pitch_shift
        self.partial_dropout = partial_dropout
        self.dropout_prob = dropout_prob

    def __call__(self, waveform: Tensor, sample_rate: int = 16000) -> Tensor:
        if self.pitch_shift:
            semitones = random.uniform(-2, 2)
            waveform = self._pitch_shift(waveform, sample_rate, semitones)
        if self.partial_dropout and random.random() < self.dropout_prob:
            waveform = self._partial_dropout(waveform)
        return waveform

    def _pitch_shift(self, waveform: Tensor, sample_rate: int, semitones: float) -> Tensor:
        # Use torchaudio resample as a crude approximation
        factor = 2 ** (semitones / 12)
        new_sr = int(sample_rate * factor)
        resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)(waveform)
        return torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=sample_rate)(resampled)

    def _partial_dropout(self, waveform: Tensor) -> Tensor:
        T = waveform.shape[0]
        drop_len = int(T * 0.2)
        start = random.randint(0, T - drop_len)
        waveform[start:start + drop_len] = 0.0
        return waveform


# --- Internal feature-level augmentations ---
class FeatureAugmenter:
    def __init__(self, time_mask: bool = False, freq_mask: bool = False,
                 time_width: float = 0.1, freq_width: float = 0.1):
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.time_width = time_width
        self.freq_width = freq_width

    def __call__(self, features: Tensor) -> Tensor:
        if self.time_mask:
            features = self._apply_time_mask(features)
        if self.freq_mask:
            features = self._apply_freq_mask(features)
        return features

    def _apply_time_mask(self, x: Tensor) -> Tensor:
        _, _, T = x.shape
        width = int(T * self.time_width)
        start = random.randint(0, max(0, T - width))
        x[:, :, start:start + width] = 0.0
        return x

    def _apply_freq_mask(self, x: Tensor) -> Tensor:
        _, F, _ = x.shape
        width = int(F * self.freq_width)
        start = random.randint(0, max(0, F - width))
        x[:, start:start + width, :] = 0.0
        return x


# --- Unified augmentation pipeline ---
class AugmentationPipeline:
    def __init__(self,
                 pitch_shift: bool = False,
                 partial_dropout: bool = False,
                 time_mask: bool = False,
                 freq_mask: bool = False,
                 prob: float = 1.0):
        """
        Args:
            pitch_shift (bool): Apply pitch shifting to waveform.
            partial_dropout (bool): Apply temporal zeroing to waveform.
            time_mask (bool): Apply time masking to features.
            freq_mask (bool): Apply frequency masking to features.
            prob (float): Probability of applying any augmentation.
        """
        self.prob = prob
        self.waveform_aug = WaveformAugmenter(pitch_shift=pitch_shift,
                                              partial_dropout=partial_dropout)
        self.feature_aug = FeatureAugmenter(time_mask=time_mask,
                                            freq_mask=freq_mask)

    def __call__(self, waveform: Tensor, features: Tensor, sample_rate: int = 16000) -> Tensor:
        if random.random() > self.prob:
            return features  # skip all augmentation

        # Apply waveform augmentations
        augmented_waveform = self.waveform_aug(waveform.clone(), sample_rate=sample_rate)

        # Re-transform into features
        from data_utils.transform import WaveletHilbertTransform  # safe import
        transform = WaveletHilbertTransform(output_len=features.shape[-1])
        transformed = transform(augmented_waveform)

        # Apply feature-level augmentations
        augmented_features = self.feature_aug(transformed)

        return augmented_features