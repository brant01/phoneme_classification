from torch import Tensor
import torch
import torchaudio


# --- Internal waveform-level augmentations ---
class WaveformAugmenter:
    def __init__(self, pitch_shift: bool = False, partial_dropout: bool = False, dropout_prob: float = 0.2):
        self.pitch_shift = pitch_shift
        self.partial_dropout = partial_dropout
        self.dropout_prob = dropout_prob

    def __call__(self, waveform: Tensor, sample_rate: int = 16000) -> Tensor:
        device = waveform.device
        if self.pitch_shift:
            semitones = torch.empty(1, device=device).uniform_(-2, 2).item()
            waveform = self._pitch_shift(waveform, sample_rate, semitones)
        if self.partial_dropout and torch.rand(1, device=device).item() < self.dropout_prob:
            waveform = self._partial_dropout(waveform)
        return waveform

    def _pitch_shift(self, waveform: Tensor, sample_rate: int, semitones: float) -> Tensor:
        factor = 2 ** (semitones / 12)
        new_sr = int(sample_rate * factor)
        resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)(waveform)
        return torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=sample_rate)(resampled)

    def _partial_dropout(self, waveform: Tensor) -> Tensor:
        T = waveform.shape[0]
        drop_len = int(T * 0.2)
        device = waveform.device
        start = torch.randint(low=0, high=T - drop_len, size=(1,), device=device).item()
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
        device = x.device
        start = torch.randint(low=0, high=max(1, T - width), size=(1,), device=device).item()
        x[:, :, start:start + width] = 0.0
        return x

    def _apply_freq_mask(self, x: Tensor) -> Tensor:
        _, F, _ = x.shape
        width = int(F * self.freq_width)
        device = x.device
        start = torch.randint(low=0, high=max(1, F - width), size=(1,), device=device).item()
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
        self.prob = prob
        self.waveform_aug = WaveformAugmenter(pitch_shift=pitch_shift,
                                              partial_dropout=partial_dropout)
        self.feature_aug = FeatureAugmenter(time_mask=time_mask,
                                            freq_mask=freq_mask)

    def __call__(self, waveform: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
        if torch.rand(1, device=waveform.device).item() > self.prob:
            return waveform, features
        waveform_aug = self.waveform_aug(waveform)
        features_aug = self.feature_aug(features)
        return waveform_aug, features_aug
