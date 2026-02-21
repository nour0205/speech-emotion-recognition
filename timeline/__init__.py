"""Timeline module for audio segmentation and windowing.

This module provides deterministic windowing of audio waveforms into overlapping
segments for time-series emotion inference.

Example:
    >>> import torch
    >>> from timeline import WindowingConfig, segment_audio
    >>> waveform = torch.randn(1, 32000)  # 2 seconds at 16kHz
    >>> config = WindowingConfig(window_sec=2.0, hop_sec=0.5, pad_mode="zero")
    >>> windows = segment_audio(waveform, sample_rate=16000, config=config)
    >>> for w in windows:
    ...     print(f"Window {w['index']}: {w['start_sec']:.2f}s - {w['end_sec']:.2f}s")
"""

from .errors import TimelineError, WindowingConfigError, WindowingRuntimeError
from .utils import samples_to_seconds, seconds_to_samples, validate_waveform_shape
from .windowing import WindowingConfig, segment_audio


__all__ = [
    # Main API
    "WindowingConfig",
    "segment_audio",
    # Errors
    "TimelineError",
    "WindowingConfigError",
    "WindowingRuntimeError",
    # Utils
    "seconds_to_samples",
    "samples_to_seconds",
    "validate_waveform_shape",
]
