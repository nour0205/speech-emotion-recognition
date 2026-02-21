"""Timeline module for audio segmentation, windowing, and emotion timeline generation.

This module provides:
- Deterministic windowing of audio waveforms into overlapping segments
- Smoothing strategies to reduce prediction jitter
- Segment merging for clean emotion timelines
- Full timeline generation pipeline

Example:
    >>> import torch
    >>> from timeline import WindowingConfig, segment_audio
    >>> waveform = torch.randn(1, 32000)  # 2 seconds at 16kHz
    >>> config = WindowingConfig(window_sec=2.0, hop_sec=0.5, pad_mode="zero")
    >>> windows = segment_audio(waveform, sample_rate=16000, config=config)
    >>> for w in windows:
    ...     print(f"Window {w['index']}: {w['start_sec']:.2f}s - {w['end_sec']:.2f}s")

Timeline Generation Example:
    >>> from timeline import generate_timeline, SmoothingConfig
    >>> result = generate_timeline("speech.wav")
    >>> for segment in result.segments:
    ...     print(f"{segment.start_sec:.2f}s - {segment.end_sec:.2f}s: {segment.emotion}")
"""

from .errors import TimelineError, WindowingConfigError, WindowingRuntimeError
from .generate import generate_timeline, generate_timeline_from_waveform
from .merge import MergeConfig, merge_windows_to_segments
from .schema import Segment, SmoothingInfo, TimelineResult, WindowPrediction
from .smooth import SmoothingConfig, smooth_windows
from .utils import samples_to_seconds, seconds_to_samples, validate_waveform_shape
from .windowing import WindowingConfig, segment_audio


__all__ = [
    # Main Timeline API (Phase 4)
    "generate_timeline",
    "generate_timeline_from_waveform",
    # Schema
    "TimelineResult",
    "Segment",
    "WindowPrediction",
    "SmoothingInfo",
    # Smoothing (Phase 4)
    "SmoothingConfig",
    "smooth_windows",
    # Merging (Phase 4)
    "MergeConfig",
    "merge_windows_to_segments",
    # Windowing (Phase 2)
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
