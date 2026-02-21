"""Audio I/O module for loading, validating, and preprocessing audio.

This module provides a production-grade audio pipeline for Speech Emotion Recognition.
It handles:
- Loading WAV audio from files or bytes
- Validating audio properties (duration, channels, silence, etc.)
- Preprocessing to canonical format (mono, 16kHz, normalized)

Example:
    >>> from audioio import load_validate_preprocess, AudioConfig
    >>> waveform, sr = load_validate_preprocess("speech.wav", AudioConfig())
    >>> waveform.shape  # [1, num_samples]
    torch.Size([1, 16000])
    >>> sr
    16000
"""

from pathlib import Path
from typing import Union

import torch

from .errors import (
    AudioDecodeError,
    AudioIOError,
    AudioPreprocessError,
    AudioValidationError,
)
from .loader import load_wav, load_wav_bytes
from .preprocess import preprocess_audio
from .utils import (
    AudioConfig,
    clamp_finite,
    compute_duration_sec,
    ensure_float32_torch,
    rms,
    safe_peak_normalize,
)
from .validate import validate_wav


__all__ = [
    # Main integration function
    "load_validate_preprocess",
    # Config
    "AudioConfig",
    # Errors
    "AudioIOError",
    "AudioDecodeError",
    "AudioValidationError",
    "AudioPreprocessError",
    # Loader
    "load_wav",
    "load_wav_bytes",
    # Validation
    "validate_wav",
    # Preprocessing
    "preprocess_audio",
    # Utils
    "compute_duration_sec",
    "rms",
    "ensure_float32_torch",
    "safe_peak_normalize",
    "clamp_finite",
]


def load_validate_preprocess(
    path_or_bytes: Union[str, Path, bytes],
    config: AudioConfig | None = None,
) -> tuple[torch.Tensor, int]:
    """Load, validate, and preprocess audio in one step.
    
    This is the main entry point for the audio pipeline. It:
    1. Detects whether input is a file path or raw bytes
    2. Loads the WAV audio
    3. Validates audio properties
    4. Preprocesses to canonical format
    
    Args:
        path_or_bytes: Either a file path (str/Path) or raw WAV bytes.
        config: Audio configuration. If None, uses default AudioConfig().
        
    Returns:
        Tuple of (processed_waveform, sample_rate) where:
        - processed_waveform: Float32 tensor with shape [1, T] (mono)
        - sample_rate: Target sample rate (default 16000)
        
    Raises:
        AudioDecodeError: If audio cannot be loaded/decoded.
        AudioValidationError: If audio fails validation.
        AudioPreprocessError: If preprocessing fails.
        
    Examples:
        Load from file:
        >>> waveform, sr = load_validate_preprocess("speech.wav")
        >>> waveform.shape
        torch.Size([1, 16000])
        
        Load from bytes:
        >>> with open("speech.wav", "rb") as f:
        ...     audio_bytes = f.read()
        >>> waveform, sr = load_validate_preprocess(audio_bytes)
        
        Custom config:
        >>> config = AudioConfig(
        ...     target_sample_rate=22050,
        ...     min_duration_sec=0.5,
        ...     reject_silence=False,
        ... )
        >>> waveform, sr = load_validate_preprocess("speech.wav", config)
    """
    if config is None:
        config = AudioConfig()
    
    # Step 1: Load
    if isinstance(path_or_bytes, bytes):
        waveform, sample_rate = load_wav_bytes(path_or_bytes)
    else:
        waveform, sample_rate = load_wav(path_or_bytes)
    
    # Step 2: Validate
    validate_wav(
        waveform=waveform,
        sample_rate=sample_rate,
        min_duration_sec=config.min_duration_sec,
        max_duration_sec=config.max_duration_sec,
        allow_stereo=config.allow_stereo,
        allow_multi_channel=config.allow_multi_channel,
        reject_silence=config.reject_silence,
        silence_rms_threshold=config.silence_rms_threshold,
    )
    
    # Step 3: Preprocess
    processed, target_sr = preprocess_audio(
        waveform=waveform,
        sample_rate=sample_rate,
        target_sample_rate=config.target_sample_rate,
        to_mono=config.to_mono,
        normalize=config.normalize,
        peak_target=config.peak_target,
    )
    
    return processed, target_sr
