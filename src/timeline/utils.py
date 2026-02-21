"""Utility functions for timeline operations."""

import torch


def seconds_to_samples(sec: float, sample_rate: int) -> int:
    """Convert seconds to sample count.
    
    Args:
        sec: Duration in seconds.
        sample_rate: Audio sample rate in Hz.
    
    Returns:
        Number of samples (rounded to nearest integer).
    
    Examples:
        >>> seconds_to_samples(2.0, 16000)
        32000
        >>> seconds_to_samples(0.5, 16000)
        8000
    """
    return round(sec * sample_rate)


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    """Convert sample count to seconds.
    
    Args:
        samples: Number of samples.
        sample_rate: Audio sample rate in Hz.
    
    Returns:
        Duration in seconds.
    
    Examples:
        >>> samples_to_seconds(32000, 16000)
        2.0
        >>> samples_to_seconds(8000, 16000)
        0.5
    """
    return samples / sample_rate


def validate_waveform_shape(waveform: torch.Tensor) -> tuple[int, int]:
    """Validate that waveform has shape [1, T] and return dimensions.
    
    Args:
        waveform: Input audio tensor.
    
    Returns:
        Tuple of (channels, num_samples).
    
    Raises:
        ValueError: If waveform does not have exactly 2 dimensions
            or does not have exactly 1 channel.
    
    Examples:
        >>> import torch
        >>> validate_waveform_shape(torch.zeros(1, 16000))
        (1, 16000)
    """
    if waveform.ndim != 2:
        raise ValueError(
            f"Waveform must have exactly 2 dimensions [1, T], got {waveform.ndim} dimensions"
        )
    
    channels, num_samples = waveform.shape
    
    if channels != 1:
        raise ValueError(
            f"Waveform must have exactly 1 channel, got {channels} channels"
        )
    
    return channels, num_samples
