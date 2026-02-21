"""Utility functions and configuration for audio I/O."""

from dataclasses import dataclass, field

import torch


@dataclass
class AudioConfig:
    """Configuration for audio loading, validation, and preprocessing.
    
    Attributes:
        min_duration_sec: Minimum allowed audio duration in seconds.
        max_duration_sec: Maximum allowed audio duration in seconds.
        target_sample_rate: Target sample rate after preprocessing.
        allow_stereo: Whether to allow stereo (2-channel) audio.
        allow_multi_channel: Whether to allow >2 channels.
        reject_silence: Whether to reject near-silent audio.
        silence_rms_threshold: RMS threshold below which audio is considered silent.
        to_mono: Whether to convert to mono during preprocessing.
        normalize: Whether to peak-normalize during preprocessing.
        peak_target: Target peak amplitude for normalization.
    """
    
    min_duration_sec: float = 0.1
    max_duration_sec: float = 600.0
    target_sample_rate: int = 16000
    allow_stereo: bool = True
    allow_multi_channel: bool = False
    reject_silence: bool = True
    silence_rms_threshold: float = 1e-4
    to_mono: bool = True
    normalize: bool = True
    peak_target: float = 0.95


def compute_duration_sec(num_samples: int, sample_rate: int) -> float:
    """Compute duration in seconds from sample count and rate.
    
    Args:
        num_samples: Number of audio samples.
        sample_rate: Sample rate in Hz.
        
    Returns:
        Duration in seconds.
        
    Examples:
        >>> compute_duration_sec(16000, 16000)
        1.0
        >>> compute_duration_sec(8000, 16000)
        0.5
    """
    if sample_rate <= 0:
        return 0.0
    return num_samples / sample_rate


def rms(waveform: torch.Tensor) -> float:
    """Compute root mean square of waveform.
    
    Args:
        waveform: Audio tensor of shape [channels, samples] or [samples].
        
    Returns:
        RMS value as float.
        
    Examples:
        >>> import torch
        >>> rms(torch.zeros(1, 1000))
        0.0
    """
    if waveform.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(waveform.float() ** 2)))


def ensure_float32_torch(waveform: torch.Tensor) -> torch.Tensor:
    """Ensure waveform is a float32 torch tensor.
    
    Args:
        waveform: Input tensor (any dtype).
        
    Returns:
        Float32 tensor with same shape.
        
    Examples:
        >>> import torch
        >>> t = ensure_float32_torch(torch.zeros(1, 100, dtype=torch.int16))
        >>> t.dtype
        torch.float32
    """
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)
    return waveform.to(dtype=torch.float32)


def safe_peak_normalize(
    waveform: torch.Tensor,
    peak_target: float = 0.95,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Peak normalize waveform to target amplitude.
    
    If max absolute value is below eps, returns waveform unchanged
    to avoid amplifying noise/silence.
    
    Args:
        waveform: Audio tensor of shape [channels, samples].
        peak_target: Target peak amplitude (0.0 to 1.0).
        eps: Minimum peak value to trigger normalization.
        
    Returns:
        Normalized waveform tensor.
        
    Examples:
        >>> import torch
        >>> w = torch.tensor([[0.5, -0.25, 0.1]])
        >>> normalized = safe_peak_normalize(w, peak_target=1.0)
        >>> float(normalized.abs().max())
        1.0
    """
    waveform = ensure_float32_torch(waveform)
    peak = waveform.abs().max()
    
    if peak < eps:
        # Signal is too quiet, don't amplify noise
        return waveform
    
    return waveform * (peak_target / peak)


def clamp_finite(waveform: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """Clamp waveform values and replace non-finite values with zero.
    
    Args:
        waveform: Input tensor.
        min_val: Minimum value after clamping.
        max_val: Maximum value after clamping.
        
    Returns:
        Clamped tensor with no NaN/Inf values.
    """
    waveform = waveform.clone()
    # Replace NaN and Inf with zero
    waveform = torch.where(torch.isfinite(waveform), waveform, torch.zeros_like(waveform))
    # Clamp to range
    return torch.clamp(waveform, min_val, max_val)
