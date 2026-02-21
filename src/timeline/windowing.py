"""Audio windowing/segmentation for timeline-based emotion recognition.

This module provides deterministic windowing of audio waveforms into overlapping
segments suitable for time-series emotion inference.

Example:
    >>> import torch
    >>> from timeline.windowing import WindowingConfig, segment_audio
    >>> waveform = torch.randn(1, 32000)  # 2 seconds at 16kHz
    >>> config = WindowingConfig(window_sec=2.0, hop_sec=0.5, pad_mode="zero")
    >>> windows = segment_audio(waveform, sample_rate=16000, config=config)
    >>> print(len(windows), windows[0]["start_sec"], windows[0]["end_sec"])
"""

from dataclasses import dataclass
from typing import Literal

import torch

from .errors import WindowingConfigError, WindowingRuntimeError
from .utils import samples_to_seconds, seconds_to_samples, validate_waveform_shape


PadMode = Literal["none", "zero", "reflect"]


@dataclass
class WindowingConfig:
    """Configuration for audio windowing/segmentation.
    
    Attributes:
        window_sec: Window duration in seconds. Default 2.0.
        hop_sec: Hop/stride duration in seconds. Default 0.5.
        pad_mode: Padding mode for partial windows. One of:
            - "none": No padding, last window may be shorter.
            - "zero": Pad with zeros to window length.
            - "reflect": Pad by reflecting the tail.
            Default "zero".
        include_partial_last_window: Whether to include a partial window
            at the end when audio doesn't divide evenly. Default True.
        min_window_sec: Minimum allowed window duration (sanity bound). Default 0.25.
        max_window_sec: Maximum allowed window duration (sanity bound). Default 10.0.
    """
    
    window_sec: float = 2.0
    hop_sec: float = 0.5
    pad_mode: PadMode = "zero"
    include_partial_last_window: bool = True
    min_window_sec: float = 0.25
    max_window_sec: float = 10.0
    
    def __post_init__(self) -> None:
        """Validate configuration on initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            WindowingConfigError: If any parameter is invalid.
        """
        # Validate window_sec > 0
        if self.window_sec <= 0:
            raise WindowingConfigError(
                message=f"window_sec must be positive, got {self.window_sec}",
                code="INVALID_CONFIG",
                details={"parameter": "window_sec", "value": self.window_sec},
            )
        
        # Validate hop_sec > 0
        if self.hop_sec <= 0:
            raise WindowingConfigError(
                message=f"hop_sec must be positive, got {self.hop_sec}",
                code="INVALID_CONFIG",
                details={"parameter": "hop_sec", "value": self.hop_sec},
            )
        
        # Validate hop_sec <= window_sec
        if self.hop_sec > self.window_sec:
            raise WindowingConfigError(
                message=f"hop_sec ({self.hop_sec}) must be <= window_sec ({self.window_sec})",
                code="INVALID_CONFIG",
                details={
                    "parameter": "hop_sec",
                    "hop_sec": self.hop_sec,
                    "window_sec": self.window_sec,
                },
            )
        
        # Validate window_sec within bounds
        if self.window_sec < self.min_window_sec:
            raise WindowingConfigError(
                message=f"window_sec ({self.window_sec}) must be >= min_window_sec ({self.min_window_sec})",
                code="INVALID_CONFIG",
                details={
                    "parameter": "window_sec",
                    "window_sec": self.window_sec,
                    "min_window_sec": self.min_window_sec,
                },
            )
        
        if self.window_sec > self.max_window_sec:
            raise WindowingConfigError(
                message=f"window_sec ({self.window_sec}) must be <= max_window_sec ({self.max_window_sec})",
                code="INVALID_CONFIG",
                details={
                    "parameter": "window_sec",
                    "window_sec": self.window_sec,
                    "max_window_sec": self.max_window_sec,
                },
            )
        
        # Validate pad_mode
        valid_pad_modes = {"none", "zero", "reflect"}
        if self.pad_mode not in valid_pad_modes:
            raise WindowingConfigError(
                message=f"pad_mode must be one of {valid_pad_modes}, got '{self.pad_mode}'",
                code="INVALID_CONFIG",
                details={"parameter": "pad_mode", "value": self.pad_mode},
            )


def segment_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    config: WindowingConfig,
) -> list[dict]:
    """Segment audio waveform into overlapping windows.
    
    Args:
        waveform: Input audio tensor of shape [1, T] (mono, float32).
        sample_rate: Audio sample rate in Hz.
        config: Windowing configuration.
    
    Returns:
        List of window dictionaries, each containing:
            - "index": int - Window index (0-based).
            - "start_sec": float - Start time in seconds.
            - "end_sec": float - End time in seconds (virtual end when padded).
            - "start_sample": int - Start sample index.
            - "end_sample": int - End sample index (virtual when padded).
            - "waveform": torch.Tensor - Window waveform [1, window_samples].
            - "is_padded": bool - Whether this window was padded.
    
    Raises:
        WindowingRuntimeError: If input has invalid shape or is empty.
    
    Example:
        >>> import torch
        >>> config = WindowingConfig(window_sec=2.0, hop_sec=0.5, pad_mode="zero")
        >>> waveform = torch.randn(1, 48000)  # 3 seconds at 16kHz
        >>> windows = segment_audio(waveform, sample_rate=16000, config=config)
        >>> len(windows)
        3
        >>> windows[0]["start_sec"], windows[0]["end_sec"]
        (0.0, 2.0)
    """
    # Validate input shape
    try:
        _, num_samples = validate_waveform_shape(waveform)
    except ValueError as e:
        raise WindowingRuntimeError(
            message=str(e),
            code="INVALID_SHAPE",
            details={"shape": list(waveform.shape)},
        ) from e
    
    # Validate non-empty input
    if num_samples == 0:
        raise WindowingRuntimeError(
            message="Input waveform has zero samples",
            code="EMPTY_INPUT",
            details={"shape": list(waveform.shape)},
        )
    
    # Convert config to samples
    window_samples = seconds_to_samples(config.window_sec, sample_rate)
    hop_samples = seconds_to_samples(config.hop_sec, sample_rate)
    
    # Ensure minimum of 1 sample
    window_samples = max(1, window_samples)
    hop_samples = max(1, hop_samples)
    
    windows: list[dict] = []
    index = 0
    start = 0
    
    while start < num_samples:
        end = start + window_samples
        
        if end <= num_samples:
            # Full window - no padding needed
            window_waveform = waveform[:, start:end].clone()
            windows.append({
                "index": index,
                "start_sec": samples_to_seconds(start, sample_rate),
                "end_sec": samples_to_seconds(end, sample_rate),
                "start_sample": start,
                "end_sample": end,
                "waveform": window_waveform,
                "is_padded": False,
            })
        else:
            # Partial window - handle based on config
            if not config.include_partial_last_window:
                # Skip partial window
                break
            
            # Get the actual samples available
            actual_end = num_samples
            actual_samples = actual_end - start
            partial_waveform = waveform[:, start:actual_end]
            
            if config.pad_mode == "none":
                # No padding - return shorter window
                window_waveform = partial_waveform.clone()
                windows.append({
                    "index": index,
                    "start_sec": samples_to_seconds(start, sample_rate),
                    "end_sec": samples_to_seconds(actual_end, sample_rate),
                    "start_sample": start,
                    "end_sample": actual_end,
                    "waveform": window_waveform,
                    "is_padded": False,
                })
            elif config.pad_mode == "zero":
                # Zero padding
                pad_length = window_samples - actual_samples
                window_waveform = torch.cat([
                    partial_waveform.clone(),
                    torch.zeros(1, pad_length, dtype=waveform.dtype, device=waveform.device),
                ], dim=1)
                windows.append({
                    "index": index,
                    "start_sec": samples_to_seconds(start, sample_rate),
                    "end_sec": samples_to_seconds(end, sample_rate),
                    "start_sample": start,
                    "end_sample": end,
                    "waveform": window_waveform,
                    "is_padded": True,
                    "actual_end_sample": actual_end,
                })
            elif config.pad_mode == "reflect":
                # Reflect padding - deterministic
                pad_length = window_samples - actual_samples
                window_waveform = _reflect_pad(partial_waveform, pad_length)
                windows.append({
                    "index": index,
                    "start_sec": samples_to_seconds(start, sample_rate),
                    "end_sec": samples_to_seconds(end, sample_rate),
                    "start_sample": start,
                    "end_sample": end,
                    "waveform": window_waveform,
                    "is_padded": True,
                    "actual_end_sample": actual_end,
                })
        
        index += 1
        start += hop_samples
    
    return windows


def _reflect_pad(waveform: torch.Tensor, pad_length: int) -> torch.Tensor:
    """Pad waveform by reflecting the tail.
    
    This is a deterministic padding method that reflects the end of the
    waveform to fill the required length.
    
    Args:
        waveform: Input tensor of shape [1, samples].
        pad_length: Number of samples to pad.
    
    Returns:
        Padded tensor of shape [1, samples + pad_length].
    """
    actual_samples = waveform.shape[1]
    
    if pad_length <= 0:
        return waveform.clone()
    
    # Build padding by reflecting
    # We may need to reflect multiple times if pad_length > actual_samples
    padding_parts: list[torch.Tensor] = []
    remaining = pad_length
    
    while remaining > 0:
        # Take up to actual_samples from the reflected tail
        take_samples = min(remaining, actual_samples)
        # Reflect: flip the last 'take_samples' of waveform
        reflected = torch.flip(waveform[:, -take_samples:], dims=[1])
        padding_parts.append(reflected)
        remaining -= take_samples
    
    padding = torch.cat(padding_parts, dim=1)[:, :pad_length]
    
    return torch.cat([waveform.clone(), padding], dim=1)
