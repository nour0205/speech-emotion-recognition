"""Audio validation functions."""

import torch

from .errors import AudioValidationError
from .utils import compute_duration_sec, rms


# Valid sample rate range
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000


def validate_wav(
    waveform: torch.Tensor,
    sample_rate: int,
    min_duration_sec: float = 0.1,
    max_duration_sec: float = 600.0,
    allow_stereo: bool = True,
    allow_multi_channel: bool = False,
    reject_silence: bool = True,
    silence_rms_threshold: float = 1e-4,
) -> None:
    """Validate audio waveform and properties.
    
    Performs comprehensive validation on audio data including:
    - Tensor type and dtype checks
    - Finite value checks (no NaN/Inf)
    - Sample rate range validation
    - Duration bounds validation
    - Channel count validation
    - Silence detection
    
    Args:
        waveform: Audio tensor with shape [channels, samples].
        sample_rate: Sample rate in Hz.
        min_duration_sec: Minimum allowed duration in seconds.
        max_duration_sec: Maximum allowed duration in seconds.
        allow_stereo: Whether to allow stereo (2-channel) audio.
        allow_multi_channel: Whether to allow >2 channels.
        reject_silence: Whether to reject near-silent audio.
        silence_rms_threshold: RMS threshold below which audio is silent.
        
    Raises:
        AudioValidationError: If validation fails, with appropriate code:
            - INVALID_DTYPE: Waveform is not a float tensor.
            - EMPTY_AUDIO: Waveform has no samples.
            - NON_FINITE: Waveform contains NaN or Inf values.
            - INVALID_SAMPLE_RATE: Sample rate outside valid range.
            - TOO_SHORT: Duration below minimum threshold.
            - TOO_LONG: Duration exceeds maximum threshold.
            - TOO_MANY_CHANNELS: More channels than allowed.
            - SILENCE: Audio is near-silent.
            
    Examples:
        >>> import torch
        >>> waveform = torch.randn(1, 16000)
        >>> validate_wav(waveform, 16000)  # No error if valid
    """
    # Check tensor type
    if not isinstance(waveform, torch.Tensor):
        raise AudioValidationError(
            message=f"Waveform must be a torch.Tensor, got {type(waveform).__name__}",
            code="INVALID_DTYPE",
            details={"actual_type": type(waveform).__name__},
        )
    
    # Check float dtype
    if not waveform.is_floating_point():
        raise AudioValidationError(
            message=f"Waveform must be float tensor, got {waveform.dtype}",
            code="INVALID_DTYPE",
            details={"actual_dtype": str(waveform.dtype)},
        )
    
    # Check non-empty
    if waveform.numel() == 0:
        raise AudioValidationError(
            message="Waveform is empty (no samples)",
            code="EMPTY_AUDIO",
            details={"shape": list(waveform.shape)},
        )
    
    # Ensure 2D shape
    if waveform.ndim != 2:
        raise AudioValidationError(
            message=f"Waveform must be 2D [channels, samples], got shape {list(waveform.shape)}",
            code="INVALID_DTYPE",
            details={"shape": list(waveform.shape), "ndim": waveform.ndim},
        )
    
    num_channels, num_samples = waveform.shape
    
    # Check for NaN/Inf
    if not torch.isfinite(waveform).all():
        nan_count = int(torch.isnan(waveform).sum())
        inf_count = int(torch.isinf(waveform).sum())
        raise AudioValidationError(
            message="Waveform contains non-finite values (NaN or Inf)",
            code="NON_FINITE",
            details={"nan_count": nan_count, "inf_count": inf_count},
        )
    
    # Check sample rate
    if not isinstance(sample_rate, int) or sample_rate < MIN_SAMPLE_RATE or sample_rate > MAX_SAMPLE_RATE:
        raise AudioValidationError(
            message=f"Sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz, got {sample_rate}",
            code="INVALID_SAMPLE_RATE",
            details={
                "sample_rate": sample_rate,
                "min_allowed": MIN_SAMPLE_RATE,
                "max_allowed": MAX_SAMPLE_RATE,
            },
        )
    
    # Check duration
    duration_sec = compute_duration_sec(num_samples, sample_rate)
    
    if duration_sec < min_duration_sec:
        raise AudioValidationError(
            message=f"Audio too short: {duration_sec:.3f}s < {min_duration_sec}s minimum",
            code="TOO_SHORT",
            details={
                "duration_sec": duration_sec,
                "min_duration_sec": min_duration_sec,
                "num_samples": num_samples,
                "sample_rate": sample_rate,
            },
        )
    
    if duration_sec > max_duration_sec:
        raise AudioValidationError(
            message=f"Audio too long: {duration_sec:.3f}s > {max_duration_sec}s maximum",
            code="TOO_LONG",
            details={
                "duration_sec": duration_sec,
                "max_duration_sec": max_duration_sec,
                "num_samples": num_samples,
                "sample_rate": sample_rate,
            },
        )
    
    # Check channels
    if num_channels == 0:
        raise AudioValidationError(
            message="Audio has no channels",
            code="EMPTY_AUDIO",
            details={"channels": num_channels},
        )
    
    if num_channels > 2 and not allow_multi_channel:
        raise AudioValidationError(
            message=f"Audio has {num_channels} channels, but multi-channel is not allowed",
            code="TOO_MANY_CHANNELS",
            details={"channels": num_channels, "max_allowed": 2},
        )
    
    if num_channels == 2 and not allow_stereo:
        raise AudioValidationError(
            message="Stereo audio not allowed",
            code="TOO_MANY_CHANNELS",
            details={"channels": num_channels, "max_allowed": 1},
        )
    
    # Check silence
    if reject_silence:
        audio_rms = rms(waveform)
        if audio_rms < silence_rms_threshold:
            raise AudioValidationError(
                message=f"Audio is near-silent (RMS={audio_rms:.6f} < {silence_rms_threshold})",
                code="SILENCE",
                details={
                    "rms": audio_rms,
                    "threshold": silence_rms_threshold,
                },
            )
