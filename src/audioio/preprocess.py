"""Audio preprocessing functions."""

import torch
import torchaudio.transforms as T

from .errors import AudioPreprocessError
from .utils import clamp_finite, ensure_float32_torch, safe_peak_normalize


def preprocess_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int = 16000,
    to_mono: bool = True,
    normalize: bool = True,
    peak_target: float = 0.95,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, int]:
    """Preprocess audio to canonical form for SER.
    
    Converts audio to a deterministic canonical format:
    - Mono channel (if to_mono=True)
    - Target sample rate (default 16kHz)
    - Float32 dtype
    - Peak normalized (if normalize=True)
    - Shape [1, T]
    
    Args:
        waveform: Input audio tensor with shape [channels, samples].
        sample_rate: Input sample rate in Hz.
        target_sample_rate: Output sample rate in Hz.
        to_mono: Whether to convert to mono.
        normalize: Whether to apply peak normalization.
        peak_target: Target peak amplitude for normalization (0.0 to 1.0).
        eps: Epsilon for numerical stability.
        
    Returns:
        Tuple of (processed_waveform, target_sample_rate) where
        processed_waveform has shape [1, T] and dtype float32.
        
    Raises:
        AudioPreprocessError: If preprocessing fails, e.g., unsupported channels.
        
    Examples:
        >>> import torch
        >>> waveform = torch.randn(2, 32000)  # Stereo, 32kHz
        >>> processed, sr = preprocess_audio(waveform, 32000)
        >>> processed.shape
        torch.Size([1, 16000])
        >>> sr
        16000
    """
    # Ensure float32
    waveform = ensure_float32_torch(waveform)
    
    # Validate input shape
    if waveform.ndim != 2:
        raise AudioPreprocessError(
            message=f"Expected 2D tensor [channels, samples], got shape {list(waveform.shape)}",
            code="INVALID_SHAPE",
            details={"shape": list(waveform.shape)},
        )
    
    num_channels = waveform.shape[0]
    
    # Convert to mono if requested
    if to_mono and num_channels > 1:
        if num_channels > 2:
            # Do not average >2 channels by default
            raise AudioPreprocessError(
                message=f"Cannot convert {num_channels}-channel audio to mono. Only mono and stereo supported.",
                code="UNSUPPORTED_CHANNELS",
                details={"channels": num_channels},
            )
        # Average stereo to mono
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample to target sample rate
    if sample_rate != target_sample_rate:
        try:
            resampler = T.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate,
                dtype=waveform.dtype,
            )
            waveform = resampler(waveform)
        except Exception as e:
            raise AudioPreprocessError(
                message=f"Resampling failed: {e}",
                code="RESAMPLE_FAILED",
                details={
                    "original_sr": sample_rate,
                    "target_sr": target_sample_rate,
                    "error": str(e),
                },
            ) from e
    
    # Ensure output is [1, T] if mono
    if to_mono and waveform.shape[0] != 1:
        waveform = waveform[:1, :]
    
    # Peak normalize
    if normalize:
        waveform = safe_peak_normalize(waveform, peak_target=peak_target, eps=eps)
    
    # Ensure no NaN/Inf and clamp to valid range
    waveform = clamp_finite(waveform, min_val=-1.0, max_val=1.0)
    
    # Final dtype check
    waveform = waveform.to(dtype=torch.float32)
    
    return waveform, target_sample_rate
