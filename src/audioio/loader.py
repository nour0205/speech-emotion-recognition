"""Audio loading functions for WAV files."""

import io
from pathlib import Path

import soundfile as sf
import torch

from .errors import AudioDecodeError


def load_wav(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load a WAV audio file from disk.
    
    Args:
        path: Path to the WAV file.
        
    Returns:
        Tuple of (waveform, sample_rate) where waveform is a float32 tensor
        with shape [channels, num_samples].
        
    Raises:
        AudioDecodeError: If file cannot be read or is not a valid WAV.
        
    Examples:
        >>> waveform, sr = load_wav("audio.wav")
        >>> waveform.shape  # [channels, samples]
        torch.Size([1, 16000])
    """
    path = Path(path)
    
    if not path.exists():
        raise AudioDecodeError(
            message=f"Audio file not found: {path}",
            code="FILE_NOT_FOUND",
            details={"path": str(path)},
        )
    
    if path.stat().st_size == 0:
        raise AudioDecodeError(
            message=f"Audio file is empty: {path}",
            code="EMPTY_FILE",
            details={"path": str(path)},
        )
    
    try:
        # soundfile returns (samples, channels) for multi-channel
        # and (samples,) for mono
        data, sample_rate = sf.read(str(path), dtype="float32")
    except Exception as e:
        raise AudioDecodeError(
            message=f"Failed to decode WAV file: {e}",
            code="INVALID_WAV",
            details={"path": str(path), "error": str(e)},
        ) from e
    
    return _numpy_to_tensor(data, sample_rate, source=str(path))


def load_wav_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    """Load WAV audio from raw bytes.
    
    Args:
        data: Raw bytes of a WAV file.
        
    Returns:
        Tuple of (waveform, sample_rate) where waveform is a float32 tensor
        with shape [channels, num_samples].
        
    Raises:
        AudioDecodeError: If bytes cannot be decoded as WAV.
        
    Examples:
        >>> with open("audio.wav", "rb") as f:
        ...     audio_bytes = f.read()
        >>> waveform, sr = load_wav_bytes(audio_bytes)
    """
    if not data:
        raise AudioDecodeError(
            message="Audio data is empty",
            code="EMPTY_FILE",
            details={"bytes_length": 0},
        )
    
    try:
        buffer = io.BytesIO(data)
        # soundfile returns (samples, channels) for multi-channel
        # and (samples,) for mono
        audio_data, sample_rate = sf.read(buffer, dtype="float32")
    except Exception as e:
        raise AudioDecodeError(
            message=f"Failed to decode WAV bytes: {e}",
            code="INVALID_WAV",
            details={"bytes_length": len(data), "error": str(e)},
        ) from e
    
    return _numpy_to_tensor(audio_data, sample_rate, source="bytes")


def _numpy_to_tensor(
    data,
    sample_rate: int,
    source: str,
) -> tuple[torch.Tensor, int]:
    """Convert numpy audio data to torch tensor.
    
    Args:
        data: Numpy array from soundfile (samples,) or (samples, channels).
        sample_rate: Sample rate in Hz.
        source: Source description for error messages.
        
    Returns:
        Tuple of (waveform, sample_rate) where waveform has shape [channels, samples].
    """
    import numpy as np
    
    if data.size == 0:
        raise AudioDecodeError(
            message="Audio contains no samples",
            code="EMPTY_AUDIO",
            details={"source": source},
        )
    
    # Convert to tensor
    waveform = torch.from_numpy(data).float()
    
    # Reshape to [channels, samples]
    if waveform.ndim == 1:
        # Mono: (samples,) -> (1, samples)
        waveform = waveform.unsqueeze(0)
    else:
        # Multi-channel: (samples, channels) -> (channels, samples)
        waveform = waveform.T
    
    return waveform, sample_rate
