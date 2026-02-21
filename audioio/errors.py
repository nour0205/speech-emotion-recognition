"""Custom exceptions for audio I/O operations."""

from typing import Any


class AudioIOError(Exception):
    """Base exception for all audio I/O errors.
    
    Attributes:
        message: Human-readable error description.
        code: Short error code string (e.g., "INVALID_WAV").
        details: Optional dictionary with additional context.
    """
    
    def __init__(
        self,
        message: str,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize AudioIOError.
        
        Args:
            message: Human-readable error description.
            code: Short error code string.
            details: Optional dictionary with additional context.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return formatted error string."""
        if self.details:
            return f"[{self.code}] {self.message} (details: {self.details})"
        return f"[{self.code}] {self.message}"
    
    def __repr__(self) -> str:
        """Return repr string."""
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code!r}, details={self.details!r})"


class AudioDecodeError(AudioIOError):
    """Raised when audio decoding fails.
    
    Common codes:
        - INVALID_WAV: File is not a valid WAV or cannot be decoded.
        - FILE_NOT_FOUND: Audio file does not exist.
        - EMPTY_FILE: File has zero bytes.
    """
    pass


class AudioValidationError(AudioIOError):
    """Raised when audio validation fails.
    
    Common codes:
        - EMPTY_AUDIO: Waveform has no samples.
        - TOO_SHORT: Duration below minimum threshold.
        - TOO_LONG: Duration exceeds maximum threshold.
        - INVALID_SAMPLE_RATE: Sample rate outside valid range.
        - TOO_MANY_CHANNELS: More channels than allowed.
        - SILENCE: Audio is near-silent (RMS below threshold).
        - NON_FINITE: Waveform contains NaN or Inf values.
        - INVALID_DTYPE: Waveform is not float tensor.
    """
    pass


class AudioPreprocessError(AudioIOError):
    """Raised when audio preprocessing fails.
    
    Common codes:
        - UNSUPPORTED_CHANNELS: Cannot process audio with this channel count.
        - RESAMPLE_FAILED: Resampling operation failed.
    """
    pass
