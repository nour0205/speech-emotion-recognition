"""Custom exceptions for model inference operations."""

from typing import Any


class ModelError(Exception):
    """Base exception for all model-related errors.
    
    Attributes:
        message: Human-readable error description.
        code: Short error code string (e.g., "MODEL_LOAD_FAILED").
        details: Optional dictionary with additional context.
    """
    
    def __init__(
        self,
        message: str,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ModelError.
        
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


class ModelLoadError(ModelError):
    """Raised when model loading fails.
    
    Common codes:
        - MODEL_NOT_FOUND: Model ID not found in registry.
        - DOWNLOAD_FAILED: Failed to download model from remote source.
        - LOAD_FAILED: Model file exists but failed to load.
        - INVALID_MODEL: Model configuration or weights are invalid.
    """
    pass


class InferenceError(ModelError):
    """Raised when inference fails.
    
    Common codes:
        - INVALID_INPUT: Input waveform/audio is invalid.
        - INFERENCE_FAILED: Model forward pass failed.
        - POSTPROCESS_FAILED: Failed to process model outputs.
    """
    pass
