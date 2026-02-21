"""Custom exceptions for timeline windowing operations."""

from typing import Any


class TimelineError(Exception):
    """Base exception for all timeline errors.
    
    Attributes:
        message: Human-readable error description.
        code: Short error code string (e.g., "INVALID_CONFIG").
        details: Optional dictionary with additional context.
    """
    
    def __init__(
        self,
        message: str,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize TimelineError.
        
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


class WindowingConfigError(TimelineError):
    """Raised when windowing configuration is invalid.
    
    Common codes:
        - INVALID_CONFIG: Configuration parameters are invalid.
    """
    
    def __init__(
        self,
        message: str,
        code: str = "INVALID_CONFIG",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize WindowingConfigError.
        
        Args:
            message: Human-readable error description.
            code: Short error code string. Defaults to "INVALID_CONFIG".
            details: Optional dictionary with additional context.
        """
        super().__init__(message, code, details)


class WindowingRuntimeError(TimelineError):
    """Raised when windowing fails at runtime.
    
    Common codes:
        - INVALID_SHAPE: Input tensor has wrong shape.
        - EMPTY_INPUT: Input tensor has zero samples.
        - WINDOW_TOO_LARGE: Window size exceeds input length.
    """
    
    def __init__(
        self,
        message: str,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize WindowingRuntimeError.
        
        Args:
            message: Human-readable error description.
            code: Short error code string.
            details: Optional dictionary with additional context.
        """
        super().__init__(message, code, details)
