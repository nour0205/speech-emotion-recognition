"""Error handling and HTTP mapping for the API.

This module provides:
- API-specific exception classes
- Mapping from internal errors to HTTP status codes
- Exception handlers for FastAPI

Error Code Mapping:
    - AudioDecodeError -> 400 INVALID_AUDIO
    - AudioValidationError -> 422 INVALID_INPUT
    - WindowingConfigError -> 422 INVALID_WINDOWING
    - ModelLoadError -> 500 MODEL_LOAD_FAILED
    - InferenceError -> 500 INFERENCE_FAILED
    - Generic exceptions -> 500 INTERNAL_ERROR
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from audioio.errors import AudioDecodeError, AudioValidationError, AudioPreprocessError
from model.errors import InferenceError, ModelLoadError
from timeline.errors import WindowingConfigError, WindowingRuntimeError

from .schemas import ApiErrorResponse, ErrorDetail


logger = logging.getLogger(__name__)


# =============================================================================
# API Exception Classes
# =============================================================================


class ApiError(Exception):
    """Base exception for API errors.
    
    Attributes:
        status_code: HTTP status code to return.
        code: Machine-readable error code.
        message: Human-readable error message.
        details: Optional additional context.
    """
    
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


class InvalidAudioError(ApiError):
    """Raised when audio cannot be decoded or read."""
    
    def __init__(
        self,
        message: str = "Failed to decode audio file",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            status_code=400,
            code="INVALID_AUDIO",
            message=message,
            details=details,
        )


class InvalidInputError(ApiError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str = "Invalid input",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            status_code=422,
            code="INVALID_INPUT",
            message=message,
            details=details,
        )


class InvalidWindowingError(ApiError):
    """Raised when windowing configuration is invalid."""
    
    def __init__(
        self,
        message: str = "Invalid windowing configuration",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            status_code=422,
            code="INVALID_WINDOWING",
            message=message,
            details=details,
        )


class ModelLoadFailedError(ApiError):
    """Raised when model loading fails."""
    
    def __init__(
        self,
        message: str = "Failed to load model",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            status_code=500,
            code="MODEL_LOAD_FAILED",
            message=message,
            details=details,
        )


class InferenceFailedError(ApiError):
    """Raised when inference fails."""
    
    def __init__(
        self,
        message: str = "Inference failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            status_code=500,
            code="INFERENCE_FAILED",
            message=message,
            details=details,
        )


class InternalError(ApiError):
    """Raised for unexpected internal errors."""
    
    def __init__(
        self,
        message: str = "Internal server error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            status_code=500,
            code="INTERNAL_ERROR",
            message=message,
            details=details,
        )


# =============================================================================
# Error Mapping Functions
# =============================================================================


def map_exception_to_api_error(exc: Exception) -> ApiError:
    """Map internal exceptions to appropriate API errors.
    
    Args:
        exc: The exception raised during processing.
        
    Returns:
        An ApiError subclass with appropriate HTTP status and code.
    """
    # Audio decode errors -> 400
    if isinstance(exc, AudioDecodeError):
        return InvalidAudioError(
            message=exc.message,
            details=exc.details,
        )
    
    # Audio validation errors -> 422
    if isinstance(exc, AudioValidationError):
        return InvalidInputError(
            message=exc.message,
            details=exc.details,
        )
    
    # Audio preprocessing errors -> 422
    if isinstance(exc, AudioPreprocessError):
        return InvalidInputError(
            message=exc.message,
            details=exc.details,
        )
    
    # Windowing config errors -> 422
    if isinstance(exc, WindowingConfigError):
        return InvalidWindowingError(
            message=exc.message,
            details=exc.details,
        )
    
    # Windowing runtime errors -> 422
    if isinstance(exc, WindowingRuntimeError):
        return InvalidWindowingError(
            message=exc.message,
            details=exc.details,
        )
    
    # Model load errors -> 500
    if isinstance(exc, ModelLoadError):
        return ModelLoadFailedError(
            message=exc.message,
            details=exc.details,
        )
    
    # Inference errors -> 500
    if isinstance(exc, InferenceError):
        return InferenceFailedError(
            message=exc.message,
            details=exc.details,
        )
    
    # Already an API error, return as-is
    if isinstance(exc, ApiError):
        return exc
    
    # Generic fallback -> 500
    return InternalError(
        message=str(exc) or "An unexpected error occurred",
        details={"exception_type": type(exc).__name__},
    )


def create_error_response(api_error: ApiError) -> ApiErrorResponse:
    """Create a structured error response from an API error.
    
    Args:
        api_error: The API error to convert.
        
    Returns:
        ApiErrorResponse with properly structured error details.
    """
    return ApiErrorResponse(
        error=ErrorDetail(
            code=api_error.code,
            message=api_error.message,
            details=api_error.details,
        )
    )


# =============================================================================
# Exception Handlers
# =============================================================================


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    """Handle ApiError exceptions.
    
    Args:
        request: The incoming request.
        exc: The ApiError exception.
        
    Returns:
        JSONResponse with error details.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        "API error: code=%s message=%s request_id=%s",
        exc.code,
        exc.message,
        request_id,
    )
    
    response = create_error_response(exc)
    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(),
        headers={"X-Request-ID": request_id},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions.
    
    Maps internal exceptions to appropriate HTTP responses.
    
    Args:
        request: The incoming request.
        exc: The unhandled exception.
        
    Returns:
        JSONResponse with error details.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Map to API error
    api_error = map_exception_to_api_error(exc)
    
    # Log with appropriate level
    if api_error.status_code >= 500:
        logger.error(
            "Internal error: code=%s message=%s request_id=%s",
            api_error.code,
            api_error.message,
            request_id,
            exc_info=True,
        )
    else:
        logger.warning(
            "Request error: code=%s message=%s request_id=%s",
            api_error.code,
            api_error.message,
            request_id,
        )
    
    response = create_error_response(api_error)
    return JSONResponse(
        status_code=api_error.status_code,
        content=response.model_dump(),
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app.
    
    Args:
        app: The FastAPI application instance.
    """
    # Handle API errors
    app.add_exception_handler(ApiError, api_error_handler)
    
    # Handle all internal errors
    app.add_exception_handler(AudioDecodeError, generic_exception_handler)
    app.add_exception_handler(AudioValidationError, generic_exception_handler)
    app.add_exception_handler(AudioPreprocessError, generic_exception_handler)
    app.add_exception_handler(WindowingConfigError, generic_exception_handler)
    app.add_exception_handler(WindowingRuntimeError, generic_exception_handler)
    app.add_exception_handler(ModelLoadError, generic_exception_handler)
    app.add_exception_handler(InferenceError, generic_exception_handler)
    
    # Catch-all for unexpected errors
    app.add_exception_handler(Exception, generic_exception_handler)
