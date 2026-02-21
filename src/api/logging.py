"""Structured logging and request middleware for the API.

This module provides:
- Structured logging configuration (JSON-like key=value format)
- Request ID middleware for tracing
- Request timing middleware

Example:
    >>> from src.api.logging import setup_logging, RequestIDMiddleware
    >>> setup_logging("INFO")
"""

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings


# Context variable for request ID (thread-safe)
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


# =============================================================================
# Custom Logging Formatter
# =============================================================================


class StructuredFormatter(logging.Formatter):
    """Structured log formatter producing key=value output.
    
    Formats log messages as:
        timestamp=ISO8601 level=LEVEL logger=NAME request_id=ID message=MSG
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as structured key=value pairs."""
        # Get ISO format timestamp
        timestamp = self.formatTime(record, self.datefmt)
        
        # Get request ID from context if available
        req_id = request_id_var.get() or "-"
        
        # Base fields
        parts = [
            f"timestamp={timestamp}",
            f"level={record.levelname}",
            f"logger={record.name}",
            f"request_id={req_id}",
        ]
        
        # Add message
        message = record.getMessage()
        # Escape any equals signs in the message
        message = message.replace('"', '\\"')
        parts.append(f'message="{message}"')
        
        # Add exception info if present
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            exc_text = exc_text.replace('\n', ' | ').replace('"', '\\"')
            parts.append(f'exception="{exc_text}"')
        
        return " ".join(parts)


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application.
    
    Sets up logging handlers with structured output format.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR).
    """
    # Create formatter
    formatter = StructuredFormatter(
        datefmt="%Y-%m-%dT%H:%M:%S%z"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(log_level.upper())
    root_logger.addHandler(stdout_handler)
    
    # Reduce noise from third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# =============================================================================
# Middleware
# =============================================================================


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to assign a unique request ID to each request.
    
    The request ID is:
    - Generated as UUID4 if not provided
    - Stored in request.state.request_id
    - Added to X-Request-ID response header
    - Made available via request_id_var context variable
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with request ID tracking."""
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Store in request state
        request.state.request_id = request_id
        
        # Store in context variable for logging
        token = request_id_var.set(request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
        finally:
            # Reset context variable
            request_id_var.reset(token)


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to track and log request timing.
    
    Logs total request duration and adds X-Response-Time header.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with timing tracking."""
        start_time = time.perf_counter()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        # Log request completion
        logger = logging.getLogger("api.timing")
        logger.info(
            "method=%s path=%s status=%d duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        
        return response


def add_middleware(app: FastAPI) -> None:
    """Add all middleware to the FastAPI application.
    
    Args:
        app: The FastAPI application instance.
    """
    # Order matters: RequestID must be added last to run first
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
