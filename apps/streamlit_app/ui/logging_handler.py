"""Custom logging handler for capturing logs in Streamlit session state.

This module provides a logging.Handler that appends formatted log lines
to a list, suitable for display in Streamlit UI.
"""

import logging
import threading
from typing import Callable


class SessionStateLogHandler(logging.Handler):
    """Logging handler that appends formatted logs to a list.
    
    This handler is designed to capture log messages during processing
    and store them in a list that can be displayed in Streamlit's UI.
    
    Example:
        >>> import logging
        >>> logs = []
        >>> handler = SessionStateLogHandler(logs.append)
        >>> logger = logging.getLogger("mylogger")
        >>> logger.addHandler(handler)
        >>> logger.info("Hello")
        >>> print(logs)
        ['INFO - mylogger - Hello']
    """
    
    def __init__(
        self,
        append_func: Callable[[str], None],
        level: int = logging.DEBUG,
        fmt: str | None = None,
    ):
        """Initialize the handler.
        
        Args:
            append_func: Function to call with each formatted log line.
            level: Minimum log level to capture.
            fmt: Log format string. Defaults to "%(levelname)s - %(name)s - %(message)s".
        """
        super().__init__(level=level)
        self._append_func = append_func
        self._lock = threading.Lock()
        
        if fmt is None:
            fmt = "%(levelname)s - %(name)s - %(message)s"
        
        self.setFormatter(logging.Formatter(fmt))
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by appending formatted message.
        
        Args:
            record: The log record to emit.
        """
        try:
            msg = self.format(record)
            with self._lock:
                self._append_func(msg)
        except Exception:
            # Don't let logging errors crash the app
            self.handleError(record)


class LogCaptureContext:
    """Context manager that captures logs to a list during a code block.
    
    Example:
        >>> with LogCaptureContext() as ctx:
        ...     logging.getLogger("test").info("Message")
        >>> print(ctx.logs)
        ['INFO - test - Message']
    """
    
    def __init__(
        self,
        logger_names: list[str] | None = None,
        level: int = logging.DEBUG,
    ):
        """Initialize the context.
        
        Args:
            logger_names: List of logger names to capture. If None,
                captures from root logger and src.* loggers.
            level: Minimum log level to capture.
        """
        self.logs: list[str] = []
        self._handlers: list[tuple[logging.Logger, SessionStateLogHandler]] = []
        self._level = level
        
        # Default loggers to capture
        self._logger_names = logger_names or [
            "",  # root logger
            "src",
            "model",
            "audioio",
            "timeline",
        ]
    
    def __enter__(self) -> "LogCaptureContext":
        """Start capturing logs."""
        for name in self._logger_names:
            logger = logging.getLogger(name)
            handler = SessionStateLogHandler(
                append_func=self.logs.append,
                level=self._level,
            )
            logger.addHandler(handler)
            self._handlers.append((logger, handler))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop capturing logs and clean up handlers."""
        for logger, handler in self._handlers:
            logger.removeHandler(handler)
        self._handlers.clear()
        return None
