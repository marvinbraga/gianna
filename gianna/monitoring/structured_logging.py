"""
Structured Logging for Gianna

This module provides structured logging capabilities with JSON output format,
making logs machine-readable for analysis with tools like Loki, Elasticsearch,
or cloud logging services.

Features:
- JSON-formatted log output
- Contextual logging with automatic context propagation
- Request/response correlation
- Performance metrics in logs
- Sensitive data redaction
- Multiple output targets (file, stdout, external services)

Usage:
    >>> from gianna.monitoring.structured_logging import configure_logging, get_logger
    >>>
    >>> # Configure at application startup
    >>> configure_logging(json_format=True, level="INFO")
    >>>
    >>> # Use the logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing request", user_id="123", action="query")
"""

import json
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

from loguru import logger as loguru_logger

T = TypeVar("T")


class LogLevel(Enum):
    """Log levels."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogConfig:
    """Configuration for structured logging."""

    # Output format
    json_format: bool = True
    pretty_print: bool = False  # Pretty print JSON (for development)
    include_timestamp: bool = True
    timestamp_format: str = "ISO"  # ISO or UNIX

    # Log level
    level: str = "INFO"
    module_levels: Dict[str, str] = field(default_factory=dict)  # Per-module levels

    # Output targets
    console_output: bool = True
    file_output: bool = False
    file_path: Optional[str] = None
    file_rotation: str = "10 MB"
    file_retention: str = "7 days"

    # Context
    include_context: bool = True
    include_caller: bool = True
    include_process: bool = True
    include_thread: bool = True

    # Security
    redact_patterns: List[str] = field(
        default_factory=lambda: [
            r"api[_-]?key",
            r"password",
            r"secret",
            r"token",
            r"auth",
            r"credential",
            r"private[_-]?key",
        ]
    )
    redact_replacement: str = "[REDACTED]"

    # Performance
    include_duration: bool = True
    slow_threshold_ms: float = 1000.0  # Log warning if operation exceeds this


# Thread-local storage for log context
_context_storage = threading.local()


def _get_context() -> Dict[str, Any]:
    """Get the current thread's log context."""
    if not hasattr(_context_storage, "context"):
        _context_storage.context = {}
    return _context_storage.context


def _set_context(context: Dict[str, Any]) -> None:
    """Set the current thread's log context."""
    _context_storage.context = context


class LogContext:
    """
    Context manager for adding contextual information to logs.

    All logs within the context will include the specified fields.

    Example:
        >>> with LogContext(request_id="abc123", user_id="user1"):
        ...     logger.info("Processing")  # Includes request_id and user_id
    """

    def __init__(self, **kwargs: Any):
        """Initialize with context fields."""
        self.new_context = kwargs
        self.previous_context: Dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        """Enter context."""
        self.previous_context = _get_context().copy()
        current = _get_context()
        current.update(self.new_context)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        _set_context(self.previous_context)


@contextmanager
def log_context(**kwargs: Any):
    """
    Context manager for adding contextual information to logs.

    Example:
        >>> with log_context(request_id="abc123"):
        ...     logger.info("Processing request")
    """
    previous_context = _get_context().copy()
    current = _get_context()
    current.update(kwargs)
    try:
        yield
    finally:
        _set_context(previous_context)


def add_context(**kwargs: Any) -> None:
    """
    Add fields to the current log context.

    These fields will be included in all subsequent logs
    until the context is cleared or overwritten.

    Example:
        >>> add_context(session_id="sess123")
        >>> logger.info("User action")  # Includes session_id
    """
    _get_context().update(kwargs)


def clear_context() -> None:
    """Clear all context fields."""
    _set_context({})


def get_correlation_id() -> str:
    """Get or create a correlation ID for the current context."""
    context = _get_context()
    if "correlation_id" not in context:
        context["correlation_id"] = str(uuid.uuid4())[:8]
    return context["correlation_id"]


class StructuredLogger:
    """
    Structured logger with JSON output support.

    Wraps loguru to provide structured logging with automatic
    context propagation and JSON formatting.
    """

    def __init__(self, name: str, config: Optional[LogConfig] = None):
        """
        Initialize structured logger.

        Args:
            name: Logger name (usually __name__)
            config: Optional configuration
        """
        self.name = name
        self.config = config or _global_config
        self._redact_patterns: Set[str] = set()
        self._compile_redact_patterns()

    def _compile_redact_patterns(self) -> None:
        """Compile regex patterns for redaction."""
        import re

        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.redact_patterns
        ]

    def _redact_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from log data."""
        result = {}
        for key, value in data.items():
            should_redact = any(
                pattern.search(key) for pattern in self._compiled_patterns
            )
            if should_redact:
                result[key] = self.config.redact_replacement
            elif isinstance(value, dict):
                result[key] = self._redact_sensitive(value)
            else:
                result[key] = value
        return result

    def _build_log_record(
        self,
        level: str,
        message: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build a structured log record."""
        record: Dict[str, Any] = {
            "level": level,
            "message": message,
            "logger": self.name,
        }

        # Add timestamp
        if self.config.include_timestamp:
            if self.config.timestamp_format == "ISO":
                record["timestamp"] = datetime.utcnow().isoformat() + "Z"
            else:
                record["timestamp"] = time.time()

        # Add caller info
        if self.config.include_caller:
            import inspect

            frame = inspect.currentframe()
            # Go up the stack to find the actual caller
            for _ in range(4):  # Skip internal frames
                if frame is not None:
                    frame = frame.f_back

            if frame:
                record["caller"] = {
                    "file": frame.f_code.co_filename,
                    "line": frame.f_lineno,
                    "function": frame.f_code.co_name,
                }

        # Add process/thread info
        if self.config.include_process:
            record["process"] = os.getpid()
        if self.config.include_thread:
            record["thread"] = threading.current_thread().name

        # Add context
        if self.config.include_context:
            context = _get_context()
            if context:
                record["context"] = context

        # Add extra fields
        if kwargs:
            record["extra"] = self._redact_sensitive(kwargs)

        return record

    def _format_output(self, record: Dict[str, Any]) -> str:
        """Format the log record for output."""
        if self.config.json_format:
            if self.config.pretty_print:
                return json.dumps(record, indent=2, default=str)
            return json.dumps(record, default=str)
        else:
            # Human-readable format
            parts = [
                record.get("timestamp", ""),
                f"[{record['level']}]",
                record["logger"],
                "-",
                record["message"],
            ]
            if "extra" in record:
                extra_str = " ".join(f"{k}={v}" for k, v in record["extra"].items())
                parts.append(f"| {extra_str}")
            return " ".join(str(p) for p in parts)

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal log method."""
        record = self._build_log_record(level, message, **kwargs)
        output = self._format_output(record)

        # Use loguru for actual output
        getattr(loguru_logger, level.lower())(output)

    def trace(self, message: str, **kwargs: Any) -> None:
        """Log at TRACE level."""
        self._log("TRACE", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log("INFO", message, **kwargs)

    def success(self, message: str, **kwargs: Any) -> None:
        """Log at SUCCESS level."""
        self._log("SUCCESS", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log("CRITICAL", message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            kwargs["exception"] = {
                "type": exc_info[0].__name__,
                "message": str(exc_info[1]),
                "traceback": traceback.format_exc(),
            }
        self._log("ERROR", message, **kwargs)

    @contextmanager
    def operation(self, name: str, **kwargs: Any):
        """
        Context manager for logging operations with timing.

        Example:
            >>> with logger.operation("process_request", user_id="123"):
            ...     do_processing()
        """
        start_time = time.perf_counter()
        operation_id = str(uuid.uuid4())[:8]

        self.info(f"Starting {name}", operation=name, operation_id=operation_id, **kwargs)

        try:
            yield
            duration_ms = (time.perf_counter() - start_time) * 1000

            log_method = self.info
            if duration_ms > self.config.slow_threshold_ms:
                log_method = self.warning

            log_method(
                f"Completed {name}",
                operation=name,
                operation_id=operation_id,
                duration_ms=round(duration_ms, 2),
                status="success",
                **kwargs,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.error(
                f"Failed {name}",
                operation=name,
                operation_id=operation_id,
                duration_ms=round(duration_ms, 2),
                status="error",
                error=str(e),
                error_type=type(e).__name__,
                **kwargs,
            )
            raise


def log_execution(
    logger: Optional[StructuredLogger] = None,
    level: str = "INFO",
    include_args: bool = False,
    include_result: bool = False,
):
    """
    Decorator to log function execution.

    Args:
        logger: Logger instance (uses global if None)
        level: Log level
        include_args: Whether to log function arguments
        include_result: Whether to log function result

    Example:
        >>> @log_execution(include_args=True)
        ... def process_data(data):
        ...     return transformed_data
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            extra: Dict[str, Any] = {"function": func.__name__}

            if include_args:
                extra["args"] = repr(args)[:200]
                extra["kwargs"] = repr(kwargs)[:200]

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                extra["duration_ms"] = round(duration_ms, 2)
                extra["status"] = "success"

                if include_result:
                    extra["result"] = repr(result)[:200]

                getattr(logger, level.lower())(
                    f"Executed {func.__name__}", **extra
                )

                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                extra["duration_ms"] = round(duration_ms, 2)
                extra["status"] = "error"
                extra["error"] = str(e)
                extra["error_type"] = type(e).__name__

                logger.error(f"Failed {func.__name__}", **extra)
                raise

        return wrapper

    return decorator


# Global configuration
_global_config = LogConfig()
_loggers: Dict[str, StructuredLogger] = {}


def configure_logging(
    json_format: bool = True,
    level: str = "INFO",
    file_path: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Configure global logging settings.

    Should be called once at application startup.

    Args:
        json_format: Whether to output JSON format
        level: Default log level
        file_path: Optional file path for log output
        **kwargs: Additional configuration options
    """
    global _global_config

    _global_config = LogConfig(
        json_format=json_format,
        level=level,
        file_output=file_path is not None,
        file_path=file_path,
        **kwargs,
    )

    # Configure loguru
    loguru_logger.remove()  # Remove default handler

    # Add console handler
    if _global_config.console_output:
        loguru_logger.add(
            sys.stderr,
            level=level,
            format="{message}" if json_format else (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
        )

    # Add file handler
    if _global_config.file_output and _global_config.file_path:
        loguru_logger.add(
            _global_config.file_path,
            level=level,
            rotation=_global_config.file_rotation,
            retention=_global_config.file_retention,
            format="{message}" if json_format else (
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
        )


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, _global_config)
    return _loggers[name]


# Convenience exports
__all__ = [
    "LogConfig",
    "LogContext",
    "LogLevel",
    "StructuredLogger",
    "add_context",
    "clear_context",
    "configure_logging",
    "get_correlation_id",
    "get_logger",
    "log_context",
    "log_execution",
]
