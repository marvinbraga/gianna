"""
Structured logging configuration for Gianna production.
Provides JSON-formatted logs with correlation IDs and contextual information.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from loguru import logger


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id

        # Add user context if available
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id

        # Add request context if available
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id

        # Add extra fields
        if hasattr(record, "extra"):
            log_entry.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": (
                    self.formatException(record.exc_info) if record.exc_info else None
                ),
            }

        return json.dumps(log_entry, ensure_ascii=False)


class CorrelationIDFilter(logging.Filter):
    """Filter to add correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to record if not present."""
        if not hasattr(record, "correlation_id"):
            record.correlation_id = getattr(self, "_correlation_id", str(uuid.uuid4()))
        return True

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for this filter."""
        self._correlation_id = correlation_id


def configure_logging(
    level: str = "INFO",
    format_type: str = "json",
    enable_file_logging: bool = True,
    log_file: Optional[str] = None,
    enable_syslog: bool = False,
) -> None:
    """
    Configure structured logging for Gianna.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (json or text)
        enable_file_logging: Whether to enable file logging
        log_file: Path to log file (defaults to logs/gianna.log)
        enable_syslog: Whether to enable syslog logging
    """
    # Clear existing handlers
    logging.root.handlers.clear()

    # Set root logger level
    logging.root.setLevel(getattr(logging, level.upper()))

    # Create formatters
    if format_type == "json":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CorrelationIDFilter())
    logging.root.addHandler(console_handler)

    # File handler
    if enable_file_logging:
        if not log_file:
            os.makedirs("logs", exist_ok=True)
            log_file = "logs/gianna.log"

        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10,
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationIDFilter())
        logging.root.addHandler(file_handler)

    # Syslog handler (Linux only)
    if enable_syslog and sys.platform.startswith("linux"):
        from logging.handlers import SysLogHandler

        syslog_handler = SysLogHandler(address="/dev/log")
        syslog_handler.setFormatter(formatter)
        syslog_handler.addFilter(CorrelationIDFilter())
        logging.root.addHandler(syslog_handler)

    # Configure loguru for backward compatibility
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "format": "{time} | {level} | {name}:{function}:{line} | {message}",
                "level": level,
            }
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding contextual information to logs."""

    def __init__(self, **context: Any):
        """Initialize with context information."""
        self.context = context
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self) -> "LogContext":
        """Enter context and set up log record factory."""

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore original log record factory."""
        logging.setLogRecordFactory(self.old_factory)


# Structured logging utilities
def log_with_context(
    logger_instance: logging.Logger, level: str, message: str, **context: Any
) -> None:
    """Log with additional context information."""
    with LogContext(**context):
        getattr(logger_instance, level.lower())(message)


def log_request_start(
    logger_instance: logging.Logger,
    request_id: str,
    method: str,
    url: str,
    user_id: Optional[str] = None,
) -> None:
    """Log the start of a request."""
    context = {
        "request_id": request_id,
        "event": "request_start",
        "http_method": method,
        "url": url,
    }
    if user_id:
        context["user_id"] = user_id

    with LogContext(**context):
        logger_instance.info(f"Starting {method} request to {url}")


def log_request_end(
    logger_instance: logging.Logger,
    request_id: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """Log the end of a request."""
    with LogContext(
        request_id=request_id,
        event="request_end",
        status_code=status_code,
        duration_ms=duration_ms,
    ):
        logger_instance.info(
            f"Request completed with status {status_code} in {duration_ms:.2f}ms"
        )


def log_error_with_context(
    logger_instance: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log an error with additional context."""
    error_context = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "event": "error_occurred",
    }
    if context:
        error_context.update(context)

    with LogContext(**error_context):
        logger_instance.error(f"Error occurred: {error}", exc_info=True)


# Performance logging
def log_performance(
    logger_instance: logging.Logger,
    operation: str,
    duration_ms: float,
    success: bool = True,
    **metadata: Any,
) -> None:
    """Log performance metrics."""
    perf_context = {
        "event": "performance_metric",
        "operation": operation,
        "duration_ms": duration_ms,
        "success": success,
    }
    perf_context.update(metadata)

    with LogContext(**perf_context):
        if success:
            logger_instance.info(
                f"Operation '{operation}' completed in {duration_ms:.2f}ms"
            )
        else:
            logger_instance.warning(
                f"Operation '{operation}' failed after {duration_ms:.2f}ms"
            )


# Security logging
def log_security_event(
    logger_instance: logging.Logger,
    event_type: str,
    severity: str,
    description: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    **metadata: Any,
) -> None:
    """Log security events."""
    security_context = {
        "event": "security_event",
        "event_type": event_type,
        "severity": severity,
        "description": description,
    }
    if user_id:
        security_context["user_id"] = user_id
    if ip_address:
        security_context["ip_address"] = ip_address
    security_context.update(metadata)

    with LogContext(**security_context):
        level = (
            "critical"
            if severity == "high"
            else "warning" if severity == "medium" else "info"
        )
        getattr(logger_instance, level)(f"Security event: {description}")


# Business logic logging
def log_business_event(
    logger_instance: logging.Logger,
    event_type: str,
    description: str,
    user_id: Optional[str] = None,
    **metadata: Any,
) -> None:
    """Log business events."""
    business_context = {
        "event": "business_event",
        "event_type": event_type,
        "description": description,
    }
    if user_id:
        business_context["user_id"] = user_id
    business_context.update(metadata)

    with LogContext(**business_context):
        logger_instance.info(f"Business event: {description}")


# Initialize default logging configuration
if not logging.root.handlers:
    configure_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format_type=os.getenv("LOG_FORMAT", "json"),
        enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true",
    )
