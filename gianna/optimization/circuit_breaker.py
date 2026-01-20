"""
Circuit Breaker Pattern Implementation for API Resilience

This module implements the Circuit Breaker pattern to prevent cascading failures
when external services (LLM providers, TTS/STT services, etc.) are unavailable.

The circuit breaker has three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is down, requests fail fast without calling the service
- HALF_OPEN: Testing if service has recovered

Usage:
    >>> breaker = CircuitBreaker(name="openai", failure_threshold=5)
    >>> @breaker
    ... def call_openai_api():
    ...     return openai.chat.completions.create(...)
"""

import asyncio
import functools
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from loguru import logger

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, not calling service
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Failure thresholds
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 3  # Successes needed to close from half-open

    # Timing
    timeout_seconds: float = 30.0  # Time to wait before trying half-open
    call_timeout_seconds: float = 10.0  # Timeout for individual calls

    # Recovery
    half_open_max_calls: int = 3  # Max calls to allow in half-open state

    # Exceptions to consider as failures
    failure_exceptions: tuple = (Exception,)  # Which exceptions trigger failure
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures

    # Monitoring
    enable_metrics: bool = True
    metrics_window_seconds: float = 300.0  # 5 minutes rolling window


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected while open
    timeout_calls: int = 0

    # State transitions
    times_opened: int = 0
    times_closed: int = 0

    # Timing
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change_time: Optional[float] = None

    # Rolling window
    recent_failures: List[float] = field(default_factory=list)
    recent_successes: List[float] = field(default_factory=list)

    def record_success(self, window_seconds: float = 300.0) -> None:
        """Record a successful call."""
        now = time.time()
        self.total_calls += 1
        self.successful_calls += 1
        self.last_success_time = now
        self.recent_successes.append(now)
        self._cleanup_window(window_seconds)

    def record_failure(self, window_seconds: float = 300.0) -> None:
        """Record a failed call."""
        now = time.time()
        self.total_calls += 1
        self.failed_calls += 1
        self.last_failure_time = now
        self.recent_failures.append(now)
        self._cleanup_window(window_seconds)

    def record_rejection(self) -> None:
        """Record a rejected call (while circuit is open)."""
        self.total_calls += 1
        self.rejected_calls += 1

    def record_timeout(self) -> None:
        """Record a timeout."""
        self.timeout_calls += 1

    def _cleanup_window(self, window_seconds: float) -> None:
        """Remove old entries from rolling windows."""
        cutoff = time.time() - window_seconds
        self.recent_failures = [t for t in self.recent_failures if t > cutoff]
        self.recent_successes = [t for t in self.recent_successes if t > cutoff]

    def get_failure_rate(self) -> float:
        """Calculate failure rate in the recent window."""
        total_recent = len(self.recent_failures) + len(self.recent_successes)
        if total_recent == 0:
            return 0.0
        return len(self.recent_failures) / total_recent

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "timeout_calls": self.timeout_calls,
            "times_opened": self.times_opened,
            "times_closed": self.times_closed,
            "failure_rate": self.get_failure_rate(),
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
        }


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and rejecting calls."""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """
    Circuit Breaker implementation for protecting external API calls.

    The circuit breaker monitors failures and prevents cascading failures
    by failing fast when a service is known to be down.

    Example:
        >>> breaker = CircuitBreaker(name="openai")
        >>>
        >>> @breaker
        ... def call_api():
        ...     return requests.get("https://api.openai.com/v1/models")
        >>>
        >>> # Or use as context manager
        >>> with breaker:
        ...     result = call_api_directly()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable[..., T]] = None,
    ):
        """
        Initialize the circuit breaker.

        Args:
            name: Identifier for this circuit breaker (e.g., "openai", "anthropic")
            config: Configuration options
            fallback: Optional fallback function to call when circuit is open
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        # Thread safety
        self._lock = threading.RLock()

        # Metrics
        self.metrics = CircuitBreakerMetrics()

        logger.debug(f"Circuit breaker '{name}' initialized in CLOSED state")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    def _check_state_transition(self) -> None:
        """Check if state should transition based on timeout."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    self._transition_to_half_open()

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self._state = CircuitState.OPEN
        self._last_failure_time = time.time()
        self.metrics.times_opened += 1
        self.metrics.last_state_change_time = time.time()
        logger.warning(
            f"Circuit breaker '{self.name}' OPENED after {self._failure_count} failures"
        )

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._success_count = 0
        self.metrics.last_state_change_time = time.time()
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self.metrics.times_closed += 1
        self.metrics.last_state_change_time = time.time()
        logger.info(f"Circuit breaker '{self.name}' CLOSED (service recovered)")

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self.config.enable_metrics:
                self.metrics.record_success(self.config.metrics_window_seconds)

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            # Check if this exception should be excluded
            if isinstance(exception, self.config.excluded_exceptions):
                return

            if self.config.enable_metrics:
                self.metrics.record_failure(self.config.metrics_window_seconds)

            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                return False
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            return False

    def _get_retry_after(self) -> float:
        """Get time until circuit might close."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.time() - self._last_failure_time
        return max(0.0, self.config.timeout_seconds - elapsed)

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function

        Raises:
            CircuitOpenError: If circuit is open and no fallback is configured
        """
        if not self._can_execute():
            self.metrics.record_rejection()
            retry_after = self._get_retry_after()

            if self.fallback is not None:
                logger.debug(
                    f"Circuit '{self.name}' open, using fallback"
                )
                return self.fallback(*args, **kwargs)

            raise CircuitOpenError(self.name, retry_after)

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.failure_exceptions as e:
            self._record_failure(e)
            raise

    async def call_async(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """
        Execute an async function through the circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function

        Raises:
            CircuitOpenError: If circuit is open and no fallback is configured
        """
        if not self._can_execute():
            self.metrics.record_rejection()
            retry_after = self._get_retry_after()

            if self.fallback is not None:
                logger.debug(f"Circuit '{self.name}' open, using fallback")
                result = self.fallback(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result

            raise CircuitOpenError(self.name, retry_after)

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.failure_exceptions as e:
            self._record_failure(e)
            raise

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Use circuit breaker as a decorator.

        Example:
            >>> @circuit_breaker
            ... def call_api():
            ...     return requests.get(url)
        """
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await self.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return self.call(func, *args, **kwargs)
            return sync_wrapper

    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry."""
        if not self._can_execute():
            raise CircuitOpenError(self.name, self._get_retry_after())
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit."""
        if exc_val is None:
            self._record_success()
        elif isinstance(exc_val, self.config.failure_exceptions):
            self._record_failure(exc_val)
        return False  # Don't suppress exceptions

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")

    def force_open(self) -> None:
        """Force the circuit to open state (for maintenance/testing)."""
        with self._lock:
            self._transition_to_open()
            logger.warning(f"Circuit breaker '{self.name}' forced OPEN")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "retry_after": self._get_retry_after() if self.is_open else 0,
                "metrics": self.metrics.to_dict() if self.config.enable_metrics else {},
            }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized access to circuit breakers for different services.

    Example:
        >>> registry = CircuitBreakerRegistry()
        >>> openai_breaker = registry.get_or_create("openai")
        >>> anthropic_breaker = registry.get_or_create("anthropic")
    """

    _instance: Optional["CircuitBreakerRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CircuitBreakerRegistry":
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._breakers: Dict[str, CircuitBreaker] = {}
                    cls._instance._configs: Dict[str, CircuitBreakerConfig] = {}
        return cls._instance

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable] = None,
    ) -> CircuitBreaker:
        """
        Get an existing circuit breaker or create a new one.

        Args:
            name: Name/identifier for the circuit breaker
            config: Optional configuration (used only for new breakers)
            fallback: Optional fallback function

        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                config=config or self._configs.get(name),
                fallback=fallback,
            )
        return self._breakers[name]

    def register_config(self, name: str, config: CircuitBreakerConfig) -> None:
        """Register a default configuration for a named circuit breaker."""
        self._configs[name] = config

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name, or None if not found."""
        return self._breakers.get(name)

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

    def remove(self, name: str) -> None:
        """Remove a circuit breaker from the registry."""
        self._breakers.pop(name, None)


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    fallback: Optional[Callable] = None,
) -> CircuitBreaker:
    """
    Convenience function to get or create a circuit breaker.

    Args:
        name: Name for the circuit breaker (e.g., "openai", "anthropic")
        config: Optional configuration
        fallback: Optional fallback function

    Returns:
        CircuitBreaker instance

    Example:
        >>> breaker = get_circuit_breaker("openai")
        >>> @breaker
        ... def call_openai():
        ...     pass
    """
    return circuit_breaker_registry.get_or_create(name, config, fallback)


# Pre-configured circuit breakers for common LLM providers
def create_llm_circuit_breaker(
    provider: str,
    failure_threshold: int = 5,
    timeout_seconds: float = 60.0,
    fallback: Optional[Callable] = None,
) -> CircuitBreaker:
    """
    Create a circuit breaker configured for LLM provider APIs.

    Args:
        provider: LLM provider name (openai, anthropic, google, etc.)
        failure_threshold: Number of failures before opening circuit
        timeout_seconds: Time to wait before trying to recover
        fallback: Optional fallback function

    Returns:
        Configured CircuitBreaker
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds,
        success_threshold=2,  # Recover quickly once working
        call_timeout_seconds=30.0,  # LLM calls can be slow
        failure_exceptions=(
            Exception,  # Catch all for simplicity
        ),
        excluded_exceptions=(
            KeyboardInterrupt,
            SystemExit,
        ),
    )
    return get_circuit_breaker(f"llm_{provider}", config, fallback)
