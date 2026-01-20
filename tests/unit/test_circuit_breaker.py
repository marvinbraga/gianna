"""
Tests for Circuit Breaker implementation.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from gianna.optimization.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    create_llm_circuit_breaker,
    get_circuit_breaker,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_seconds == 30.0
        assert config.enable_metrics is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=60.0,
            enable_metrics=False,
        )
        assert config.failure_threshold == 3
        assert config.timeout_seconds == 60.0
        assert config.enable_metrics is False


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Test that circuit starts in CLOSED state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_successful_call(self):
        """Test successful call through circuit breaker."""
        breaker = CircuitBreaker(name="test")

        def success_func():
            return "success"

        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.is_closed

    def test_circuit_opens_after_failures(self):
        """Test that circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func():
            raise ValueError("Test error")

        # Cause failures
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)

        assert breaker.is_open

    def test_circuit_rejects_when_open(self):
        """Test that circuit rejects calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # Next call should be rejected
        def success_func():
            return "success"

        with pytest.raises(CircuitOpenError):
            breaker.call(success_func)

    def test_fallback_when_open(self):
        """Test fallback function when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)

        def fallback():
            return "fallback_value"

        breaker = CircuitBreaker(name="test", config=config, fallback=fallback)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # Fallback should be used
        result = breaker.call(failing_func)
        assert result == "fallback_value"

    def test_circuit_transitions_to_half_open(self):
        """Test transition to HALF_OPEN after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,  # Short timeout for testing
        )
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.is_open

        # Wait for timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert breaker.is_half_open

    def test_circuit_closes_after_success_in_half_open(self):
        """Test that circuit closes after successful calls in HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,
            success_threshold=2,
        )
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func():
            raise ValueError("Test error")

        def success_func():
            return "success"

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        # Wait for timeout
        time.sleep(0.15)

        # Successful calls should close circuit
        breaker.call(success_func)
        breaker.call(success_func)

        assert breaker.is_closed

    def test_decorator_usage(self):
        """Test using circuit breaker as decorator."""
        breaker = CircuitBreaker(name="test")

        @breaker
        def decorated_func():
            return "decorated"

        result = decorated_func()
        assert result == "decorated"

    def test_context_manager_usage(self):
        """Test using circuit breaker as context manager."""
        breaker = CircuitBreaker(name="test")

        with breaker:
            pass  # Success

        assert breaker.is_closed

    def test_metrics_tracking(self):
        """Test that metrics are tracked correctly."""
        config = CircuitBreakerConfig(enable_metrics=True)
        breaker = CircuitBreaker(name="test", config=config)

        def success_func():
            return "success"

        # Make some successful calls
        for _ in range(5):
            breaker.call(success_func)

        status = breaker.get_status()
        assert status["metrics"]["successful_calls"] == 5
        assert status["metrics"]["total_calls"] == 5

    def test_reset(self):
        """Test circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(name="test", config=config)

        def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        with pytest.raises(ValueError):
            breaker.call(failing_func)

        assert breaker.is_open

        # Reset
        breaker.reset()

        assert breaker.is_closed

    def test_force_open(self):
        """Test forcing circuit open."""
        breaker = CircuitBreaker(name="test")
        assert breaker.is_closed

        breaker.force_open()

        assert breaker.is_open


class TestCircuitBreakerAsync:
    """Tests for async circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_async_call(self):
        """Test async call through circuit breaker."""
        breaker = CircuitBreaker(name="test")

        async def async_func():
            return "async_result"

        result = await breaker.call_async(async_func)
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test async decorator usage."""
        breaker = CircuitBreaker(name="test")

        @breaker
        async def async_decorated():
            return "async_decorated"

        result = await async_decorated()
        assert result == "async_decorated"


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        registry1 = CircuitBreakerRegistry()
        registry2 = CircuitBreakerRegistry()
        assert registry1 is registry2

    def test_get_or_create(self):
        """Test getting or creating circuit breakers."""
        registry = CircuitBreakerRegistry()
        registry._breakers.clear()  # Reset for test

        breaker1 = registry.get_or_create("test1")
        breaker2 = registry.get_or_create("test1")

        assert breaker1 is breaker2

    def test_get_all_status(self):
        """Test getting status of all breakers."""
        registry = CircuitBreakerRegistry()
        registry._breakers.clear()  # Reset for test

        registry.get_or_create("test1")
        registry.get_or_create("test2")

        status = registry.get_all_status()
        assert "test1" in status
        assert "test2" in status

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()
        registry._breakers.clear()

        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = registry.get_or_create("test", config=config)

        # Open it
        breaker.force_open()
        assert breaker.is_open

        registry.reset_all()

        assert breaker.is_closed


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_circuit_breaker(self):
        """Test get_circuit_breaker function."""
        breaker = get_circuit_breaker("convenience_test")
        assert isinstance(breaker, CircuitBreaker)

    def test_create_llm_circuit_breaker(self):
        """Test create_llm_circuit_breaker function."""
        breaker = create_llm_circuit_breaker("openai")
        assert isinstance(breaker, CircuitBreaker)
        assert "llm_openai" in breaker.name
