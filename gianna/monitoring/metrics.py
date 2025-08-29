"""
Metrics collection and monitoring for Gianna production.
Provides Prometheus-compatible metrics and custom application metrics.
"""

import os
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Generator, List, Optional

import psutil

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
    )
    from prometheus_client import Counter as PromCounter
    from prometheus_client import (
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsCollector:
    """Central metrics collector for Gianna."""

    def __init__(self, enable_prometheus: bool = True):
        """Initialize metrics collector."""
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self._metrics_lock = Lock()
        self._custom_metrics: Dict[str, Any] = defaultdict(int)
        self._counters = Counter()

        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        else:
            self.registry = None

        # Start background metrics collection
        self._collection_thread = threading.Thread(
            target=self._collect_system_metrics, daemon=True
        )
        self._collection_thread.start()

    def _setup_prometheus_metrics(self):
        """Set up Prometheus metrics."""
        # Request metrics
        self.request_count = PromCounter(
            "gianna_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "gianna_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            registry=self.registry,
        )

        # LLM metrics
        self.llm_requests = PromCounter(
            "gianna_llm_requests_total",
            "Total number of LLM requests",
            ["provider", "model", "status"],
            registry=self.registry,
        )

        self.llm_tokens = PromCounter(
            "gianna_llm_tokens_total",
            "Total number of LLM tokens",
            ["provider", "model", "type"],  # type: input/output
            registry=self.registry,
        )

        self.llm_duration = Histogram(
            "gianna_llm_duration_seconds",
            "LLM request duration in seconds",
            ["provider", "model"],
            registry=self.registry,
        )

        # Audio metrics
        self.audio_processing = PromCounter(
            "gianna_audio_processing_total",
            "Total audio processing operations",
            ["operation", "format", "status"],
            registry=self.registry,
        )

        self.audio_duration = Histogram(
            "gianna_audio_processing_duration_seconds",
            "Audio processing duration in seconds",
            ["operation", "format"],
            registry=self.registry,
        )

        # Memory and learning metrics
        self.memory_operations = PromCounter(
            "gianna_memory_operations_total",
            "Total memory operations",
            ["operation", "type", "status"],
            registry=self.registry,
        )

        self.conversation_length = Histogram(
            "gianna_conversation_length",
            "Length of conversations in turns",
            buckets=[1, 5, 10, 25, 50, 100, float("inf")],
            registry=self.registry,
        )

        # System metrics
        self.cpu_usage = Gauge(
            "gianna_cpu_usage_percent", "CPU usage percentage", registry=self.registry
        )

        self.memory_usage = Gauge(
            "gianna_memory_usage_bytes", "Memory usage in bytes", registry=self.registry
        )

        self.disk_usage = Gauge(
            "gianna_disk_usage_percent", "Disk usage percentage", registry=self.registry
        )

        # Cache metrics
        self.cache_operations = PromCounter(
            "gianna_cache_operations_total",
            "Total cache operations",
            ["operation", "status"],
            registry=self.registry,
        )

        self.cache_hit_rate = Gauge(
            "gianna_cache_hit_rate", "Cache hit rate (0-1)", registry=self.registry
        )

        # Error metrics
        self.errors = PromCounter(
            "gianna_errors_total",
            "Total number of errors",
            ["type", "component"],
            registry=self.registry,
        )

        # Application info
        self.app_info = Info(
            "gianna_app_info", "Application information", registry=self.registry
        )
        self.app_info.info(
            {
                "version": os.getenv("VERSION", "0.1.4"),
                "environment": os.getenv("ENVIRONMENT", "production"),
                "python_version": os.sys.version.split()[0],
            }
        )

    def _collect_system_metrics(self):
        """Collect system metrics in background."""
        while True:
            try:
                if self.enable_prometheus:
                    # Update system metrics
                    self.cpu_usage.set(psutil.cpu_percent(interval=1))

                    memory_info = psutil.virtual_memory()
                    self.memory_usage.set(memory_info.used)

                    disk_info = psutil.disk_usage("/")
                    self.disk_usage.set(disk_info.percent)

                time.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                # Log error but continue collecting
                print(f"Error collecting system metrics: {e}")
                time.sleep(60)  # Wait longer on error

    # Request tracking
    def track_request(self, method: str, endpoint: str, status: int, duration: float):
        """Track HTTP request metrics."""
        if self.enable_prometheus:
            self.request_count.labels(
                method=method, endpoint=endpoint, status=status
            ).inc()
            self.request_duration.labels(method=method, endpoint=endpoint).observe(
                duration
            )

        with self._metrics_lock:
            self._counters[f"requests.{method}.{endpoint}.{status}"] += 1
            self._custom_metrics[f"request_duration.{method}.{endpoint}"].append(
                duration
            )

    @contextmanager
    def track_request_context(self, method: str, endpoint: str) -> Generator:
        """Context manager for tracking request duration."""
        start_time = time.time()
        status = 500  # Default to error

        try:
            yield
            status = 200  # Success
        except Exception as e:
            # Determine status based on exception type
            status = getattr(e, "status_code", 500)
            raise
        finally:
            duration = time.time() - start_time
            self.track_request(method, endpoint, status, duration)

    # LLM tracking
    def track_llm_request(
        self,
        provider: str,
        model: str,
        status: str,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Track LLM request metrics."""
        if self.enable_prometheus:
            self.llm_requests.labels(
                provider=provider, model=model, status=status
            ).inc()
            self.llm_duration.labels(provider=provider, model=model).observe(duration)

            if input_tokens > 0:
                self.llm_tokens.labels(
                    provider=provider, model=model, type="input"
                ).inc(input_tokens)
            if output_tokens > 0:
                self.llm_tokens.labels(
                    provider=provider, model=model, type="output"
                ).inc(output_tokens)

        with self._metrics_lock:
            self._counters[f"llm.{provider}.{model}.{status}"] += 1
            self._custom_metrics[f"llm_duration.{provider}.{model}"] = duration
            if input_tokens > 0:
                self._custom_metrics[
                    f"llm_tokens.{provider}.{model}.input"
                ] += input_tokens
            if output_tokens > 0:
                self._custom_metrics[
                    f"llm_tokens.{provider}.{model}.output"
                ] += output_tokens

    @contextmanager
    def track_llm_context(self, provider: str, model: str) -> Generator:
        """Context manager for tracking LLM request duration."""
        start_time = time.time()
        status = "error"

        try:
            yield
            status = "success"
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            self.track_llm_request(provider, model, status, duration)

    # Audio tracking
    def track_audio_processing(
        self,
        operation: str,
        audio_format: str,
        status: str,
        duration: float,
    ):
        """Track audio processing metrics."""
        if self.enable_prometheus:
            self.audio_processing.labels(
                operation=operation, format=audio_format, status=status
            ).inc()
            self.audio_duration.labels(
                operation=operation, format=audio_format
            ).observe(duration)

        with self._metrics_lock:
            self._counters[f"audio.{operation}.{audio_format}.{status}"] += 1
            self._custom_metrics[f"audio_duration.{operation}.{audio_format}"] = (
                duration
            )

    # Memory and cache tracking
    def track_memory_operation(
        self,
        operation: str,
        memory_type: str,
        status: str,
    ):
        """Track memory system operations."""
        if self.enable_prometheus:
            self.memory_operations.labels(
                operation=operation, type=memory_type, status=status
            ).inc()

        with self._metrics_lock:
            self._counters[f"memory.{operation}.{memory_type}.{status}"] += 1

    def track_cache_operation(self, operation: str, hit: bool):
        """Track cache operations."""
        status = "hit" if hit else "miss"

        if self.enable_prometheus:
            self.cache_operations.labels(operation=operation, status=status).inc()

            # Update hit rate
            with self._metrics_lock:
                hits = self._counters.get("cache.hits", 0)
                total = self._counters.get("cache.total", 0)
                if hit:
                    hits += 1
                total += 1
                self._counters["cache.hits"] = hits
                self._counters["cache.total"] = total

                hit_rate = hits / total if total > 0 else 0
                self.cache_hit_rate.set(hit_rate)

        with self._metrics_lock:
            self._counters[f"cache.{operation}.{status}"] += 1

    def track_conversation_length(self, length: int):
        """Track conversation length."""
        if self.enable_prometheus:
            self.conversation_length.observe(length)

        with self._metrics_lock:
            self._custom_metrics["conversation_lengths"].append(length)

    # Error tracking
    def track_error(self, error_type: str, component: str):
        """Track application errors."""
        if self.enable_prometheus:
            self.errors.labels(type=error_type, component=component).inc()

        with self._metrics_lock:
            self._counters[f"errors.{component}.{error_type}"] += 1

    # Custom metrics
    def increment_counter(
        self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1
    ):
        """Increment a custom counter."""
        key = name
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in labels.items())
            key = f"{name}.{label_str}"

        with self._metrics_lock:
            self._custom_metrics[key] += value

    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """Set a custom gauge value."""
        key = name
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in labels.items())
            key = f"{name}.{label_str}"

        with self._metrics_lock:
            self._custom_metrics[key] = value

    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ):
        """Record a histogram value."""
        key = name
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in labels.items())
            key = f"{name}.{label_str}"

        with self._metrics_lock:
            if key not in self._custom_metrics:
                self._custom_metrics[key] = []
            self._custom_metrics[key].append(value)

    # Metrics export
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if self.enable_prometheus:
            return generate_latest(self.registry)
        return ""

    def get_custom_metrics(self) -> Dict[str, Any]:
        """Get custom metrics as dictionary."""
        with self._metrics_lock:
            return dict(self._custom_metrics)

    def get_counters(self) -> Dict[str, int]:
        """Get counters as dictionary."""
        with self._metrics_lock:
            return dict(self._counters)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {
            "timestamp": time.time(),
            "prometheus_enabled": self.enable_prometheus,
        }

        if self.enable_prometheus:
            summary["prometheus_metrics_count"] = len(list(self.registry.collect()))

        summary["custom_metrics"] = self.get_custom_metrics()
        summary["counters"] = self.get_counters()

        return summary


# Decorators for automatic metrics collection
def track_duration(
    metrics_collector: MetricsCollector,
    operation: str,
    component: str = "unknown",
):
    """Decorator to track function execution duration."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_histogram(
                    f"{component}.{operation}.duration", duration, {"status": "success"}
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_histogram(
                    f"{component}.{operation}.duration", duration, {"status": "error"}
                )
                metrics_collector.track_error(
                    error_type=type(e).__name__, component=component
                )
                raise

        return wrapper

    return decorator


def track_calls(
    metrics_collector: MetricsCollector,
    operation: str,
    component: str = "unknown",
):
    """Decorator to track function calls."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                metrics_collector.increment_counter(
                    f"{component}.{operation}.calls", {"status": "success"}
                )
                return result
            except Exception as e:
                metrics_collector.increment_counter(
                    f"{component}.{operation}.calls", {"status": "error"}
                )
                metrics_collector.track_error(
                    error_type=type(e).__name__, component=component
                )
                raise

        return wrapper

    return decorator


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector(
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"
        )
    return _global_metrics_collector


def reset_metrics_collector():
    """Reset global metrics collector (mainly for testing)."""
    global _global_metrics_collector
    _global_metrics_collector = None
