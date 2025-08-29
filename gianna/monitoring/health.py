"""
Health checking system for Gianna production.
Provides comprehensive health checks for all system components.
"""

import asyncio
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil
import requests


class HealthStatus(Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: float
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


class HealthChecker:
    """Comprehensive health checker for Gianna components."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
        self.check_timeouts: Dict[str, float] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Register default health checks
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default system health checks."""
        self.register_check(
            "system_resources", self._check_system_resources, timeout=5.0
        )
        self.register_check("disk_space", self._check_disk_space, timeout=5.0)
        self.register_check("memory_usage", self._check_memory_usage, timeout=5.0)
        self.register_check("database", self._check_database, timeout=10.0)
        self.register_check("cache", self._check_cache, timeout=5.0)
        self.register_check("llm_providers", self._check_llm_providers, timeout=15.0)
        self.register_check("audio_system", self._check_audio_system, timeout=10.0)
        self.register_check(
            "file_permissions", self._check_file_permissions, timeout=5.0
        )

    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
        timeout: float = 10.0,
    ):
        """Register a health check function."""
        self.checks[name] = check_func
        self.check_timeouts[name] = timeout

    def run_check(self, name: str) -> HealthCheckResult:
        """Run a single health check."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
                duration_ms=0.0,
                timestamp=time.time(),
            )

        start_time = time.time()
        try:
            # Run check with timeout
            future = self.executor.submit(self.checks[name])
            timeout = self.check_timeouts.get(name, 10.0)
            result = future.result(timeout=timeout)

            # Ensure proper timing
            result.duration_ms = (time.time() - start_time) * 1000
            result.timestamp = time.time()

            return result

        except TimeoutError:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {timeout}s",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
            )

    def run_all_checks(self, parallel: bool = True) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}

        if parallel:
            # Run checks in parallel
            future_to_name = {
                self.executor.submit(self.run_check, name): name
                for name in self.checks.keys()
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check execution failed: {str(e)}",
                        duration_ms=0.0,
                        timestamp=time.time(),
                    )
        else:
            # Run checks sequentially
            for name in self.checks.keys():
                results[name] = self.run_check(name)

        return results

    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall health status from individual checks."""
        if not results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in results.values()]

        # If any check is unhealthy, overall is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY

        # If any check is degraded, overall is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        # If all checks are healthy, overall is healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        results = self.run_all_checks()
        overall_status = self.get_overall_status(results)

        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "checks": {name: result.to_dict() for name, result in results.items()},
            "summary": {
                "total_checks": len(results),
                "healthy": len(
                    [r for r in results.values() if r.status == HealthStatus.HEALTHY]
                ),
                "degraded": len(
                    [r for r in results.values() if r.status == HealthStatus.DEGRADED]
                ),
                "unhealthy": len(
                    [r for r in results.values() if r.status == HealthStatus.UNHEALTHY]
                ),
                "unknown": len(
                    [r for r in results.values() if r.status == HealthStatus.UNKNOWN]
                ),
            },
        }

    # Default health check implementations
    def _check_system_resources(self) -> HealthCheckResult:
        """Check overall system resource health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            load_avg = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "load_average": load_avg,
            }

            # Determine status based on resource usage
            if cpu_percent > 90 or memory_percent > 90 or load_avg > psutil.cpu_count():
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.UNHEALTHY,
                    message=f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            elif cpu_percent > 70 or memory_percent > 70:
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.DEGRADED,
                    message=f"Moderate resource usage: CPU {cpu_percent}%, Memory {memory_percent}%",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            else:
                return HealthCheckResult(
                    name="system_resources",
                    status=HealthStatus.HEALTHY,
                    message=f"Resources normal: CPU {cpu_percent}%, Memory {memory_percent}%",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {str(e)}",
                duration_ms=0.0,
                timestamp=0.0,
            )

    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        try:
            disk_usage = psutil.disk_usage("/")
            used_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024**3)

            details = {
                "used_percent": used_percent,
                "free_gb": free_gb,
                "total_gb": disk_usage.total / (1024**3),
            }

            if used_percent > 95 or free_gb < 1:
                return HealthCheckResult(
                    name="disk_space",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Critical disk space: {used_percent:.1f}% used, {free_gb:.1f}GB free",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            elif used_percent > 85 or free_gb < 5:
                return HealthCheckResult(
                    name="disk_space",
                    status=HealthStatus.DEGRADED,
                    message=f"Low disk space: {used_percent:.1f}% used, {free_gb:.1f}GB free",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            else:
                return HealthCheckResult(
                    name="disk_space",
                    status=HealthStatus.HEALTHY,
                    message=f"Sufficient disk space: {used_percent:.1f}% used, {free_gb:.1f}GB free",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check disk space: {str(e)}",
                duration_ms=0.0,
                timestamp=0.0,
            )

    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage details."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            details = {
                "rss_mb": memory_info.rss / (1024**2),
                "vms_mb": memory_info.vms / (1024**2),
                "percent": memory_percent,
            }

            if memory_percent > 80:
                return HealthCheckResult(
                    name="memory_usage",
                    status=HealthStatus.UNHEALTHY,
                    message=f"High memory usage: {memory_percent:.1f}%",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            elif memory_percent > 60:
                return HealthCheckResult(
                    name="memory_usage",
                    status=HealthStatus.DEGRADED,
                    message=f"Moderate memory usage: {memory_percent:.1f}%",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            else:
                return HealthCheckResult(
                    name="memory_usage",
                    status=HealthStatus.HEALTHY,
                    message=f"Normal memory usage: {memory_percent:.1f}%",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check memory usage: {str(e)}",
                duration_ms=0.0,
                timestamp=0.0,
            )

    def _check_database(self) -> HealthCheckResult:
        """Check database connectivity and health."""
        try:
            # Check SQLite databases
            db_paths = ["gianna_state.db", "gianna_optimization.db"]
            healthy_dbs = 0
            total_dbs = len(db_paths)

            details = {"databases": {}}

            for db_path in db_paths:
                try:
                    if os.path.exists(db_path):
                        conn = sqlite3.connect(db_path, timeout=5.0)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT name FROM sqlite_master WHERE type='table';"
                        )
                        tables = cursor.fetchall()
                        conn.close()

                        details["databases"][db_path] = {
                            "status": "healthy",
                            "tables": len(tables),
                        }
                        healthy_dbs += 1
                    else:
                        details["databases"][db_path] = {
                            "status": "missing",
                            "tables": 0,
                        }

                except Exception as e:
                    details["databases"][db_path] = {"status": "error", "error": str(e)}

            if healthy_dbs == total_dbs:
                return HealthCheckResult(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message=f"All {total_dbs} databases healthy",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            elif healthy_dbs > 0:
                return HealthCheckResult(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    message=f"{healthy_dbs}/{total_dbs} databases healthy",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            else:
                return HealthCheckResult(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message="No databases accessible",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                duration_ms=0.0,
                timestamp=0.0,
            )

    def _check_cache(self) -> HealthCheckResult:
        """Check cache system health (Redis)."""
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))

            try:
                import redis

                client = redis.Redis(host=redis_host, port=redis_port, socket_timeout=5)

                # Test basic operations
                test_key = "health_check_test"
                client.set(test_key, "test_value", ex=10)
                value = client.get(test_key)
                client.delete(test_key)

                info = client.info()

                return HealthCheckResult(
                    name="cache",
                    status=HealthStatus.HEALTHY,
                    message="Redis cache is healthy",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details={
                        "redis_version": info.get("redis_version"),
                        "connected_clients": info.get("connected_clients"),
                        "used_memory_human": info.get("used_memory_human"),
                    },
                )

            except ImportError:
                return HealthCheckResult(
                    name="cache",
                    status=HealthStatus.DEGRADED,
                    message="Redis not available (library not installed)",
                    duration_ms=0.0,
                    timestamp=0.0,
                )
            except Exception as e:
                return HealthCheckResult(
                    name="cache",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Redis connection failed: {str(e)}",
                    duration_ms=0.0,
                    timestamp=0.0,
                )

        except Exception as e:
            return HealthCheckResult(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache check failed: {str(e)}",
                duration_ms=0.0,
                timestamp=0.0,
            )

    def _check_llm_providers(self) -> HealthCheckResult:
        """Check LLM provider availability."""
        try:
            from gianna.assistants.models.registers import LLMRegister

            register = LLMRegister()
            available_providers = register.list_all_models()

            healthy_providers = 0
            total_providers = len(available_providers)
            provider_status = {}

            for provider in available_providers[:3]:  # Check first 3 providers only
                try:
                    # Simple availability check (don't make actual API calls in health check)
                    api_key_env = f"{provider.upper()}_API_KEY"
                    has_api_key = bool(os.getenv(api_key_env))

                    if has_api_key:
                        provider_status[provider] = "configured"
                        healthy_providers += 1
                    else:
                        provider_status[provider] = "missing_api_key"

                except Exception as e:
                    provider_status[provider] = f"error: {str(e)}"

            details = {
                "total_providers": total_providers,
                "checked_providers": len(provider_status),
                "provider_status": provider_status,
            }

            if healthy_providers >= 1:
                return HealthCheckResult(
                    name="llm_providers",
                    status=HealthStatus.HEALTHY,
                    message=f"{healthy_providers} LLM providers configured",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            else:
                return HealthCheckResult(
                    name="llm_providers",
                    status=HealthStatus.UNHEALTHY,
                    message="No LLM providers configured",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                name="llm_providers",
                status=HealthStatus.UNHEALTHY,
                message=f"LLM provider check failed: {str(e)}",
                duration_ms=0.0,
                timestamp=0.0,
            )

    def _check_audio_system(self) -> HealthCheckResult:
        """Check audio system availability."""
        try:
            details = {"components": {}}
            healthy_components = 0
            total_components = 3  # PyAudio, PyDub, FFmpeg

            # Check PyAudio
            try:
                import pyaudio

                details["components"]["pyaudio"] = "available"
                healthy_components += 1
            except ImportError:
                details["components"]["pyaudio"] = "missing"

            # Check PyDub
            try:
                import pydub

                details["components"]["pydub"] = "available"
                healthy_components += 1
            except ImportError:
                details["components"]["pydub"] = "missing"

            # Check FFmpeg
            try:
                import subprocess

                result = subprocess.run(
                    ["ffmpeg", "-version"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    details["components"]["ffmpeg"] = "available"
                    healthy_components += 1
                else:
                    details["components"]["ffmpeg"] = "error"
            except (FileNotFoundError, subprocess.TimeoutExpired):
                details["components"]["ffmpeg"] = "missing"
            except Exception as e:
                details["components"]["ffmpeg"] = f"error: {str(e)}"

            if healthy_components == total_components:
                return HealthCheckResult(
                    name="audio_system",
                    status=HealthStatus.HEALTHY,
                    message="All audio components available",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            elif healthy_components >= 2:
                return HealthCheckResult(
                    name="audio_system",
                    status=HealthStatus.DEGRADED,
                    message=f"{healthy_components}/{total_components} audio components available",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            else:
                return HealthCheckResult(
                    name="audio_system",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Only {healthy_components}/{total_components} audio components available",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                name="audio_system",
                status=HealthStatus.UNHEALTHY,
                message=f"Audio system check failed: {str(e)}",
                duration_ms=0.0,
                timestamp=0.0,
            )

    def _check_file_permissions(self) -> HealthCheckResult:
        """Check file system permissions."""
        try:
            critical_paths = [
                "/app/data",
                "/app/logs",
                "/tmp/gianna",
            ]

            details = {"paths": {}}
            issues = []

            for path in critical_paths:
                try:
                    if os.path.exists(path):
                        readable = os.access(path, os.R_OK)
                        writable = os.access(path, os.W_OK)
                        executable = os.access(path, os.X_OK)

                        details["paths"][path] = {
                            "exists": True,
                            "readable": readable,
                            "writable": writable,
                            "executable": executable,
                        }

                        if not (readable and writable):
                            issues.append(f"{path} permissions issue")
                    else:
                        details["paths"][path] = {
                            "exists": False,
                            "readable": False,
                            "writable": False,
                            "executable": False,
                        }
                        issues.append(f"{path} does not exist")

                except Exception as e:
                    details["paths"][path] = {"error": str(e)}
                    issues.append(f"{path} check failed: {str(e)}")

            if not issues:
                return HealthCheckResult(
                    name="file_permissions",
                    status=HealthStatus.HEALTHY,
                    message="All file permissions correct",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            elif len(issues) <= len(critical_paths) // 2:
                return HealthCheckResult(
                    name="file_permissions",
                    status=HealthStatus.DEGRADED,
                    message=f"Some file permission issues: {'; '.join(issues)}",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )
            else:
                return HealthCheckResult(
                    name="file_permissions",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Critical file permission issues: {'; '.join(issues)}",
                    duration_ms=0.0,
                    timestamp=0.0,
                    details=details,
                )

        except Exception as e:
            return HealthCheckResult(
                name="file_permissions",
                status=HealthStatus.UNHEALTHY,
                message=f"File permissions check failed: {str(e)}",
                duration_ms=0.0,
                timestamp=0.0,
            )


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


def reset_health_checker():
    """Reset global health checker (mainly for testing)."""
    global _global_health_checker
    _global_health_checker = None
