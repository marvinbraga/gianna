"""
Rate limiting implementation for Gianna production security.
Provides protection against abuse and ensures fair resource usage.
"""

import asyncio
import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Types of rate limiting strategies."""

    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimit:
    """Rate limit configuration."""

    requests: int  # Number of requests
    window: int  # Time window in seconds
    burst: Optional[int] = None  # Burst allowance
    rate_type: RateLimitType = RateLimitType.SLIDING_WINDOW


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[int] = None
    limit_type: str = ""
    limit_name: str = ""


class SlidingWindowLimiter:
    """Sliding window rate limiter implementation."""

    def __init__(self):
        self.windows: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, key: str, limit: RateLimit) -> RateLimitResult:
        """Check if request is allowed under sliding window."""
        current_time = time.time()
        window_start = current_time - limit.window

        with self._lock:
            # Clean old entries
            while self.windows[key] and self.windows[key][0] < window_start:
                self.windows[key].popleft()

            # Check if limit exceeded
            current_count = len(self.windows[key])

            if current_count >= limit.requests:
                # Calculate reset time (when oldest request will expire)
                reset_time = (
                    self.windows[key][0] + limit.window
                    if self.windows[key]
                    else current_time
                )
                retry_after = int(reset_time - current_time)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=retry_after,
                    limit_type="sliding_window",
                    limit_name=f"{limit.requests}/{limit.window}s",
                )

            # Add current request
            self.windows[key].append(current_time)

            return RateLimitResult(
                allowed=True,
                remaining=limit.requests - current_count - 1,
                reset_time=current_time + limit.window,
                limit_type="sliding_window",
                limit_name=f"{limit.requests}/{limit.window}s",
            )


class TokenBucketLimiter:
    """Token bucket rate limiter implementation."""

    def __init__(self):
        self.buckets: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str, limit: RateLimit) -> RateLimitResult:
        """Check if request is allowed under token bucket."""
        current_time = time.time()

        with self._lock:
            if key not in self.buckets:
                self.buckets[key] = {
                    "tokens": limit.requests,
                    "last_refill": current_time,
                    "capacity": limit.requests,
                    "refill_rate": limit.requests / limit.window,  # tokens per second
                }

            bucket = self.buckets[key]

            # Calculate tokens to add based on elapsed time
            time_passed = current_time - bucket["last_refill"]
            tokens_to_add = time_passed * bucket["refill_rate"]

            # Add tokens but don't exceed capacity
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = current_time

            # Check if we have tokens available
            if bucket["tokens"] < 1:
                # Calculate when next token will be available
                time_to_token = (1 - bucket["tokens"]) / bucket["refill_rate"]
                retry_after = int(time_to_token) + 1

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=current_time + time_to_token,
                    retry_after=retry_after,
                    limit_type="token_bucket",
                    limit_name=f"{limit.requests}/{limit.window}s",
                )

            # Consume token
            bucket["tokens"] -= 1

            return RateLimitResult(
                allowed=True,
                remaining=int(bucket["tokens"]),
                reset_time=current_time
                + (bucket["capacity"] - bucket["tokens"]) / bucket["refill_rate"],
                limit_type="token_bucket",
                limit_name=f"{limit.requests}/{limit.window}s",
            )


class FixedWindowLimiter:
    """Fixed window rate limiter implementation."""

    def __init__(self):
        self.windows: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str, limit: RateLimit) -> RateLimitResult:
        """Check if request is allowed under fixed window."""
        current_time = time.time()
        window_start = int(current_time // limit.window) * limit.window

        with self._lock:
            if (
                key not in self.windows
                or self.windows[key]["window_start"] != window_start
            ):
                self.windows[key] = {"count": 0, "window_start": window_start}

            current_count = self.windows[key]["count"]

            if current_count >= limit.requests:
                reset_time = window_start + limit.window
                retry_after = int(reset_time - current_time)

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=retry_after,
                    limit_type="fixed_window",
                    limit_name=f"{limit.requests}/{limit.window}s",
                )

            self.windows[key]["count"] += 1

            return RateLimitResult(
                allowed=True,
                remaining=limit.requests - current_count - 1,
                reset_time=window_start + limit.window,
                limit_type="fixed_window",
                limit_name=f"{limit.requests}/{limit.window}s",
            )


class RateLimiter:
    """Comprehensive rate limiter with multiple strategies and Redis support."""

    def __init__(self, redis_client=None):
        """Initialize rate limiter."""
        self.redis_client = redis_client
        self.limits: Dict[str, RateLimit] = {}

        # Initialize local limiters
        self.sliding_window = SlidingWindowLimiter()
        self.token_bucket = TokenBucketLimiter()
        self.fixed_window = FixedWindowLimiter()

        # Setup default limits
        self._setup_default_limits()

        # Cleanup thread for local storage
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired, daemon=True
        )
        self._cleanup_thread.start()

    def _setup_default_limits(self):
        """Setup default rate limits."""
        self.add_limit("api", RateLimit(100, 60, burst=20))  # 100 req/min
        self.add_limit("auth", RateLimit(10, 60))  # 10 auth attempts/min
        self.add_limit("llm", RateLimit(50, 60))  # 50 LLM requests/min
        self.add_limit("audio", RateLimit(30, 60))  # 30 audio operations/min
        self.add_limit("upload", RateLimit(10, 300))  # 10 uploads/5min
        self.add_limit("health", RateLimit(1000, 60))  # 1000 health checks/min

    def add_limit(self, name: str, limit: RateLimit):
        """Add a rate limit configuration."""
        self.limits[name] = limit
        logger.info(f"Added rate limit '{name}': {limit.requests}/{limit.window}s")

    def remove_limit(self, name: str):
        """Remove a rate limit configuration."""
        if name in self.limits:
            del self.limits[name]
            logger.info(f"Removed rate limit '{name}'")

    def get_limit(self, name: str) -> Optional[RateLimit]:
        """Get rate limit configuration."""
        return self.limits.get(name)

    def check_limit(
        self, limit_name: str, identifier: str, cost: int = 1
    ) -> RateLimitResult:
        """Check if request is within rate limit."""
        if limit_name not in self.limits:
            logger.warning(f"Rate limit '{limit_name}' not found")
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                reset_time=time.time() + 3600,
                limit_type="none",
                limit_name=limit_name,
            )

        limit = self.limits[limit_name]
        key = f"{limit_name}:{identifier}"

        # Use Redis if available, otherwise local storage
        if self.redis_client:
            return self._check_redis_limit(key, limit, cost)
        else:
            return self._check_local_limit(key, limit, cost)

    def _check_local_limit(
        self, key: str, limit: RateLimit, cost: int
    ) -> RateLimitResult:
        """Check limit using local storage."""
        if limit.rate_type == RateLimitType.SLIDING_WINDOW:
            return self.sliding_window.is_allowed(key, limit)
        elif limit.rate_type == RateLimitType.TOKEN_BUCKET:
            return self.token_bucket.is_allowed(key, limit)
        elif limit.rate_type == RateLimitType.FIXED_WINDOW:
            return self.fixed_window.is_allowed(key, limit)
        else:
            # Default to sliding window
            return self.sliding_window.is_allowed(key, limit)

    def _check_redis_limit(
        self, key: str, limit: RateLimit, cost: int
    ) -> RateLimitResult:
        """Check limit using Redis storage."""
        try:
            if limit.rate_type == RateLimitType.SLIDING_WINDOW:
                return self._redis_sliding_window(key, limit, cost)
            elif limit.rate_type == RateLimitType.FIXED_WINDOW:
                return self._redis_fixed_window(key, limit, cost)
            else:
                # Fallback to local for unsupported types
                return self._check_local_limit(key, limit, cost)

        except Exception as e:
            logger.error(f"Redis rate limiting failed: {e}")
            # Fallback to local storage
            return self._check_local_limit(key, limit, cost)

    def _redis_sliding_window(
        self, key: str, limit: RateLimit, cost: int
    ) -> RateLimitResult:
        """Sliding window implementation using Redis."""
        current_time = time.time()
        window_start = current_time - limit.window

        pipe = self.redis_client.pipeline()

        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current entries
        pipe.zcard(key)

        # Execute pipeline
        results = pipe.execute()
        current_count = results[1]

        if current_count >= limit.requests:
            # Get reset time from oldest entry
            oldest_entries = self.redis_client.zrange(key, 0, 0, withscores=True)
            reset_time = (
                oldest_entries[0][1] + limit.window if oldest_entries else current_time
            )
            retry_after = int(reset_time - current_time)

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after,
                limit_type="sliding_window_redis",
                limit_name=f"{limit.requests}/{limit.window}s",
            )

        # Add current request
        unique_id = f"{current_time}:{hash(key) % 10000}"
        self.redis_client.zadd(key, {unique_id: current_time})
        self.redis_client.expire(key, limit.window)

        return RateLimitResult(
            allowed=True,
            remaining=limit.requests - current_count - cost,
            reset_time=current_time + limit.window,
            limit_type="sliding_window_redis",
            limit_name=f"{limit.requests}/{limit.window}s",
        )

    def _redis_fixed_window(
        self, key: str, limit: RateLimit, cost: int
    ) -> RateLimitResult:
        """Fixed window implementation using Redis."""
        current_time = time.time()
        window_start = int(current_time // limit.window) * limit.window
        window_key = f"{key}:{window_start}"

        # Get current count
        current_count = self.redis_client.get(window_key) or 0
        current_count = int(current_count)

        if current_count >= limit.requests:
            reset_time = window_start + limit.window
            retry_after = int(reset_time - current_time)

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after,
                limit_type="fixed_window_redis",
                limit_name=f"{limit.requests}/{limit.window}s",
            )

        # Increment counter
        pipe = self.redis_client.pipeline()
        pipe.incr(window_key, cost)
        pipe.expire(window_key, limit.window)
        pipe.execute()

        return RateLimitResult(
            allowed=True,
            remaining=limit.requests - current_count - cost,
            reset_time=window_start + limit.window,
            limit_type="fixed_window_redis",
            limit_name=f"{limit.requests}/{limit.window}s",
        )

    def get_limit_status(self, limit_name: str, identifier: str) -> Dict:
        """Get current status for a rate limit."""
        key = f"{limit_name}:{identifier}"
        limit = self.limits.get(limit_name)

        if not limit:
            return {"error": "Limit not found"}

        # Perform a check without consuming quota
        result = self.check_limit(limit_name, identifier, cost=0)

        return {
            "limit_name": limit_name,
            "identifier": identifier,
            "allowed": result.allowed,
            "remaining": result.remaining,
            "reset_time": result.reset_time,
            "limit_type": result.limit_type,
            "limit_config": f"{limit.requests}/{limit.window}s",
        }

    def reset_limit(self, limit_name: str, identifier: str):
        """Reset rate limit for specific identifier."""
        key = f"{limit_name}:{identifier}"

        if self.redis_client:
            self.redis_client.delete(key)
        else:
            # Reset local storage
            if (
                hasattr(self.sliding_window, "windows")
                and key in self.sliding_window.windows
            ):
                del self.sliding_window.windows[key]
            if (
                hasattr(self.token_bucket, "buckets")
                and key in self.token_bucket.buckets
            ):
                del self.token_bucket.buckets[key]
            if (
                hasattr(self.fixed_window, "windows")
                and key in self.fixed_window.windows
            ):
                del self.fixed_window.windows[key]

        logger.info(f"Reset rate limit for {key}")

    def get_statistics(self) -> Dict:
        """Get rate limiter statistics."""
        stats = {
            "limits_configured": len(self.limits),
            "redis_enabled": self.redis_client is not None,
            "limits": {},
        }

        for name, limit in self.limits.items():
            stats["limits"][name] = {
                "requests": limit.requests,
                "window": limit.window,
                "type": limit.rate_type.value,
                "burst": limit.burst,
            }

        if not self.redis_client:
            # Add local storage stats
            stats["local_storage"] = {
                "sliding_windows": len(self.sliding_window.windows),
                "token_buckets": len(self.token_bucket.buckets),
                "fixed_windows": len(self.fixed_window.windows),
            }

        return stats

    def _cleanup_expired(self):
        """Cleanup expired entries from local storage."""
        while True:
            try:
                current_time = time.time()

                # Cleanup sliding windows
                for key, window in list(self.sliding_window.windows.items()):
                    if window and current_time - window[-1] > 3600:  # 1 hour old
                        del self.sliding_window.windows[key]

                # Cleanup token buckets
                for key, bucket in list(self.token_bucket.buckets.items()):
                    if current_time - bucket["last_refill"] > 3600:  # 1 hour old
                        del self.token_bucket.buckets[key]

                # Cleanup fixed windows
                for key, window in list(self.fixed_window.windows.items()):
                    if current_time - window["window_start"] > 3600:  # 1 hour old
                        del self.fixed_window.windows[key]

                time.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                time.sleep(60)  # Wait longer on error


# Utility functions for common rate limiting patterns
def create_ip_limiter(requests: int = 100, window: int = 60) -> RateLimiter:
    """Create rate limiter for IP-based limiting."""
    limiter = RateLimiter()
    limiter.add_limit("ip", RateLimit(requests, window))
    return limiter


def create_user_limiter(requests: int = 50, window: int = 60) -> RateLimiter:
    """Create rate limiter for user-based limiting."""
    limiter = RateLimiter()
    limiter.add_limit("user", RateLimit(requests, window))
    return limiter


def create_api_limiter() -> RateLimiter:
    """Create comprehensive API rate limiter."""
    limiter = RateLimiter()

    # Different limits for different endpoints
    limiter.add_limit("api_general", RateLimit(1000, 3600))  # 1000/hour
    limiter.add_limit("api_auth", RateLimit(10, 300))  # 10/5min
    limiter.add_limit("api_upload", RateLimit(5, 300))  # 5/5min
    limiter.add_limit("api_llm", RateLimit(100, 3600))  # 100/hour

    return limiter


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(redis_client=None) -> RateLimiter:
    """Get global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter(redis_client)
    return _global_rate_limiter


def reset_rate_limiter():
    """Reset global rate limiter (mainly for testing)."""
    global _global_rate_limiter
    _global_rate_limiter = None
