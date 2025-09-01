"""
Advanced caching system for Gianna production.
Provides multi-level caching with Redis, memory, and disk storage.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache storage levels."""

    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class CacheStrategy(Enum):
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[int]
    size: int

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if not self.ttl:
            return False
        return time.time() - self.created_at > self.ttl

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MemoryCache:
    """In-memory cache with LRU eviction."""

    def __init__(self, max_size: int = 1000, max_memory: int = 100 * 1024 * 1024):
        """Initialize memory cache."""
        self.max_size = max_size
        self.max_memory = max_memory
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._current_memory = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                self._remove(key)
                return None

            # Update access info
            entry.accessed_at = time.time()
            entry.access_count += 1

            # Move to end for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            # Calculate size
            try:
                size = len(pickle.dumps(value))
            except:
                size = 1024  # Default size if can't serialize

            # Check if value is too large
            if size > self.max_memory:
                logger.warning(f"Cache entry too large: {size} bytes")
                return False

            # Evict if necessary
            while (
                len(self._cache) >= self.max_size
                or self._current_memory + size > self.max_memory
            ):
                if not self._evict_one():
                    break

            # Remove existing entry if present
            if key in self._cache:
                self._remove(key)

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1,
                ttl=ttl,
                size=size,
            )

            self._cache[key] = entry
            self._access_order.append(key)
            self._current_memory += size

            return True

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_memory = 0

    def _remove(self, key: str):
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory -= entry.size
            if key in self._access_order:
                self._access_order.remove(key)

    def _evict_one(self) -> bool:
        """Evict one entry using LRU."""
        if not self._access_order:
            return False

        # Remove least recently used
        key_to_remove = self._access_order[0]
        self._remove(key_to_remove)
        return True

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "memory_usage": self._current_memory,
                "memory_limit": self.max_memory,
                "size_limit": self.max_size,
                "memory_utilization": self._current_memory / self.max_memory,
            }


class DiskCache:
    """Disk-based cache for persistent storage."""

    def __init__(self, cache_dir: str = "cache", max_size_gb: float = 1.0):
        """Initialize disk cache."""
        self.cache_dir = cache_dir
        self.max_size = max_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self._lock = threading.RLock()

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key."""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")

    def _get_metadata_path(self, key: str) -> str:
        """Get metadata file path for cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.meta")

    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)

            if not os.path.exists(file_path) or not os.path.exists(meta_path):
                return None

            try:
                # Read metadata
                with open(meta_path, "r") as f:
                    metadata = json.load(f)

                # Check expiration
                if metadata.get("ttl"):
                    if time.time() - metadata["created_at"] > metadata["ttl"]:
                        self._remove_files(key)
                        return None

                # Update access info
                metadata["accessed_at"] = time.time()
                metadata["access_count"] = metadata.get("access_count", 0) + 1

                with open(meta_path, "w") as f:
                    json.dump(metadata, f)

                # Read value
                with open(file_path, "rb") as f:
                    return pickle.load(f)

            except Exception as e:
                logger.error(f"Error reading from disk cache: {e}")
                self._remove_files(key)
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache."""
        with self._lock:
            try:
                # Serialize value
                serialized = pickle.dumps(value)

                # Check size limits
                if len(serialized) > self.max_size:
                    logger.warning(
                        f"Cache entry too large for disk: {len(serialized)} bytes"
                    )
                    return False

                # Ensure we have space
                self._ensure_space(len(serialized))

                # Write files
                file_path = self._get_file_path(key)
                meta_path = self._get_metadata_path(key)

                # Write value
                with open(file_path, "wb") as f:
                    f.write(serialized)

                # Write metadata
                metadata = {
                    "key": key,
                    "created_at": time.time(),
                    "accessed_at": time.time(),
                    "access_count": 1,
                    "ttl": ttl,
                    "size": len(serialized),
                }

                with open(meta_path, "w") as f:
                    json.dump(metadata, f)

                return True

            except Exception as e:
                logger.error(f"Error writing to disk cache: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        with self._lock:
            return self._remove_files(key)

    def clear(self):
        """Clear all disk cache entries."""
        with self._lock:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    def _remove_files(self, key: str) -> bool:
        """Remove cache and metadata files."""
        file_path = self._get_file_path(key)
        meta_path = self._get_metadata_path(key)

        removed = False
        for path in [file_path, meta_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed = True
                except OSError:
                    pass

        return removed

    def _ensure_space(self, needed_size: int):
        """Ensure enough space by evicting old entries."""
        current_size = self._get_total_size()

        while current_size + needed_size > self.max_size:
            if not self._evict_oldest():
                break
            current_size = self._get_total_size()

    def _get_total_size(self) -> int:
        """Get total size of cache directory."""
        total_size = 0
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
        return total_size

    def _evict_oldest(self) -> bool:
        """Evict oldest cache entry."""
        oldest_time = float("inf")
        oldest_key = None

        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".meta"):
                meta_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(meta_path, "r") as f:
                        metadata = json.load(f)

                    if metadata["accessed_at"] < oldest_time:
                        oldest_time = metadata["accessed_at"]
                        oldest_key = metadata["key"]

                except Exception:
                    continue

        if oldest_key:
            return self._remove_files(oldest_key)
        return False

    def get_stats(self) -> Dict:
        """Get disk cache statistics."""
        with self._lock:
            total_size = self._get_total_size()
            file_count = len(
                [f for f in os.listdir(self.cache_dir) if f.endswith(".cache")]
            )

            return {
                "entries": file_count,
                "total_size": total_size,
                "size_limit": self.max_size,
                "utilization": total_size / self.max_size,
            }


class CacheManager:
    """Multi-level cache manager with Redis, memory, and disk storage."""

    def __init__(
        self,
        redis_client=None,
        memory_cache_size: int = 1000,
        memory_cache_mb: int = 100,
        disk_cache_gb: float = 1.0,
        cache_dir: str = "cache",
    ):
        """Initialize cache manager."""
        self.redis_client = redis_client

        # Initialize cache levels
        self.memory_cache = MemoryCache(
            memory_cache_size, memory_cache_mb * 1024 * 1024
        )
        self.disk_cache = DiskCache(cache_dir, disk_cache_gb)

        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.redis_ttl = 86400  # 24 hours

        # Statistics
        self._stats = {
            "hits": {"memory": 0, "redis": 0, "disk": 0},
            "misses": {"memory": 0, "redis": 0, "disk": 0},
            "sets": {"memory": 0, "redis": 0, "disk": 0},
        }
        self._stats_lock = threading.RLock()

        # Background cleanup
        self._cleanup_thread = threading.Thread(
            target=self._background_cleanup, daemon=True
        )
        self._cleanup_thread.start()

    def get(self, key: str, levels: List[CacheLevel] = None) -> Optional[Any]:
        """Get value from cache with multi-level lookup."""
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]

        value = None
        found_level = None

        # Try each cache level
        for level in levels:
            if level == CacheLevel.MEMORY:
                value = self.memory_cache.get(key)
                if value is not None:
                    found_level = "memory"
                    break
                self._increment_stat("misses", "memory")

            elif level == CacheLevel.REDIS and self.redis_client:
                try:
                    cached = self.redis_client.get(key)
                    if cached:
                        value = pickle.loads(cached)
                        found_level = "redis"
                        break
                except Exception as e:
                    logger.error(f"Redis cache error: {e}")
                self._increment_stat("misses", "redis")

            elif level == CacheLevel.DISK:
                value = self.disk_cache.get(key)
                if value is not None:
                    found_level = "disk"
                    break
                self._increment_stat("misses", "disk")

        # If found, populate higher-priority caches
        if value is not None and found_level:
            self._increment_stat("hits", found_level)

            # Populate higher-priority caches
            if found_level == "disk":
                self.memory_cache.set(key, value, self.default_ttl)
                if self.redis_client:
                    self._set_redis(key, value, self.redis_ttl)
            elif found_level == "redis":
                self.memory_cache.set(key, value, self.default_ttl)

        return value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        levels: List[CacheLevel] = None,
    ) -> bool:
        """Set value in cache at specified levels."""
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]

        ttl = ttl or self.default_ttl
        success = True

        for level in levels:
            if level == CacheLevel.MEMORY:
                if self.memory_cache.set(key, value, ttl):
                    self._increment_stat("sets", "memory")
                else:
                    success = False

            elif level == CacheLevel.REDIS and self.redis_client:
                if self._set_redis(key, value, ttl):
                    self._increment_stat("sets", "redis")
                else:
                    success = False

            elif level == CacheLevel.DISK:
                if self.disk_cache.set(key, value, ttl):
                    self._increment_stat("sets", "disk")
                else:
                    success = False

        return success

    def delete(self, key: str, levels: List[CacheLevel] = None) -> bool:
        """Delete value from cache at specified levels."""
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]

        deleted = False

        for level in levels:
            if level == CacheLevel.MEMORY:
                deleted |= self.memory_cache.delete(key)

            elif level == CacheLevel.REDIS and self.redis_client:
                try:
                    deleted |= bool(self.redis_client.delete(key))
                except Exception as e:
                    logger.error(f"Redis delete error: {e}")

            elif level == CacheLevel.DISK:
                deleted |= self.disk_cache.delete(key)

        return deleted

    def clear(self, levels: List[CacheLevel] = None):
        """Clear cache at specified levels."""
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]

        for level in levels:
            if level == CacheLevel.MEMORY:
                self.memory_cache.clear()

            elif level == CacheLevel.REDIS and self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.error(f"Redis clear error: {e}")

            elif level == CacheLevel.DISK:
                self.disk_cache.clear()

    def _set_redis(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in Redis cache."""
        try:
            serialized = pickle.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def _increment_stat(self, stat_type: str, level: str):
        """Increment cache statistics."""
        with self._stats_lock:
            self._stats[stat_type][level] += 1

    def get_statistics(self) -> Dict:
        """Get comprehensive cache statistics."""
        with self._stats_lock:
            stats = {
                "operations": dict(self._stats),
                "memory": self.memory_cache.get_stats(),
                "disk": self.disk_cache.get_stats(),
            }

            if self.redis_client:
                try:
                    redis_info = self.redis_client.info("memory")
                    stats["redis"] = {
                        "used_memory": redis_info.get("used_memory", 0),
                        "used_memory_human": redis_info.get("used_memory_human", "N/A"),
                        "connected_clients": self.redis_client.info().get(
                            "connected_clients", 0
                        ),
                    }
                except Exception:
                    stats["redis"] = {"error": "Unable to get Redis stats"}
            else:
                stats["redis"] = {"enabled": False}

            # Calculate hit rates
            for level in ["memory", "redis", "disk"]:
                hits = stats["operations"]["hits"][level]
                misses = stats["operations"]["misses"][level]
                total = hits + misses
                if total > 0:
                    stats[level]["hit_rate"] = hits / total
                else:
                    stats[level]["hit_rate"] = 0.0

            return stats

    def _background_cleanup(self):
        """Background cleanup task."""
        while True:
            try:
                # Clean up every 5 minutes
                time.sleep(300)

                # Cleanup memory cache (remove expired entries)
                current_time = time.time()
                with self.memory_cache._lock:
                    expired_keys = []
                    for key, entry in self.memory_cache._cache.items():
                        if entry.is_expired():
                            expired_keys.append(key)

                    for key in expired_keys:
                        self.memory_cache._remove(key)

                logger.info(
                    f"Cache cleanup: removed {len(expired_keys)} expired entries"
                )

            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")


# Decorators for caching
def cached(
    ttl: int = 3600,
    key_prefix: str = "",
    levels: List[CacheLevel] = None,
    cache_manager: CacheManager = None,
):
    """Decorator for caching function results."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cache_manager
            if cache_manager is None:
                cache_manager = get_cache_manager()

            # Generate cache key
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            result = cache_manager.get(cache_key, levels)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl, levels)

            return result

        return wrapper

    return decorator


def cache_async(
    ttl: int = 3600,
    key_prefix: str = "",
    levels: List[CacheLevel] = None,
    cache_manager: CacheManager = None,
):
    """Decorator for caching async function results."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal cache_manager
            if cache_manager is None:
                cache_manager = get_cache_manager()

            # Generate cache key
            key_parts = [key_prefix or func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            result = cache_manager.get(cache_key, levels)
            if result is not None:
                return result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl, levels)

            return result

        return wrapper

    return decorator


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager(redis_client=None) -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(redis_client=redis_client)
    return _global_cache_manager


def reset_cache_manager():
    """Reset global cache manager (mainly for testing)."""
    global _global_cache_manager
    _global_cache_manager = None
