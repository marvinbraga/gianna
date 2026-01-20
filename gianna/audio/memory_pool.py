"""
Memory Pool for Audio Processing

This module implements a memory pool for audio buffers to reduce garbage collection
pressure during real-time audio processing. Pre-allocating and reusing buffers
significantly improves performance and reduces latency spikes.

Usage:
    >>> pool = AudioBufferPool(chunk_size=1024, pool_size=32)
    >>> buffer = pool.acquire()
    >>> # ... use buffer for audio processing ...
    >>> pool.release(buffer)

The pool automatically grows if needed and tracks usage statistics.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Union

import numpy as np
from loguru import logger


class PoolExhaustedError(Exception):
    """Raised when pool is exhausted and blocking is disabled."""

    pass


class BufferState(Enum):
    """State of a buffer in the pool."""

    AVAILABLE = "available"
    IN_USE = "in_use"
    CORRUPTED = "corrupted"


@dataclass
class PoolStatistics:
    """Statistics for memory pool monitoring."""

    # Allocation stats
    total_allocations: int = 0
    total_releases: int = 0
    pool_hits: int = 0  # Reused from pool
    pool_misses: int = 0  # Had to allocate new

    # Current state
    pool_size: int = 0
    buffers_in_use: int = 0
    buffers_available: int = 0

    # Growth stats
    times_grown: int = 0
    peak_usage: int = 0

    # Timing
    average_acquire_time_us: float = 0.0
    average_release_time_us: float = 0.0

    # Memory
    total_memory_bytes: int = 0
    memory_in_use_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_allocations": self.total_allocations,
            "total_releases": self.total_releases,
            "pool_hits": self.pool_hits,
            "pool_misses": self.pool_misses,
            "hit_rate": self.pool_hits / max(1, self.total_allocations),
            "pool_size": self.pool_size,
            "buffers_in_use": self.buffers_in_use,
            "buffers_available": self.buffers_available,
            "times_grown": self.times_grown,
            "peak_usage": self.peak_usage,
            "average_acquire_time_us": self.average_acquire_time_us,
            "average_release_time_us": self.average_release_time_us,
            "total_memory_bytes": self.total_memory_bytes,
            "memory_in_use_bytes": self.memory_in_use_bytes,
        }


@dataclass
class BufferMetadata:
    """Metadata for tracking individual buffers."""

    buffer_id: int
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    times_used: int = 0
    state: BufferState = BufferState.AVAILABLE


class AudioBufferPool:
    """
    Memory pool for audio processing buffers.

    Pre-allocates numpy arrays for audio data to avoid frequent allocations
    during real-time processing. Buffers are reused to minimize GC pressure.

    Features:
    - Pre-allocation of buffers at startup
    - Automatic pool growth when exhausted
    - Thread-safe acquire/release
    - Usage statistics and monitoring
    - Automatic buffer zeroing on release

    Example:
        >>> pool = AudioBufferPool(
        ...     chunk_size=1024,
        ...     pool_size=32,
        ...     dtype=np.float32,
        ... )
        >>>
        >>> # Acquire a buffer
        >>> buffer = pool.acquire()
        >>> buffer[:] = audio_data  # Fill with audio data
        >>>
        >>> # Process audio...
        >>>
        >>> # Release back to pool
        >>> pool.release(buffer)
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        pool_size: int = 32,
        dtype: np.dtype = np.float32,
        channels: int = 1,
        max_pool_size: int = 256,
        grow_factor: float = 1.5,
        auto_grow: bool = True,
        zero_on_release: bool = True,
        blocking: bool = True,
        timeout_seconds: float = 5.0,
    ):
        """
        Initialize the audio buffer pool.

        Args:
            chunk_size: Size of each audio chunk (samples per channel)
            pool_size: Initial number of buffers to pre-allocate
            dtype: NumPy data type for audio samples
            channels: Number of audio channels
            max_pool_size: Maximum pool size (limits growth)
            grow_factor: Factor to grow pool when exhausted
            auto_grow: Whether to automatically grow pool when exhausted
            zero_on_release: Whether to zero buffers when released
            blocking: Whether acquire() blocks when pool is exhausted
            timeout_seconds: Timeout for blocking acquire
        """
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.channels = channels
        self.max_pool_size = max_pool_size
        self.grow_factor = grow_factor
        self.auto_grow = auto_grow
        self.zero_on_release = zero_on_release
        self.blocking = blocking
        self.timeout_seconds = timeout_seconds

        # Calculate buffer shape
        if channels == 1:
            self.buffer_shape = (chunk_size,)
        else:
            self.buffer_shape = (chunk_size, channels)

        # Calculate buffer size in bytes
        self.buffer_size_bytes = np.prod(self.buffer_shape) * np.dtype(dtype).itemsize

        # Thread safety
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)

        # Pool storage
        self._available: Deque[np.ndarray] = deque()
        self._in_use: Set[int] = set()  # Track buffer IDs in use
        self._buffer_metadata: Dict[int, BufferMetadata] = {}
        self._buffer_counter = 0

        # Statistics
        self.stats = PoolStatistics()
        self._acquire_times: Deque[float] = deque(maxlen=100)
        self._release_times: Deque[float] = deque(maxlen=100)

        # Pre-allocate buffers
        self._preallocate(pool_size)

        logger.debug(
            f"AudioBufferPool initialized: {pool_size} buffers of "
            f"{chunk_size}x{channels} {dtype.__name__} ({self.buffer_size_bytes} bytes each)"
        )

    def _preallocate(self, count: int) -> None:
        """Pre-allocate buffers."""
        with self._lock:
            for _ in range(count):
                if self.stats.pool_size >= self.max_pool_size:
                    break
                buffer = self._create_buffer()
                self._available.append(buffer)

    def _create_buffer(self) -> np.ndarray:
        """Create a new buffer with metadata."""
        buffer = np.zeros(self.buffer_shape, dtype=self.dtype)
        buffer_id = self._buffer_counter
        self._buffer_counter += 1

        # Store metadata using buffer id
        self._buffer_metadata[id(buffer)] = BufferMetadata(
            buffer_id=buffer_id,
            created_at=time.time(),
        )

        self.stats.pool_size += 1
        self.stats.buffers_available += 1
        self.stats.total_memory_bytes += self.buffer_size_bytes

        return buffer

    def acquire(self, timeout: Optional[float] = None) -> np.ndarray:
        """
        Acquire a buffer from the pool.

        Args:
            timeout: Override default timeout (None uses pool default)

        Returns:
            numpy.ndarray: Buffer ready for use

        Raises:
            PoolExhaustedError: If pool is exhausted and blocking is disabled
            TimeoutError: If timeout expires while waiting
        """
        start_time = time.perf_counter()
        timeout = timeout if timeout is not None else self.timeout_seconds

        with self._not_empty:
            # Try to get an available buffer
            while not self._available:
                if self.auto_grow and self.stats.pool_size < self.max_pool_size:
                    # Grow the pool
                    self._grow_pool()
                    self.stats.pool_misses += 1
                elif self.blocking:
                    # Wait for a buffer to become available
                    if not self._not_empty.wait(timeout):
                        raise TimeoutError(
                            f"Timeout waiting for buffer after {timeout}s"
                        )
                else:
                    raise PoolExhaustedError(
                        f"Pool exhausted ({self.stats.buffers_in_use} in use)"
                    )

            # Get buffer from available pool
            buffer = self._available.popleft()
            self._in_use.add(id(buffer))

            # Update metadata
            metadata = self._buffer_metadata.get(id(buffer))
            if metadata:
                metadata.state = BufferState.IN_USE
                metadata.last_used_at = time.time()
                metadata.times_used += 1

            # Update statistics
            self.stats.total_allocations += 1
            self.stats.pool_hits += 1
            self.stats.buffers_available -= 1
            self.stats.buffers_in_use += 1
            self.stats.memory_in_use_bytes += self.buffer_size_bytes
            self.stats.peak_usage = max(self.stats.peak_usage, self.stats.buffers_in_use)

            # Track timing
            elapsed_us = (time.perf_counter() - start_time) * 1_000_000
            self._acquire_times.append(elapsed_us)
            self.stats.average_acquire_time_us = sum(self._acquire_times) / len(
                self._acquire_times
            )

            return buffer

    def release(self, buffer: np.ndarray) -> None:
        """
        Release a buffer back to the pool.

        Args:
            buffer: Buffer to release (must have been acquired from this pool)
        """
        start_time = time.perf_counter()

        with self._not_empty:
            buffer_ptr = id(buffer)

            if buffer_ptr not in self._in_use:
                logger.warning("Attempted to release a buffer not from this pool")
                return

            # Remove from in-use set
            self._in_use.discard(buffer_ptr)

            # Zero the buffer if configured
            if self.zero_on_release:
                buffer.fill(0)

            # Update metadata
            metadata = self._buffer_metadata.get(buffer_ptr)
            if metadata:
                metadata.state = BufferState.AVAILABLE

            # Return to available pool
            self._available.append(buffer)

            # Update statistics
            self.stats.total_releases += 1
            self.stats.buffers_available += 1
            self.stats.buffers_in_use -= 1
            self.stats.memory_in_use_bytes -= self.buffer_size_bytes

            # Track timing
            elapsed_us = (time.perf_counter() - start_time) * 1_000_000
            self._release_times.append(elapsed_us)
            self.stats.average_release_time_us = sum(self._release_times) / len(
                self._release_times
            )

            # Notify waiters
            self._not_empty.notify()

    def _grow_pool(self) -> None:
        """Grow the pool when exhausted."""
        current_size = self.stats.pool_size
        target_size = min(
            int(current_size * self.grow_factor),
            self.max_pool_size,
        )
        new_buffers = target_size - current_size

        if new_buffers > 0:
            for _ in range(new_buffers):
                buffer = self._create_buffer()
                self._available.append(buffer)

            self.stats.times_grown += 1
            logger.debug(f"AudioBufferPool grew from {current_size} to {target_size}")

    def shrink(self, target_size: Optional[int] = None) -> int:
        """
        Shrink the pool by releasing unused buffers.

        Args:
            target_size: Target pool size (None = shrink to min needed)

        Returns:
            Number of buffers freed
        """
        with self._lock:
            if target_size is None:
                # Keep only what's in use plus a small margin
                target_size = max(self.stats.buffers_in_use + 4, 8)

            freed = 0
            while self.stats.pool_size > target_size and self._available:
                buffer = self._available.pop()
                buffer_ptr = id(buffer)
                self._buffer_metadata.pop(buffer_ptr, None)
                self.stats.pool_size -= 1
                self.stats.buffers_available -= 1
                self.stats.total_memory_bytes -= self.buffer_size_bytes
                freed += 1

            if freed > 0:
                logger.debug(f"AudioBufferPool shrunk by {freed} buffers")

            return freed

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return self.stats.to_dict()

    def reset_statistics(self) -> None:
        """Reset pool statistics (keeps buffers)."""
        with self._lock:
            current_pool_size = self.stats.pool_size
            current_in_use = self.stats.buffers_in_use
            current_available = self.stats.buffers_available
            total_memory = self.stats.total_memory_bytes
            memory_in_use = self.stats.memory_in_use_bytes

            self.stats = PoolStatistics(
                pool_size=current_pool_size,
                buffers_in_use=current_in_use,
                buffers_available=current_available,
                total_memory_bytes=total_memory,
                memory_in_use_bytes=memory_in_use,
            )
            self._acquire_times.clear()
            self._release_times.clear()

    def __enter__(self) -> np.ndarray:
        """Context manager entry - acquire a buffer."""
        return self.acquire()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - this won't work as expected, use BufferContext."""
        pass  # Can't release here as we don't have the buffer reference


class BufferContext:
    """
    Context manager for automatic buffer acquisition and release.

    Example:
        >>> pool = AudioBufferPool()
        >>> with BufferContext(pool) as buffer:
        ...     buffer[:] = audio_data
        ...     process(buffer)
        >>> # Buffer automatically released
    """

    def __init__(self, pool: AudioBufferPool):
        """Initialize with a pool reference."""
        self.pool = pool
        self.buffer: Optional[np.ndarray] = None

    def __enter__(self) -> np.ndarray:
        """Acquire buffer on entry."""
        self.buffer = self.pool.acquire()
        return self.buffer

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release buffer on exit."""
        if self.buffer is not None:
            self.pool.release(self.buffer)
            self.buffer = None


class MultiSizeBufferPool:
    """
    Memory pool supporting multiple buffer sizes.

    Useful when audio processing requires different chunk sizes
    at different stages of the pipeline.

    Example:
        >>> pool = MultiSizeBufferPool()
        >>> pool.register_size(512, pool_size=16)
        >>> pool.register_size(1024, pool_size=32)
        >>> pool.register_size(4096, pool_size=8)
        >>>
        >>> small_buffer = pool.acquire(512)
        >>> large_buffer = pool.acquire(4096)
    """

    def __init__(
        self,
        dtype: np.dtype = np.float32,
        channels: int = 1,
        **pool_kwargs: Any,
    ):
        """
        Initialize multi-size pool.

        Args:
            dtype: Data type for all buffers
            channels: Number of channels
            **pool_kwargs: Additional arguments passed to individual pools
        """
        self.dtype = dtype
        self.channels = channels
        self.pool_kwargs = pool_kwargs
        self._pools: Dict[int, AudioBufferPool] = {}
        self._lock = threading.Lock()

    def register_size(self, chunk_size: int, pool_size: int = 16) -> None:
        """
        Register a new buffer size.

        Args:
            chunk_size: Size of buffers
            pool_size: Number of buffers to pre-allocate
        """
        with self._lock:
            if chunk_size not in self._pools:
                self._pools[chunk_size] = AudioBufferPool(
                    chunk_size=chunk_size,
                    pool_size=pool_size,
                    dtype=self.dtype,
                    channels=self.channels,
                    **self.pool_kwargs,
                )
                logger.debug(f"Registered buffer pool for size {chunk_size}")

    def acquire(self, chunk_size: int) -> np.ndarray:
        """
        Acquire a buffer of the specified size.

        Args:
            chunk_size: Required buffer size

        Returns:
            Buffer from the appropriate pool

        Raises:
            KeyError: If size not registered
        """
        if chunk_size not in self._pools:
            raise KeyError(
                f"Buffer size {chunk_size} not registered. "
                f"Available sizes: {list(self._pools.keys())}"
            )
        return self._pools[chunk_size].acquire()

    def release(self, buffer: np.ndarray) -> None:
        """
        Release a buffer back to its pool.

        Args:
            buffer: Buffer to release
        """
        # Find the right pool based on buffer size
        size = buffer.shape[0]
        if size in self._pools:
            self._pools[size].release(buffer)
        else:
            logger.warning(f"No pool for buffer size {size}, buffer will be GC'd")

    def get_all_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics from all pools."""
        return {
            size: pool.get_statistics() for size, pool in self._pools.items()
        }


# Global default pool instance
_default_pool: Optional[AudioBufferPool] = None
_default_pool_lock = threading.Lock()


def get_default_pool(
    chunk_size: int = 1024,
    pool_size: int = 32,
    **kwargs: Any,
) -> AudioBufferPool:
    """
    Get or create the default global audio buffer pool.

    Args:
        chunk_size: Chunk size for the pool
        pool_size: Initial pool size
        **kwargs: Additional pool configuration

    Returns:
        Global AudioBufferPool instance
    """
    global _default_pool

    with _default_pool_lock:
        if _default_pool is None:
            _default_pool = AudioBufferPool(
                chunk_size=chunk_size,
                pool_size=pool_size,
                **kwargs,
            )
        return _default_pool


def reset_default_pool() -> None:
    """Reset the default global pool."""
    global _default_pool

    with _default_pool_lock:
        _default_pool = None
