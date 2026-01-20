"""
Tests for Audio Memory Pool implementation.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from gianna.audio.memory_pool import (
    AudioBufferPool,
    BufferContext,
    MultiSizeBufferPool,
    PoolExhaustedError,
    get_default_pool,
    reset_default_pool,
)


class TestAudioBufferPool:
    """Tests for AudioBufferPool class."""

    def test_initialization(self):
        """Test pool initialization."""
        pool = AudioBufferPool(chunk_size=1024, pool_size=16)
        stats = pool.get_statistics()

        assert stats["pool_size"] == 16
        assert stats["buffers_available"] == 16
        assert stats["buffers_in_use"] == 0

    def test_acquire_release(self):
        """Test basic acquire and release."""
        pool = AudioBufferPool(chunk_size=512, pool_size=8)

        buffer = pool.acquire()

        assert isinstance(buffer, np.ndarray)
        assert buffer.shape == (512,)

        stats = pool.get_statistics()
        assert stats["buffers_in_use"] == 1
        assert stats["buffers_available"] == 7

        pool.release(buffer)

        stats = pool.get_statistics()
        assert stats["buffers_in_use"] == 0
        assert stats["buffers_available"] == 8

    def test_buffer_reuse(self):
        """Test that buffers are reused from pool."""
        pool = AudioBufferPool(chunk_size=256, pool_size=4)

        # Acquire and release a few times
        for _ in range(10):
            buffer = pool.acquire()
            pool.release(buffer)

        stats = pool.get_statistics()
        # Pool hits should be high since we're reusing
        assert stats["pool_hits"] == 10
        assert stats["pool_size"] == 4  # Shouldn't have grown

    def test_zero_on_release(self):
        """Test that buffers are zeroed on release."""
        pool = AudioBufferPool(chunk_size=128, pool_size=2, zero_on_release=True)

        buffer = pool.acquire()
        buffer.fill(1.0)  # Fill with non-zero values

        pool.release(buffer)

        # Acquire same buffer (should be the only one available)
        buffer2 = pool.acquire()

        # Should be zeroed
        assert np.allclose(buffer2, 0.0)

    def test_auto_grow(self):
        """Test automatic pool growth."""
        pool = AudioBufferPool(
            chunk_size=256,
            pool_size=2,
            max_pool_size=10,
            auto_grow=True,
        )

        # Acquire more than initial size
        buffers = []
        for _ in range(5):
            buffers.append(pool.acquire())

        stats = pool.get_statistics()
        assert stats["pool_size"] > 2
        assert stats["times_grown"] > 0

        # Release all
        for buf in buffers:
            pool.release(buf)

    def test_max_pool_size_limit(self):
        """Test that pool respects max size limit."""
        pool = AudioBufferPool(
            chunk_size=256,
            pool_size=2,
            max_pool_size=4,
            auto_grow=True,
            blocking=False,
        )

        # Acquire up to max
        buffers = []
        for _ in range(4):
            buffers.append(pool.acquire())

        stats = pool.get_statistics()
        assert stats["pool_size"] == 4

        # Next acquire should fail
        with pytest.raises(PoolExhaustedError):
            pool.acquire()

        # Release all
        for buf in buffers:
            pool.release(buf)

    def test_blocking_acquire(self):
        """Test blocking acquire when pool is exhausted."""
        pool = AudioBufferPool(
            chunk_size=256,
            pool_size=1,
            max_pool_size=1,
            auto_grow=False,
            blocking=True,
            timeout_seconds=0.5,
        )

        # Acquire the only buffer
        buffer = pool.acquire()

        # Try to acquire with timeout - should fail
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.1)

        # Release
        pool.release(buffer)

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        pool = AudioBufferPool(chunk_size=256, pool_size=10)
        errors = []
        operations = 100

        def worker():
            try:
                for _ in range(operations // 10):
                    buffer = pool.acquire()
                    time.sleep(0.001)  # Simulate some work
                    pool.release(buffer)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = pool.get_statistics()
        assert stats["buffers_in_use"] == 0

    def test_multi_channel_buffers(self):
        """Test buffers with multiple channels."""
        pool = AudioBufferPool(chunk_size=1024, pool_size=4, channels=2)

        buffer = pool.acquire()

        assert buffer.shape == (1024, 2)

        pool.release(buffer)

    def test_different_dtypes(self):
        """Test buffers with different data types."""
        for dtype in [np.float32, np.float64, np.int16]:
            pool = AudioBufferPool(chunk_size=512, pool_size=2, dtype=dtype)
            buffer = pool.acquire()

            assert buffer.dtype == dtype

            pool.release(buffer)

    def test_shrink(self):
        """Test pool shrinking."""
        pool = AudioBufferPool(chunk_size=256, pool_size=20)

        initial_size = pool.stats.pool_size

        freed = pool.shrink(target_size=5)

        assert freed == initial_size - 5
        assert pool.stats.pool_size == 5

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        pool = AudioBufferPool(chunk_size=256, pool_size=4)

        # Do some operations
        for _ in range(5):
            buf = pool.acquire()
            pool.release(buf)

        stats = pool.get_statistics()

        assert stats["total_allocations"] == 5
        assert stats["total_releases"] == 5
        assert stats["pool_hits"] == 5
        assert stats["average_acquire_time_us"] > 0
        assert stats["average_release_time_us"] > 0


class TestBufferContext:
    """Tests for BufferContext context manager."""

    def test_context_manager(self):
        """Test automatic acquire/release with context manager."""
        pool = AudioBufferPool(chunk_size=256, pool_size=4)

        with BufferContext(pool) as buffer:
            assert isinstance(buffer, np.ndarray)
            assert pool.stats.buffers_in_use == 1

        # Buffer should be released after context
        assert pool.stats.buffers_in_use == 0

    def test_context_manager_exception(self):
        """Test that buffer is released even on exception."""
        pool = AudioBufferPool(chunk_size=256, pool_size=4)

        try:
            with BufferContext(pool) as buffer:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Buffer should still be released
        assert pool.stats.buffers_in_use == 0


class TestMultiSizeBufferPool:
    """Tests for MultiSizeBufferPool class."""

    def test_register_and_acquire(self):
        """Test registering sizes and acquiring buffers."""
        pool = MultiSizeBufferPool()
        pool.register_size(256, pool_size=4)
        pool.register_size(1024, pool_size=2)

        small_buf = pool.acquire(256)
        large_buf = pool.acquire(1024)

        assert small_buf.shape == (256,)
        assert large_buf.shape == (1024,)

        pool.release(small_buf)
        pool.release(large_buf)

    def test_unregistered_size_error(self):
        """Test error when acquiring unregistered size."""
        pool = MultiSizeBufferPool()
        pool.register_size(256)

        with pytest.raises(KeyError):
            pool.acquire(512)

    def test_get_all_statistics(self):
        """Test getting statistics from all pools."""
        pool = MultiSizeBufferPool()
        pool.register_size(256, pool_size=4)
        pool.register_size(512, pool_size=2)

        # Use some buffers
        buf1 = pool.acquire(256)
        buf2 = pool.acquire(512)

        stats = pool.get_all_statistics()

        assert 256 in stats
        assert 512 in stats
        assert stats[256]["buffers_in_use"] == 1
        assert stats[512]["buffers_in_use"] == 1

        pool.release(buf1)
        pool.release(buf2)


class TestGlobalPool:
    """Tests for global pool functions."""

    def test_get_default_pool(self):
        """Test getting default pool."""
        reset_default_pool()

        pool = get_default_pool(chunk_size=512, pool_size=8)

        assert isinstance(pool, AudioBufferPool)

        # Should return same instance
        pool2 = get_default_pool()
        assert pool is pool2

    def test_reset_default_pool(self):
        """Test resetting default pool."""
        pool1 = get_default_pool()
        reset_default_pool()
        pool2 = get_default_pool()

        # Should be different instances
        assert pool1 is not pool2
