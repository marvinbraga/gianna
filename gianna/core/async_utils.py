"""
Async Utilities for Gianna

This module provides async infrastructure for the Gianna project, including:
- Event loop management
- Async execution utilities
- Sync-to-async bridges
- Async context managers
- Concurrent task management

Usage:
    >>> from gianna.core.async_utils import run_async, AsyncExecutor
    >>>
    >>> # Run async function from sync context
    >>> result = run_async(async_function())
    >>>
    >>> # Or use AsyncExecutor for batch operations
    >>> async with AsyncExecutor() as executor:
    ...     results = await executor.gather([task1(), task2(), task3()])
"""

import asyncio
import concurrent.futures
import functools
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from loguru import logger

T = TypeVar("T")
R = TypeVar("R")


class ExecutionMode(Enum):
    """Execution mode for async operations."""

    CONCURRENT = "concurrent"  # Run all tasks concurrently
    SEQUENTIAL = "sequential"  # Run tasks one after another
    BATCH = "batch"  # Run in batches with concurrency limit


@dataclass
class AsyncConfig:
    """Configuration for async operations."""

    # Concurrency limits
    max_concurrent_tasks: int = 10
    batch_size: int = 5

    # Timeouts
    default_timeout_seconds: float = 30.0
    task_timeout_seconds: float = 60.0

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0

    # Error handling
    fail_fast: bool = False  # Stop on first error
    collect_exceptions: bool = True  # Collect all exceptions


@dataclass
class TaskResult(Generic[T]):
    """Result of an async task execution."""

    success: bool
    value: Optional[T] = None
    error: Optional[Exception] = None
    execution_time_seconds: float = 0.0
    retries: int = 0


class AsyncEventLoopManager:
    """
    Manages event loops for async operations.

    Handles the complexity of running async code from sync contexts,
    managing thread-local event loops, and proper cleanup.
    """

    _thread_loops: Dict[int, asyncio.AbstractEventLoop] = {}
    _main_loop: Optional[asyncio.AbstractEventLoop] = None
    _lock = threading.Lock()

    @classmethod
    def get_or_create_loop(cls) -> asyncio.AbstractEventLoop:
        """
        Get or create an event loop for the current thread.

        Returns:
            Event loop for the current thread
        """
        thread_id = threading.get_ident()

        with cls._lock:
            # Check if we already have a loop for this thread
            if thread_id in cls._thread_loops:
                loop = cls._thread_loops[thread_id]
                if not loop.is_closed():
                    return loop

            # Try to get the running loop
            try:
                loop = asyncio.get_running_loop()
                cls._thread_loops[thread_id] = loop
                return loop
            except RuntimeError:
                pass

            # Try to get the existing event loop
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    cls._thread_loops[thread_id] = loop
                    return loop
            except RuntimeError:
                pass

            # Create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            cls._thread_loops[thread_id] = loop
            logger.debug(f"Created new event loop for thread {thread_id}")
            return loop

    @classmethod
    def run_coroutine(cls, coro: Coroutine[Any, Any, T]) -> T:
        """
        Run a coroutine from sync context.

        Args:
            coro: Coroutine to run

        Returns:
            Result from the coroutine
        """
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # We're in an async context, need to use different approach
            # Create a new thread to run the coroutine
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(cls._run_in_new_loop, coro)
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run or loop.run_until_complete
            loop = cls.get_or_create_loop()
            return loop.run_until_complete(coro)

    @classmethod
    def _run_in_new_loop(cls, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine in a new event loop (for nested async calls)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @classmethod
    def cleanup_thread_loop(cls) -> None:
        """Cleanup the event loop for the current thread."""
        thread_id = threading.get_ident()

        with cls._lock:
            if thread_id in cls._thread_loops:
                loop = cls._thread_loops.pop(thread_id)
                if not loop.is_closed():
                    loop.close()
                logger.debug(f"Cleaned up event loop for thread {thread_id}")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from synchronous code.

    This is the main entry point for calling async code from sync code.

    Args:
        coro: Coroutine to execute

    Returns:
        Result from the coroutine

    Example:
        >>> async def fetch_data():
        ...     return await some_async_operation()
        >>> result = run_async(fetch_data())
    """
    return AsyncEventLoopManager.run_coroutine(coro)


def async_to_sync(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """
    Decorator to convert an async function to sync.

    Args:
        func: Async function to wrap

    Returns:
        Sync wrapper function

    Example:
        >>> @async_to_sync
        ... async def fetch_data():
        ...     return await some_async_operation()
        >>> result = fetch_data()  # Can be called synchronously
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return run_async(func(*args, **kwargs))

    return wrapper


def sync_to_async(
    func: Callable[..., T],
    executor: Optional[concurrent.futures.Executor] = None,
) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Convert a sync function to async by running in executor.

    Args:
        func: Sync function to wrap
        executor: Optional executor (defaults to thread pool)

    Returns:
        Async wrapper function

    Example:
        >>> def blocking_io():
        ...     return requests.get(url)
        >>> async_fetch = sync_to_async(blocking_io)
        >>> result = await async_fetch()
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_event_loop()
        partial_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, partial_func)

    return wrapper


class AsyncExecutor:
    """
    Executor for managing concurrent async tasks.

    Provides utilities for running multiple async tasks with
    concurrency limits, retries, and error handling.

    Example:
        >>> async with AsyncExecutor(max_concurrent=5) as executor:
        ...     results = await executor.map(process, items)
    """

    def __init__(
        self,
        config: Optional[AsyncConfig] = None,
        executor: Optional[concurrent.futures.Executor] = None,
    ):
        """
        Initialize the async executor.

        Args:
            config: Configuration for async operations
            executor: Optional thread pool for sync operations
        """
        self.config = config or AsyncConfig()
        self._executor = executor
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._active_tasks: Set[asyncio.Task] = set()

    async def __aenter__(self) -> "AsyncExecutor":
        """Async context manager entry."""
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        # Wait for all active tasks to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._active_tasks.clear()

    async def run(
        self,
        coro: Coroutine[Any, Any, T],
        timeout: Optional[float] = None,
    ) -> TaskResult[T]:
        """
        Run a single coroutine with retry and timeout support.

        Args:
            coro: Coroutine to run
            timeout: Optional timeout override

        Returns:
            TaskResult with success/failure info
        """
        timeout = timeout or self.config.default_timeout_seconds
        start_time = time.time()
        retries = 0
        last_error: Optional[Exception] = None

        while retries <= self.config.max_retries:
            try:
                if self._semaphore:
                    async with self._semaphore:
                        result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await asyncio.wait_for(coro, timeout=timeout)

                return TaskResult(
                    success=True,
                    value=result,
                    execution_time_seconds=time.time() - start_time,
                    retries=retries,
                )

            except asyncio.TimeoutError as e:
                last_error = e
                retries += 1
                if retries <= self.config.max_retries:
                    delay = self.config.retry_delay_seconds * (
                        self.config.retry_backoff_factor ** (retries - 1)
                    )
                    logger.warning(
                        f"Task timed out, retrying in {delay:.1f}s "
                        f"(attempt {retries}/{self.config.max_retries})"
                    )
                    await asyncio.sleep(delay)

            except Exception as e:
                last_error = e
                if self.config.fail_fast:
                    break
                retries += 1
                if retries <= self.config.max_retries:
                    delay = self.config.retry_delay_seconds * (
                        self.config.retry_backoff_factor ** (retries - 1)
                    )
                    logger.warning(
                        f"Task failed with {type(e).__name__}, retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        return TaskResult(
            success=False,
            error=last_error,
            execution_time_seconds=time.time() - start_time,
            retries=retries,
        )

    async def gather(
        self,
        coros: List[Coroutine[Any, Any, T]],
        return_exceptions: bool = True,
    ) -> List[TaskResult[T]]:
        """
        Run multiple coroutines concurrently.

        Args:
            coros: List of coroutines to run
            return_exceptions: Whether to return exceptions or raise

        Returns:
            List of TaskResult objects
        """
        tasks = [self.run(coro) for coro in coros]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    async def map(
        self,
        func: Callable[[Any], Coroutine[Any, Any, T]],
        items: List[Any],
        mode: ExecutionMode = ExecutionMode.CONCURRENT,
    ) -> List[TaskResult[T]]:
        """
        Map an async function over items with concurrency control.

        Args:
            func: Async function to apply
            items: Items to process
            mode: Execution mode

        Returns:
            List of results
        """
        if mode == ExecutionMode.SEQUENTIAL:
            results = []
            for item in items:
                result = await self.run(func(item))
                results.append(result)
            return results

        elif mode == ExecutionMode.BATCH:
            results = []
            for i in range(0, len(items), self.config.batch_size):
                batch = items[i : i + self.config.batch_size]
                batch_results = await self.gather([func(item) for item in batch])
                results.extend(batch_results)
            return results

        else:  # CONCURRENT
            return await self.gather([func(item) for item in items])

    async def run_sync(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Run a sync function asynchronously in a thread pool.

        Args:
            func: Sync function to run
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from the function
        """
        loop = asyncio.get_event_loop()
        partial_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self._executor, partial_func)


@asynccontextmanager
async def async_timeout(seconds: float) -> AsyncGenerator[None, None]:
    """
    Async context manager for timeouts.

    Args:
        seconds: Timeout in seconds

    Example:
        >>> async with async_timeout(5.0):
        ...     await long_running_operation()
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {seconds}s")
        raise


class AsyncRetry:
    """
    Decorator for async function retry logic.

    Example:
        >>> @AsyncRetry(max_retries=3, delay=1.0)
        ... async def flaky_operation():
        ...     return await external_api_call()
    """

    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: Tuple[type, ...] = (Exception,),
    ):
        """
        Initialize retry decorator.

        Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries
            backoff_factor: Factor to multiply delay after each retry
            exceptions: Tuple of exceptions to catch and retry
        """
        self.max_retries = max_retries
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions

    def __call__(
        self, func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        """Apply retry logic to async function."""

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            current_delay = self.delay

            for attempt in range(self.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    if attempt < self.max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/"
                            f"{self.max_retries + 1}): {e}. Retrying in {current_delay:.1f}s"
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= self.backoff_factor

            raise last_exception  # type: ignore

        return wrapper


class AsyncThrottle:
    """
    Rate limiter for async operations.

    Limits the rate of operations to prevent overwhelming external services.

    Example:
        >>> throttle = AsyncThrottle(rate=10, period=1.0)  # 10 ops/second
        >>> async with throttle:
        ...     await api_call()
    """

    def __init__(self, rate: int, period: float = 1.0):
        """
        Initialize throttle.

        Args:
            rate: Maximum number of operations per period
            period: Time period in seconds
        """
        self.rate = rate
        self.period = period
        self._semaphore = asyncio.Semaphore(rate)
        self._tokens: List[float] = []
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncThrottle":
        """Acquire a slot."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release is automatic after period."""
        pass

    async def acquire(self) -> None:
        """Acquire a rate limit slot."""
        async with self._lock:
            now = time.time()
            # Remove expired tokens
            self._tokens = [t for t in self._tokens if now - t < self.period]

            if len(self._tokens) >= self.rate:
                # Need to wait
                oldest = self._tokens[0]
                wait_time = self.period - (now - oldest)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            self._tokens.append(time.time())


class AsyncBatcher:
    """
    Batches multiple async calls for efficiency.

    Collects calls and executes them in batches to reduce overhead.

    Example:
        >>> batcher = AsyncBatcher(batch_func, max_size=10, max_wait=0.1)
        >>> result = await batcher.add(item)
    """

    def __init__(
        self,
        batch_func: Callable[[List[T]], Coroutine[Any, Any, List[R]]],
        max_size: int = 10,
        max_wait_seconds: float = 0.1,
    ):
        """
        Initialize batcher.

        Args:
            batch_func: Function that processes a batch of items
            max_size: Maximum batch size
            max_wait_seconds: Maximum time to wait for batch to fill
        """
        self.batch_func = batch_func
        self.max_size = max_size
        self.max_wait = max_wait_seconds

        self._queue: List[Tuple[T, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._process_task: Optional[asyncio.Task] = None

    async def add(self, item: T) -> R:
        """
        Add an item to be batched.

        Args:
            item: Item to process

        Returns:
            Result for this item
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        async with self._lock:
            self._queue.append((item, future))

            if len(self._queue) >= self.max_size:
                await self._flush()
            elif self._process_task is None or self._process_task.done():
                self._process_task = asyncio.create_task(self._delayed_flush())

        return await future

    async def _delayed_flush(self) -> None:
        """Wait and then flush."""
        await asyncio.sleep(self.max_wait)
        async with self._lock:
            if self._queue:
                await self._flush()

    async def _flush(self) -> None:
        """Process the current batch."""
        if not self._queue:
            return

        items = [item for item, _ in self._queue]
        futures = [future for _, future in self._queue]
        self._queue.clear()

        try:
            results = await self.batch_func(items)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)
