"""
Sistema de Gerenciamento de Recursos para Gianna

Implementa controle de recursos (CPU, memória, I/O), throttling inteligente,
pool de workers assíncronos, circuit breaker e rate limiting.
"""

import asyncio
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import Lock, Semaphore, Thread
from typing import Any, Callable, Dict, List, Optional, Union

import psutil
from loguru import logger


class ResourceType(Enum):
    """Tipos de recursos"""

    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    CUSTOM = "custom"


class CircuitState(Enum):
    """Estados do Circuit Breaker"""

    CLOSED = "closed"  # Funcionando normalmente
    OPEN = "open"  # Falhas detectadas, rejeitando requisições
    HALF_OPEN = "half_open"  # Testando se pode voltar ao normal


@dataclass
class ResourceUsage:
    """Informações de uso de recursos"""

    type: ResourceType
    current: float
    maximum: float
    percentage: float
    unit: str
    timestamp: float


@dataclass
class WorkerTask:
    """Tarefa para worker assíncrono"""

    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 5
    timeout: Optional[float] = None
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class ResourceMonitor:
    """Monitor de recursos do sistema"""

    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.thresholds = {
            ResourceType.CPU: 80.0,
            ResourceType.MEMORY: 85.0,
            ResourceType.IO: 90.0,
        }
        self.callbacks = defaultdict(list)
        self._monitoring = False
        self._monitor_thread = None
        self._lock = Lock()

    def set_threshold(self, resource_type: ResourceType, percentage: float):
        """Define limite para tipo de recurso"""
        self.thresholds[resource_type] = percentage

    def add_threshold_callback(
        self, resource_type: ResourceType, callback: Callable[[ResourceUsage], None]
    ):
        """Adiciona callback para quando limite é atingido"""
        self.callbacks[resource_type].append(callback)

    def get_current_usage(self) -> Dict[ResourceType, ResourceUsage]:
        """Obtém uso atual de recursos"""
        usage = {}
        current_time = time.time()

        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        usage[ResourceType.CPU] = ResourceUsage(
            type=ResourceType.CPU,
            current=cpu_percent,
            maximum=100.0,
            percentage=cpu_percent,
            unit="%",
            timestamp=current_time,
        )

        # Memory
        memory = psutil.virtual_memory()
        usage[ResourceType.MEMORY] = ResourceUsage(
            type=ResourceType.MEMORY,
            current=memory.used,
            maximum=memory.total,
            percentage=memory.percent,
            unit="bytes",
            timestamp=current_time,
        )

        # I/O (disk)
        disk_io = psutil.disk_io_counters()
        if disk_io:
            # Estimativa simples baseada em atividade
            io_percent = min(
                (disk_io.read_bytes + disk_io.write_bytes) / (1024**3) * 10, 100
            )
            usage[ResourceType.IO] = ResourceUsage(
                type=ResourceType.IO,
                current=disk_io.read_bytes + disk_io.write_bytes,
                maximum=float("inf"),
                percentage=io_percent,
                unit="bytes",
                timestamp=current_time,
            )

        return usage

    def start_monitoring(self):
        """Inicia monitoramento de recursos"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Para monitoramento de recursos"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        while self._monitoring:
            try:
                usage = self.get_current_usage()

                for resource_type, resource_usage in usage.items():
                    threshold = self.thresholds.get(resource_type, 100.0)

                    if resource_usage.percentage > threshold:
                        # Executa callbacks
                        for callback in self.callbacks[resource_type]:
                            try:
                                callback(resource_usage)
                            except Exception as e:
                                logger.error(f"Error in resource callback: {e}")

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.check_interval)


class Throttler:
    """Sistema de throttling inteligente"""

    def __init__(
        self,
        base_rate: float = 10.0,  # requests per second
        burst_size: int = 20,
        adaptive: bool = True,
    ):

        self.base_rate = base_rate
        self.burst_size = burst_size
        self.adaptive = adaptive

        # Token bucket
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = Lock()

        # Adaptive throttling
        self.current_rate = base_rate
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = time.time()

        # Estatísticas
        self.total_requests = 0
        self.throttled_requests = 0

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Adquire permissão para executar operação

        Returns:
            True se pode executar, False se throttled
        """
        with self.lock:
            self._update_tokens()

            if self.tokens >= 1:
                self.tokens -= 1
                self.total_requests += 1
                return True
            else:
                self.throttled_requests += 1

                if timeout and timeout > 0:
                    # Espera até ter token disponível
                    wait_time = 1.0 / self.current_rate
                    if wait_time <= timeout:
                        time.sleep(wait_time)
                        return self.acquire(timeout - wait_time)

                return False

    def _update_tokens(self):
        """Atualiza tokens baseado no tempo"""
        now = time.time()
        elapsed = now - self.last_update

        # Adiciona tokens baseado na taxa
        tokens_to_add = elapsed * self.current_rate
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_update = now

    def record_success(self):
        """Registra operação bem-sucedida"""
        self.success_count += 1
        if self.adaptive:
            self._adjust_rate()

    def record_error(self):
        """Registra erro de operação"""
        self.error_count += 1
        if self.adaptive:
            self._adjust_rate()

    def _adjust_rate(self):
        """Ajusta taxa baseado no desempenho"""
        now = time.time()

        # Ajusta a cada 30 segundos
        if now - self.last_adjustment < 30:
            return

        total_ops = self.success_count + self.error_count
        if total_ops == 0:
            return

        error_rate = self.error_count / total_ops

        if error_rate > 0.05:  # Mais de 5% de erros
            # Diminui taxa
            self.current_rate = max(self.base_rate * 0.1, self.current_rate * 0.8)
            logger.info(
                f"Throttling: Rate decreased to {self.current_rate:.2f}/s (error rate: {error_rate:.1%})"
            )
        elif error_rate < 0.01:  # Menos de 1% de erros
            # Aumenta taxa
            self.current_rate = min(self.base_rate * 2, self.current_rate * 1.2)
            logger.debug(f"Throttling: Rate increased to {self.current_rate:.2f}/s")

        # Reset counters
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = now

    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do throttler"""
        with self.lock:
            throttle_rate = 0.0
            if self.total_requests > 0:
                throttle_rate = self.throttled_requests / self.total_requests

            return {
                "base_rate": self.base_rate,
                "current_rate": self.current_rate,
                "burst_size": self.burst_size,
                "current_tokens": self.tokens,
                "total_requests": self.total_requests,
                "throttled_requests": self.throttled_requests,
                "throttle_rate": throttle_rate,
            }


class CircuitBreaker:
    """Circuit Breaker para proteção contra falhas"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ):

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = Lock()

    def call(self, func: Callable, *args, **kwargs):
        """
        Executa função com circuit breaker

        Raises:
            CircuitBreakerOpenError: Quando circuit está aberto
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
                else:
                    # Transição para HALF_OPEN
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except Exception as e:
            self._record_failure()
            raise

    def _record_success(self):
        """Registra operação bem-sucedida"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    def _record_failure(self):
        """Registra falha de operação"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker: {self.state} -> OPEN")

    def get_state(self) -> CircuitState:
        """Obtém estado atual do circuit breaker"""
        return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do circuit breaker"""
        with self.lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "success_count": self.success_count,
                "recovery_timeout": self.recovery_timeout,
                "last_failure_time": self.last_failure_time,
            }


class RateLimiter:
    """Rate limiter por usuário/session com sliding window"""

    def __init__(self, requests_per_minute: int = 60, window_size: int = 60):

        self.requests_per_minute = requests_per_minute
        self.window_size = window_size

        # Sliding window para cada cliente
        self.clients = defaultdict(lambda: deque())
        self.lock = Lock()

    def is_allowed(self, client_id: str) -> bool:
        """
        Verifica se cliente pode fazer requisição

        Args:
            client_id: Identificador único do cliente

        Returns:
            True se permitido, False se rate limited
        """
        now = time.time()

        with self.lock:
            client_requests = self.clients[client_id]

            # Remove requisições antigas (fora da janela)
            cutoff_time = now - self.window_size
            while client_requests and client_requests[0] < cutoff_time:
                client_requests.popleft()

            # Verifica se pode adicionar nova requisição
            if len(client_requests) < self.requests_per_minute:
                client_requests.append(now)
                return True
            else:
                return False

    def get_remaining(self, client_id: str) -> int:
        """Obtém número de requisições restantes"""
        now = time.time()

        with self.lock:
            client_requests = self.clients[client_id]

            # Limpa requisições antigas
            cutoff_time = now - self.window_size
            while client_requests and client_requests[0] < cutoff_time:
                client_requests.popleft()

            return max(0, self.requests_per_minute - len(client_requests))

    def get_reset_time(self, client_id: str) -> float:
        """Obtém tempo até reset da janela"""
        with self.lock:
            client_requests = self.clients[client_id]

            if not client_requests:
                return 0.0

            oldest_request = client_requests[0]
            return max(0.0, oldest_request + self.window_size - time.time())


class WorkerPool:
    """Pool de workers assíncronos com prioridades"""

    def __init__(
        self,
        max_workers: int = 4,
        queue_size: int = 1000,
        enable_priorities: bool = True,
    ):

        self.max_workers = max_workers
        self.queue_size = queue_size
        self.enable_priorities = enable_priorities

        # Queue de tarefas (implementação simples com lista)
        self.task_queue = []
        self.completed_tasks = {}
        self.failed_tasks = {}

        # Workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}

        # Controle
        self.lock = Lock()
        self.running = True

        # Estatísticas
        self.total_submitted = 0
        self.total_completed = 0
        self.total_failed = 0

    def submit_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        priority: int = 5,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Submete tarefa para execução

        Args:
            task_id: ID único da tarefa
            func: Função a executar
            priority: Prioridade (1=alta, 10=baixa)
            timeout: Timeout em segundos

        Returns:
            task_id
        """
        task = WorkerTask(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
        )

        with self.lock:
            if len(self.task_queue) >= self.queue_size:
                raise RuntimeError("Worker queue is full")

            self.task_queue.append(task)
            self.total_submitted += 1

            # Ordena por prioridade se habilitado
            if self.enable_priorities:
                self.task_queue.sort(key=lambda t: t.priority)

        # Processa próxima tarefa
        self._process_next_task()

        return task_id

    def _process_next_task(self):
        """Processa próxima tarefa da queue"""
        with self.lock:
            if not self.task_queue or len(self.active_tasks) >= self.max_workers:
                return

            task = self.task_queue.pop(0)

            # Submete para executor
            future = self.executor.submit(self._execute_task, task)
            self.active_tasks[task.id] = {
                "task": task,
                "future": future,
                "started_at": time.time(),
            }

    def _execute_task(self, task: WorkerTask):
        """Executa tarefa individual"""
        try:
            start_time = time.time()

            # Verifica timeout
            if task.timeout:

                def timeout_handler():
                    raise TimeoutError(f"Task {task.id} timed out")

                # Implementação simples de timeout
                # Em produção, usaria asyncio ou signal

            result = task.func(*task.args, **task.kwargs)

            execution_time = time.time() - start_time

            with self.lock:
                self.completed_tasks[task.id] = {
                    "result": result,
                    "execution_time": execution_time,
                    "completed_at": time.time(),
                }
                self.total_completed += 1

                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]

            logger.debug(f"Task {task.id} completed in {execution_time:.2f}s")

            # Processa próxima tarefa
            self._process_next_task()

        except Exception as e:
            with self.lock:
                self.failed_tasks[task.id] = {"error": str(e), "failed_at": time.time()}
                self.total_failed += 1

                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]

            logger.error(f"Task {task.id} failed: {e}")

            # Processa próxima tarefa
            self._process_next_task()

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Obtém status de uma tarefa"""
        with self.lock:
            if task_id in self.active_tasks:
                return "running"
            elif task_id in self.completed_tasks:
                return "completed"
            elif task_id in self.failed_tasks:
                return "failed"
            elif any(t.id == task_id for t in self.task_queue):
                return "queued"
            else:
                return None

    def get_task_result(self, task_id: str) -> Any:
        """Obtém resultado de tarefa completada"""
        with self.lock:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]["result"]
            elif task_id in self.failed_tasks:
                raise RuntimeError(
                    f"Task failed: {self.failed_tasks[task_id]['error']}"
                )
            else:
                raise ValueError(f"Task {task_id} not found or not completed")

    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do pool"""
        with self.lock:
            return {
                "max_workers": self.max_workers,
                "active_tasks": len(self.active_tasks),
                "queued_tasks": len(self.task_queue),
                "total_submitted": self.total_submitted,
                "total_completed": self.total_completed,
                "total_failed": self.total_failed,
                "queue_utilization": len(self.task_queue) / self.queue_size,
                "worker_utilization": len(self.active_tasks) / self.max_workers,
            }

    def shutdown(self, wait: bool = True):
        """Finaliza pool de workers"""
        self.running = False
        self.executor.shutdown(wait=wait)
        logger.info("Worker pool shutdown")


class ResourceManager:
    """Gerenciador principal de recursos"""

    def __init__(
        self,
        max_workers: int = 4,
        throttle_rate: float = 10.0,
        rate_limit_per_minute: int = 60,
    ):

        # Componentes
        self.resource_monitor = ResourceMonitor()
        self.throttler = Throttler(base_rate=throttle_rate)
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit_per_minute)
        self.worker_pool = WorkerPool(max_workers=max_workers)

        # Auto-scaling
        self.auto_scaling_enabled = True
        self.scaling_factor = 1.0

        # Setup callbacks
        self._setup_resource_callbacks()

    def _setup_resource_callbacks(self):
        """Configura callbacks para monitoramento de recursos"""

        def cpu_callback(usage: ResourceUsage):
            if usage.percentage > 90:
                # CPU crítico - reduz throttling drasticamente
                self.throttler.current_rate = self.throttler.base_rate * 0.2
                logger.warning(
                    f"CPU critical ({usage.percentage:.1f}%) - throttling to {self.throttler.current_rate:.1f}/s"
                )

        def memory_callback(usage: ResourceUsage):
            if usage.percentage > 90:
                # Memória crítica - força limpeza
                import gc

                gc.collect()
                logger.warning(
                    f"Memory critical ({usage.percentage:.1f}%) - forced garbage collection"
                )

        self.resource_monitor.add_threshold_callback(ResourceType.CPU, cpu_callback)
        self.resource_monitor.add_threshold_callback(
            ResourceType.MEMORY, memory_callback
        )

    def execute_with_protection(
        self, func: Callable, client_id: str = "default", *args, **kwargs
    ):
        """
        Executa função com proteção completa de recursos

        Args:
            func: Função a executar
            client_id: ID do cliente (para rate limiting)
        """
        # Rate limiting
        if not self.rate_limiter.is_allowed(client_id):
            raise RateLimitError(f"Rate limit exceeded for client {client_id}")

        # Throttling
        if not self.throttler.acquire(timeout=1.0):
            raise ThrottleError("System is throttled, try again later")

        try:
            # Circuit breaker + execução
            result = self.circuit_breaker.call(func, *args, **kwargs)
            self.throttler.record_success()
            return result

        except Exception as e:
            self.throttler.record_error()
            raise

    @contextmanager
    def resource_context(self, client_id: str = "default"):
        """Context manager para proteção de recursos"""
        if not self.rate_limiter.is_allowed(client_id):
            raise RateLimitError(f"Rate limit exceeded for client {client_id}")

        if not self.throttler.acquire(timeout=1.0):
            raise ThrottleError("System is throttled")

        try:
            yield
            self.throttler.record_success()
        except Exception:
            self.throttler.record_error()
            raise

    def submit_async_task(self, task_id: str, func: Callable, *args, **kwargs) -> str:
        """Submete tarefa assíncrona"""
        return self.worker_pool.submit_task(task_id, func, *args, **kwargs)

    def start_monitoring(self):
        """Inicia todos os sistemas de monitoramento"""
        self.resource_monitor.start_monitoring()
        logger.info("Resource management started")

    def stop_monitoring(self):
        """Para todos os sistemas de monitoramento"""
        self.resource_monitor.stop_monitoring()
        self.worker_pool.shutdown()
        logger.info("Resource management stopped")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas completas"""
        return {
            "resource_usage": self.resource_monitor.get_current_usage(),
            "throttler": self.throttler.get_stats(),
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "worker_pool": self.worker_pool.get_stats(),
            "scaling_factor": self.scaling_factor,
        }

    def optimize_resources(self):
        """Otimiza recursos baseado no uso atual"""
        usage = self.resource_monitor.get_current_usage()

        cpu_usage = usage.get(ResourceType.CPU)
        memory_usage = usage.get(ResourceType.MEMORY)

        if cpu_usage and memory_usage:
            # Calcula fator de escala baseado no uso
            avg_usage = (cpu_usage.percentage + memory_usage.percentage) / 2

            if avg_usage > 80:
                # Alto uso - reduz recursos
                self.scaling_factor = max(0.5, self.scaling_factor * 0.9)
                self.throttler.current_rate = (
                    self.throttler.base_rate * self.scaling_factor
                )
            elif avg_usage < 40:
                # Baixo uso - aumenta recursos
                self.scaling_factor = min(2.0, self.scaling_factor * 1.1)
                self.throttler.current_rate = (
                    self.throttler.base_rate * self.scaling_factor
                )

            logger.debug(
                f"Resource optimization: scaling factor = {self.scaling_factor:.2f}"
            )


# Exceções personalizadas
class ResourceManagementError(Exception):
    """Erro base para gerenciamento de recursos"""

    pass


class ThrottleError(ResourceManagementError):
    """Erro de throttling"""

    pass


class RateLimitError(ResourceManagementError):
    """Erro de rate limiting"""

    pass


class CircuitBreakerOpenError(ResourceManagementError):
    """Erro quando circuit breaker está aberto"""

    pass
