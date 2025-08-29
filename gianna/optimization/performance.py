"""
Sistema de Otimização de Performance para Gianna

Implementa cache de respostas LLM, pool de conexões, processamento paralelo
e monitoramento de métricas para otimizar performance do sistema.
"""

import asyncio
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil
from loguru import logger

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis não disponível - cache apenas em memória")


@dataclass
class PerformanceMetrics:
    """Métricas de performance do sistema"""

    response_time: float
    memory_usage: float
    cpu_usage: float
    cache_hit_rate: float
    active_connections: int
    timestamp: float


class ConnectionPool:
    """Pool de conexões para diferentes providers LLM"""

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.pools: Dict[str, List[Any]] = {}
        self.locks: Dict[str, Lock] = {}
        self.usage_count: Dict[str, int] = {}

    def get_connection(self, provider: str) -> Optional[Any]:
        """Obtém conexão do pool para um provider"""
        if provider not in self.pools:
            self.pools[provider] = []
            self.locks[provider] = Lock()
            self.usage_count[provider] = 0

        with self.locks[provider]:
            if self.pools[provider]:
                connection = self.pools[provider].pop()
                self.usage_count[provider] += 1
                return connection

        # Se não há conexões disponíveis, cria nova se abaixo do limite
        if len(self.pools[provider]) < self.max_connections:
            connection = self._create_connection(provider)
            self.usage_count[provider] += 1
            return connection

        return None

    def return_connection(self, provider: str, connection: Any):
        """Retorna conexão ao pool"""
        if provider in self.pools:
            with self.locks[provider]:
                if len(self.pools[provider]) < self.max_connections:
                    self.pools[provider].append(connection)
                    self.usage_count[provider] -= 1

    def _create_connection(self, provider: str) -> Any:
        """Cria nova conexão para o provider"""
        # Implementação específica seria baseada no provider
        # Por now, retorna placeholder
        return f"connection_{provider}_{time.time()}"

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Obtém estatísticas do pool"""
        stats = {}
        for provider in self.pools:
            with self.locks[provider]:
                stats[provider] = {
                    "available": len(self.pools[provider]),
                    "in_use": self.usage_count[provider],
                    "total": len(self.pools[provider]) + self.usage_count[provider],
                }
        return stats


class LRUCacheManager:
    """Gerenciador de cache LRU com suporte a Redis"""

    def __init__(
        self, max_size: int = 1000, ttl: int = 3600, redis_url: Optional[str] = None
    ):
        self.max_size = max_size
        self.ttl = ttl
        self.redis_client = None
        self._memory_cache = {}
        self._cache_times = {}
        self._lock = Lock()

        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis cache conectado")
            except Exception as e:
                logger.warning(f"Falha ao conectar Redis: {e}")

    @lru_cache(maxsize=128)
    def _generate_cache_key(
        self, input_text: str, model: str, temperature: float = 0.7
    ) -> str:
        """Gera chave única para cache"""
        return f"llm:{model}:{hash(input_text)}:{temperature}"

    def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache"""
        # Verifica cache em memória primeiro
        with self._lock:
            if key in self._memory_cache:
                cache_time = self._cache_times.get(key, 0)
                if time.time() - cache_time < self.ttl:
                    return self._memory_cache[key]
                else:
                    # Cache expirado
                    del self._memory_cache[key]
                    del self._cache_times[key]

        # Verifica Redis se disponível
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    import pickle

                    return pickle.loads(value)
            except Exception as e:
                logger.warning(f"Erro ao ler Redis: {e}")

        return None

    def set(self, key: str, value: Any):
        """Define valor no cache"""
        current_time = time.time()

        # Cache em memória
        with self._lock:
            # Remove entradas antigas se atingiu limite
            if len(self._memory_cache) >= self.max_size:
                oldest_key = min(self._cache_times.keys(), key=self._cache_times.get)
                del self._memory_cache[oldest_key]
                del self._cache_times[oldest_key]

            self._memory_cache[key] = value
            self._cache_times[key] = current_time

        # Cache no Redis se disponível
        if self.redis_client:
            try:
                import pickle

                self.redis_client.setex(key, self.ttl, pickle.dumps(value))
            except Exception as e:
                logger.warning(f"Erro ao escrever Redis: {e}")

    def invalidate(self, pattern: str = None):
        """Invalida entradas do cache"""
        if pattern:
            # Remove entradas que correspondem ao pattern
            with self._lock:
                keys_to_remove = [k for k in self._memory_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self._memory_cache[key]
                    del self._cache_times[key]

            if self.redis_client:
                try:
                    keys = self.redis_client.keys(f"*{pattern}*")
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as e:
                    logger.warning(f"Erro ao invalidar Redis: {e}")
        else:
            # Limpa todo o cache
            with self._lock:
                self._memory_cache.clear()
                self._cache_times.clear()

            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Erro ao limpar Redis: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do cache"""
        with self._lock:
            memory_stats = {
                "size": len(self._memory_cache),
                "max_size": self.max_size,
                "hit_rate": getattr(self, "_hit_count", 0)
                / max(getattr(self, "_access_count", 1), 1),
            }

        redis_stats = {}
        if self.redis_client:
            try:
                info = self.redis_client.info()
                redis_stats = {
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                }
            except Exception as e:
                logger.warning(f"Erro ao obter stats Redis: {e}")

        return {"memory_cache": memory_stats, "redis_cache": redis_stats}


class PerformanceOptimizer:
    """Classe principal para otimização de performance do sistema"""

    def __init__(
        self,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        max_connections: int = 10,
        max_workers: int = 4,
        redis_url: Optional[str] = None,
    ):

        self.cache_manager = LRUCacheManager(cache_size, cache_ttl, redis_url)
        self.connection_pool = ConnectionPool(max_connections)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Métricas
        self.metrics_history: List[PerformanceMetrics] = []
        self.metrics_lock = Lock()

        # Estatísticas
        self._total_requests = 0
        self._cache_hits = 0
        self._start_time = time.time()

        # Monitoramento automático
        self._monitoring_active = False
        self._monitoring_thread = None

        # Cleanup automático
        self._cleanup_refs = weakref.WeakSet()

    def cached_llm_call(
        self,
        input_text: str,
        model: str,
        llm_callable,
        temperature: float = 0.7,
        force_refresh: bool = False,
    ) -> Any:
        """
        Executa chamada LLM com cache inteligente

        Args:
            input_text: Texto de entrada
            model: Nome do modelo
            llm_callable: Função que faz a chamada ao LLM
            temperature: Temperatura do modelo
            force_refresh: Força atualização do cache

        Returns:
            Resposta do LLM (cached ou fresh)
        """
        self._total_requests += 1
        start_time = time.time()

        cache_key = self.cache_manager._generate_cache_key(
            input_text, model, temperature
        )

        # Verifica cache se não forçando refresh
        if not force_refresh:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                logger.debug(f"Cache hit para modelo {model}")
                return cached_result

        # Executa chamada LLM
        try:
            connection = self.connection_pool.get_connection(model)

            if connection:
                try:
                    result = llm_callable(input_text)

                    # Cache o resultado
                    self.cache_manager.set(cache_key, result)

                    # Registra métricas
                    response_time = time.time() - start_time
                    self._record_metrics(response_time)

                    logger.debug(
                        f"Nova resposta LLM para {model} em {response_time:.2f}s"
                    )
                    return result

                finally:
                    self.connection_pool.return_connection(model, connection)
            else:
                # Pool cheio, executa diretamente
                result = llm_callable(input_text)
                self.cache_manager.set(cache_key, result)

                response_time = time.time() - start_time
                self._record_metrics(response_time)

                return result

        except Exception as e:
            logger.error(f"Erro na chamada LLM: {e}")
            raise

    def parallel_processing(
        self, tasks: List[Tuple[callable, tuple]], timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Executa tarefas em paralelo com pool de threads

        Args:
            tasks: Lista de tuplas (função, argumentos)
            timeout: Timeout em segundos

        Returns:
            Lista de resultados na mesma ordem
        """
        if not tasks:
            return []

        start_time = time.time()
        results = [None] * len(tasks)

        # Submete tarefas
        future_to_index = {}
        for i, (func, args) in enumerate(tasks):
            future = self.executor.submit(func, *args)
            future_to_index[future] = i

        # Coleta resultados
        try:
            for future in as_completed(future_to_index, timeout=timeout):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Erro na tarefa {index}: {e}")
                    results[index] = e

        except asyncio.TimeoutError:
            logger.warning(f"Timeout em processamento paralelo após {timeout}s")

        processing_time = time.time() - start_time
        logger.debug(f"Processamento paralelo completado em {processing_time:.2f}s")

        return results

    def _record_metrics(self, response_time: float):
        """Registra métricas de performance"""
        try:
            # Métricas do sistema
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()

            # Taxa de hit do cache
            cache_hit_rate = self._cache_hits / max(self._total_requests, 1)

            # Conexões ativas
            pool_stats = self.connection_pool.get_stats()
            active_connections = sum(stats["in_use"] for stats in pool_stats.values())

            metrics = PerformanceMetrics(
                response_time=response_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                cache_hit_rate=cache_hit_rate,
                active_connections=active_connections,
                timestamp=time.time(),
            )

            with self.metrics_lock:
                self.metrics_history.append(metrics)

                # Mantém apenas últimas 1000 métricas
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)

        except Exception as e:
            logger.warning(f"Erro ao registrar métricas: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas completas de performance"""
        with self.metrics_lock:
            recent_metrics = self.metrics_history[-100:] if self.metrics_history else []

        if not recent_metrics:
            return {
                "total_requests": self._total_requests,
                "cache_hit_rate": 0.0,
                "uptime": time.time() - self._start_time,
            }

        avg_response_time = sum(m.response_time for m in recent_metrics) / len(
            recent_metrics
        )
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)

        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(self._total_requests, 1),
            "avg_response_time": avg_response_time,
            "avg_memory_usage": avg_memory,
            "avg_cpu_usage": avg_cpu,
            "uptime": time.time() - self._start_time,
            "cache_stats": self.cache_manager.get_stats(),
            "connection_pool_stats": self.connection_pool.get_stats(),
            "thread_pool_active": len(
                [t for t in self.executor._threads if t.is_alive()]
            ),
        }

    def optimize_for_workload(self, workload_type: str = "balanced"):
        """
        Otimiza configurações baseado no tipo de workload

        Args:
            workload_type: 'cpu_intensive', 'io_intensive', 'balanced'
        """
        if workload_type == "cpu_intensive":
            # Mais threads, cache menor
            self.max_workers = min(psutil.cpu_count(), 8)
            self.cache_manager.max_size = 500

        elif workload_type == "io_intensive":
            # Mais conexões, cache maior
            self.connection_pool.max_connections = 20
            self.cache_manager.max_size = 2000

        elif workload_type == "balanced":
            # Configuração equilibrada
            self.max_workers = psutil.cpu_count() // 2
            self.cache_manager.max_size = 1000

        # Recria executor se necessário
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        logger.info(f"Otimização aplicada para workload: {workload_type}")

    def start_monitoring(self, interval: int = 30):
        """Inicia monitoramento automático de performance"""
        if self._monitoring_active:
            return

        self._monitoring_active = True

        def monitor():
            while self._monitoring_active:
                try:
                    self._record_metrics(0.0)  # Registra métricas do sistema
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Erro no monitoramento: {e}")

        self._monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self._monitoring_thread.start()
        logger.info(f"Monitoramento iniciado (intervalo: {interval}s)")

    def stop_monitoring(self):
        """Para o monitoramento automático"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Monitoramento parado")

    def cleanup(self):
        """Limpeza de recursos"""
        self.stop_monitoring()

        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        # Limpa cache se necessário
        if (
            hasattr(self.cache_manager, "redis_client")
            and self.cache_manager.redis_client
        ):
            try:
                self.cache_manager.redis_client.close()
            except:
                pass

        logger.info("Recursos de performance limpos")

    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
