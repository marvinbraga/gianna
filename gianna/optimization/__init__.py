"""
Sistema de Otimização de Performance para Gianna

Este módulo fornece ferramentas abrangentes para otimização de performance,
incluindo cache multi-layer, monitoramento em tempo real, gerenciamento de recursos
e proteção contra falhas.

Componentes principais:
- PerformanceOptimizer: Otimização principal com cache LLM e processamento paralelo
- MultiLayerCache: Sistema de cache inteligente (memória + Redis + SQLite)
- PerformanceMonitor: Monitoramento em tempo real com alertas
- ResourceManager: Gerenciamento de recursos com throttling e circuit breaker

Exemplo de uso básico:
    from gianna.optimization import PerformanceOptimizer, PerformanceMonitor

    # Setup básico
    optimizer = PerformanceOptimizer(cache_size=1000, max_workers=4)
    monitor = PerformanceMonitor()

    # Usar cache para chamadas LLM
    def my_llm_call(text):
        return "response"

    cached_result = optimizer.cached_llm_call("input", "gpt-4", my_llm_call)

    # Monitoramento
    monitor.start_monitoring()
    monitor.increment_counter("api_calls")

Exemplo avançado:
    from gianna.optimization import (
        PerformanceOptimizer, PerformanceMonitor,
        ResourceManager, MultiLayerCache
    )

    # Setup completo
    cache = MultiLayerCache(
        memory_size=1000,
        redis_url="redis://localhost:6379",
        sqlite_path="gianna_cache.db"
    )

    optimizer = PerformanceOptimizer(
        cache_size=2000,
        max_workers=8,
        redis_url="redis://localhost:6379"
    )

    monitor = PerformanceMonitor(enable_profiling=True)
    resource_mgr = ResourceManager(max_workers=8, throttle_rate=20.0)

    # Context managers para proteção
    with resource_mgr.resource_context("user123"):
        result = optimizer.cached_llm_call("query", "gpt-4", llm_function)
        monitor.increment_counter("protected_calls")
"""

from .caching import (
    CacheEntry,
    CacheLayer,
    CacheWarmer,
    MemoryCache,
    MultiLayerCache,
    RedisCache,
    SQLiteCache,
)
from .monitoring import (
    Alert,
    AlertLevel,
    AlertManager,
    MetricData,
    MetricType,
    PerformanceDashboard,
    PerformanceMonitor,
    Profiler,
    TimerContext,
)
from .performance import (
    ConnectionPool,
    LRUCacheManager,
    PerformanceMetrics,
    PerformanceOptimizer,
)
from .resource_management import (  # Exceções
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    RateLimiter,
    RateLimitError,
    ResourceManagementError,
    ResourceManager,
    ResourceMonitor,
    ResourceType,
    ResourceUsage,
    ThrottleError,
    Throttler,
    WorkerPool,
    WorkerTask,
)

# Versão do módulo
__version__ = "1.0.0"

# Classes principais para importação simplificada
__all__ = [
    # Performance
    "PerformanceOptimizer",
    "PerformanceMetrics",
    "ConnectionPool",
    "LRUCacheManager",
    # Caching
    "MultiLayerCache",
    "CacheEntry",
    "CacheLayer",
    "MemoryCache",
    "RedisCache",
    "SQLiteCache",
    "CacheWarmer",
    # Monitoring
    "PerformanceMonitor",
    "PerformanceDashboard",
    "AlertManager",
    "Alert",
    "AlertLevel",
    "MetricType",
    "MetricData",
    "TimerContext",
    "Profiler",
    # Resource Management
    "ResourceManager",
    "ResourceMonitor",
    "Throttler",
    "CircuitBreaker",
    "RateLimiter",
    "WorkerPool",
    "ResourceType",
    "CircuitState",
    "ResourceUsage",
    "WorkerTask",
    # Exceptions
    "ResourceManagementError",
    "ThrottleError",
    "RateLimitError",
    "CircuitBreakerOpenError",
]


def create_default_optimizer(
    redis_url: str = None, cache_size: int = 1000, max_workers: int = 4
) -> PerformanceOptimizer:
    """
    Cria otimizador com configuração padrão

    Args:
        redis_url: URL do Redis (opcional)
        cache_size: Tamanho do cache em memória
        max_workers: Número máximo de workers

    Returns:
        PerformanceOptimizer configurado
    """
    return PerformanceOptimizer(
        cache_size=cache_size, max_workers=max_workers, redis_url=redis_url
    )


def create_default_monitor(
    enable_profiling: bool = True, collection_interval: int = 30
) -> PerformanceMonitor:
    """
    Cria monitor com configuração padrão

    Args:
        enable_profiling: Habilita profiling automático
        collection_interval: Intervalo de coleta em segundos

    Returns:
        PerformanceMonitor configurado
    """
    return PerformanceMonitor(
        enable_profiling=enable_profiling, collection_interval=collection_interval
    )


def create_default_resource_manager(
    max_workers: int = 4, throttle_rate: float = 10.0, rate_limit_per_minute: int = 60
) -> ResourceManager:
    """
    Cria gerenciador de recursos com configuração padrão

    Args:
        max_workers: Número máximo de workers
        throttle_rate: Taxa de throttling (req/s)
        rate_limit_per_minute: Rate limit por minuto

    Returns:
        ResourceManager configurado
    """
    return ResourceManager(
        max_workers=max_workers,
        throttle_rate=throttle_rate,
        rate_limit_per_minute=rate_limit_per_minute,
    )


def create_complete_optimization_suite(redis_url: str = None) -> dict:
    """
    Cria suite completa de otimização com todos os componentes

    Args:
        redis_url: URL do Redis (opcional)

    Returns:
        Dict com todos os componentes configurados
    """
    # Cache multi-layer
    cache = MultiLayerCache(
        memory_size=1000, redis_url=redis_url, sqlite_path="gianna_optimization.db"
    )

    # Otimizador principal
    optimizer = PerformanceOptimizer(
        cache_size=1000, max_workers=4, redis_url=redis_url
    )

    # Monitor de performance
    monitor = PerformanceMonitor(enable_profiling=True, collection_interval=30)

    # Gerenciador de recursos
    resource_mgr = ResourceManager(
        max_workers=4, throttle_rate=10.0, rate_limit_per_minute=60
    )

    return {
        "cache": cache,
        "optimizer": optimizer,
        "monitor": monitor,
        "resource_manager": resource_mgr,
    }


# Configuração de logging padrão
import logging

from loguru import logger


def setup_optimization_logging(level: str = "INFO", file_path: str = None):
    """
    Configura logging para o módulo de otimização

    Args:
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
        file_path: Caminho para arquivo de log (opcional)
    """
    # Remove handlers padrão do loguru
    logger.remove()

    # Adiciona handler para console
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>optimization</cyan> | <level>{message}</level>",
        colorize=True,
    )

    # Adiciona handler para arquivo se especificado
    if file_path:
        logger.add(
            file_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | optimization | {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )
