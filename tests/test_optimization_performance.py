"""
Testes de Performance para Sistema de Otimização Gianna

Testes completos para validar funcionalidade e performance do sistema de otimização,
incluindo cache, monitoramento, gerenciamento de recursos e proteção contra falhas.
"""

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import pytest

# Import dos módulos de otimização
from gianna.optimization import (
    AlertLevel,
    CacheEntry,
    CircuitBreakerOpenError,
    CircuitState,
    MemoryCache,
    MultiLayerCache,
    PerformanceMonitor,
    PerformanceOptimizer,
    RateLimitError,
    ResourceManager,
    ResourceType,
    ThrottleError,
)


class TestPerformanceOptimizer:
    """Testes para PerformanceOptimizer"""

    def test_creation(self):
        """Testa criação do otimizador"""
        optimizer = PerformanceOptimizer(cache_size=100, max_workers=2)
        assert optimizer.cache_manager.max_size == 100
        assert optimizer.max_workers == 2

    def test_cached_llm_call_hit(self):
        """Testa cache hit em chamada LLM"""
        optimizer = PerformanceOptimizer(cache_size=100)

        def mock_llm(text):
            return f"response_to_{text}"

        # Primeira chamada
        result1 = optimizer.cached_llm_call("test", "model1", mock_llm)
        assert result1 == "response_to_test"

        # Segunda chamada (deve usar cache)
        result2 = optimizer.cached_llm_call("test", "model1", mock_llm)
        assert result2 == "response_to_test"

        # Verifica estatísticas
        stats = optimizer.get_performance_stats()
        assert stats["cache_hits"] > 0

    def test_cached_llm_call_miss(self):
        """Testa cache miss em chamada LLM"""
        optimizer = PerformanceOptimizer(cache_size=100)

        def mock_llm(text):
            return f"response_to_{text}"

        # Chamadas diferentes
        result1 = optimizer.cached_llm_call("test1", "model1", mock_llm)
        result2 = optimizer.cached_llm_call("test2", "model1", mock_llm)

        assert result1 == "response_to_test1"
        assert result2 == "response_to_test2"

        stats = optimizer.get_performance_stats()
        assert stats["total_requests"] == 2

    def test_parallel_processing(self):
        """Testa processamento paralelo"""
        optimizer = PerformanceOptimizer(max_workers=3)

        def slow_task(n):
            time.sleep(0.1)  # Simula operação lenta
            return n * 2

        tasks = [(slow_task, (i,)) for i in range(5)]

        start_time = time.time()
        results = optimizer.parallel_processing(tasks)
        end_time = time.time()

        # Verifica resultados
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result == i * 2

        # Deve ser mais rápido que execução sequencial
        assert end_time - start_time < 0.5  # 5 * 0.1 seria 0.5s sequencial

    def test_context_manager(self):
        """Testa uso como context manager"""
        with PerformanceOptimizer() as optimizer:
            assert optimizer._monitoring_active

        # Deve ter parado o monitoramento
        assert not optimizer._monitoring_active

    def test_optimize_for_workload(self):
        """Testa otimização por tipo de workload"""
        optimizer = PerformanceOptimizer()

        # CPU intensive
        optimizer.optimize_for_workload("cpu_intensive")
        assert optimizer.cache_manager.max_size == 500

        # IO intensive
        optimizer.optimize_for_workload("io_intensive")
        assert optimizer.connection_pool.max_connections == 20
        assert optimizer.cache_manager.max_size == 2000


class TestMultiLayerCache:
    """Testes para sistema de cache multi-layer"""

    def test_memory_cache_basic(self):
        """Testa operações básicas do cache em memória"""
        cache = MemoryCache(max_size=3)

        # Set e Get
        entry = CacheEntry("key1", "value1", time.time(), 3600)
        cache.set(entry)

        result = cache.get("key1")
        assert result is not None
        assert result.value == "value1"

        # Miss
        result = cache.get("nonexistent")
        assert result is None

    def test_memory_cache_lru_eviction(self):
        """Testa eviction LRU"""
        cache = MemoryCache(max_size=2)

        # Adiciona 3 entradas (deve evict a primeira)
        entries = [
            CacheEntry("key1", "value1", time.time(), 3600),
            CacheEntry("key2", "value2", time.time(), 3600),
            CacheEntry("key3", "value3", time.time(), 3600),
        ]

        for entry in entries:
            cache.set(entry)

        # key1 deve ter sido removida
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_cache_expiration(self):
        """Testa expiração de cache"""
        cache = MemoryCache()

        # Entry expirada
        expired_entry = CacheEntry(
            "expired", "value", time.time() - 3600, 1800
        )  # TTL 30min, criada há 1h
        cache.set(expired_entry)

        result = cache.get("expired")
        assert result is None

    def test_multilayer_cache_creation(self):
        """Testa criação de cache multi-layer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = os.path.join(tmpdir, "test.db")
            cache = MultiLayerCache(memory_size=100, sqlite_path=sqlite_path)

            assert len(cache.layers) >= 2  # Memory + SQLite

    def test_multilayer_cache_operations(self):
        """Testa operações em cache multi-layer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sqlite_path = os.path.join(tmpdir, "test.db")
            cache = MultiLayerCache(memory_size=100, sqlite_path=sqlite_path)

            # Set
            cache.set("test_key", "test_value", ttl=3600)

            # Get
            result = cache.get("test_key")
            assert result == "test_value"

            # Delete
            cache.delete("test_key")
            result = cache.get("test_key")
            assert result is None


class TestPerformanceMonitor:
    """Testes para monitor de performance"""

    def test_creation(self):
        """Testa criação do monitor"""
        monitor = PerformanceMonitor(collection_interval=10)
        assert monitor.collection_interval == 10
        assert not monitor.active

    def test_metrics_collection(self):
        """Testa coleta de métricas"""
        monitor = PerformanceMonitor()

        # Incrementa algumas métricas
        monitor.increment_counter("test_counter", 5)
        monitor.set_gauge("test_gauge", 42.5)
        monitor.record_histogram("test_histogram", 100.0)
        monitor.record_timer("test_timer", 250.5)

        # Coleta métricas
        metrics = monitor.app_collector.collect()

        metric_names = [m.name for m in metrics]
        assert "test_counter" in metric_names
        assert "test_gauge" in metric_names
        assert "test_histogram.mean" in metric_names
        assert "test_timer.mean" in metric_names

    def test_timer_context(self):
        """Testa context manager para timing"""
        monitor = PerformanceMonitor()

        with monitor.timer_context("test_operation"):
            time.sleep(0.05)  # 50ms

        # Verifica se timer foi registrado
        metrics = monitor.app_collector.collect()
        timer_metrics = [m for m in metrics if "test_operation" in m.name]
        assert len(timer_metrics) > 0

        # Deve ter registrado tempo próximo a 50ms
        mean_metric = next(m for m in timer_metrics if m.name.endswith(".mean"))
        assert 40 < mean_metric.value < 80  # Entre 40-80ms (margem de erro)

    def test_monitoring_lifecycle(self):
        """Testa ciclo de vida do monitoramento"""
        monitor = PerformanceMonitor(collection_interval=1)

        # Inicia
        monitor.start_monitoring()
        assert monitor.active

        # Aguarda coleta
        time.sleep(1.5)

        # Para
        monitor.stop_monitoring()
        assert not monitor.active

    def test_alert_system(self):
        """Testa sistema de alertas"""
        monitor = PerformanceMonitor()
        alerts_received = []

        def alert_handler(alert):
            alerts_received.append(alert)

        # Adiciona regra de alerta
        monitor.alert_manager.add_rule(
            "test_metric",
            ">",
            100,
            AlertLevel.WARNING,
            "Test Alert",
            "Value is {current_value}",
        )
        monitor.alert_manager.add_handler(AlertLevel.WARNING, alert_handler)

        # Simula métrica que deve disparar alerta
        from gianna.optimization.monitoring import MetricData, MetricType

        metric = MetricData("test_metric", MetricType.GAUGE, 150, time.time())
        monitor.alert_manager.check_metrics([metric])

        # Verifica se alerta foi disparado
        assert len(alerts_received) == 1
        assert alerts_received[0].level == AlertLevel.WARNING


class TestResourceManager:
    """Testes para gerenciador de recursos"""

    def test_creation(self):
        """Testa criação do gerenciador"""
        mgr = ResourceManager(max_workers=4, throttle_rate=5.0)
        assert mgr.worker_pool.max_workers == 4
        assert mgr.throttler.base_rate == 5.0

    def test_throttling(self):
        """Testa sistema de throttling"""
        mgr = ResourceManager(throttle_rate=2.0)  # 2 req/s

        # Primeiras requisições devem passar
        assert mgr.throttler.acquire()
        assert mgr.throttler.acquire()

        # Próximas podem ser throttled (dependendo do burst)
        start_time = time.time()
        for _ in range(10):
            mgr.throttler.acquire(timeout=0.1)
        end_time = time.time()

        # Deve ter levado tempo devido ao throttling
        assert end_time - start_time > 0.5

    def test_rate_limiting(self):
        """Testa rate limiting por cliente"""
        mgr = ResourceManager(rate_limit_per_minute=5)

        client_id = "test_client"

        # Primeiras 5 requisições devem passar
        for i in range(5):
            assert mgr.rate_limiter.is_allowed(client_id)

        # 6ª requisição deve ser rejeitada
        assert not mgr.rate_limiter.is_allowed(client_id)

        # Verifica remaining
        assert mgr.rate_limiter.get_remaining(client_id) == 0

    def test_circuit_breaker(self):
        """Testa circuit breaker"""
        mgr = ResourceManager()
        cb = mgr.circuit_breaker

        def failing_function():
            raise Exception("Test failure")

        # Circuit deve estar fechado inicialmente
        assert cb.get_state() == CircuitState.CLOSED

        # Força falhas até abrir circuit
        for _ in range(6):  # failure_threshold = 5
            try:
                cb.call(failing_function)
            except:
                pass

        # Circuit deve estar aberto
        assert cb.get_state() == CircuitState.OPEN

        # Próxima chamada deve ser rejeitada
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(failing_function)

    def test_worker_pool(self):
        """Testa pool de workers"""
        mgr = ResourceManager(max_workers=2)

        def slow_task(n):
            time.sleep(0.1)
            return n * 2

        # Submete tarefas
        task_ids = []
        for i in range(3):
            task_id = mgr.submit_async_task(f"task_{i}", slow_task, i)
            task_ids.append(task_id)

        # Aguarda conclusão
        time.sleep(0.5)

        # Verifica resultados
        for i, task_id in enumerate(task_ids):
            status = mgr.worker_pool.get_task_status(task_id)
            assert status == "completed"

            result = mgr.worker_pool.get_task_result(task_id)
            assert result == i * 2

    def test_resource_protection(self):
        """Testa proteção completa de recursos"""
        mgr = ResourceManager(throttle_rate=10.0, rate_limit_per_minute=100)

        def test_function():
            return "success"

        # Deve executar normalmente
        result = mgr.execute_with_protection(test_function, "client1")
        assert result == "success"

        # Context manager
        with mgr.resource_context("client2"):
            result = test_function()
            assert result == "success"

    def test_resource_monitoring(self):
        """Testa monitoramento de recursos"""
        mgr = ResourceManager()

        # Inicia monitoramento
        mgr.start_monitoring()

        # Aguarda coleta
        time.sleep(1.5)

        # Obtém estatísticas
        stats = mgr.get_comprehensive_stats()

        assert "resource_usage" in stats
        assert "throttler" in stats
        assert "circuit_breaker" in stats
        assert "worker_pool" in stats

        # Para monitoramento
        mgr.stop_monitoring()


class TestPerformanceBenchmarks:
    """Benchmarks de performance do sistema"""

    def test_cache_performance(self):
        """Benchmark de performance do cache"""
        cache = MemoryCache(max_size=1000)

        # Benchmark set operations
        start_time = time.time()
        for i in range(1000):
            entry = CacheEntry(f"key_{i}", f"value_{i}", time.time(), 3600)
            cache.set(entry)
        set_time = time.time() - start_time

        # Benchmark get operations
        start_time = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        get_time = time.time() - start_time

        print(
            f"Cache performance: {1000/set_time:.0f} sets/s, {1000/get_time:.0f} gets/s"
        )

        # Assertions básicas (deve ser rápido)
        assert set_time < 1.0  # Menos de 1s para 1000 sets
        assert get_time < 1.0  # Menos de 1s para 1000 gets

    def test_concurrent_cache_access(self):
        """Teste de acesso concorrente ao cache"""
        cache = MemoryCache(max_size=1000)
        results = []

        def cache_operations():
            for i in range(100):
                entry = CacheEntry(f"key_{i}", f"value_{i}", time.time(), 3600)
                cache.set(entry)
                result = cache.get(f"key_{i}")
                results.append(result is not None)

        # Executa operações concorrentes
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=cache_operations)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verifica que todas as operações foram bem-sucedidas
        assert all(results)
        assert len(results) == 500  # 5 threads * 100 ops

    def test_llm_cache_effectiveness(self):
        """Teste de efetividade do cache LLM"""
        optimizer = PerformanceOptimizer(cache_size=100)

        call_count = 0

        def mock_llm(text):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simula latência LLM
            return f"response_to_{text}"

        # Faz chamadas repetidas
        texts = ["query1", "query2", "query1", "query3", "query2", "query1"]

        start_time = time.time()
        for text in texts:
            optimizer.cached_llm_call(text, "gpt-4", mock_llm)
        end_time = time.time()

        # Verifica que cache foi efetivo
        assert call_count == 3  # Apenas chamadas únicas

        stats = optimizer.get_performance_stats()
        hit_rate = stats["cache_hit_rate"]
        assert hit_rate > 0.4  # Mais de 40% hit rate

        print(
            f"LLM Cache effectiveness: {hit_rate:.1%} hit rate, {call_count} unique calls for {len(texts)} requests"
        )


if __name__ == "__main__":
    # Executa testes básicos se rodado diretamente
    print("Executando testes de performance do sistema de otimização...")

    # Teste básico do otimizador
    optimizer = PerformanceOptimizer()

    def test_llm(text):
        return f"AI response to: {text}"

    result = optimizer.cached_llm_call("test input", "gpt-4", test_llm)
    print(f"✓ Cache LLM: {result}")

    # Teste básico do monitor
    monitor = PerformanceMonitor()
    monitor.increment_counter("test_counter")
    monitor.set_gauge("test_gauge", 42.0)
    print("✓ Monitoramento funcionando")

    # Teste básico do gerenciador de recursos
    resource_mgr = ResourceManager()
    with resource_mgr.resource_context():
        print("✓ Proteção de recursos funcionando")

    print(
        "\nTodos os testes básicos passaram! Execute 'pytest tests/test_optimization_performance.py -v' para testes completos."
    )
