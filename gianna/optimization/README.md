# Sistema de Otimização de Performance Gianna

Sistema abrangente de otimização de performance para o framework Gianna, fornecendo cache inteligente, monitoramento em tempo real, gerenciamento de recursos e proteção contra falhas.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Componentes Principais](#componentes-principais)
- [Instalação](#instalação)
- [Uso Básico](#uso-básico)
- [Configuração Avançada](#configuração-avançada)
- [Monitoramento](#monitoramento)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## 🎯 Visão Geral

O Sistema de Otimização de Performance é projetado para:

- **Acelerar chamadas LLM** com cache inteligente multi-layer
- **Monitorar performance** em tempo real com alertas automáticos
- **Gerenciar recursos** com throttling e circuit breaker
- **Processar tarefas** em paralelo com pool de workers
- **Proteger contra falhas** com rate limiting e recuperação automática

### Benefícios Principais

- ⚡ **50-90% redução** no tempo de resposta (cache hits)
- 📊 **Monitoramento completo** com métricas e alertas
- 🛡️ **Proteção robusta** contra sobrecarga e falhas
- 🔧 **Configuração flexível** para diferentes cenários
- 📈 **Escalabilidade automática** baseada em carga

## 🏗️ Componentes Principais

### 1. PerformanceOptimizer
**Classe principal de otimização**

```python
from gianna.optimization import PerformanceOptimizer

optimizer = PerformanceOptimizer(
    cache_size=1000,
    max_workers=4,
    redis_url="redis://localhost:6379"
)

# Cache LLM com fallback automático
result = optimizer.cached_llm_call("input", "gpt-4", llm_function)

# Processamento paralelo
tasks = [(func1, (arg1,)), (func2, (arg2,))]
results = optimizer.parallel_processing(tasks)
```

**Características:**
- Cache LRU em memória com fallback Redis
- Pool de conexões para providers LLM
- Processamento paralelo com timeout
- Métricas de performance integradas
- Context manager com cleanup automático

### 2. MultiLayerCache
**Sistema de cache inteligente em múltiplas camadas**

```python
from gianna.optimization import MultiLayerCache

cache = MultiLayerCache(
    memory_size=1000,
    redis_url="redis://localhost:6379",
    sqlite_path="cache.db"
)

# Operações básicas
cache.set("key", value, ttl=3600, tags={"user", "session"})
result = cache.get("key")
cache.invalidate_by_tags({"user"})
```

**Camadas (em ordem de velocidade):**
1. **Memória**: LRU cache com eviction inteligente
2. **Redis**: Cache distribuído com compressão
3. **SQLite**: Persistência local para long-term cache

**Características:**
- Promoção automática entre camadas
- TTL configurável por entrada
- Invalidação por tags
- Cache warming automático
- Estatísticas detalhadas

### 3. PerformanceMonitor
**Monitoramento em tempo real com alertas**

```python
from gianna.optimization import PerformanceMonitor

monitor = PerformanceMonitor(enable_profiling=True)
monitor.start_monitoring()

# Métricas customizadas
monitor.increment_counter("api_calls")
monitor.set_gauge("active_users", 42)
monitor.record_histogram("request_size", 1024)

# Context manager para timing
with monitor.timer_context("database_query"):
    result = db.query("SELECT * FROM users")
```

**Tipos de Métricas:**
- **Counter**: Valores incrementais (requests, errors)
- **Gauge**: Valores instantâneos (CPU, memory)
- **Histogram**: Distribuições (response times, sizes)
- **Timer**: Medição de duração automática

**Sistema de Alertas:**
- Regras configuráveis por métrica
- Múltiplos níveis (INFO, WARNING, CRITICAL)
- Handlers customizáveis
- Cooldown automático

### 4. ResourceManager
**Gerenciamento avançado de recursos**

```python
from gianna.optimization import ResourceManager

resource_mgr = ResourceManager(
    max_workers=8,
    throttle_rate=20.0,
    rate_limit_per_minute=100
)

# Execução protegida
with resource_mgr.resource_context("user123"):
    result = expensive_operation()

# Worker pool assíncrono
task_id = resource_mgr.submit_async_task("task1", cpu_intensive_func, arg1, arg2)
result = resource_mgr.worker_pool.get_task_result(task_id)
```

**Componentes:**
- **Throttler**: Controle de taxa com adaptive throttling
- **Circuit Breaker**: Proteção contra falhas em cascata
- **Rate Limiter**: Sliding window por cliente
- **Worker Pool**: Processamento assíncrono com prioridades

## 🚀 Instalação

### Dependências Básicas
```bash
pip install psutil loguru
```

### Dependências Opcionais
```bash
# Para Redis cache
pip install redis

# Para plotting de métricas
pip install matplotlib numpy

# Para compressão avançada
pip install lz4
```

### Instalação via Poetry (recomendado)
```bash
cd gianna
poetry install
```

## 📖 Uso Básico

### Setup Rápido

```python
from gianna.optimization import create_complete_optimization_suite

# Cria suite completa
suite = create_complete_optimization_suite(redis_url="redis://localhost:6379")

optimizer = suite['optimizer']
monitor = suite['monitor']
resource_mgr = suite['resource_manager']
cache = suite['cache']

# Inicia monitoramento
monitor.start_monitoring()
resource_mgr.start_monitoring()
```

### Cache para LLM

```python
def my_llm_call(input_text):
    # Sua implementação de chamada LLM
    return llm_chain.invoke({"input": input_text})

# Primeira chamada (lenta)
result1 = optimizer.cached_llm_call("Hello", "gpt-4", my_llm_call)

# Segunda chamada (rápida - cache hit)
result2 = optimizer.cached_llm_call("Hello", "gpt-4", my_llm_call)

print(f"Cache hit rate: {optimizer.get_performance_stats()['cache_hit_rate']:.1%}")
```

### Monitoramento Simples

```python
# Registra métricas
monitor.increment_counter("requests_total", labels={"endpoint": "/api/chat"})
monitor.set_gauge("active_connections", 15)

# Timing automático
with monitor.timer_context("llm_processing"):
    response = process_llm_request(user_input)

# Gera relatório
report = monitor.dashboard.generate_report(minutes=60)
print(f"Requests processadas: {report['metrics']['requests_total']['count']}")
```

### Proteção de Recursos

```python
# Context manager com proteção completa
try:
    with resource_mgr.resource_context(client_id="user123"):
        result = expensive_computation()
except RateLimitError:
    print("Rate limit exceeded")
except ThrottleError:
    print("System is throttled")
```

## ⚙️ Configuração Avançada

### Configuração por Ambiente

```python
# Desenvolvimento
optimizer_dev = PerformanceOptimizer(
    cache_size=100,
    max_workers=2,
    cache_ttl=1800
)

# Produção
optimizer_prod = PerformanceOptimizer(
    cache_size=5000,
    max_workers=8,
    cache_ttl=3600,
    redis_url="redis://redis-cluster:6379"
)

# Alto volume
optimizer_hv = PerformanceOptimizer(
    cache_size=10000,
    max_workers=16,
    cache_ttl=7200,
    redis_url="redis://redis-cluster:6379"
)
```

### Configuração de Alertas

```python
monitor = PerformanceMonitor()

# Alerta para CPU alto
monitor.alert_manager.add_rule(
    metric_name="system.cpu.usage_percent",
    condition=">",
    threshold=80,
    level=AlertLevel.WARNING,
    title="High CPU Usage",
    message="CPU usage is {current_value}%, threshold: {threshold}%"
)

# Handler customizado
def slack_alert_handler(alert):
    send_slack_message(f"🚨 {alert.title}: {alert.message}")

monitor.alert_manager.add_handler(AlertLevel.CRITICAL, slack_alert_handler)
```

### Cache Warming

```python
cache = MultiLayerCache()

# Adiciona padrões de warming
cache.cache_warmer.add_warming_pattern(
    key_pattern="common_query_*",
    callable_func=lambda: preload_common_responses(),
    interval=3600,  # A cada hora
    tags={"warming", "common"}
)

cache.start_background_tasks()
```

### Configuração de Throttling Adaptativo

```python
resource_mgr = ResourceManager()

# Throttling que se adapta à performance
throttler = resource_mgr.throttler
throttler.adaptive = True  # Habilita adaptação automática

# Callbacks para condições de recursos
def cpu_callback(usage):
    if usage.percentage > 90:
        throttler.current_rate = throttler.base_rate * 0.5

resource_mgr.resource_monitor.add_threshold_callback(
    ResourceType.CPU,
    cpu_callback
)
```

## 📊 Monitoramento

### Métricas do Sistema

O sistema coleta automaticamente:

**Métricas de Sistema:**
- `system.cpu.usage_percent`: Uso de CPU (%)
- `system.memory.usage_percent`: Uso de memória (%)
- `system.disk.usage_percent`: Uso de disco (%)
- `system.network.bytes_sent/recv`: Tráfego de rede

**Métricas de Aplicação:**
- `application.llm.response_time`: Tempo de resposta LLM
- `application.cache.hit_rate`: Taxa de cache hit
- `application.requests.total`: Total de requests
- `application.errors.total`: Total de erros

### Dashboard de Métricas

```python
# Relatório completo
report = monitor.dashboard.generate_report(minutes=60)

print(f"Período: {report['time_range_minutes']} minutos")
for metric, stats in report['metrics'].items():
    print(f"{metric}:")
    print(f"  Média: {stats['mean']:.2f}")
    print(f"  Min/Max: {stats['min']:.2f}/{stats['max']:.2f}")
```

### Visualização (se matplotlib disponível)

```python
# Gera gráfico de métrica
monitor.dashboard.plot_metric(
    "system.cpu.usage_percent",
    minutes=120,
    save_path="cpu_usage.png"
)
```

### Profiling Automático

```python
monitor = PerformanceMonitor(enable_profiling=True)

# Operações lentas são automaticamente perfiladas
slow_operations = monitor.profiler.get_slow_operations(limit=10)

for op in slow_operations:
    print(f"Função: {op['function']}")
    print(f"Duração: {op['duration']:.2f}s")
    print(f"Profile:\n{op['profile'][:500]}...")
```

## ⚡ Performance

### Benchmarks Típicos

**Cache Performance:**
- Memória: ~100k ops/sec
- Redis: ~10k ops/sec
- SQLite: ~1k ops/sec

**LLM Caching:**
- Cache hit: < 1ms
- Cache miss: tempo original da chamada
- Hit rate típica: 60-80% em produção

**Resource Management:**
- Throttling overhead: < 0.1ms
- Circuit breaker overhead: < 0.01ms
- Rate limiting overhead: < 0.1ms

### Otimizações Disponíveis

```python
# Otimização automática por workload
optimizer.optimize_for_workload("cpu_intensive")  # Mais threads
optimizer.optimize_for_workload("io_intensive")   # Mais conexões
optimizer.optimize_for_workload("balanced")       # Equilibrado

# Otimização manual de recursos
resource_mgr.optimize_resources()  # Ajusta baseado no uso atual
```

### Tuning para Alto Volume

```python
# Configuração para milhares de requests/minuto
high_volume_optimizer = PerformanceOptimizer(
    cache_size=50000,      # Cache muito grande
    max_workers=32,        # Muitos workers
    cache_ttl=14400,       # TTL longo (4h)
    redis_url="redis://redis-cluster:6379"
)

# Resource manager agressivo
high_volume_resources = ResourceManager(
    max_workers=32,
    throttle_rate=1000.0,   # 1000 req/s
    rate_limit_per_minute=5000
)
```

## 🔧 Troubleshooting

### Problemas Comuns

**1. Cache Hit Rate Baixo**
```python
# Verifica TTL
stats = optimizer.cache_manager.get_stats()
print(f"TTL atual: {optimizer.cache_manager.ttl}s")

# Aumenta TTL se apropriado
optimizer.cache_manager.ttl = 7200  # 2 horas

# Verifica se chaves estão sendo geradas consistentemente
# (mesmo input deve gerar mesma chave)
```

**2. Circuit Breaker Aberto Frequentemente**
```python
cb_stats = resource_mgr.circuit_breaker.get_stats()
print(f"Estado: {cb_stats['state']}")
print(f"Falhas: {cb_stats['failure_count']}/{cb_stats['failure_threshold']}")

# Ajusta threshold se necessário
resource_mgr.circuit_breaker.failure_threshold = 10
```

**3. Throttling Excessivo**
```python
throttle_stats = resource_mgr.throttler.get_stats()
print(f"Taxa atual: {throttle_stats['current_rate']:.2f} req/s")
print(f"Taxa de throttling: {throttle_stats['throttle_rate']:.1%}")

# Ajusta taxa base
resource_mgr.throttler.base_rate = 50.0
```

**4. Memória Alta**
```python
# Verifica uso de cache
cache_stats = optimizer.cache_manager.get_stats()
memory_usage = cache_stats['memory_cache']['size']

# Limpa cache se necessário
optimizer.cache_manager.invalidate()

# Reduz tamanho do cache
optimizer.cache_manager.max_size = 500
```

### Debug e Logging

```python
# Habilita logging detalhado
from gianna.optimization import setup_optimization_logging

setup_optimization_logging(
    level="DEBUG",
    file_path="optimization_debug.log"
)

# Monitora métricas específicas
monitor.set_gauge("debug.cache_size", len(optimizer.cache_manager._memory_cache))
monitor.set_gauge("debug.active_workers", len(resource_mgr.worker_pool.active_tasks))
```

### Health Checks

```python
def health_check():
    """Verifica saúde do sistema de otimização"""
    issues = []

    # Verifica optimizer
    optimizer_stats = optimizer.get_performance_stats()
    if optimizer_stats['cache_hit_rate'] < 0.3:
        issues.append("Low cache hit rate")

    # Verifica resources
    resource_stats = resource_mgr.get_comprehensive_stats()
    cpu_usage = resource_stats['resource_usage'].get('cpu')
    if cpu_usage and cpu_usage.percentage > 90:
        issues.append("High CPU usage")

    # Verifica circuit breaker
    cb_state = resource_mgr.circuit_breaker.get_state()
    if cb_state != CircuitState.CLOSED:
        issues.append(f"Circuit breaker {cb_state.value}")

    return issues

# Execute periodicamente
issues = health_check()
if issues:
    print("⚠️ Issues detectadas:", issues)
```

## 📚 API Reference

### PerformanceOptimizer

#### Métodos Principais
```python
cached_llm_call(input_text, model, llm_callable, temperature=0.7, force_refresh=False)
parallel_processing(tasks, timeout=None)
optimize_for_workload(workload_type)
get_performance_stats()
cleanup()
```

#### Context Manager
```python
with PerformanceOptimizer() as optimizer:
    # Monitoramento automático iniciado
    result = optimizer.cached_llm_call(...)
# Cleanup automático
```

### MultiLayerCache

#### Operações Básicas
```python
get(key)                           # Obtém valor
set(key, value, ttl=3600, tags=None)  # Define valor
delete(key)                        # Remove valor
clear()                           # Limpa tudo
invalidate_by_tags(tags)          # Invalida por tags
```

#### Estatísticas
```python
get_comprehensive_stats()         # Stats de todas as camadas
```

### PerformanceMonitor

#### Métricas
```python
increment_counter(name, value=1, labels=None)
set_gauge(name, value, labels=None)
record_histogram(name, value, labels=None)
record_timer(name, duration, labels=None)
```

#### Monitoramento
```python
start_monitoring()                # Inicia coleta
stop_monitoring()                 # Para coleta
get_status()                      # Status atual
```

#### Context Managers
```python
timer_context(name, labels=None)  # Timing automático
```

### ResourceManager

#### Proteção de Recursos
```python
execute_with_protection(func, client_id, *args, **kwargs)
resource_context(client_id)      # Context manager
```

#### Worker Pool
```python
submit_async_task(task_id, func, *args, priority=5, **kwargs)
get_task_status(task_id)
get_task_result(task_id)
```

#### Estatísticas
```python
get_comprehensive_stats()        # Stats completas
optimize_resources()             # Otimização automática
```

### Exceções

```python
# Resource Management
ResourceManagementError          # Base exception
ThrottleError                    # Throttling ativo
RateLimitError                   # Rate limit excedido
CircuitBreakerOpenError          # Circuit breaker aberto
```

## 🎯 Exemplos Avançados

### Integração Completa

```python
# Setup completo para produção
suite = create_complete_optimization_suite(
    redis_url="redis://redis-cluster:6379"
)

optimizer = suite['optimizer']
monitor = suite['monitor']
resource_mgr = suite['resource_manager']

# Configuração de alertas
def critical_alert_handler(alert):
    # Enviar para sistema de monitoramento
    send_to_datadog(alert)
    send_slack_notification(alert)

monitor.alert_manager.add_handler(AlertLevel.CRITICAL, critical_alert_handler)

# Inicia tudo
monitor.start_monitoring()
resource_mgr.start_monitoring()

# Use em toda a aplicação
class OptimizedLLMService:
    def __init__(self):
        self.optimizer = optimizer
        self.monitor = monitor
        self.resource_mgr = resource_mgr

    def process_request(self, user_input, user_id):
        # Proteção de recursos
        with self.resource_mgr.resource_context(user_id):
            # Monitoring
            self.monitor.increment_counter("llm_requests", labels={"user": user_id})

            # Cache otimizado
            with self.monitor.timer_context("llm_processing"):
                response = self.optimizer.cached_llm_call(
                    user_input,
                    "gpt-4",
                    self._call_llm
                )

            return response

    def _call_llm(self, input_text):
        # Sua implementação de LLM
        return llm_chain.invoke({"input": input_text})
```

### Auto-Scaling Baseado em Métricas

```python
class AutoScaler:
    def __init__(self, optimizer, monitor):
        self.optimizer = optimizer
        self.monitor = monitor

    def auto_scale(self):
        """Ajusta recursos baseado em métricas"""
        # Obtém métricas recentes
        report = self.monitor.dashboard.generate_report(minutes=5)

        # CPU usage
        cpu_metric = report['metrics'].get('system.cpu.usage_percent', {})
        avg_cpu = cpu_metric.get('mean', 0)

        # Response time
        rt_metric = report['metrics'].get('application.llm.response_time.mean', {})
        avg_rt = rt_metric.get('mean', 0)

        # Decisões de scaling
        if avg_cpu > 80 or avg_rt > 2000:  # Alta carga
            # Reduz workers, aumenta cache
            new_workers = max(2, self.optimizer.max_workers - 1)
            new_cache_size = min(10000, self.optimizer.cache_manager.max_size * 1.2)

            print(f"Scaling down: workers={new_workers}, cache={new_cache_size}")

        elif avg_cpu < 40 and avg_rt < 500:  # Baixa carga
            # Aumenta workers, otimiza cache
            new_workers = min(16, self.optimizer.max_workers + 1)

            print(f"Scaling up: workers={new_workers}")

# Use periodicamente
scaler = AutoScaler(optimizer, monitor)
threading.Thread(
    target=lambda: [scaler.auto_scale() for _ in iter(lambda: time.sleep(300), None)],
    daemon=True
).start()
```

---

## 📄 Licença

Este sistema está incluído no projeto Gianna sob a licença Apache 2.0.

## 🤝 Contribuição

Para contribuir com melhorias:

1. Faça fork do projeto
2. Crie branch para feature (`git checkout -b feature/optimization-improvement`)
3. Commit suas mudanças (`git commit -am 'Add new optimization feature'`)
4. Push para branch (`git push origin feature/optimization-improvement`)
5. Abra Pull Request

## 📞 Suporte

- **Issues**: Use GitHub Issues para bugs e features
- **Documentação**: Este README e código com docstrings
- **Exemplos**: Veja `/examples/optimization_examples.py`
- **Testes**: Execute `pytest tests/test_optimization_performance.py`
