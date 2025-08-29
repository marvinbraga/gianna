"""
Exemplos de Uso do Sistema de Otimização de Performance Gianna

Este arquivo demonstra como usar os diferentes componentes do sistema de otimização
para melhorar a performance da aplicação Gianna.
"""

import asyncio
import os
import time
from pathlib import Path

# Imports da aplicação Gianna
from gianna.assistants.models.factory_method import get_chain_instance
from gianna.core.state import CoreState

# Imports do sistema de otimização
from gianna.optimization import (
    MultiLayerCache,
    PerformanceMonitor,
    PerformanceOptimizer,
    ResourceManager,
    create_complete_optimization_suite,
    setup_optimization_logging,
)


def example_1_basic_llm_caching():
    """
    Exemplo 1: Cache básico para chamadas LLM

    Demonstra como usar o cache para acelerar chamadas LLM repetidas
    """
    print("=== Exemplo 1: Cache Básico para LLM ===")

    # Cria otimizador com cache
    optimizer = PerformanceOptimizer(
        cache_size=1000, cache_ttl=3600, max_workers=4  # 1 hora
    )

    # Função que simula chamada LLM
    def simulate_llm_call(input_text: str) -> str:
        """Simula uma chamada LLM lenta"""
        time.sleep(0.5)  # Simula latência
        return f"AI Response to: {input_text}"

    # Primeira chamada (será lenta)
    print("Primeira chamada LLM...")
    start_time = time.time()
    result1 = optimizer.cached_llm_call(
        "What is machine learning?", "gpt-4", simulate_llm_call
    )
    first_call_time = time.time() - start_time
    print(f"Resultado: {result1}")
    print(f"Tempo: {first_call_time:.2f}s")

    # Segunda chamada (será rápida - cache hit)
    print("\nSegunda chamada LLM (mesma entrada)...")
    start_time = time.time()
    result2 = optimizer.cached_llm_call(
        "What is machine learning?", "gpt-4", simulate_llm_call
    )
    second_call_time = time.time() - start_time
    print(f"Resultado: {result2}")
    print(f"Tempo: {second_call_time:.2f}s")

    # Estatísticas
    stats = optimizer.get_performance_stats()
    print(f"\nEstatísticas:")
    print(f"- Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
    print(f"- Total Requests: {stats['total_requests']}")
    print(f"- Speed Up: {first_call_time/second_call_time:.1f}x mais rápido")

    optimizer.cleanup()


def example_2_real_llm_integration():
    """
    Exemplo 2: Integração com LLM real do Gianna

    Demonstra integração com o sistema de modelos real do Gianna
    """
    print("\n=== Exemplo 2: Integração LLM Real ===")

    # Configuração básica (só funciona se API keys estiverem configuradas)
    try:
        # Cria otimizador
        optimizer = PerformanceOptimizer(cache_size=500)

        # Obtém chain LLM do Gianna
        chain = get_chain_instance("gpt35", "You are a helpful assistant.")

        def cached_llm_invoke(input_text: str) -> str:
            """Wrapper para cache do LLM"""
            response = chain.invoke({"input": input_text})
            return response

        # Perguntas de teste
        questions = [
            "What is Python?",
            "Explain machine learning briefly",
            "What is Python?",  # Repetida - deve usar cache
            "How does caching work?",
        ]

        print("Fazendo chamadas LLM com cache...")
        total_time = 0

        for i, question in enumerate(questions, 1):
            start_time = time.time()

            answer = optimizer.cached_llm_call(
                question, "gpt35", cached_llm_invoke, temperature=0.7
            )

            call_time = time.time() - start_time
            total_time += call_time

            print(f"\n{i}. Pergunta: {question}")
            print(f"   Resposta: {answer[:100]}...")
            print(f"   Tempo: {call_time:.2f}s")

        # Estatísticas finais
        stats = optimizer.get_performance_stats()
        print(f"\n=== Estatísticas Finais ===")
        print(f"Tempo total: {total_time:.2f}s")
        print(f"Hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"Chamadas únicas: {stats['total_requests'] - stats['cache_hits']}")

        optimizer.cleanup()

    except Exception as e:
        print(f"Erro na integração LLM (verifique API keys): {e}")


def example_3_multilayer_cache():
    """
    Exemplo 3: Cache multi-layer com persistência

    Demonstra cache em múltiplas camadas (memória, Redis, SQLite)
    """
    print("\n=== Exemplo 3: Cache Multi-Layer ===")

    # Cria cache multi-layer com SQLite (Redis opcional)
    cache_dir = Path("temp_cache")
    cache_dir.mkdir(exist_ok=True)

    cache = MultiLayerCache(
        memory_size=100,
        redis_url=None,  # Redis opcional
        sqlite_path=str(cache_dir / "gianna_cache.db"),
    )

    print("Cache multi-layer criado com camadas:")
    for i, layer in enumerate(cache.layers):
        print(f"  {i+1}. {type(layer).__name__}")

    # Teste de operações básicas
    test_data = {
        "user:123": {"name": "João", "preferences": {"lang": "pt"}},
        "conversation:abc": ["Olá", "Como posso ajudar?"],
        "model_response:xyz": "Esta é uma resposta cached do modelo",
    }

    print("\nArmazenando dados no cache...")
    for key, value in test_data.items():
        cache.set(key, value, ttl=3600, tags={"example", "test"})
        print(f"✓ Armazenado: {key}")

    print("\nRecuperando dados do cache...")
    for key in test_data.keys():
        value = cache.get(key)
        print(f"✓ {key}: {str(value)[:50]}...")

    # Estatísticas
    stats = cache.get_comprehensive_stats()
    print(f"\nEstatísticas do cache:")
    print(f"Hit rate global: {stats['hit_rate']:.1%}")
    print(f"Camadas ativas: {stats['layer_count']}")

    # Limpeza
    cache.clear()
    import shutil

    shutil.rmtree(cache_dir, ignore_errors=True)

    print("✓ Cache limpo")


def example_4_performance_monitoring():
    """
    Exemplo 4: Monitoramento de performance em tempo real

    Demonstra coleta de métricas, alertas e dashboards
    """
    print("\n=== Exemplo 4: Monitoramento de Performance ===")

    # Cria monitor com profiling
    monitor = PerformanceMonitor(
        collection_interval=2, enable_profiling=True  # Coleta a cada 2s
    )

    # Setup de alertas
    def alert_handler(alert):
        print(f"🚨 ALERTA [{alert.level.value}]: {alert.title} - {alert.message}")

    # Adiciona regras de alerta
    monitor.alert_manager.add_rule(
        "app.response_time",
        ">",
        1000,  # > 1s
        AlertLevel.WARNING,
        "Slow Response",
        "Response time is {current_value}ms",
    )

    from gianna.optimization.monitoring import AlertLevel

    monitor.alert_manager.add_handler(AlertLevel.WARNING, alert_handler)

    # Inicia monitoramento
    monitor.start_monitoring()
    print("Monitor iniciado - coletando métricas...")

    # Simula atividade da aplicação
    for i in range(10):
        # Simula operações variadas
        if i % 3 == 0:
            # Operação lenta
            monitor.increment_counter("app.slow_operations")
            with monitor.timer_context("app.response_time"):
                time.sleep(0.2)  # Simula operação lenta
        else:
            # Operação rápida
            monitor.increment_counter("app.fast_operations")
            with monitor.timer_context("app.response_time"):
                time.sleep(0.05)  # Simula operação rápida

        # Métricas de sistema
        monitor.set_gauge("app.active_users", i * 10)
        monitor.record_histogram("app.request_size", i * 100)

        time.sleep(0.5)

    # Para monitoramento
    time.sleep(2)  # Aguarda última coleta
    monitor.stop_monitoring()

    # Relatório
    print("\n=== Relatório de Performance ===")
    status = monitor.get_status()
    print(f"Métricas coletadas: {status['metrics_collected']}")
    print(f"Alertas ativos: {status['active_alerts']}")

    # Gera relatório
    report = monitor.dashboard.generate_report(minutes=5)
    print(f"\nResumo dos últimos 5 minutos:")
    for metric_name, data in report["metrics"].items():
        if not metric_name.startswith("system"):
            print(f"  {metric_name}:")
            print(f"    - Média: {data.get('mean', 0):.2f}")
            print(f"    - Mín/Máx: {data.get('min', 0):.2f}/{data.get('max', 0):.2f}")


def example_5_resource_management():
    """
    Exemplo 5: Gerenciamento avançado de recursos

    Demonstra throttling, circuit breaker, rate limiting e worker pools
    """
    print("\n=== Exemplo 5: Gerenciamento de Recursos ===")

    # Cria gerenciador de recursos
    resource_mgr = ResourceManager(
        max_workers=3,
        throttle_rate=5.0,  # 5 req/s
        rate_limit_per_minute=20,  # 20 req/min por cliente
    )

    # Inicia monitoramento
    resource_mgr.start_monitoring()

    print("Gerenciador de recursos configurado:")
    print("- Throttling: 5 req/s")
    print("- Rate limit: 20 req/min por cliente")
    print("- Workers: 3 máximo")
    print("- Circuit breaker ativo")

    # Teste 1: Proteção básica
    print("\n1. Testando proteção básica...")

    def safe_operation():
        time.sleep(0.1)
        return "Operação concluída com sucesso"

    try:
        result = resource_mgr.execute_with_protection(
            safe_operation, client_id="client1"
        )
        print(f"✓ {result}")
    except Exception as e:
        print(f"❌ Erro: {e}")

    # Teste 2: Rate limiting
    print("\n2. Testando rate limiting...")

    client_id = "test_client"
    success_count = 0

    for i in range(25):  # Tenta mais que o limite (20)
        try:
            with resource_mgr.resource_context(client_id):
                success_count += 1
        except Exception as e:
            print(f"Request {i+1} bloqueada: {type(e).__name__}")
            break

    print(f"Requests permitidas: {success_count}/25")
    remaining = resource_mgr.rate_limiter.get_remaining(client_id)
    print(f"Requests restantes: {remaining}")

    # Teste 3: Worker pool assíncrono
    print("\n3. Testando worker pool...")

    def cpu_intensive_task(n):
        """Simula tarefa CPU-intensiva"""
        total = 0
        for i in range(n * 10000):
            total += i
        return f"Task {n}: resultado = {total}"

    # Submete tarefas
    task_ids = []
    start_time = time.time()

    for i in range(6):
        task_id = resource_mgr.submit_async_task(
            f"cpu_task_{i}", cpu_intensive_task, i + 1
        )
        task_ids.append(task_id)
        print(f"✓ Tarefa submetida: {task_id}")

    # Aguarda conclusão
    print("\nAguardando conclusão das tarefas...")
    completed = 0

    while completed < len(task_ids):
        time.sleep(0.5)

        for task_id in task_ids:
            status = resource_mgr.worker_pool.get_task_status(task_id)
            if status == "completed":
                if task_id not in [
                    t
                    for t in task_ids
                    if resource_mgr.worker_pool.get_task_status(t) == "completed"
                ]:
                    try:
                        result = resource_mgr.worker_pool.get_task_result(task_id)
                        print(f"✓ {result}")
                        completed += 1
                    except:
                        pass

    execution_time = time.time() - start_time
    print(f"\nTempo total: {execution_time:.2f}s")

    # Estatísticas finais
    stats = resource_mgr.get_comprehensive_stats()
    print(f"\n=== Estatísticas de Recursos ===")
    print(f"Worker pool utilização: {stats['worker_pool']['worker_utilization']:.1%}")
    print(f"Throttling rate atual: {stats['throttler']['current_rate']:.1f} req/s")
    print(f"Circuit breaker: {stats['circuit_breaker']['state']}")

    resource_mgr.stop_monitoring()


def example_6_complete_integration():
    """
    Exemplo 6: Integração completa com CoreState

    Demonstra integração completa com o sistema de estado do Gianna
    """
    print("\n=== Exemplo 6: Integração Completa ===")

    # Cria suite completa de otimização
    optimization_suite = create_complete_optimization_suite()

    print("Suite de otimização criada com componentes:")
    for name, component in optimization_suite.items():
        print(f"  - {name}: {type(component).__name__}")

    # Integração com CoreState
    try:
        core_state = CoreState()

        # Simulação de workflow otimizado
        print("\nExecutando workflow otimizado...")

        optimizer = optimization_suite["optimizer"]
        monitor = optimization_suite["monitor"]
        resource_mgr = optimization_suite["resource_manager"]

        # Inicia monitoramento
        monitor.start_monitoring()
        resource_mgr.start_monitoring()

        # Simula sessão de conversação com cache
        conversation_data = [
            ("user", "Olá, como você pode me ajudar?"),
            ("assistant", "Olá! Eu posso ajudar com várias tarefas..."),
            ("user", "O que você sabe sobre Python?"),
            ("assistant", "Python é uma linguagem de programação..."),
            ("user", "Olá, como você pode me ajudar?"),  # Repetida
        ]

        print("\nSimulando conversação com cache...")

        for speaker, message in conversation_data:
            # Registra métricas
            monitor.increment_counter(
                "conversation.messages", labels={"speaker": speaker}
            )

            if speaker == "user":
                # Simula processamento de entrada do usuário
                with monitor.timer_context("processing.user_input"):
                    time.sleep(0.1)

                # Cache de respostas baseadas na entrada
                def generate_response(user_input):
                    time.sleep(0.3)  # Simula geração de resposta
                    return f"Resposta otimizada para: {user_input[:30]}..."

                with resource_mgr.resource_context("user_session"):
                    response = optimizer.cached_llm_call(
                        message, "conversation_model", generate_response
                    )

                print(f"User: {message}")
                print(f"AI: {response}")

            time.sleep(0.2)

        # Estatísticas finais
        print(f"\n=== Estatísticas da Sessão ===")

        # Performance
        perf_stats = optimizer.get_performance_stats()
        print(f"Cache hit rate: {perf_stats['cache_hit_rate']:.1%}")
        print(f"Tempo médio de resposta: {perf_stats.get('avg_response_time', 0):.3f}s")

        # Recursos
        resource_stats = resource_mgr.get_comprehensive_stats()
        cpu_usage = resource_stats["resource_usage"].get("cpu")
        if cpu_usage:
            print(f"CPU usage: {cpu_usage.percentage:.1f}%")

        # Para monitoramento
        monitor.stop_monitoring()
        resource_mgr.stop_monitoring()

        # Limpeza
        optimizer.cleanup()

        print("✓ Workflow completo executado com sucesso")

    except Exception as e:
        print(f"Erro na integração: {e}")


def example_7_configuration_templates():
    """
    Exemplo 7: Templates de configuração para diferentes cenários

    Demonstra configurações otimizadas para diferentes tipos de uso
    """
    print("\n=== Exemplo 7: Templates de Configuração ===")

    # Configuração para desenvolvimento
    def dev_config():
        return {
            "optimizer": PerformanceOptimizer(
                cache_size=100,  # Cache pequeno
                max_workers=2,  # Poucos workers
                cache_ttl=1800,  # TTL curto (30min)
            ),
            "monitor": PerformanceMonitor(
                collection_interval=10,  # Coleta mais espaçada
                enable_profiling=True,  # Profiling para debug
            ),
            "resource_mgr": ResourceManager(
                max_workers=2,
                throttle_rate=100.0,  # Sem throttling agressivo
                rate_limit_per_minute=1000,  # Rate limit generoso
            ),
        }

    # Configuração para produção
    def prod_config():
        return {
            "optimizer": PerformanceOptimizer(
                cache_size=5000,  # Cache grande
                max_workers=8,  # Muitos workers
                cache_ttl=3600,  # TTL longo (1h)
                redis_url="redis://localhost:6379",  # Redis para produção
            ),
            "monitor": PerformanceMonitor(
                collection_interval=30,  # Coleta regular
                enable_profiling=False,  # Sem profiling em prod
            ),
            "resource_mgr": ResourceManager(
                max_workers=8,
                throttle_rate=20.0,  # Throttling moderado
                rate_limit_per_minute=100,  # Rate limit restritivo
            ),
        }

    # Configuração para alto volume
    def high_volume_config():
        return {
            "optimizer": PerformanceOptimizer(
                cache_size=10000,  # Cache muito grande
                max_workers=16,  # Muitos workers
                cache_ttl=7200,  # TTL longo (2h)
                redis_url="redis://localhost:6379",
            ),
            "monitor": PerformanceMonitor(
                collection_interval=60,  # Coleta menos frequente
                enable_profiling=False,  # Sem overhead
            ),
            "resource_mgr": ResourceManager(
                max_workers=16,
                throttle_rate=50.0,  # Throttling alto
                rate_limit_per_minute=200,  # Rate limit alto
            ),
        }

    configs = {
        "Desenvolvimento": dev_config(),
        "Produção": prod_config(),
        "Alto Volume": high_volume_config(),
    }

    print("Templates de configuração disponíveis:")

    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  - Cache size: {config['optimizer'].cache_manager.max_size}")
        print(f"  - Workers: {config['optimizer'].max_workers}")
        print(f"  - Throttling: {config['resource_mgr'].throttler.base_rate} req/s")
        print(f"  - Monitoring: {config['monitor'].collection_interval}s interval")

        # Cleanup
        config["optimizer"].cleanup()

    print("\n💡 Use estes templates como base para sua configuração!")


def main():
    """Função principal que executa todos os exemplos"""

    # Setup de logging
    setup_optimization_logging(level="INFO")

    print("🚀 Exemplos do Sistema de Otimização de Performance Gianna")
    print("=" * 60)

    # Executa exemplos
    examples = [
        example_1_basic_llm_caching,
        example_2_real_llm_integration,
        example_3_multilayer_cache,
        example_4_performance_monitoring,
        example_5_resource_management,
        example_6_complete_integration,
        example_7_configuration_templates,
    ]

    for example in examples:
        try:
            example()
            time.sleep(1)  # Pausa entre exemplos
        except KeyboardInterrupt:
            print("\n⏹️  Exemplos interrompidos pelo usuário")
            break
        except Exception as e:
            print(f"\n❌ Erro no exemplo {example.__name__}: {e}")
            continue

    print("\n✅ Todos os exemplos concluídos!")
    print("\nPara usar o sistema de otimização:")
    print("1. Importe os componentes necessários")
    print("2. Configure baseado no seu cenário (dev/prod/high-volume)")
    print("3. Use context managers para proteção automática")
    print("4. Monitore métricas e ajuste conforme necessário")


if __name__ == "__main__":
    main()
