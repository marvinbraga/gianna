# PR: Implementar melhorias abrangentes e manual do usuário

## Título
```
Implementar melhorias abrangentes e manual do usuário
```

## Descrição (copie abaixo)

---

## Resumo

Este PR implementa melhorias significativas na arquitetura e infraestrutura do Gianna, além de adicionar documentação completa para usuários.

## Novas Funcionalidades

### 🔧 Infraestrutura

1. **Router Configuration (YAML-based)**
   - `config/routing_rules.yaml`: Regras de roteamento configuráveis
   - `gianna/coordination/router_config.py`: Validador Pydantic
   - Refatorado `router.py` para carregar do YAML com fallback

2. **Circuit Breaker para APIs**
   - `gianna/optimization/circuit_breaker.py`
   - Estados: CLOSED → OPEN → HALF_OPEN
   - Suporte sync/async com decorators
   - Registry singleton para gerenciamento centralizado

3. **Memory Pool para Audio**
   - `gianna/audio/memory_pool.py`
   - Pool de buffers pré-alocados para reduzir GC
   - Thread-safe com suporte a concorrência

4. **Infraestrutura Async/Await**
   - `gianna/core/async_utils.py`
   - `AsyncExecutor`, `AsyncRetry`, `AsyncThrottle`, `AsyncBatcher`

5. **Logging Estruturado (JSON)**
   - `gianna/monitoring/structured_logging.py`
   - Formato JSON para Loki/Elasticsearch
   - Contexto automático e redação de dados sensíveis

6. **Sistema de Configurações Centralizado**
   - `gianna/config/settings.py`
   - Settings unificadas com Pydantic
   - Suporte a YAML/JSON e variáveis de ambiente

7. **CLI Administrativa**
   - `gianna/cli.py`
   - Comandos: config, health, benchmark, cache, secrets, info

8. **Tipos para VAD Callbacks**
   - Protocols expandidos em `gianna/audio/vad/types.py`
   - `VADCallbackConfig` para configuração centralizada

### 📚 Documentação

- **Manual do Usuário completo** (`docs/user-guide/MANUAL_USUARIO.md`)
  - ~1.150 linhas de documentação em português
  - Instalação, configuração, uso básico e avançado
  - Guias para todos os provedores LLM
  - Sistema de áudio, agentes, memória semântica
  - CLI, otimização e performance
  - 4 exemplos práticos completos
  - Troubleshooting e referência de API

### ✅ Testes

- `tests/unit/test_circuit_breaker.py` (~200 linhas)
- `tests/unit/test_memory_pool.py` (~250 linhas)
- `tests/unit/test_router_config.py` (~180 linhas)

## Estatísticas

| Métrica | Valor |
|---------|-------|
| Arquivos alterados | 15 |
| Linhas adicionadas | ~6.500 |
| Novos módulos | 9 |
| Novos testes | ~50 |

## Benefícios

- ⚡ **Performance**: Memory pool reduz GC, circuit breaker previne falhas em cascata
- 🔧 **Configurabilidade**: Routing rules em YAML, settings centralizadas
- 🔍 **Observabilidade**: Logging estruturado JSON, métricas detalhadas
- 📖 **Documentação**: Manual completo para onboarding de novos usuários
- ✅ **Qualidade**: Mais testes, melhor cobertura

## Test plan

- [x] Testes unitários passando
- [x] Validação de configuração YAML
- [x] Circuit breaker testado com falhas simuladas
- [x] Memory pool testado com concorrência
- [x] Manual revisado e completo
