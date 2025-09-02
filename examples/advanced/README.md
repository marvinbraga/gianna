# Exemplos Avançados - Gianna

Este diretório contém exemplos avançados de uso do Gianna, demonstrando casos de uso complexos e integrações sofisticadas.

## Estrutura dos Exemplos

### 🧠 Multi-Agent Systems
- [`multi_agent_orchestration.py`](./multi_agent_orchestration.py) - Coordenação complexa entre agentes
- [`specialized_agents.py`](./specialized_agents.py) - Agentes especializados por domínio
- [`agent_collaboration.py`](./agent_collaboration.py) - Colaboração entre múltiplos agentes

### 🎤 Voice Processing
- [`real_time_voice_assistant.py`](./real_time_voice_assistant.py) - Assistente de voz em tempo real
- [`voice_command_system.py`](./voice_command_system.py) - Sistema avançado de comandos por voz
- [`multilingual_voice.py`](./multilingual_voice.py) - Processamento de voz multilíngue

### 🧠 Memória e Aprendizado
- [`semantic_memory_system.py`](./semantic_memory_system.py) - Sistema de memória semântica avançado
- [`user_adaptation_engine.py`](./user_adaptation_engine.py) - Engine de adaptação ao usuário
- [`context_aware_responses.py`](./context_aware_responses.py) - Respostas conscientes do contexto

### ⚡ Performance e Otimização
- [`high_performance_deployment.py`](./high_performance_deployment.py) - Deploy de alta performance
- [`caching_strategies.py`](./caching_strategies.py) - Estratégias avançadas de cache
- [`resource_optimization.py`](./resource_optimization.py) - Otimização de recursos

### 🔧 Integrações
- [`external_apis_integration.py`](./external_apis_integration.py) - Integração com APIs externas
- [`database_integration.py`](./database_integration.py) - Integração com bancos de dados
- [`cloud_services_integration.py`](./cloud_services_integration.py) - Integração com serviços cloud

### 🔐 Enterprise Features
- [`enterprise_security.py`](./enterprise_security.py) - Recursos de segurança empresarial
- [`audit_logging.py`](./audit_logging.py) - Sistema de auditoria e logging
- [`user_management.py`](./user_management.py) - Gerenciamento de usuários

## Pré-requisitos

### Dependências Básicas
```bash
poetry install
```

### Configurações Específicas

Cada exemplo pode ter configurações específicas. Crie `.env.advanced`:

```env
# APIs básicas
OPENAI_API_KEY=sua_chave
GOOGLE_API_KEY=sua_chave
ELEVEN_LABS_API_KEY=sua_chave

# Configurações avançadas
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost:5432/gianna
ELASTIC_URL=http://localhost:9200

# Integrações externas
WEBHOOK_SECRET=seu_webhook_secret
API_RATE_LIMIT=1000
MAX_CONCURRENT_SESSIONS=100

# Performance
CACHE_ENABLED=true
PERFORMANCE_MONITORING=true
DISTRIBUTED_PROCESSING=true
```

### Serviços Externos

Alguns exemplos requerem serviços externos:

```bash
# Redis (cache e mensageria)
docker run -d --name redis -p 6379:6379 redis:alpine

# PostgreSQL (persistência)
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=gianna \
  -p 5432:5432 postgres:13

# Elasticsearch (busca semântica)
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  elasticsearch:7.14.0
```

## Como Usar os Exemplos

### 1. Executar Exemplo Individual
```bash
cd examples/advanced
python multi_agent_orchestration.py
```

### 2. Executar com Configurações Específicas
```bash
# Configurar variáveis de ambiente
export GIANNA_CONFIG=production
export LOG_LEVEL=DEBUG

python real_time_voice_assistant.py
```

### 3. Modo Interativo
```bash
# Executar em modo interativo para testes
python -i specialized_agents.py
```

### 4. Com Docker
```bash
# Executar em container isolado
docker run -it --rm \
  -v $(pwd):/app \
  -w /app/examples/advanced \
  python:3.11 python semantic_memory_system.py
```

## Níveis de Complexidade

### Iniciante
- `basic_conversation.py` - Conversação simples com estado
- `simple_voice_commands.py` - Comandos de voz básicos
- `file_processing.py` - Processamento básico de arquivos

### Intermediário
- `multi_agent_orchestration.py` - Coordenação de agentes
- `real_time_voice_assistant.py` - Assistente de voz tempo real
- `semantic_memory_system.py` - Memória semântica

### Avançado
- `distributed_system.py` - Sistema distribuído
- `enterprise_deployment.py` - Deploy empresarial
- `ai_powered_automation.py` - Automação com IA

### Expert
- `custom_llm_integration.py` - Integração de LLM customizado
- `advanced_optimization.py` - Otimizações avançadas
- `production_monitoring.py` - Monitoramento de produção

## Casos de Uso por Setor

### 🏢 Corporativo
- Assistente executivo inteligente
- Automação de workflows empresariais
- Sistema de knowledge management
- Análise de documentos corporativos

### 🎓 Educacional
- Tutor personalizado adaptativo
- Sistema de avaliação inteligente
- Geração de conteúdo educacional
- Assistente para pesquisa acadêmica

### 🏥 Saúde
- Assistente médico especializado
- Análise de prontuários
- Sistema de triagem inteligente
- Monitoramento de pacientes

### 💼 Atendimento
- Chatbot inteligente multicamada
- Sistema de tickets automatizado
- Análise de sentimento em tempo real
- Escalação automática de problemas

### 🔬 Pesquisa & Desenvolvimento
- Assistente de pesquisa científica
- Análise de literatura acadêmica
- Geração de hipóteses
- Processamento de dados experimentais

## Performance Benchmarks

### Latência Típica
```
Conversação simples:     100-300ms
Multi-agente:           300-800ms
Voz tempo real:         150-400ms
Processamento semântico: 200-600ms
```

### Throughput
```
Mensagens por segundo:   50-200
Sessões simultâneas:     100-1000
Processamento de voz:    10-50 streams
```

### Recursos
```
RAM por sessão:          50-200MB
CPU por processamento:   10-30%
Armazenamento por usuário: 1-10MB
```

## Troubleshooting

### Problemas Comuns

1. **Exemplo não executa:**
   ```bash
   # Verificar dependências
   poetry check
   poetry install

   # Verificar configuração
   python -c "from gianna.core.state import GiannaState; print('OK')"
   ```

2. **Erro de API:**
   ```bash
   # Verificar chaves
   python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT_SET'))"
   ```

3. **Performance ruim:**
   ```bash
   # Verificar recursos
   htop

   # Configurar cache
   export CACHE_ENABLED=true
   ```

### Debug Mode

Executar exemplos em modo debug:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
python -u exemplo.py 2>&1 | tee debug.log
```

### Profiling

Para análise de performance:

```bash
python -m cProfile -o profile.stats exemplo.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats()"
```

## Contribuindo

### Adicionando Novos Exemplos

1. **Estrutura do arquivo:**
```python
"""
Título do Exemplo - Descrição breve

Este exemplo demonstra como usar [funcionalidade] para [objetivo].

Pré-requisitos:
- Dependência 1
- Dependência 2
- Serviço externo (se necessário)

Uso:
    python exemplo.py
"""

import asyncio
from pathlib import Path
from gianna.core.langgraph_chain import LangGraphChain

# Configurações
CONFIG = {
    "model": "gpt4",
    "temperature": 0.7,
    "max_examples": 10
}

def main():
    """Função principal do exemplo."""
    print("🚀 Iniciando exemplo...")

    # Implementação

    print("✅ Exemplo concluído!")

if __name__ == "__main__":
    main()
```

2. **Documentação:**
   - Adicionar ao README.md
   - Incluir comentários detalhados
   - Exemplos de saída esperada
   - Troubleshooting específico

3. **Testes:**
   - Adicionar teste básico
   - Verificar com diferentes configurações
   - Testar cenários de erro

### Guidelines

- Use nomes descritivos para variáveis e funções
- Inclua tratamento de erros apropriado
- Adicione logs informativos
- Documente parâmetros configuráveis
- Forneça exemplos de uso

## Próximos Passos

1. **Execute os exemplos básicos** para entender os conceitos
2. **Modifique os parâmetros** para ver diferentes comportamentos
3. **Combine exemplos** para criar soluções mais complexas
4. **Contribua** com novos exemplos baseados em seus casos de uso

Para mais informações, consulte:
- [Guia do Usuário](../../docs/user-guide/)
- [Guia do Desenvolvedor](../../docs/developer-guide/)
- [Documentação da API](../../docs/api/)
