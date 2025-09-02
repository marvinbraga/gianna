# Guia do Desenvolvedor - Gianna

Este guia contém informações técnicas para desenvolvedores que querem contribuir ou estender o projeto Gianna.

## Índice

1. [Arquitetura do Sistema](#arquitetura-do-sistema)
2. [Configuração do Ambiente de Desenvolvimento](#configuração-do-ambiente-de-desenvolvimento)
3. [Padrões de Código](#padrões-de-código)
4. [Extensão do Sistema](#extensão-do-sistema)
5. [Testes e Validação](#testes-e-validação)
6. [Deploy e Produção](#deploy-e-produção)

## Arquitetura do Sistema

### Visão Geral

O Gianna segue uma arquitetura modular baseada em:

1. **Factory Method Pattern**: Para criação dinâmica de componentes
2. **Registry Pattern**: Para registro e descoberta de serviços
3. **State Management**: Estado centralizado com LangGraph
4. **Multi-Agent System**: Agentes especializados coordenados
5. **Plugin Architecture**: Extensibilidade através de plugins

```
gianna/
├── core/           # Estado e coordenação central (Fase 1)
├── agents/         # Sistema multi-agente ReAct (Fase 2)
├── audio/          # Processamento de áudio streaming (Fase 3)
├── memory/         # Sistema de memória semântica (Fase 4)
├── learning/       # Aprendizado adaptativo (Fase 4)
├── coordination/   # Coordenação entre componentes (Fase 4)
├── optimization/   # Otimizações de produção (Fase 5)
├── workflows/      # Fluxos de trabalho complexos (Fase 5)
└── tools/          # Ferramentas integradas (Fase 2)
```

### Fase 1: Core State Management

**Estado Centralizado**
```python
# gianna/core/state.py
from typing_extensions import TypedDict
from pydantic import BaseModel

class GiannaState(TypedDict):
    conversation: ConversationState
    audio: AudioState
    commands: CommandState
    metadata: Dict[str, Any]
```

**LangGraph Integration**
```python
# gianna/core/langgraph_chain.py
from langgraph.graph import StateGraph, END

class LangGraphChain(AbstractBasicChain):
    def _build_workflow(self):
        graph = StateGraph(GiannaState)
        graph.add_node("process_input", self._process_input)
        graph.add_node("llm_processing", self._llm_processing)
        graph.add_node("format_output", self._format_output)
        return graph.compile(checkpointer=self.checkpointer)
```

### Fase 2: Multi-Agent ReAct System

**Agentes Especializados**
```python
# gianna/agents/react_agents.py
class CommandAgent(GiannaReActAgent):
    def __init__(self, llm: BaseLanguageModel):
        tools = [ShellExecutorTool(), FileSystemTool()]
        super().__init__("command_agent", llm, tools)

class AudioAgent(GiannaReActAgent):
    def __init__(self, llm: BaseLanguageModel):
        tools = [TTSTool(), STTTool(), AudioRecorderTool()]
        super().__init__("audio_agent", llm, tools)
```

**Coordenação Multi-Agente**
```python
# gianna/coordination/orchestrator.py
class AgentOrchestrator:
    def route_request(self, state: GiannaState) -> str:
        # Análise inteligente de contexto
        # Roteamento para agente apropriado
        # Coordenação de execução paralela
```

### Fase 3: Voice Processing Pipeline

**Streaming de Áudio**
```python
# gianna/audio/streaming.py
class StreamingAudioProcessor:
    async def process_stream(self, audio_stream):
        # VAD (Voice Activity Detection)
        # STT em tempo real
        # Processamento LLM
        # TTS streaming
```

## Configuração do Ambiente de Desenvolvimento

### Setup Inicial

```bash
# Clone e configuração
git clone <repository-url>
cd gianna

# Ambiente Python
poetry install
poetry shell

# Configuração desenvolvimento
invoke dev-setup

# Pre-commit hooks
pre-commit install
```

### Configuração do Editor

**VS Code** (`settings.json`):
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

### Variáveis de Ambiente de Desenvolvimento

```env
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG
TESTING_MODE=true

# APIs de teste (use keys de desenvolvimento)
OPENAI_API_KEY=test_key
GOOGLE_API_KEY=test_key

# Configurações de desenvolvimento
LLM_DEFAULT_MODEL=gpt35
CACHE_ENABLED=false
PERFORMANCE_MONITORING=true
```

## Padrões de Código

### Convenções de Nomenclatura

```python
# Classes: PascalCase
class AudioProcessor:
    pass

# Funções e variáveis: snake_case
def process_audio_stream():
    user_input = "exemplo"

# Constantes: UPPER_CASE
DEFAULT_TIMEOUT = 30
MAX_RETRY_COUNT = 3

# Arquivos: snake_case
# audio_processor.py
# voice_workflows.py
```

### Estrutura de Classes

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

class AbstractComponent(ABC):
    """Classe base abstrata para componentes do sistema."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._initialize()

    def _initialize(self):
        """Inicialização específica do componente."""
        pass

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Método principal de processamento."""
        pass

    def validate_input(self, input_data: Any) -> bool:
        """Validação de entrada."""
        return True

    def get_status(self) -> Dict[str, Any]:
        """Status atual do componente."""
        return {
            "name": self.name,
            "status": "active",
            "config": self.config
        }
```

### Tratamento de Erros

```python
# Exceções customizadas
class GiannaException(Exception):
    """Exceção base do sistema Gianna."""
    pass

class AudioProcessingError(GiannaException):
    """Erro no processamento de áudio."""
    pass

class LLMError(GiannaException):
    """Erro na comunicação com LLM."""
    pass

# Tratamento padrão
import logging
logger = logging.getLogger(__name__)

def safe_process(func):
    """Decorator para processamento seguro com logging."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Erro em {func.__name__}: {str(e)}")
            raise GiannaException(f"Falha no processamento: {str(e)}")
    return wrapper
```

### Documentação de Código

```python
def process_voice_input(
    audio_data: bytes,
    language: str = "pt-br",
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Processa entrada de voz e retorna texto transcrito.

    Args:
        audio_data: Dados de áudio em formato WAV
        language: Idioma para transcrição (padrão: pt-br)
        config: Configurações opcionais do processamento

    Returns:
        Dict contendo:
        - text: Texto transcrito
        - confidence: Nível de confiança (0-1)
        - duration: Duração do áudio em segundos
        - metadata: Informações adicionais

    Raises:
        AudioProcessingError: Se falhar no processamento
        ValueError: Se audio_data estiver inválido

    Example:
        >>> audio = load_audio_file("teste.wav")
        >>> result = process_voice_input(audio, "pt-br")
        >>> print(result["text"])
        "Olá, como você está?"
    """
    if not audio_data:
        raise ValueError("audio_data não pode estar vazio")

    # Implementação...
```

## Extensão do Sistema

### Adicionando Novo Provedor LLM

1. **Criar classe do modelo:**
```python
# gianna/assistants/models/novo_provedor.py
from .basics import AbstractBasicChain, ModelsEnum
from langchain.chat_models import ChatNovoProvedor

class NovoProvedorModels(ModelsEnum):
    MODELO1 = "modelo1"
    MODELO2 = "modelo2"

class NovoProvedorChain(AbstractBasicChain):
    def __init__(self, model: NovoProvedorModels, prompt: str):
        self.model = model
        super().__init__(PromptTemplate.from_template(prompt))

        self.chain = (
            self.prompt_template
            | ChatNovoProvedor(model_name=model.model_name)
            | StrOutputParser()
        )
```

2. **Criar factory:**
```python
class NovoProvedorFactory(AbstractLLMFactory):
    models_enum = NovoProvedorModels

    def create_chain(self, model: str, prompt: str) -> AbstractBasicChain:
        model_enum = self.models_enum(model)
        return NovoProvedorChain(model_enum, prompt)
```

3. **Registrar no sistema:**
```python
# gianna/assistants/models/__init__.py
from .novo_provedor import NovoProvedorFactory

# Registro automático
novo_provedor_factory = NovoProvedorFactory()
LLMRegister().register_factory("novo_provedor", novo_provedor_factory)
```

### Adicionando Nova Ferramenta

```python
# gianna/tools/minha_ferramenta.py
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class MinhaFerramentaInput(BaseModel):
    parametro1: str = Field(description="Descrição do parâmetro 1")
    parametro2: Optional[int] = Field(default=None, description="Parâmetro opcional")

class MinhaFerramenta(BaseTool):
    name = "minha_ferramenta"
    description = """Descrição detalhada da ferramenta.
    Input: JSON com parametros necessários
    Output: Resultado da operação"""
    args_schema: Type[BaseModel] = MinhaFerramentaInput

    def _run(self, parametro1: str, parametro2: Optional[int] = None) -> str:
        # Implementação da ferramenta
        resultado = self._processar(parametro1, parametro2)
        return json.dumps(resultado)

    def _processar(self, param1: str, param2: Optional[int]) -> Dict:
        # Lógica específica
        return {"resultado": f"Processado {param1}"}
```

### Criando Novo Agente Especializado

```python
# gianna/agents/meu_agente.py
from .react_agents import GiannaReActAgent
from ..tools.minha_ferramenta import MinhaFerramenta

class MeuAgente(GiannaReActAgent):
    """Agente especializado em tarefa específica."""

    def __init__(self, llm: BaseLanguageModel):
        tools = [
            MinhaFerramenta(),
            # Outras ferramentas relevantes
        ]
        super().__init__("meu_agente", llm, tools)

        self.system_message = """Você é um especialista em [domínio].
        Suas responsabilidades:
        1. Analisar requisitos específicos
        2. Usar ferramentas apropriadas
        3. Fornecer resultados precisos
        4. Documentar ações realizadas
        """

    def _prepare_agent_state(self, state: GiannaState) -> Dict:
        """Preparação específica do estado para este agente."""
        base_state = super()._prepare_agent_state(state)

        # Adicionar contexto específico
        base_state["domain_context"] = self._extract_domain_context(state)

        return base_state

    def _extract_domain_context(self, state: GiannaState) -> Dict:
        """Extrair contexto específico do domínio."""
        return {"domain": "meu_dominio"}
```

## Testes e Validação

### Estrutura de Testes

```python
# tests/unit/test_meu_modulo.py
import pytest
from unittest.mock import Mock, patch
from gianna.meu_modulo import MinhaClasse

class TestMinhaClasse:
    @pytest.fixture
    def minha_instancia(self):
        return MinhaClasse("configuracao_teste")

    def test_processamento_basico(self, minha_instancia):
        """Teste de processamento básico."""
        input_data = {"teste": "valor"}
        resultado = minha_instancia.processar(input_data)

        assert resultado is not None
        assert "resultado" in resultado

    @patch('gianna.meu_modulo.external_service')
    def test_com_mock(self, mock_service, minha_instancia):
        """Teste com mock de serviço externo."""
        mock_service.return_value = {"mock": "data"}

        resultado = minha_instancia.processar_com_servico_externo()

        assert resultado["mock"] == "data"
        mock_service.assert_called_once()

    def test_tratamento_erro(self, minha_instancia):
        """Teste de tratamento de erro."""
        with pytest.raises(ValueError):
            minha_instancia.processar(None)
```

### Testes de Integração

```python
# tests/integration/test_voice_workflow.py
import pytest
import asyncio
from gianna.workflows.voice_interaction import VoiceWorkflow

@pytest.mark.asyncio
class TestVoiceWorkflow:
    async def test_pipeline_completo(self):
        """Teste do pipeline completo de voz."""
        workflow = VoiceWorkflow()

        # Simular entrada de áudio
        audio_mock = self._create_audio_mock()

        resultado = await workflow.process_voice_input(audio_mock)

        assert "text" in resultado
        assert "response" in resultado
        assert resultado["success"] is True

    def _create_audio_mock(self):
        """Criar mock de dados de áudio."""
        # Implementação do mock
        pass
```

### Validação de Performance

```python
# tests/performance/test_load.py
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

def test_response_time():
    """Testar tempo de resposta sob carga."""

    def single_request():
        start_time = time.time()
        # Fazer requisição ao sistema
        response = make_request()
        end_time = time.time()
        return end_time - start_time

    # Executar múltiplas requisições
    with ThreadPoolExecutor(max_workers=10) as executor:
        times = list(executor.map(lambda _: single_request(), range(50)))

    avg_time = statistics.mean(times)
    p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile

    assert avg_time < 2.0  # Menos de 2s em média
    assert p95_time < 5.0  # 95% das requisições em menos de 5s
```

## Deploy e Produção

### Configuração de Produção

```python
# config/production.py
PRODUCTION_CONFIG = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "handlers": ["file", "console"]
    },
    "performance": {
        "cache_enabled": True,
        "connection_pool_size": 20,
        "timeout_seconds": 30,
        "max_concurrent_requests": 100
    },
    "security": {
        "api_key_rotation": True,
        "input_validation": True,
        "output_sanitization": True,
        "audit_logging": True
    },
    "monitoring": {
        "health_check_interval": 60,
        "metrics_collection": True,
        "alert_thresholds": {
            "response_time": 5.0,
            "error_rate": 0.05,
            "memory_usage": 0.80
        }
    }
}
```

### Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Dependências Python
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

# Código da aplicação
COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  gianna:
    build: .
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "8000:8000"

  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Monitoramento

```python
# gianna/optimization/monitoring.py
class ProductionMonitor:
    def __init__(self):
        self.metrics = {}
        self.alerts = []

    def collect_metrics(self):
        """Coletar métricas do sistema."""
        return {
            "response_time": self._get_avg_response_time(),
            "memory_usage": self._get_memory_usage(),
            "active_sessions": self._get_active_sessions(),
            "error_rate": self._get_error_rate(),
            "cache_hit_rate": self._get_cache_hit_rate()
        }

    def check_health(self):
        """Verificação de saúde do sistema."""
        metrics = self.collect_metrics()
        health_status = "healthy"

        if metrics["response_time"] > 5.0:
            health_status = "degraded"
            self._send_alert("High response time")

        if metrics["error_rate"] > 0.05:
            health_status = "unhealthy"
            self._send_alert("High error rate")

        return {"status": health_status, "metrics": metrics}
```

## Contribuindo

### Fluxo de Contribuição

1. **Fork do repositório**
2. **Criar branch feature**: `git checkout -b feature/nova-funcionalidade`
3. **Implementar mudanças** seguindo padrões de código
4. **Adicionar testes** para nova funcionalidade
5. **Executar validações**: `invoke ci`
6. **Commit com mensagem descritiva**
7. **Push e criar Pull Request**

### Code Review Checklist

- [ ] Código segue padrões estabelecidos
- [ ] Documentação atualizada
- [ ] Testes adicionados e passando
- [ ] Performance não foi degradada
- [ ] Segurança foi considerada
- [ ] Compatibilidade mantida
- [ ] Logs apropriados adicionados

Para mais informações sobre contribuição, veja [CONTRIBUTING.md](../../CONTRIBUTING.md).
