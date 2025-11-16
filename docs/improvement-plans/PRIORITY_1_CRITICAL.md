# Plano de Melhorias - Prioridade 1 (CRÍTICO)

**Status:** Pendente
**Estimativa:** 3-5 dias
**Impacto:** Alto
**Risco de não fazer:** Alto

---

## 1. Refatorar `_build_routing_rules()` - 278 linhas

**Arquivo:** `gianna/coordination/router.py`
**Linhas:** 1-278
**Severidade:** 🔴 CRÍTICO

### Problema

Método com 278 linhas contendo múltiplas listas hardcoded de keywords/patterns:
- 300+ keywords para agent COMMAND
- 200+ keywords para agent AUDIO
- Estrutura repetitiva
- Difícil de manter e estender

### Impacto

- **Manutenibilidade:** Difícil adicionar novos agentes ou modificar keywords
- **Testabilidade:** Impossível testar regras isoladamente
- **Extensibilidade:** Adicionar novo agente requer modificar código-fonte

### Solução Proposta

#### 1.1. Criar arquivo de configuração YAML

```yaml
# config/routing_rules.yaml
version: "1.0"
rules:
  - agent_type: "COMMAND"
    priority: 1
    keywords:
      - "executar"
      - "rodar"
      - "criar"
      # ... resto das keywords
    patterns:
      - "^(execute|run|create)\\s+"
      - "^make\\s+\\w+"

  - agent_type: "AUDIO"
    priority: 2
    keywords:
      - "reproduzir"
      - "tocar"
      - "gravar"
      # ... resto das keywords
    patterns:
      - "^(play|record)\\s+"

  - agent_type: "MEMORY"
    priority: 3
    keywords:
      - "lembrar"
      - "buscar"
      - "pesquisar"
    patterns:
      - "^(remember|search|find)\\s+"
```

#### 1.2. Implementar loader de configuração

```python
# gianna/coordination/router.py

import yaml
from pathlib import Path
from typing import List, Dict, Any

class AgentRouter:
    def __init__(
        self,
        llm: BaseChatModel,
        agents: Dict[AgentType, BaseReActAgent],
        config_path: Optional[Path] = None,
    ):
        self.llm = llm
        self.agents = agents
        self.config_path = config_path or Path(__file__).parent.parent.parent / "config" / "routing_rules.yaml"
        self.routing_rules = self._load_routing_rules_from_config()

    def _load_routing_rules_from_config(self) -> List[RoutingRule]:
        """Load routing rules from YAML configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            rules = []
            for rule_config in config.get('rules', []):
                # Compile regex patterns
                compiled_patterns = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in rule_config.get('patterns', [])
                ]

                rule = RoutingRule(
                    agent_type=AgentType[rule_config['agent_type'].upper()],
                    keywords=rule_config.get('keywords', []),
                    patterns=compiled_patterns,
                    priority=rule_config.get('priority', 1),
                )
                rules.append(rule)

            logger.info(f"Loaded {len(rules)} routing rules from {self.config_path}")
            return rules

        except FileNotFoundError:
            logger.error(f"Routing rules config not found: {self.config_path}")
            return self._get_default_rules()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing routing rules YAML: {e}")
            return self._get_default_rules()

    def _get_default_rules(self) -> List[RoutingRule]:
        """Fallback to minimal default rules if config cannot be loaded."""
        return [
            RoutingRule(
                agent_type=AgentType.COMMAND,
                keywords=["executar", "rodar", "criar"],
                patterns=[re.compile(r"^(execute|run|create)\s+", re.IGNORECASE)],
                priority=1,
            ),
            # ... minimal defaults
        ]
```

#### 1.3. Adicionar validação de configuração

```python
# gianna/coordination/router_validator.py

from pydantic import BaseModel, validator, Field
from typing import List, Optional
from enum import Enum

class AgentTypeEnum(str, Enum):
    COMMAND = "COMMAND"
    AUDIO = "AUDIO"
    MEMORY = "MEMORY"
    LEARNING = "LEARNING"
    GENERAL = "GENERAL"

class RoutingRuleConfig(BaseModel):
    agent_type: AgentTypeEnum
    priority: int = Field(ge=1, le=10)
    keywords: List[str]
    patterns: Optional[List[str]] = []

    @validator('keywords')
    def keywords_not_empty(cls, v):
        if not v:
            raise ValueError('Keywords list cannot be empty')
        return v

    @validator('patterns')
    def validate_regex_patterns(cls, v):
        if v:
            for pattern in v:
                try:
                    re.compile(pattern)
                except re.error as e:
                    raise ValueError(f'Invalid regex pattern "{pattern}": {e}')
        return v

class RoutingRulesConfig(BaseModel):
    version: str
    rules: List[RoutingRuleConfig]

    @validator('rules')
    def rules_not_empty(cls, v):
        if not v:
            raise ValueError('Rules list cannot be empty')
        return v
```

### Passos de Implementação

1. **Criar estrutura de diretórios**
   ```bash
   mkdir -p config
   mkdir -p gianna/coordination/validators
   ```

2. **Criar arquivo de configuração YAML**
   - Migrar todas as keywords/patterns atuais
   - Organizar por agent_type
   - Adicionar comentários explicativos

3. **Implementar validator Pydantic**
   - Validar estrutura do YAML
   - Validar regex patterns
   - Validar tipos de agentes

4. **Refatorar classe AgentRouter**
   - Adicionar método `_load_routing_rules_from_config()`
   - Implementar fallback para regras padrão
   - Adicionar tratamento de erros robusto

5. **Adicionar testes**
   ```python
   # tests/unit/test_router_config.py

   def test_load_routing_rules_from_valid_config():
       router = AgentRouter(llm, agents, config_path=Path("tests/fixtures/valid_routing_rules.yaml"))
       assert len(router.routing_rules) > 0

   def test_load_routing_rules_from_invalid_yaml():
       router = AgentRouter(llm, agents, config_path=Path("tests/fixtures/invalid_routing_rules.yaml"))
       # Should fallback to default rules
       assert len(router.routing_rules) > 0

   def test_routing_rules_validator():
       config = {
           "version": "1.0",
           "rules": [
               {
                   "agent_type": "COMMAND",
                   "priority": 1,
                   "keywords": ["executar"],
                   "patterns": ["^execute\\s+"]
               }
           ]
       }
       validated = RoutingRulesConfig(**config)
       assert validated.version == "1.0"
   ```

6. **Atualizar documentação**
   - Adicionar seção em README sobre configuração de routing
   - Documentar formato do YAML
   - Adicionar exemplos de customização

### Benefícios

- ✅ **Manutenibilidade:** Fácil adicionar/remover keywords sem tocar no código
- ✅ **Extensibilidade:** Novos agentes podem ser configurados sem recompilação
- ✅ **Testabilidade:** Regras podem ser testadas com diferentes configurações
- ✅ **Flexibilidade:** Usuários podem customizar routing sem modificar código-fonte
- ✅ **Separação de responsabilidades:** Dados separados de lógica

### Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|-------|--------------|-----------|
| Arquivo YAML inválido | Médio | Validação com Pydantic + fallback para defaults |
| Quebra de compatibilidade | Baixo | Manter defaults funcionais no código |
| Performance de I/O | Baixo | Cache de configuração em memória |

### Critérios de Aceitação

- [ ] Arquivo `config/routing_rules.yaml` criado com todas as regras atuais
- [ ] Classe `RoutingRulesConfig` implementada com validação Pydantic
- [ ] Método `_load_routing_rules_from_config()` implementado
- [ ] Fallback para regras padrão em caso de erro
- [ ] Testes unitários com coverage >= 90%
- [ ] Documentação atualizada
- [ ] Não há regressões em testes existentes

---

## 2. Adicionar Tipos Específicos em Callbacks

**Arquivos Afetados:**
- `gianna/audio/vad/base.py`
- Todos os módulos que usam callbacks

**Severidade:** 🔴 CRÍTICO

### Problema

Callbacks são definidos como `callable` genérico sem type hints específicos:

```python
def set_speech_start_callback(self, callback: Optional[callable]) -> None:
    self.speech_start_callback = callback
```

### Impacto

- **Type Safety:** IDE não consegue inferir tipos de parâmetros
- **Documentação:** Não é claro quais parâmetros o callback deve receber
- **Erros em Runtime:** Assinatura incorreta só é detectada ao executar

### Solução Proposta

```python
# gianna/audio/vad/types.py

from typing import Protocol, Callable, Optional
from gianna.audio.vad.models import VADResult

class VADCallbackProtocol(Protocol):
    """Protocol for VAD event callbacks."""
    def __call__(self, result: VADResult) -> None:
        """
        Called when a VAD event occurs.

        Args:
            result: VAD detection result containing audio data and metadata
        """
        ...

# Definir tipos específicos
VADCallback = Callable[[VADResult], None]
SpeechStartCallback = Callable[[VADResult], None]
SpeechEndCallback = Callable[[VADResult], None]
ActivityCallback = Callable[[VADResult], None]

# Callbacks opcionais
OptionalVADCallback = Optional[VADCallback]
```

```python
# gianna/audio/vad/base.py

from gianna.audio.vad.types import (
    SpeechStartCallback,
    SpeechEndCallback,
    ActivityCallback,
)

class BaseVAD:
    def set_speech_start_callback(
        self,
        callback: Optional[SpeechStartCallback]
    ) -> None:
        """
        Set callback for speech start events.

        Args:
            callback: Function to call when speech starts.
                     Must accept VADResult as parameter.

        Example:
            >>> def on_speech_start(result: VADResult) -> None:
            ...     print(f"Speech detected: {result.is_speech}")
            >>> vad.set_speech_start_callback(on_speech_start)
        """
        self.speech_start_callback = callback
```

### Passos de Implementação

1. **Criar módulo de tipos**
   - `gianna/audio/vad/types.py`
   - Definir todos os callback protocols

2. **Atualizar BaseVAD**
   - Adicionar type hints específicos
   - Atualizar docstrings

3. **Atualizar todas as implementações**
   - EnergyVAD, SpectralVAD, WebRtcVAD, SileroVAD, AdaptiveVAD

4. **Adicionar testes de tipo**
   ```python
   # tests/unit/test_vad_types.py

   from gianna.audio.vad.types import VADCallback
   from gianna.audio.vad.models import VADResult

   def test_vad_callback_signature():
       def valid_callback(result: VADResult) -> None:
           pass

       # Type checker should pass
       callback: VADCallback = valid_callback
   ```

### Benefícios

- ✅ Type safety melhorada
- ✅ Melhor experiência de desenvolvimento (autocomplete)
- ✅ Documentação clara de assinaturas
- ✅ Detecção de erros em tempo de desenvolvimento

---

## 3. Remover Bare Except Statements

**Arquivos Afetados:**
- `gianna/optimization/performance.py`
- `gianna/production/cache_manager.py`
- `gianna/learning/adaptation_engine.py` (2 instâncias)

**Severidade:** 🔴 CRÍTICO

### Problema

Uso de `except:` sem especificar exceção captura **todos** os erros, incluindo:
- `KeyboardInterrupt` (Ctrl+C)
- `SystemExit`
- `GeneratorExit`
- Bugs no código

```python
try:
    do_something()
except:  # ❌ MAU - captura tudo
    log_error()
```

### Impacto

- **Debugging:** Dificulta encontrar bugs
- **Controle:** Impossível interromper com Ctrl+C
- **Silencia erros:** Esconde problemas reais

### Solução Proposta

```python
# PADRÃO RECOMENDADO

try:
    do_something()
except (SpecificError1, SpecificError2) as e:
    # Tratar erros esperados
    logger.error(f"Expected error: {e}")
except Exception as e:
    # Tratar erros inesperados (mas não KeyboardInterrupt, etc)
    logger.error(f"Unexpected error: {e}", exc_info=True)
    # Opcionalmente re-raise
    raise
```

### Implementação por Arquivo

#### 3.1. `gianna/optimization/performance.py`

Localizar e corrigir:
```python
# Localizar com grep
grep -n "except:" gianna/optimization/performance.py

# Substituir por:
except (CacheError, SerializationError) as e:
    logger.error(f"Performance monitoring error: {e}", exc_info=True)
except Exception as e:
    logger.critical(f"Unexpected error in performance monitor: {e}", exc_info=True)
    raise
```

#### 3.2. `gianna/production/cache_manager.py`

```python
# Localizar
grep -n "except:" gianna/production/cache_manager.py

# Substituir por:
except (CacheConnectionError, SerializationError) as e:
    logger.warning(f"Cache operation failed: {e}")
    # Fallback gracioso
    return None
except Exception as e:
    logger.error(f"Unexpected cache error: {e}", exc_info=True)
    raise
```

#### 3.3. `gianna/learning/adaptation_engine.py`

```python
# 2 instâncias
grep -n "except:" gianna/learning/adaptation_engine.py

# Substituir por:
except (PatternAnalysisError, ModelUpdateError) as e:
    logger.warning(f"Learning adaptation failed: {e}")
    # Continue com fallback
except Exception as e:
    logger.error(f"Unexpected learning error: {e}", exc_info=True)
    # Não raise - learning é opcional
```

### Passos de Implementação

1. **Identificar todas as instâncias**
   ```bash
   grep -rn "except:" gianna/ --include="*.py" | grep -v "except Exception"
   ```

2. **Para cada instância:**
   - Identificar possíveis exceções específicas
   - Adicionar tratamento apropriado
   - Adicionar logging estruturado
   - Decidir: re-raise ou handle gracefully

3. **Adicionar custom exceptions se necessário**
   ```python
   # gianna/exceptions.py

   class GiannaError(Exception):
       """Base exception for Gianna."""

   class CacheError(GiannaError):
       """Cache-related errors."""

   class PerformanceMonitoringError(GiannaError):
       """Performance monitoring errors."""

   class LearningError(GiannaError):
       """Learning system errors."""
   ```

4. **Adicionar testes**
   ```python
   def test_no_bare_except_in_performance_monitor():
       # Verificar que exceções específicas são capturadas
       monitor = PerformanceMonitor()

       with pytest.raises(SpecificError):
           monitor.some_method_that_should_fail()
   ```

5. **Configurar linter para detectar**
   ```python
   # .flake8
   [flake8]
   select = E,W,F,C,B,B9
   # B001: bare except
   ```

### Critérios de Aceitação

- [ ] Todas as 4 instâncias de bare except corrigidas
- [ ] Custom exceptions criadas se necessário
- [ ] Logging estruturado adicionado
- [ ] Testes atualizados
- [ ] Flake8 configurado para detectar bare except
- [ ] CI passa com novas regras

---

## 4. Corrigir Hardcoded Test API Keys

**Arquivo:** `tests/conftest.py` (linhas 45-54)
**Severidade:** 🔴 CRÍTICO (Segurança)

### Problema

```python
test_api_keys = {
    "OPENAI_API_KEY": "test-openai-key",
    "GOOGLE_API_KEY": "test-google-key",
    "ELEVEN_LABS_API_KEY": "test-elevenlabs-key",
    # ... mais keys hardcoded
}
```

Embora sejam keys de teste, isso é má prática porque:
- Pode acidentalmente vazar em logs
- Não segue best practices de segurança
- Dificulta uso de keys reais em testes de integração

### Solução Proposta

```python
# tests/conftest.py

import os
import pytest

@pytest.fixture
def mock_api_keys(monkeypatch):
    """
    Mock API keys for testing.

    Can be overridden with environment variables:
    - TEST_USE_REAL_KEYS=1: Use real API keys from environment
    - TEST_OPENAI_API_KEY: Override OpenAI key for tests
    """
    use_real_keys = os.getenv("TEST_USE_REAL_KEYS", "0") == "1"

    keys = {
        "OPENAI_API_KEY": os.getenv("TEST_OPENAI_API_KEY", "sk-test-fake-key-123"),
        "GOOGLE_API_KEY": os.getenv("TEST_GOOGLE_API_KEY", "test-google-fake"),
        "ELEVEN_LABS_API_KEY": os.getenv("TEST_ELEVEN_LABS_API_KEY", "test-el-fake"),
        "ANTHROPIC_API_KEY": os.getenv("TEST_ANTHROPIC_API_KEY", "sk-ant-test-fake"),
        "GROQ_API_KEY": os.getenv("TEST_GROQ_API_KEY", "gsk-test-fake"),
        "NVIDIA_API_KEY": os.getenv("TEST_NVIDIA_API_KEY", "nvapi-test-fake"),
        "COHERE_API_KEY": os.getenv("TEST_COHERE_API_KEY", "test-cohere-fake"),
        "XAI_API_KEY": os.getenv("TEST_XAI_API_KEY", "xai-test-fake"),
    }

    # Se usar keys reais, pegar do environment
    if use_real_keys:
        for key in keys.keys():
            real_key = os.getenv(key)
            if real_key:
                keys[key] = real_key

    # Set environment variables
    for key, value in keys.items():
        monkeypatch.setenv(key, value)

    return keys
```

### Passos de Implementação

1. **Atualizar fixture de API keys**
   - Usar monkeypatch do pytest
   - Suportar override com env vars
   - Adicionar modo de teste com keys reais

2. **Criar arquivo .env.test.example**
   ```bash
   # .env.test.example

   # Set to 1 to use real API keys in integration tests
   TEST_USE_REAL_KEYS=0

   # Override specific test keys (optional)
   # TEST_OPENAI_API_KEY=sk-...
   # TEST_ANTHROPIC_API_KEY=sk-ant-...
   ```

3. **Atualizar documentação de testes**
   ```markdown
   ## Running Tests

   ### Unit Tests (default - uses mock keys)
   ```bash
   pytest tests/unit/
   ```

   ### Integration Tests (with real APIs)
   ```bash
   # Set up real API keys
   cp .env.test.example .env.test
   # Edit .env.test with real keys

   TEST_USE_REAL_KEYS=1 pytest tests/integration/ --external-api
   ```
   ```

4. **Adicionar .env.test ao .gitignore**
   ```bash
   echo ".env.test" >> .gitignore
   ```

### Benefícios

- ✅ Segurança melhorada
- ✅ Flexibilidade para testes de integração
- ✅ Segue best practices
- ✅ Mais fácil de debugar com keys reais

---

## Estimativa de Esforço

| Task | Esforço | Prioridade |
|------|---------|-----------|
| 1. Refatorar routing rules | 2 dias | P0 |
| 2. Tipos em callbacks | 1 dia | P0 |
| 3. Remover bare except | 0.5 dia | P0 |
| 4. Corrigir test API keys | 0.5 dia | P0 |

**Total:** 4 dias

---

## Ordem de Execução Recomendada

1. **Dia 1:** Task 4 (API keys) + Task 3 (bare except) - Correções rápidas
2. **Dia 2-3:** Task 1 (Routing rules) - Refatoração grande
3. **Dia 4:** Task 2 (Callbacks) - Tipos

---

## Métricas de Sucesso

- [ ] Todos os 4 tasks concluídos
- [ ] Coverage de testes mantido >= 80%
- [ ] CI/CD passando
- [ ] Sem regressões em testes existentes
- [ ] Code review aprovado
- [ ] Documentação atualizada
