# Plano de Melhorias - Prioridade 3 (ENHANCEMENTS)

**Status:** Pendente
**Estimativa:** 7-10 dias
**Impacto:** Médio
**Dependências:** Prioridades 1 e 2 concluídas

---

## 1. Melhorar Test Coverage

**Coverage Atual:** 80% (mínimo requisito)
**Coverage Alvo:** 90%+
**Severidade:** 🟢 BAIXO (mas importante para qualidade)

### Situação Atual

Análise de coverage mostra:
- **Total de arquivos-fonte:** 129
- **Total de arquivos de teste:** 13
- **Razão teste/source:** 0.10 (10%)
- **Coverage atual:** 80% (pytest --cov-fail-under=80)

### Módulos com Coverage Insuficiente

Executar análise detalhada:
```bash
pytest --cov=gianna --cov-report=term-missing --cov-report=html
```

Módulos típicos com coverage < 80%:
1. Edge cases em error handling
2. Callbacks e event handlers
3. Código de inicialização/shutdown
4. Integrações com APIs externas
5. Código de UI/CLI

### Estratégia de Melhoria

#### 1.1. Testes de Edge Cases

```python
# tests/unit/test_vad_edge_cases.py

import pytest
from gianna.audio.vad import EnergyVAD, SpectralVAD

class TestVADEdgeCases:
    """Test VAD edge cases and error conditions."""

    def test_empty_audio_data(self):
        """Test VAD with empty audio data."""
        vad = EnergyVAD()

        with pytest.raises(ValueError, match="Audio data cannot be empty"):
            vad.process_stream(b"")

    def test_corrupted_audio_data(self):
        """Test VAD with corrupted audio data."""
        vad = EnergyVAD()
        corrupted_data = b"\xff" * 100  # Invalid audio

        # Should handle gracefully
        result = vad.process_stream(corrupted_data)
        assert result is not None

    def test_extremely_loud_audio(self):
        """Test VAD with clipping audio."""
        vad = EnergyVAD(threshold=0.5)

        # Generate clipping audio (max int16 values)
        import numpy as np
        loud_audio = np.full(1024, 32767, dtype=np.int16).tobytes()

        result = vad.process_stream(loud_audio)
        assert result.is_speech == True

    def test_extremely_quiet_audio(self):
        """Test VAD with near-silent audio."""
        vad = EnergyVAD(threshold=0.5)

        # Generate near-silent audio
        import numpy as np
        quiet_audio = np.full(1024, 10, dtype=np.int16).tobytes()

        result = vad.process_stream(quiet_audio)
        assert result.is_speech == False

    def test_rapid_state_transitions(self):
        """Test VAD with rapid speech/silence transitions."""
        vad = EnergyVAD()

        transitions = []
        def on_state_change(result):
            transitions.append(result.is_speech)

        vad.set_speech_start_callback(on_state_change)

        # Simulate rapid transitions
        import numpy as np
        for i in range(100):
            if i % 2 == 0:
                audio = np.full(1024, 10000, dtype=np.int16).tobytes()
            else:
                audio = np.full(1024, 10, dtype=np.int16).tobytes()

            vad.process_stream(audio)

        # Should have detected transitions
        assert len(transitions) > 0
```

#### 1.2. Testes de Callbacks e Events

```python
# tests/unit/test_callbacks.py

import pytest
from unittest.mock import Mock, call
from gianna.audio.vad import EnergyVAD

class TestVADCallbacks:
    """Test VAD callback functionality."""

    def test_speech_start_callback_called(self):
        """Test that speech start callback is invoked."""
        vad = EnergyVAD()
        callback = Mock()
        vad.set_speech_start_callback(callback)

        # Generate speech audio
        import numpy as np
        speech_audio = np.full(1024, 10000, dtype=np.int16).tobytes()
        vad.process_stream(speech_audio)

        callback.assert_called_once()

    def test_speech_end_callback_called(self):
        """Test that speech end callback is invoked."""
        vad = EnergyVAD(min_silence_duration=0.1)
        callback = Mock()
        vad.set_speech_end_callback(callback)

        import numpy as np

        # Start speech
        speech = np.full(1024, 10000, dtype=np.int16).tobytes()
        vad.process_stream(speech)

        # End speech
        silence = np.full(1024, 10, dtype=np.int16).tobytes()
        for _ in range(10):  # Enough silence
            vad.process_stream(silence)

        callback.assert_called()

    def test_callback_exception_handled(self):
        """Test that exceptions in callbacks are handled."""
        vad = EnergyVAD()

        def failing_callback(result):
            raise ValueError("Callback error")

        vad.set_speech_start_callback(failing_callback)

        # Should not crash
        import numpy as np
        speech = np.full(1024, 10000, dtype=np.int16).tobytes()
        vad.process_stream(speech)  # Should not raise

    def test_multiple_callbacks(self):
        """Test multiple callbacks on same event."""
        vad = EnergyVAD()
        callback1 = Mock()
        callback2 = Mock()

        vad.add_event_callback("speech_start", callback1)
        vad.add_event_callback("speech_start", callback2)

        import numpy as np
        speech = np.full(1024, 10000, dtype=np.int16).tobytes()
        vad.process_stream(speech)

        callback1.assert_called_once()
        callback2.assert_called_once()
```

#### 1.3. Testes de Integração com APIs Reais (Opcional)

```python
# tests/integration/test_llm_apis.py

import pytest
import os

@pytest.mark.skipif(
    os.getenv("TEST_USE_REAL_KEYS") != "1",
    reason="Requires real API keys"
)
@pytest.mark.external_api
class TestLLMAPIsIntegration:
    """Integration tests with real LLM APIs."""

    def test_openai_gpt4_real(self):
        """Test real OpenAI GPT-4 call."""
        from gianna.assistants.models import get_chain_instance

        chain = get_chain_instance(
            model_name="gpt-4",
            prompt="You are a helpful assistant"
        )

        response = chain.invoke("What is 2+2?")
        assert "4" in response.lower()

    def test_anthropic_claude_real(self):
        """Test real Anthropic Claude call."""
        from gianna.assistants.models import get_chain_instance

        chain = get_chain_instance(
            model_name="claude-3-sonnet",
            prompt="You are a helpful assistant"
        )

        response = chain.invoke("What is the capital of France?")
        assert "paris" in response.lower()
```

#### 1.4. Testes de Property-Based Testing

```python
# tests/unit/test_vad_properties.py

from hypothesis import given, strategies as st
import numpy as np

@given(
    threshold=st.floats(min_value=0.0, max_value=1.0),
    audio_length=st.integers(min_value=1, max_value=10000),
)
def test_vad_threshold_property(threshold, audio_length):
    """Property test: VAD should respect threshold."""
    from gianna.audio.vad import EnergyVAD

    vad = EnergyVAD(threshold=threshold)

    # Generate random audio
    audio = np.random.randint(-32768, 32767, audio_length, dtype=np.int16)

    # Should not crash
    result = vad.process_stream(audio.tobytes())
    assert result is not None

@given(
    sample_rate=st.sampled_from([8000, 16000, 44100, 48000]),
    channels=st.integers(min_value=1, max_value=2),
)
def test_vad_audio_format_property(sample_rate, channels):
    """Property test: VAD should handle various audio formats."""
    from gianna.audio.vad import EnergyVAD

    vad = EnergyVAD(sample_rate=sample_rate)

    # Generate audio
    audio = np.random.randint(-100, 100, 1024, dtype=np.int16)

    result = vad.process_stream(audio.tobytes())
    assert result is not None
```

### Passos de Implementação

1. **Análise de Coverage (0.5 dia)**
   ```bash
   pytest --cov=gianna --cov-report=html
   # Identificar módulos com coverage < 80%
   ```

2. **Criar testes para gaps identificados (3 dias)**
   - Edge cases
   - Error paths
   - Callbacks
   - Integrações

3. **Property-based testing (1 dia)**
   - Instalar hypothesis
   - Adicionar property tests

4. **Testes de integração (1 dia)**
   - APIs reais (opcional)
   - End-to-end workflows

5. **Revisão e refinamento (0.5 dia)**
   - Verificar coverage final
   - Otimizar testes lentos

### Ferramentas

```bash
# Instalar ferramentas de coverage
pip install coverage pytest-cov

# Instalar hypothesis para property testing
pip install hypothesis

# Gerar relatório HTML
pytest --cov=gianna --cov-report=html --cov-report=term-missing

# Abrir relatório
open htmlcov/index.html
```

### Métricas de Sucesso

- [ ] Coverage >= 90% overall
- [ ] Coverage >= 85% em todos os módulos principais
- [ ] Todos os edge cases cobertos
- [ ] Property tests implementados
- [ ] CI/CD passando

---

## 2. Adicionar Logging Estruturado

**Severidade:** 🟢 BAIXO
**Impacto:** Observabilidade e debugging

### Problema Atual

Logging atual usa strings formatadas:
```python
logger.info(f"VAD initialized: threshold={threshold}")
```

**Problemas:**
- Difícil de parsear automaticamente
- Não é machine-readable
- Difícil fazer queries estruturadas

### Solução Proposta

```python
# gianna/monitoring/structured_logger.py

import json
from loguru import logger
from typing import Any, Dict
from datetime import datetime

class StructuredLogger:
    """Wrapper for structured logging."""

    def __init__(self, service_name: str = "gianna"):
        self.service_name = service_name
        self.logger = logger

        # Configure loguru for structured logging
        self.logger.add(
            "logs/gianna_{time}.json",
            format=self._json_formatter,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
        )

    def _json_formatter(self, record: dict) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "service": self.service_name,
            "message": record["message"],
            "extra": record.get("extra", {}),
        }

        if record.get("exception"):
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback,
            }

        return json.dumps(log_entry)

    def info(self, event: str, **kwargs):
        """Log info event with structured data."""
        self.logger.bind(**kwargs).info(event)

    def warning(self, event: str, **kwargs):
        """Log warning event with structured data."""
        self.logger.bind(**kwargs).warning(event)

    def error(self, event: str, **kwargs):
        """Log error event with structured data."""
        self.logger.bind(**kwargs).error(event)

# Usage
logger = StructuredLogger()

# ANTES
logger.info(f"VAD initialized: threshold={threshold}, sample_rate={sample_rate}")

# DEPOIS
logger.info(
    "vad_initialized",
    threshold=threshold,
    sample_rate=sample_rate,
    vad_type="energy",
)

# Output JSON:
# {
#   "timestamp": "2025-01-15T10:30:00",
#   "level": "INFO",
#   "service": "gianna",
#   "message": "vad_initialized",
#   "extra": {
#     "threshold": 0.5,
#     "sample_rate": 16000,
#     "vad_type": "energy"
#   }
# }
```

### Benefícios

- ✅ Logs machine-readable
- ✅ Fácil integração com ELK/Loki
- ✅ Queries estruturadas
- ✅ Melhor debugging

---

## 3. Database Connection Pooling

**Severidade:** 🟢 BAIXO
**Impacto:** Performance em alta carga

### Problema

Atualmente cada operação pode abrir nova conexão:
```python
# Potencial problema
def save_state(state: GiannaState):
    conn = sqlite3.connect(db_path)  # Nova conexão
    # ...
    conn.close()
```

### Solução

```python
# gianna/core/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

class DatabaseManager:
    """Manage database connections with pooling."""

    def __init__(self, db_url: str):
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=10,  # Connections to keep
            max_overflow=20,  # Additional connections when busy
            pool_timeout=30,  # Timeout waiting for connection
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,  # Set True for debugging
        )

        self.SessionFactory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.SessionFactory)

    @contextmanager
    def get_session(self):
        """Get database session from pool."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_pool_status(self) -> dict:
        """Get connection pool statistics."""
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total": pool.size() + pool.overflow(),
        }

# Usage
db_manager = DatabaseManager("sqlite:///gianna.db")

def save_state(state: GiannaState):
    with db_manager.get_session() as session:
        session.add(state)
        # Auto-commit on exit
```

---

## 4. Data Validation com Pydantic em Mais Lugares

**Severidade:** 🟢 BAIXO
**Impacto:** Robustez e type safety

### Expandir Uso de Pydantic

```python
# gianna/audio/models.py

from pydantic import BaseModel, validator, Field
from typing import Literal

class AudioConfig(BaseModel):
    """Configuration for audio processing."""

    sample_rate: Literal[8000, 16000, 44100, 48000] = Field(
        default=16000,
        description="Audio sample rate in Hz"
    )
    channels: Literal[1, 2] = Field(
        default=1,
        description="Number of audio channels"
    )
    chunk_size: int = Field(
        default=1024,
        ge=128,
        le=8192,
        description="Size of audio chunks in samples"
    )
    format: Literal["int16", "float32"] = Field(
        default="int16",
        description="Audio data format"
    )

    @validator('chunk_size')
    def chunk_size_power_of_2(cls, v):
        """Validate chunk size is power of 2 for FFT efficiency."""
        import math
        if v & (v - 1) != 0:
            raise ValueError(f"Chunk size must be power of 2, got {v}")
        return v

class VADConfig(BaseModel):
    """Configuration for VAD."""

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice activity threshold"
    )
    min_speech_duration: float = Field(
        default=0.3,
        ge=0.0,
        description="Minimum speech duration in seconds"
    )
    min_silence_duration: float = Field(
        default=0.5,
        ge=0.0,
        description="Minimum silence duration in seconds"
    )

    @validator('min_speech_duration')
    def speech_duration_reasonable(cls, v):
        if v > 10.0:
            raise ValueError("min_speech_duration > 10s is unrealistic")
        return v
```

---

## Estimativa de Esforço

| Task | Esforço | Prioridade |
|------|---------|-----------|
| 1. Melhorar coverage | 5 dias | P1 |
| 2. Logging estruturado | 2 dias | P2 |
| 3. DB pooling | 1 dia | P3 |
| 4. Mais Pydantic | 2 dias | P3 |

**Total:** 10 dias

---

## Métricas de Sucesso

- [ ] Coverage >= 90%
- [ ] Logs em formato JSON estruturado
- [ ] DB pool configurado e monitorado
- [ ] Validação Pydantic em 90%+ dos inputs
- [ ] CI/CD passando
