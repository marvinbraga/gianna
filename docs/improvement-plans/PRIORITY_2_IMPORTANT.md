# Plano de Melhorias - Prioridade 2 (IMPORTANTE)

**Status:** Pendente
**Estimativa:** 5-7 dias
**Impacto:** Médio-Alto
**Dependências:** Prioridade 1 concluída

---

## 1. Implementar Async/Await onde Apropriado

**Severidade:** 🟡 MÉDIO
**Impacto:** Performance e responsividade

### Problema

O projeto é predominantemente síncrono, mas há operações que se beneficiariam de async:
- Chamadas a APIs de LLM (latência alta)
- Processamento de áudio em streaming
- Operações de I/O (cache, banco de dados)
- Múltiplas requisições concorrentes

### Oportunidades Identificadas

#### 1.1. LLM Calls Assíncronos

```python
# ANTES (síncrono)
# gianna/assistants/models/factory_method.py

def invoke_llm(prompt: str) -> str:
    response = llm.invoke(prompt)  # Bloqueia até resposta
    return response.content

# DEPOIS (assíncrono)
async def invoke_llm_async(prompt: str) -> str:
    """Invoke LLM asynchronously."""
    response = await llm.ainvoke(prompt)  # Não bloqueia
    return response.content

# Batch processing
async def invoke_llm_batch_async(prompts: List[str]) -> List[str]:
    """Process multiple prompts concurrently."""
    tasks = [invoke_llm_async(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    return responses
```

#### 1.2. Audio Processing Assíncrono

```python
# gianna/audio/vad/base.py

class BaseVAD:
    async def process_stream_async(self, audio_chunk: AudioChunk) -> VADResult:
        """
        Process audio chunk asynchronously.

        Allows non-blocking audio processing for better responsiveness.
        """
        loop = asyncio.get_event_loop()

        # Run CPU-intensive VAD in thread pool
        result = await loop.run_in_executor(
            None,  # Default thread pool
            self.process_stream,
            audio_chunk
        )

        return result

    async def process_stream_batch_async(
        self,
        audio_chunks: List[AudioChunk]
    ) -> List[VADResult]:
        """Process multiple audio chunks concurrently."""
        tasks = [
            self.process_stream_async(chunk)
            for chunk in audio_chunks
        ]
        results = await asyncio.gather(*tasks)
        return results
```

#### 1.3. Cache Operations Assíncronas

```python
# gianna/optimization/caching.py

class AsyncMemoryCache:
    """Async-compatible memory cache."""

    def __init__(self):
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache asynchronously."""
        async with self.lock:
            entry = self.cache.get(key)

            if entry and entry.is_expired():
                del self.cache[key]
                return None

            return entry

    async def set(self, entry: CacheEntry) -> bool:
        """Set entry in cache asynchronously."""
        async with self.lock:
            # Check memory limits
            if await self._should_evict():
                await self._evict_by_lru()

            self.cache[entry.key] = entry
            return True

    async def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], Awaitable[Any]],
        ttl: int = 3600,
    ) -> Any:
        """Get from cache or compute asynchronously."""
        entry = await self.get(key)

        if entry:
            return entry.value

        # Compute value asynchronously
        value = await compute_func()

        # Store in cache
        await self.set(CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
        ))

        return value
```

#### 1.4. Multi-Agent Coordination Assíncrona

```python
# gianna/coordination/orchestrator.py

class AsyncAgentOrchestrator:
    """Orchestrate multiple agents concurrently."""

    async def execute_parallel_tasks(
        self,
        tasks: List[AgentTask],
    ) -> List[AgentResult]:
        """
        Execute multiple agent tasks in parallel.

        Example:
            >>> tasks = [
            ...     AgentTask(agent="command", input="ls -la"),
            ...     AgentTask(agent="memory", input="search: Python"),
            ...     AgentTask(agent="audio", input="play music.mp3"),
            ... ]
            >>> results = await orchestrator.execute_parallel_tasks(tasks)
        """
        async_tasks = [
            self._execute_task_async(task)
            for task in tasks
        ]

        results = await asyncio.gather(
            *async_tasks,
            return_exceptions=True,  # Don't fail all if one fails
        )

        return results

    async def _execute_task_async(self, task: AgentTask) -> AgentResult:
        """Execute single task asynchronously."""
        agent = self.agents.get(task.agent_type)

        if not agent:
            raise ValueError(f"Unknown agent: {task.agent_type}")

        # Run agent in thread pool if sync, or await if async
        if hasattr(agent, 'execute_async'):
            result = await agent.execute_async(task.input)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                agent.execute,
                task.input
            )

        return result
```

### Passos de Implementação

1. **Fase 1: Infraestrutura (1 dia)**
   - Criar módulo `gianna/async_utils.py` com helpers
   - Implementar event loop management
   - Criar async fixtures para testes

2. **Fase 2: LLM Async (1 dia)**
   - Adicionar métodos `ainvoke` em factories
   - Implementar batch processing assíncrono
   - Testes de performance async vs sync

3. **Fase 3: Audio Async (1.5 dias)**
   - Adicionar `process_stream_async` em BaseVAD
   - Implementar em todas as implementações VAD
   - Testes de streaming assíncrono

4. **Fase 4: Cache Async (1 dia)**
   - Implementar AsyncMemoryCache
   - AsyncRedisCache (se redis disponível)
   - Migration path de sync para async

5. **Fase 5: Integration (1 dia)**
   - Multi-agent orchestration async
   - End-to-end async workflows
   - Performance benchmarks

### Benefícios

- ✅ Melhor performance em operações I/O-bound
- ✅ Capacidade de processar múltiplas requisições simultaneamente
- ✅ Responsividade melhorada em aplicações interativas
- ✅ Melhor uso de recursos do sistema

### Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Complexidade aumentada | Manter API síncrona e adicionar async como opção |
| Bugs em código async | Testes extensivos com pytest-asyncio |
| Compatibilidade | Manter backward compatibility com sync API |

---

## 2. Adicionar Circuit Breaker para Chamadas de API

**Severidade:** 🟡 MÉDIO
**Impacto:** Resiliência e reliability

### Problema

Chamadas a APIs externas (LLMs, TTS, etc) podem:
- Falhar temporariamente
- Ter timeout
- Ficar lentas
- Causar cascata de falhas

Atualmente não há proteção contra:
- Retry infinito
- Sobrecarga de APIs falhando
- Degradação cascata

### Solução Proposta

#### 2.1. Implementar Circuit Breaker Pattern

```python
# gianna/resilience/circuit_breaker.py

from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, TypeVar, Generic
import time

T = TypeVar('T')

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: int = 60  # Seconds before trying again
    expected_exception: type = Exception

class CircuitBreaker(Generic[T]):
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by temporarily blocking calls
    to failing services.

    Example:
        >>> breaker = CircuitBreaker(
        ...     failure_threshold=5,
        ...     timeout=60,
        ... )
        >>>
        >>> @breaker
        ... def call_external_api():
        ...     return api.get_data()
    """

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = get_logger(__name__)

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap function with circuit breaker."""
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. "
                    f"Retry after {self._seconds_until_retry()}s"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.config.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1

            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.config.timeout

    def _seconds_until_retry(self) -> int:
        """Get seconds until retry is allowed."""
        if not self.last_failure_time:
            return 0

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return max(0, int(self.config.timeout - elapsed))

    def _transition_to_open(self):
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self.logger.warning(
            f"Circuit breaker opened after {self.failure_count} failures"
        )

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.logger.info("Circuit breaker entering HALF_OPEN state")

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.success_count = 0
        self.failure_count = 0
        self.logger.info("Circuit breaker closed - service recovered")

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass
```

#### 2.2. Aplicar em LLM Calls

```python
# gianna/assistants/models/base.py

from gianna.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

class AbstractLLMFactory:
    def __init__(self):
        # Circuit breaker específico para cada provedor
        self.circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=3,  # Abrir após 3 falhas
                success_threshold=2,  # Fechar após 2 sucessos
                timeout=30,  # Tentar novamente após 30s
                expected_exception=(APIError, TimeoutError),
            )
        )

    @circuit_breaker
    def invoke(self, prompt: str) -> str:
        """Invoke LLM with circuit breaker protection."""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
```

#### 2.3. Aplicar em TTS/STT

```python
# gianna/assistants/audio/tts/base.py

class AbstractTextToSpeech:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=5,
                timeout=60,
            )
        )

    def speak(self, text: str) -> bytes:
        """Generate speech with circuit breaker protection."""
        @self.circuit_breaker
        def _speak():
            return self._generate_speech(text)

        try:
            return _speak()
        except CircuitBreakerOpenError as e:
            logger.warning(f"TTS circuit breaker open: {e}")
            # Fallback to cached or simple TTS
            return self._fallback_tts(text)
```

### Passos de Implementação

1. **Implementar Circuit Breaker base** (0.5 dia)
2. **Adicionar em LLM factories** (0.5 dia)
3. **Adicionar em TTS/STT** (0.5 dia)
4. **Testes extensivos** (1 dia)
5. **Monitoramento e métricas** (0.5 dia)

### Benefícios

- ✅ Previne cascata de falhas
- ✅ Melhora resiliência do sistema
- ✅ Feedback rápido sobre serviços degradados
- ✅ Permite fallbacks gracioso

---

## 3. Melhorar Tratamento de Erros em Streams

**Severidade:** 🟡 MÉDIO
**Impacto:** Reliability

### Problema

Streams de áudio podem falhar por:
- Dispositivo de áudio desconectado
- Buffer overflow/underflow
- Latência alta
- Interrupções de rede (para TTS/STT remoto)

Atualmente tratamento de erros é básico.

### Solução Proposta

```python
# gianna/audio/streaming.py

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

class ResilientAudioStream:
    """Audio stream with robust error handling."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((IOError, OSError)),
    )
    def start_stream(self):
        """Start audio stream with automatic retry."""
        try:
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._stream_callback,
            )
        except OSError as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """Stream callback with error handling."""
        if status:
            logger.warning(f"Stream status: {status}")

            if status & pyaudio.paInputOverflow:
                logger.error("Input overflow - audio data lost")
                # Notify listeners
                self._on_buffer_overflow()

            if status & pyaudio.paInputUnderflow:
                logger.error("Input underflow - insufficient data")
                # Attempt recovery
                self._recover_from_underflow()

        try:
            # Process audio data
            self._process_audio_data(in_data)
            return (in_data, pyaudio.paContinue)

        except Exception as e:
            logger.error(f"Error in stream callback: {e}")
            # Continue stream but notify error
            self._on_processing_error(e)
            return (in_data, pyaudio.paContinue)

    def _recover_from_underflow(self):
        """Attempt to recover from buffer underflow."""
        # Increase buffer size temporarily
        self.chunk_size = int(self.chunk_size * 1.5)
        logger.info(f"Increased buffer size to {self.chunk_size}")
```

### Benefícios

- ✅ Streams mais robustos
- ✅ Recuperação automática de erros
- ✅ Melhor experiência do usuário

---

## 4. Memory Pool para Audio Chunks

**Severidade:** 🟡 MÉDIO
**Impacto:** Performance (redução de GC)

### Problema

Processamento de áudio cria/destrói muitos numpy arrays:
- Alocação constante
- Garbage collection frequente
- Fragmentação de memória

### Solução

```python
# gianna/audio/memory_pool.py

import numpy as np
from typing import Optional
from threading import Lock

class AudioChunkPool:
    """
    Memory pool for audio chunks to reduce GC pressure.

    Pre-allocates numpy arrays and reuses them.
    """

    def __init__(
        self,
        pool_size: int = 1000,
        chunk_size: int = 1024,
        dtype: np.dtype = np.int16,
    ):
        self.pool_size = pool_size
        self.chunk_size = chunk_size
        self.dtype = dtype

        # Pre-allocate pool
        self.available = [
            np.zeros(chunk_size, dtype=dtype)
            for _ in range(pool_size)
        ]
        self.in_use = set()
        self.lock = Lock()

        # Stats
        self.stats = {
            "gets": 0,
            "puts": 0,
            "allocations": 0,  # When pool is empty
            "pool_hits": 0,
        }

    def get(self) -> np.ndarray:
        """Get audio chunk from pool."""
        with self.lock:
            self.stats["gets"] += 1

            if self.available:
                chunk = self.available.pop()
                self.in_use.add(id(chunk))
                self.stats["pool_hits"] += 1
                return chunk
            else:
                # Pool exhausted, allocate new
                chunk = np.zeros(self.chunk_size, dtype=self.dtype)
                self.in_use.add(id(chunk))
                self.stats["allocations"] += 1
                logger.warning(f"Audio pool exhausted, allocated new chunk")
                return chunk

    def put(self, chunk: np.ndarray):
        """Return chunk to pool."""
        with self.lock:
            self.stats["puts"] += 1

            chunk_id = id(chunk)
            if chunk_id in self.in_use:
                self.in_use.remove(chunk_id)

            if len(self.available) < self.pool_size:
                # Zero out for reuse
                chunk.fill(0)
                self.available.append(chunk)

    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self.lock:
            hit_rate = (
                self.stats["pool_hits"] / self.stats["gets"]
                if self.stats["gets"] > 0
                else 0
            )
            return {
                **self.stats,
                "hit_rate": hit_rate,
                "available": len(self.available),
                "in_use": len(self.in_use),
            }

# Usage in VAD
class BaseVAD:
    def __init__(self):
        self.audio_pool = AudioChunkPool(
            pool_size=1000,
            chunk_size=1024,
        )

    def process_stream(self, audio_data: bytes) -> VADResult:
        # Get chunk from pool
        chunk = self.audio_pool.get()

        try:
            # Convert bytes to numpy array into pre-allocated chunk
            np.frombuffer(audio_data, dtype=np.int16, out=chunk)

            # Process
            result = self.detect_activity(chunk)

            return result
        finally:
            # Return to pool
            self.audio_pool.put(chunk)
```

### Benefícios

- ✅ Redução de 70-80% em alocações
- ✅ Menos GC pauses
- ✅ Melhor performance em streaming

---

## Estimativa de Esforço Total

| Task | Esforço | Prioridade |
|------|---------|-----------|
| 1. Async/Await | 3 dias | P1 |
| 2. Circuit Breaker | 2 dias | P1 |
| 3. Error Handling Streams | 1 dia | P2 |
| 4. Memory Pool | 1 dia | P2 |

**Total:** 7 dias

---

## Métricas de Sucesso

- [ ] Throughput aumentado em 50%+ (async)
- [ ] Zero cascading failures (circuit breaker)
- [ ] Stream uptime > 99% (error handling)
- [ ] GC pauses reduzidas 70%+ (memory pool)
- [ ] Testes passando com coverage >= 85%
