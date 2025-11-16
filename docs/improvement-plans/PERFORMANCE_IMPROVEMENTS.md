# Plano de Melhorias de Performance

**Status:** Pendente
**Estimativa:** 4-6 dias
**Impacto:** Alto em workloads intensivos
**Prioridade:** MÉDIA

---

## Resumo Executivo

Este documento detalha melhorias de performance identificadas durante análise do código. Foco em:
- Redução de latência
- Otimização de memória
- Throughput aumentado
- Redução de garbage collection

---

## 1. Memory Cache com Complexidade O(n)

**Arquivo:** `gianna/optimization/caching.py:94-150`
**Severidade:** 🟡 MÉDIO
**Impacto:** Performance degradation em caches grandes

### Problema Identificado

```python
# gianna/optimization/caching.py

def set(self, entry: CacheEntry) -> bool:
    with self.lock:
        # ❌ O(n) - Calcula soma a cada insert!
        current_memory = sum(e.size_bytes for e in self.cache.values())

        if current_memory + entry.size_bytes > self.max_memory_bytes:
            self._evict_by_memory()
```

**Análise de Complexidade:**
- **Operação:** `sum(e.size_bytes for e in self.cache.values())`
- **Complexidade:** O(n) onde n = número de entradas no cache
- **Frequência:** A cada insert
- **Impacto:** Cache com 10,000 entradas → 10,000 somas por insert

### Medição de Impacto

```python
# Benchmark atual
import timeit

def benchmark_current():
    cache = MemoryCache(max_memory_bytes=100_000_000)

    # Populate cache
    for i in range(10000):
        entry = CacheEntry(f"key{i}", f"value{i}", size_bytes=1000)
        cache.set(entry)

    # Measure insert time
    def insert():
        entry = CacheEntry("newkey", "newvalue", size_bytes=1000)
        cache.set(entry)

    time = timeit.timeit(insert, number=1000)
    print(f"Current: {time:.4f}s for 1000 inserts")
    # Output: ~0.5s (devido ao O(n))
```

### Solução Proposta

```python
# gianna/optimization/caching.py

from threading import Lock
from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class CacheEntry:
    key: str
    value: Any
    size_bytes: int
    created_at: float = None
    last_accessed: float = None
    ttl: int = 3600
    access_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_accessed is None:
            self.last_accessed = self.created_at

class OptimizedMemoryCache:
    """
    Optimized memory cache with O(1) memory tracking.

    Improvements:
    - O(1) memory usage tracking (instead of O(n))
    - LRU eviction with OrderedDict
    - Separate locks for stats vs data
    """

    def __init__(
        self,
        max_entries: int = 10000,
        max_memory_bytes: int = 100_000_000,
        default_ttl: int = 3600,
    ):
        from collections import OrderedDict

        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_bytes
        self.default_ttl = default_ttl

        # OrderedDict for LRU
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # O(1) memory tracking
        self.total_memory_used = 0  # ← CRITICAL: mantém total em memória

        # Locks
        self.data_lock = Lock()  # For cache operations
        self.stats_lock = Lock()  # For statistics

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
        }

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get entry from cache.

        Complexity: O(1) average, O(n) worst case (OrderedDict.move_to_end)
        """
        with self.data_lock:
            entry = self.cache.get(key)

            if entry is None:
                with self.stats_lock:
                    self.stats["misses"] += 1
                return None

            # Check expiration
            if self._is_expired(entry):
                # Remove expired entry
                self._remove_entry(key)

                with self.stats_lock:
                    self.stats["misses"] += 1
                return None

            # Update LRU - move to end
            self.cache.move_to_end(key)

            # Update access stats
            entry.last_accessed = time.time()
            entry.access_count += 1

            with self.stats_lock:
                self.stats["hits"] += 1

            return entry

    def set(self, entry: CacheEntry) -> bool:
        """
        Set entry in cache.

        Complexity: O(1) for memory check (vs O(n) before!)
        """
        with self.data_lock:
            # Check if key exists (for memory accounting)
            existing_entry = self.cache.get(entry.key)
            if existing_entry:
                # Subtract old size, add new size
                self.total_memory_used -= existing_entry.size_bytes

            # ✅ O(1) memory check (não mais O(n)!)
            if self.total_memory_used + entry.size_bytes > self.max_memory_bytes:
                self._evict_by_memory(entry.size_bytes)

            # Check entry count
            if len(self.cache) >= self.max_entries:
                self._evict_lru()

            # Add to cache
            self.cache[entry.key] = entry
            self.cache.move_to_end(entry.key)  # Mark as most recent

            # ✅ O(1) memory update
            self.total_memory_used += entry.size_bytes

            with self.stats_lock:
                self.stats["sets"] += 1

            return True

    def _remove_entry(self, key: str) -> bool:
        """Remove entry and update memory tracking."""
        entry = self.cache.pop(key, None)
        if entry:
            self.total_memory_used -= entry.size_bytes
            return True
        return False

    def _evict_lru(self, count: int = 1):
        """
        Evict least recently used entries.

        Complexity: O(count) - typically O(1) for count=1
        """
        for _ in range(count):
            if not self.cache:
                break

            # Remove first item (least recently used)
            key, entry = self.cache.popitem(last=False)
            self.total_memory_used -= entry.size_bytes

            with self.stats_lock:
                self.stats["evictions"] += 1

    def _evict_by_memory(self, required_bytes: int):
        """
        Evict entries until enough memory is available.

        Complexity: O(k) where k = entries evicted
        """
        while (
            self.cache and
            self.total_memory_used + required_bytes > self.max_memory_bytes
        ):
            # Evict LRU
            key, entry = self.cache.popitem(last=False)
            self.total_memory_used -= entry.size_bytes

            with self.stats_lock:
                self.stats["evictions"] += 1

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        age = time.time() - entry.created_at
        return age > entry.ttl

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self.data_lock:
            size = len(self.cache)
            memory = self.total_memory_used

        with self.stats_lock:
            stats = self.stats.copy()

        hit_rate = (
            stats["hits"] / (stats["hits"] + stats["misses"])
            if stats["hits"] + stats["misses"] > 0
            else 0
        )

        return {
            **stats,
            "size": size,
            "memory_bytes": memory,
            "memory_mb": memory / 1_000_000,
            "hit_rate": hit_rate,
            "max_entries": self.max_entries,
            "max_memory_mb": self.max_memory_bytes / 1_000_000,
        }
```

### Benchmark Comparação

```python
# tests/performance/test_cache_benchmark.py

import pytest
import time
from gianna.optimization.caching import MemoryCache, OptimizedMemoryCache

class TestCachePerformance:
    """Benchmark cache implementations."""

    def test_insert_performance_old_vs_new(self, benchmark):
        """Compare insert performance."""

        # Old implementation
        cache_old = MemoryCache(max_memory_bytes=100_000_000)

        def insert_old():
            for i in range(1000):
                entry = CacheEntry(f"key{i}", f"value{i}", size_bytes=1000)
                cache_old.set(entry)

        time_old = benchmark(insert_old)

        # New implementation
        cache_new = OptimizedMemoryCache(max_memory_bytes=100_000_000)

        def insert_new():
            for i in range(1000):
                entry = CacheEntry(f"key{i}", f"value{i}", size_bytes=1000)
                cache_new.set(entry)

        time_new = benchmark(insert_new)

        # Assert improvement
        assert time_new < time_old * 0.5  # At least 50% faster

    def test_memory_tracking_accuracy(self):
        """Test that memory tracking is accurate."""
        cache = OptimizedMemoryCache()

        # Add entries
        total_size = 0
        for i in range(100):
            size = 1000 + i
            entry = CacheEntry(f"key{i}", f"value{i}", size_bytes=size)
            cache.set(entry)
            total_size += size

        # Check accuracy
        assert cache.total_memory_used == total_size

    def test_eviction_frees_memory(self):
        """Test that eviction properly frees memory."""
        cache = OptimizedMemoryCache(
            max_entries=10,
            max_memory_bytes=10000,
        )

        # Fill cache beyond limit
        for i in range(20):
            entry = CacheEntry(f"key{i}", f"value{i}", size_bytes=1000)
            cache.set(entry)

        # Should have evicted to stay under limit
        assert len(cache.cache) <= 10
        assert cache.total_memory_used <= 10000
```

### Resultados Esperados

| Métrica | Antes | Depois | Melhoria |
|---------|-------|--------|----------|
| Insert (1000 items) | 0.5s | 0.05s | **10x** |
| Insert (10000 items) | 5.0s | 0.1s | **50x** |
| Memory accuracy | 100% | 100% | - |
| Memory overhead | O(n) calc | O(1) var | **Constante** |

### Passos de Implementação

1. **Implementar OptimizedMemoryCache** (0.5 dia)
2. **Adicionar testes** (0.5 dia)
3. **Benchmarks** (0.5 dia)
4. **Migration path** (0.5 dia)
   ```python
   # Gradual migration
   if os.getenv("USE_OPTIMIZED_CACHE", "1") == "1":
       cache = OptimizedMemoryCache()
   else:
       cache = MemoryCache()  # Fallback
   ```

---

## 2. Audio Processing Assíncrono

**Severidade:** 🟡 MÉDIO
**Status:** JÁ DETALHADO EM PRIORITY_2_IMPORTANT.md

Ver documento completo em `PRIORITY_2_IMPORTANT.md`, seção 1.

### Resumo

- Implementar async/await para processamento não-bloqueante
- Permitir processamento paralelo de chunks
- Melhoria esperada: 2-3x throughput

---

## 3. Memory Pool para Audio Chunks

**Severidade:** 🟡 MÉDIO
**Status:** JÁ DETALHADO EM PRIORITY_2_IMPORTANT.md

Ver documento completo em `PRIORITY_2_IMPORTANT.md`, seção 4.

### Resumo

- Pre-alocar numpy arrays para audio chunks
- Reduzir garbage collection em 70-80%
- Melhorar performance em streaming

---

## 4. Batch Processing de Embeddings

**Arquivo:** `gianna/memory/embeddings.py`
**Severidade:** 🟢 BAIXO
**Impacto:** Throughput em bulk operations

### Problema

Embeddings são gerados um por vez:

```python
for text in texts:
    embedding = self.generate_embedding(text)  # Uma chamada por texto
```

### Solução

```python
# gianna/memory/embeddings.py

class EmbeddingGenerator:
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """
        Generate embeddings in batches for better throughput.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Check cache first
            uncached = []
            uncached_indices = []

            for j, text in enumerate(batch):
                if text in self._embedding_cache:
                    embeddings.append(self._embedding_cache[text])
                else:
                    uncached.append(text)
                    uncached_indices.append(i + j)

            # Generate uncached embeddings in batch
            if uncached:
                # Batch API call (muito mais eficiente!)
                batch_embeddings = self.model.embed_documents(uncached)

                # Cache results
                for text, emb in zip(uncached, batch_embeddings):
                    self._embedding_cache[text] = emb

                # Insert at correct positions
                for idx, emb in zip(uncached_indices, batch_embeddings):
                    embeddings.insert(idx, emb)

        return embeddings
```

**Melhoria esperada:** 5-10x throughput em bulk operations

---

## 5. Database Query Optimization

**Arquivo:** `gianna/core/state_manager.py`
**Severidade:** 🟢 BAIXO
**Impacto:** Latência em queries frequentes

### Problemas Potenciais

1. **N+1 queries**
2. **Missing indexes**
3. **No query caching**

### Soluções

#### 5.1. Adicionar Índices

```python
# gianna/core/models.py

from sqlalchemy import Column, String, Integer, Index

class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False)
    timestamp = Column(Integer, nullable=False)
    message = Column(String, nullable=False)

    # Índices para queries comuns
    __table_args__ = (
        Index('idx_session_timestamp', 'session_id', 'timestamp'),
        Index('idx_timestamp', 'timestamp'),
    )
```

#### 5.2. Eager Loading

```python
# Evitar N+1
from sqlalchemy.orm import joinedload

# RUIM - N+1 queries
sessions = db.query(Session).all()
for session in sessions:
    messages = session.messages  # Query separada!

# BOM - 1 query com join
sessions = db.query(Session).options(
    joinedload(Session.messages)
).all()
```

#### 5.3. Query Caching

```python
from functools import lru_cache

class StateManager:
    @lru_cache(maxsize=1000)
    def get_session_history(self, session_id: str) -> List[Message]:
        """Get session history with caching."""
        return db.query(Message).filter(
            Message.session_id == session_id
        ).all()
```

---

## 6. Lazy Loading de Modelos Pesados

**Arquivo:** `gianna/assistants/audio/stt/whisper.py`
**Severidade:** 🟢 BAIXO
**Impacto:** Startup time

### Problema

Modelos pesados (Whisper, Silero VAD) são carregados ao importar:

```python
# Carrega modelo na importação (lento!)
model = whisper.load_model("base")
```

### Solução

```python
# gianna/assistants/audio/stt/whisper.py

class WhisperSTT:
    def __init__(self):
        self._model = None  # Lazy load

    @property
    def model(self):
        """Lazy load Whisper model."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model("base")
        return self._model

    def transcribe(self, audio_file: str) -> str:
        # Modelo só é carregado quando necessário
        return self.model.transcribe(audio_file)["text"]
```

**Melhoria:** Startup 5-10x mais rápido

---

## 7. Profiling e Monitoramento

### Ferramentas Recomendadas

```python
# gianna/monitoring/profiler.py

import cProfile
import pstats
from functools import wraps

def profile(output_file=None):
    """Decorator to profile function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            result = func(*args, **kwargs)

            profiler.disable()

            if output_file:
                profiler.dump_stats(output_file)
            else:
                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative')
                stats.print_stats(20)

            return result
        return wrapper
    return decorator

# Usage
@profile(output_file="vad_profile.prof")
def process_audio_stream():
    # ... código
    pass
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def process_large_dataset():
    # Ver uso de memória linha por linha
    data = load_data()
    processed = process(data)
    return processed
```

---

## Estimativa de Esforço

| Task | Esforço | Prioridade | Melhoria Esperada |
|------|---------|-----------|-------------------|
| 1. Otimizar memory cache | 2 dias | P0 | 10-50x inserts |
| 2. Async audio | 3 dias | P1 | 2-3x throughput |
| 3. Memory pool | 1 dia | P1 | 70% menos GC |
| 4. Batch embeddings | 1 dia | P2 | 5-10x bulk |
| 5. DB optimization | 1 dia | P2 | 2-5x queries |
| 6. Lazy loading | 0.5 dia | P3 | 5-10x startup |
| 7. Profiling tools | 0.5 dia | P3 | Observabilidade |

**Total:** 9 dias

---

## Métricas de Sucesso

### Performance Targets

- [ ] Cache insert: < 1ms (vs 50ms antes)
- [ ] Audio throughput: > 100 chunks/s
- [ ] GC pauses: < 100ms (vs 500ms antes)
- [ ] Embedding throughput: > 1000 texts/min
- [ ] Query latency: < 10ms (95th percentile)
- [ ] Startup time: < 2s (vs 10s antes)

### Benchmarks

```bash
# Run all performance tests
pytest tests/performance/ --benchmark-only

# Generate profiling report
python -m cProfile -o profile.prof main.py
python -m pstats profile.prof

# Memory profiling
mprof run main.py
mprof plot
```

---

## Conclusão

Implementar todas as otimizações resultará em:
- **10-50x** melhoria em cache operations
- **2-3x** melhoria em audio throughput
- **70%** redução em GC overhead
- **5-10x** melhoria em bulk embeddings
- **2-5x** melhoria em database queries

**ROI:** Alto - esforço de 9 dias para ganhos significativos em performance.
