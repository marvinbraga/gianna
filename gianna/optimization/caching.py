"""
Sistema de Cache Multi-Layer para Gianna

Implementa cache inteligente com múltiplas camadas (memória, Redis, SQLite),
cache warming, invalidação automática e análise de padrões de uso.
"""

import hashlib
import json
import pickle
import sqlite3
import time
import zlib
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Set, Union

from loguru import logger

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class CacheEntry:
    """Entrada do cache com metadados"""

    key: str
    value: Any
    timestamp: float
    ttl: int
    access_count: int = 0
    last_access: float = 0
    size_bytes: int = 0
    tags: Set[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.last_access == 0:
            self.last_access = self.timestamp
        if self.size_bytes == 0:
            self.size_bytes = len(pickle.dumps(self.value))

    def is_expired(self) -> bool:
        """Verifica se a entrada está expirada"""
        return time.time() > (self.timestamp + self.ttl)

    def access(self):
        """Registra um acesso à entrada"""
        self.access_count += 1
        self.last_access = time.time()


class CacheLayer(ABC):
    """Interface abstrata para camadas de cache"""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Obtém entrada do cache"""
        pass

    @abstractmethod
    def set(self, entry: CacheEntry) -> bool:
        """Define entrada no cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove entrada do cache"""
        pass

    @abstractmethod
    def clear(self):
        """Limpa todo o cache"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas da camada"""
        pass


class MemoryCache(CacheLayer):
    """Cache em memória com LRU e compressão"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.lock = Lock()
        self.stats = defaultdict(int)

    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    # Move para o fim (mais recente)
                    self.cache.move_to_end(key)
                    entry.access()
                    self.stats["hits"] += 1
                    return entry
                else:
                    # Remove entrada expirada
                    del self.cache[key]
                    self.stats["expirations"] += 1

        self.stats["misses"] += 1
        return None

    def set(self, entry: CacheEntry) -> bool:
        with self.lock:
            # Verifica limites de memória
            current_memory = sum(e.size_bytes for e in self.cache.values())
            if current_memory + entry.size_bytes > self.max_memory_bytes:
                self._evict_by_memory()

            # Verifica limite de itens
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            self.cache[entry.key] = entry
            self.stats["sets"] += 1
            return True

    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats["deletes"] += 1
                return True
        return False

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.stats["clears"] += 1

    def _evict_lru(self):
        """Remove entrada menos recentemente usada"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            self.stats["evictions"] += 1
            logger.debug(f"Evicted LRU entry: {key}")

    def _evict_by_memory(self):
        """Remove entradas para liberar memória"""
        target_memory = self.max_memory_bytes * 0.8  # Libera 20%

        # Ordena por último acesso (menos recente primeiro)
        entries_by_access = sorted(self.cache.items(), key=lambda x: x[1].last_access)

        current_memory = sum(e.size_bytes for _, e in self.cache.items())

        for key, entry in entries_by_access:
            if current_memory <= target_memory:
                break
            del self.cache[key]
            current_memory -= entry.size_bytes
            self.stats["memory_evictions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "type": "memory",
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": sum(e.size_bytes for e in self.cache.values())
                / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                **dict(self.stats),
            }


class RedisCache(CacheLayer):
    """Cache Redis com compressão e serialização"""

    def __init__(
        self, redis_url: str, compression: bool = True, key_prefix: str = "gianna:"
    ):
        self.compression = compression
        self.key_prefix = key_prefix
        self.stats = defaultdict(int)

        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis não está disponível")

        try:
            self.client = redis.from_url(redis_url)
            self.client.ping()
            logger.info("Redis cache conectado")
        except Exception as e:
            logger.error(f"Erro ao conectar Redis: {e}")
            raise

    def _serialize(self, entry: CacheEntry) -> bytes:
        """Serializa entrada com compressão opcional"""
        data = pickle.dumps(entry)
        if self.compression:
            data = zlib.compress(data)
        return data

    def _deserialize(self, data: bytes) -> CacheEntry:
        """Deserializa entrada com descompressão opcional"""
        if self.compression:
            data = zlib.decompress(data)
        return pickle.loads(data)

    def _make_key(self, key: str) -> str:
        """Cria chave com prefixo"""
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[CacheEntry]:
        try:
            redis_key = self._make_key(key)
            data = self.client.get(redis_key)

            if data:
                entry = self._deserialize(data)
                if not entry.is_expired():
                    entry.access()
                    # Atualiza no Redis com nova contagem de acesso
                    self.client.setex(
                        redis_key,
                        entry.ttl - int(time.time() - entry.timestamp),
                        self._serialize(entry),
                    )
                    self.stats["hits"] += 1
                    return entry
                else:
                    # Remove entrada expirada
                    self.client.delete(redis_key)
                    self.stats["expirations"] += 1

        except Exception as e:
            logger.warning(f"Erro ao ler do Redis: {e}")

        self.stats["misses"] += 1
        return None

    def set(self, entry: CacheEntry) -> bool:
        try:
            redis_key = self._make_key(entry.key)
            data = self._serialize(entry)

            # Define com TTL
            self.client.setex(redis_key, entry.ttl, data)
            self.stats["sets"] += 1
            return True

        except Exception as e:
            logger.warning(f"Erro ao escrever no Redis: {e}")
            return False

    def delete(self, key: str) -> bool:
        try:
            redis_key = self._make_key(key)
            result = self.client.delete(redis_key)
            if result:
                self.stats["deletes"] += 1
            return bool(result)
        except Exception as e:
            logger.warning(f"Erro ao deletar do Redis: {e}")
            return False

    def clear(self):
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            self.stats["clears"] += 1
        except Exception as e:
            logger.warning(f"Erro ao limpar Redis: {e}")

    def get_stats(self) -> Dict[str, Any]:
        try:
            info = self.client.info()
            return {
                "type": "redis",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                **dict(self.stats),
            }
        except Exception:
            return {"type": "redis", "error": "Unable to get stats"}


class SQLiteCache(CacheLayer):
    """Cache SQLite para persistência de longo prazo"""

    def __init__(self, db_path: str = "cache.db", max_entries: int = 10000):
        self.db_path = Path(db_path)
        self.max_entries = max_entries
        self.stats = defaultdict(int)
        self._init_db()

    def _init_db(self):
        """Inicializa banco de dados SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    ttl INTEGER,
                    access_count INTEGER,
                    last_access REAL,
                    size_bytes INTEGER,
                    tags TEXT
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_access ON cache_entries(last_access)"
            )

    def get(self, key: str) -> Optional[CacheEntry]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM cache_entries WHERE key = ?", (key,)
                )
                row = cursor.fetchone()

                if row:
                    entry = CacheEntry(
                        key=row["key"],
                        value=pickle.loads(row["value"]),
                        timestamp=row["timestamp"],
                        ttl=row["ttl"],
                        access_count=row["access_count"],
                        last_access=row["last_access"],
                        size_bytes=row["size_bytes"],
                        tags=set(json.loads(row["tags"]) if row["tags"] else []),
                    )

                    if not entry.is_expired():
                        entry.access()
                        # Atualiza contagem de acesso
                        conn.execute(
                            """UPDATE cache_entries
                               SET access_count = ?, last_access = ?
                               WHERE key = ?""",
                            (entry.access_count, entry.last_access, key),
                        )
                        self.stats["hits"] += 1
                        return entry
                    else:
                        # Remove entrada expirada
                        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        self.stats["expirations"] += 1

        except Exception as e:
            logger.warning(f"Erro ao ler do SQLite: {e}")

        self.stats["misses"] += 1
        return None

    def set(self, entry: CacheEntry) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Verifica limite de entradas
                count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
                if count >= self.max_entries:
                    self._evict_old_entries(conn)

                # Insert or replace
                conn.execute(
                    """INSERT OR REPLACE INTO cache_entries
                       (key, value, timestamp, ttl, access_count, last_access, size_bytes, tags)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.key,
                        pickle.dumps(entry.value),
                        entry.timestamp,
                        entry.ttl,
                        entry.access_count,
                        entry.last_access,
                        entry.size_bytes,
                        json.dumps(list(entry.tags)),
                    ),
                )
                self.stats["sets"] += 1
                return True

        except Exception as e:
            logger.warning(f"Erro ao escrever no SQLite: {e}")
            return False

    def delete(self, key: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                if cursor.rowcount > 0:
                    self.stats["deletes"] += 1
                    return True
        except Exception as e:
            logger.warning(f"Erro ao deletar do SQLite: {e}")
        return False

    def clear(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
            self.stats["clears"] += 1
        except Exception as e:
            logger.warning(f"Erro ao limpar SQLite: {e}")

    def _evict_old_entries(self, conn):
        """Remove entradas antigas para fazer espaço"""
        target = self.max_entries * 0.8  # Remove 20%
        conn.execute(
            """DELETE FROM cache_entries WHERE key IN (
                SELECT key FROM cache_entries
                ORDER BY last_access ASC
                LIMIT ?
            )""",
            (int(self.max_entries - target),),
        )

    def get_stats(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                size = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
                total_size = (
                    conn.execute(
                        "SELECT SUM(size_bytes) FROM cache_entries"
                    ).fetchone()[0]
                    or 0
                )

            return {
                "type": "sqlite",
                "size": size,
                "max_entries": self.max_entries,
                "total_size_mb": total_size / (1024 * 1024),
                "db_size_mb": self.db_path.stat().st_size / (1024 * 1024),
                **dict(self.stats),
            }
        except Exception as e:
            return {"type": "sqlite", "error": str(e)}


class CacheWarmer:
    """Sistema de aquecimento de cache baseado em padrões"""

    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.warming_patterns = []
        self.warming_thread = None
        self.active = False

    def add_warming_pattern(
        self,
        key_pattern: str,
        callable_func,
        interval: int = 3600,
        tags: Set[str] = None,
    ):
        """
        Adiciona padrão de aquecimento

        Args:
            key_pattern: Padrão da chave
            callable_func: Função para gerar valor
            interval: Intervalo de aquecimento em segundos
            tags: Tags para categorização
        """
        self.warming_patterns.append(
            {
                "pattern": key_pattern,
                "callable": callable_func,
                "interval": interval,
                "last_warmed": 0,
                "tags": tags or set(),
            }
        )

    def start_warming(self):
        """Inicia aquecimento automático"""
        if self.active:
            return

        self.active = True
        self.warming_thread = Thread(target=self._warming_loop, daemon=True)
        self.warming_thread.start()
        logger.info("Cache warming iniciado")

    def stop_warming(self):
        """Para aquecimento automático"""
        self.active = False
        if self.warming_thread:
            self.warming_thread.join(timeout=5)
        logger.info("Cache warming parado")

    def _warming_loop(self):
        """Loop principal de aquecimento"""
        while self.active:
            try:
                current_time = time.time()

                for pattern in self.warming_patterns:
                    if current_time - pattern["last_warmed"] >= pattern["interval"]:
                        self._warm_pattern(pattern)
                        pattern["last_warmed"] = current_time

                time.sleep(60)  # Verifica a cada minuto

            except Exception as e:
                logger.error(f"Erro no cache warming: {e}")
                time.sleep(60)

    def _warm_pattern(self, pattern):
        """Aquece cache para um padrão específico"""
        try:
            key = pattern["pattern"]
            value = pattern["callable"]()

            # Cria entrada de cache
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=pattern["interval"],
                tags=pattern["tags"],
            )

            self.cache_manager.set(entry)
            logger.debug(f"Cache warmed for pattern: {key}")

        except Exception as e:
            logger.warning(f"Erro ao aquecer cache para {pattern['pattern']}: {e}")


class MultiLayerCache:
    """Cache inteligente com múltiplas camadas"""

    def __init__(
        self,
        memory_size: int = 1000,
        redis_url: Optional[str] = None,
        sqlite_path: str = "cache.db",
    ):

        # Inicializa camadas
        self.layers = []

        # Camada L1: Memória (mais rápida)
        self.memory_cache = MemoryCache(memory_size)
        self.layers.append(self.memory_cache)

        # Camada L2: Redis (rede)
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_cache = RedisCache(redis_url)
                self.layers.append(self.redis_cache)
                logger.info("Redis cache layer ativado")
            except Exception as e:
                logger.warning(f"Redis não disponível: {e}")

        # Camada L3: SQLite (persistência)
        try:
            self.sqlite_cache = SQLiteCache(sqlite_path)
            self.layers.append(self.sqlite_cache)
            logger.info("SQLite cache layer ativado")
        except Exception as e:
            logger.warning(f"SQLite cache não disponível: {e}")

        # Cache warming
        self.cache_warmer = CacheWarmer(self)

        # Estatísticas globais
        self.global_stats = defaultdict(int)

    def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache (verifica todas as camadas)"""
        self.global_stats["total_gets"] += 1

        for i, layer in enumerate(self.layers):
            entry = layer.get(key)
            if entry:
                # Promove para camadas superiores
                self._promote_entry(entry, i)
                self.global_stats["hits"] += 1
                return entry.value

        self.global_stats["misses"] += 1
        return None

    def set(self, key: str, value: Any, ttl: int = 3600, tags: Set[str] = None):
        """Define valor em todas as camadas"""
        entry = CacheEntry(
            key=key, value=value, timestamp=time.time(), ttl=ttl, tags=tags or set()
        )

        # Define em todas as camadas
        for layer in self.layers:
            try:
                layer.set(entry)
            except Exception as e:
                logger.warning(f"Erro ao definir em camada {type(layer).__name__}: {e}")

        self.global_stats["sets"] += 1

    def delete(self, key: str):
        """Remove de todas as camadas"""
        for layer in self.layers:
            try:
                layer.delete(key)
            except Exception as e:
                logger.warning(f"Erro ao deletar de camada {type(layer).__name__}: {e}")

        self.global_stats["deletes"] += 1

    def invalidate_by_tags(self, tags: Set[str]):
        """Invalida entradas por tags"""
        # Implementação básica - cada camada precisa implementar
        # busca por tags para ser mais eficiente
        logger.info(f"Invalidating cache entries with tags: {tags}")

    def clear(self):
        """Limpa todas as camadas"""
        for layer in self.layers:
            try:
                layer.clear()
            except Exception as e:
                logger.warning(f"Erro ao limpar camada {type(layer).__name__}: {e}")

        self.global_stats["clears"] += 1

    def _promote_entry(self, entry: CacheEntry, found_at_layer: int):
        """Promove entrada para camadas superiores"""
        for i in range(found_at_layer):
            try:
                self.layers[i].set(entry)
            except Exception as e:
                logger.warning(f"Erro ao promover entrada: {e}")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas de todas as camadas"""
        layer_stats = []
        for layer in self.layers:
            layer_stats.append(layer.get_stats())

        hit_rate = self.global_stats["hits"] / max(self.global_stats["total_gets"], 1)

        return {
            "global_stats": dict(self.global_stats),
            "hit_rate": hit_rate,
            "layer_count": len(self.layers),
            "layer_stats": layer_stats,
        }

    def optimize_cache(self):
        """Otimiza cache baseado em padrões de uso"""
        # Análise de padrões seria implementada aqui
        # Por exemplo: ajustar tamanhos de cache, TTLs, etc.
        logger.info("Cache optimization executada")

    def start_background_tasks(self):
        """Inicia tarefas em background (warming, limpeza, etc.)"""
        self.cache_warmer.start_warming()

    def stop_background_tasks(self):
        """Para tarefas em background"""
        self.cache_warmer.stop_warming()

    def __enter__(self):
        self.start_background_tasks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_background_tasks()
