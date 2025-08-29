"""
Sistema Avançado de Memória Semântica - Gianna

Este exemplo demonstra um sistema completo de memória semântica com:

- Armazenamento vetorial de conversações e contextos
- Busca semântica inteligente por similaridade
- Clustering automático de tópicos relacionados
- Sumarização automática de contextos longos
- Indexação incremental em tempo real
- Sistema de tags e metadados
- Análise temporal de conversações
- Exportação e importação de memória

Pré-requisitos:
- Gianna instalado
- ChromaDB (pip install chromadb)
- Sentence-transformers (pip install sentence-transformers)
- Chaves de API para embeddings (OpenAI recomendado)

Uso:
    python semantic_memory_system.py
"""

import asyncio
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gianna.assistants.models.factory_method import get_chain_instance

# Imports principais
from gianna.core.langgraph_chain import LangGraphChain
from gianna.core.state_manager import StateManager
from gianna.memory.embeddings import EmbeddingGenerator
from gianna.memory.semantic_memory import SemanticMemory

# Imports opcionais para funcionalidades avançadas
try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    print("⚠️  ChromaDB não disponível. Usando fallback simples.")
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Sentence-transformers não disponível. Usando embeddings básicos.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  Scikit-learn não disponível. Clustering desabilitado.")
    SKLEARN_AVAILABLE = False


@dataclass
class MemoryEntry:
    """Entrada de memória com metadados ricos."""

    id: str
    content: str
    timestamp: datetime
    session_id: str
    user_id: str
    entry_type: str  # 'conversation', 'fact', 'preference', 'context'
    tags: List[str]
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self):
        """Converter para dicionário."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        if self.last_accessed:
            data["last_accessed"] = self.last_accessed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict):
        """Criar instância a partir de dicionário."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("last_accessed"):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        return cls(**data)


@dataclass
class TopicCluster:
    """Cluster de tópicos relacionados."""

    id: str
    name: str
    entries: List[str]  # IDs das entradas
    centroid: List[float]
    keywords: List[str]
    created_at: datetime
    last_updated: datetime
    confidence_score: float


class AdvancedSemanticMemory:
    """Sistema avançado de memória semântica."""

    def __init__(
        self,
        persist_directory: str = "gianna_advanced_memory",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Armazenamento de entradas
        self.entries: Dict[str, MemoryEntry] = {}
        self.user_profiles: Dict[str, Dict] = defaultdict(dict)
        self.topic_clusters: Dict[str, TopicCluster] = {}

        # Componentes de embedding
        self.embedding_model_name = embedding_model
        self.local_model = None
        self.openai_embeddings = None

        # Base de dados vetorial
        self.chroma_client = None
        self.chroma_collection = None

        # Configurações
        self.max_entries = 10000
        self.clustering_threshold = 0.7
        self.importance_decay = 0.95  # Decay diário
        self.min_cluster_size = 3

        # Métricas
        self.metrics = {
            "total_entries": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "clusters_created": 0,
            "embedding_time": 0.0,
        }

        # LLM para sumarização
        self.summarizer_chain = None

        self._initialize()

    def _initialize(self):
        """Inicializar sistema de memória."""
        print("🧠 Inicializando Sistema de Memória Semântica Avançado...")

        try:
            # Carregar dados existentes
            self._load_persistent_data()

            # Configurar embeddings
            self._setup_embeddings()

            # Configurar base de dados vetorial
            self._setup_vector_database()

            # Configurar LLM para sumarização
            self._setup_summarizer()

            print("✅ Sistema de memória inicializado com sucesso")
            print(f"📊 Entradas carregadas: {len(self.entries)}")
            print(f"🏷️  Clusters de tópicos: {len(self.topic_clusters)}")

        except Exception as e:
            print(f"❌ Erro na inicialização: {str(e)}")
            raise

    def _setup_embeddings(self):
        """Configurar sistema de embeddings."""
        # Tentar usar sentence-transformers local primeiro
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"🔄 Carregando modelo local: {self.embedding_model_name}")
                self.local_model = SentenceTransformer(self.embedding_model_name)
                print("✅ Modelo local carregado")
                return
            except Exception as e:
                print(f"⚠️  Falha no modelo local: {str(e)}")

        # Fallback para OpenAI embeddings
        try:
            from gianna.memory.embeddings import EmbeddingGenerator

            self.openai_embeddings = EmbeddingGenerator()
            print("✅ OpenAI embeddings configurado")
        except Exception as e:
            print(f"❌ Falha ao configurar embeddings: {str(e)}")
            raise RuntimeError("Nenhum sistema de embeddings disponível")

    def _setup_vector_database(self):
        """Configurar base de dados vetorial."""
        if not CHROMA_AVAILABLE:
            print("⚠️  ChromaDB não disponível, usando armazenamento simples")
            return

        try:
            # Configurar ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory / "chroma_db")
            )

            # Criar ou obter coleção
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="gianna_memory", metadata={"hnsw:space": "cosine"}
            )

            print("✅ ChromaDB configurado")

        except Exception as e:
            print(f"⚠️  Falha no ChromaDB: {str(e)}, usando fallback")

    def _setup_summarizer(self):
        """Configurar LLM para sumarização."""
        try:
            self.summarizer_chain = LangGraphChain(
                "gpt4",
                """Você é um especialista em sumarização de conversações. Sua tarefa é:

1. Extrair os pontos principais de conversações longas
2. Identificar temas recorrentes e insights importantes
3. Manter o contexto e nuances importantes
4. Gerar resumos concisos mas informativos

Formato de saída:
- Tópicos principais (até 5)
- Insights importantes
- Preferências do usuário identificadas
- Contexto relevante para conversações futuras

Seja preciso e mantenha informações importantes.""",
            )
            print("✅ Sumarizador configurado")
        except Exception as e:
            print(f"⚠️  Falha ao configurar sumarizador: {str(e)}")

    async def add_memory(
        self,
        content: str,
        session_id: str,
        user_id: str,
        entry_type: str = "conversation",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Adicionar nova entrada à memória."""

        entry_id = f"{user_id}_{int(time.time())}_{len(self.entries)}"

        # Calcular importância inicial
        importance = self._calculate_importance(content, entry_type, tags or [])

        # Criar entrada
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            entry_type=entry_type,
            tags=tags or [],
            metadata=metadata or {},
            importance_score=importance,
        )

        # Gerar embedding
        start_time = time.time()
        entry.embedding = await self._generate_embedding(content)
        self.metrics["embedding_time"] += time.time() - start_time

        # Adicionar à memória
        self.entries[entry_id] = entry
        self.metrics["total_entries"] += 1

        # Adicionar ao banco vetorial
        if self.chroma_collection and entry.embedding:
            self.chroma_collection.add(
                embeddings=[entry.embedding],
                documents=[content],
                metadatas=[
                    {
                        "entry_id": entry_id,
                        "user_id": user_id,
                        "entry_type": entry_type,
                        "timestamp": entry.timestamp.isoformat(),
                        "importance": importance,
                        "tags": ",".join(tags) if tags else "",
                    }
                ],
                ids=[entry_id],
            )

        # Atualizar perfil do usuário
        await self._update_user_profile(user_id, entry)

        # Clustering periódico
        if len(self.entries) % 50 == 0:
            await self._update_topic_clusters()

        # Limpeza periódica
        if len(self.entries) > self.max_entries:
            await self._cleanup_old_entries()

        print(f"💾 Memória adicionada: {entry_id} (importância: {importance:.2f})")
        return entry_id

    async def search_memories(
        self,
        query: str,
        user_id: str = None,
        entry_types: List[str] = None,
        tags: List[str] = None,
        limit: int = 10,
        min_similarity: float = 0.5,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Buscar memórias por similaridade semântica."""

        try:
            results = []

            # Gerar embedding da query
            query_embedding = await self._generate_embedding(query)
            if not query_embedding:
                return []

            # Buscar usando ChromaDB se disponível
            if self.chroma_collection:
                chroma_results = self.chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit * 2,  # Pegar mais para filtrar
                    where=(
                        {
                            "$and": [
                                {"user_id": {"$eq": user_id}} if user_id else {},
                                (
                                    {"entry_type": {"$in": entry_types}}
                                    if entry_types
                                    else {}
                                ),
                            ]
                        }
                        if user_id or entry_types
                        else None
                    ),
                )

                # Processar resultados
                for i, entry_id in enumerate(chroma_results["ids"][0]):
                    if entry_id in self.entries:
                        entry = self.entries[entry_id]
                        similarity = (
                            1 - chroma_results["distances"][0][i]
                        )  # ChromaDB retorna distância

                        if similarity >= min_similarity:
                            results.append((entry, similarity))

            # Fallback: busca simples por similaridade
            else:
                for entry in self.entries.values():
                    # Filtrar por critérios
                    if user_id and entry.user_id != user_id:
                        continue
                    if entry_types and entry.entry_type not in entry_types:
                        continue
                    if tags and not any(tag in entry.tags for tag in tags):
                        continue

                    if entry.embedding:
                        similarity = self._cosine_similarity(
                            query_embedding, entry.embedding
                        )
                        if similarity >= min_similarity:
                            results.append((entry, similarity))

            # Ordenar por similaridade e relevância
            results.sort(key=lambda x: x[1] * x[0].importance_score, reverse=True)

            # Atualizar estatísticas de acesso
            for entry, _ in results[:limit]:
                entry.access_count += 1
                entry.last_accessed = datetime.now()

            self.metrics["successful_retrievals"] += 1
            return results[:limit]

        except Exception as e:
            print(f"❌ Erro na busca: {str(e)}")
            self.metrics["failed_retrievals"] += 1
            return []

    async def get_conversation_context(
        self, user_id: str, session_id: str = None, max_entries: int = 20
    ) -> Dict[str, Any]:
        """Obter contexto de conversação para um usuário."""

        # Buscar entradas recentes
        recent_entries = []
        for entry in sorted(
            self.entries.values(), key=lambda x: x.timestamp, reverse=True
        ):

            if entry.user_id != user_id:
                continue

            if session_id and entry.session_id != session_id:
                continue

            recent_entries.append(entry)
            if len(recent_entries) >= max_entries:
                break

        if not recent_entries:
            return {"summary": "", "topics": [], "preferences": {}}

        # Gerar resumo automático se muitas entradas
        if len(recent_entries) > 5 and self.summarizer_chain:
            conversation_text = "\\n".join(
                [
                    f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M')}] {entry.content}"
                    for entry in recent_entries
                ]
            )

            summary_result = await self.summarizer_chain.ainvoke(
                {"input": f"Sumarize esta conversação:\\n\\n{conversation_text}"}
            )
            summary = summary_result.get("output", "")
        else:
            summary = ""

        # Extrair tópicos comuns
        all_tags = []
        for entry in recent_entries:
            all_tags.extend(entry.tags)

        topic_counts = Counter(all_tags)
        top_topics = [topic for topic, _ in topic_counts.most_common(10)]

        # Obter preferências do usuário
        user_preferences = self.user_profiles.get(user_id, {})

        return {
            "summary": summary,
            "topics": top_topics,
            "preferences": user_preferences,
            "recent_entries": len(recent_entries),
            "oldest_entry": (
                recent_entries[-1].timestamp.isoformat() if recent_entries else None
            ),
            "most_recent": (
                recent_entries[0].timestamp.isoformat() if recent_entries else None
            ),
        }

    async def find_related_topics(
        self, topic: str, limit: int = 10
    ) -> List[Tuple[str, float]]:
        """Encontrar tópicos relacionados."""

        # Buscar entradas relacionadas ao tópico
        related_entries = await self.search_memories(
            query=topic, limit=50, min_similarity=0.3
        )

        if not related_entries:
            return []

        # Extrair e contar tags relacionadas
        related_tags = Counter()
        for entry, similarity in related_entries:
            for tag in entry.tags:
                if tag.lower() != topic.lower():
                    related_tags[tag] += similarity

        # Retornar tópicos mais relacionados
        return [(tag, score) for tag, score in related_tags.most_common(limit)]

    async def _generate_embedding(self, text: str) -> List[float]:
        """Gerar embedding para texto."""
        try:
            if self.local_model:
                # Usar modelo local
                embedding = self.local_model.encode(text).tolist()
                return embedding

            elif self.openai_embeddings:
                # Usar OpenAI embeddings
                embedding = await self.openai_embeddings.get_embedding(text)
                return embedding

            else:
                print("❌ Nenhum sistema de embedding disponível")
                return []

        except Exception as e:
            print(f"❌ Erro ao gerar embedding: {str(e)}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcular similaridade cosseno."""
        if not vec1 or not vec2:
            return 0.0

        try:
            if SKLEARN_AVAILABLE:
                return cosine_similarity([vec1], [vec2])[0][0]
            else:
                # Implementação manual
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                norm_a = sum(a * a for a in vec1) ** 0.5
                norm_b = sum(b * b for b in vec2) ** 0.5

                if norm_a == 0 or norm_b == 0:
                    return 0.0

                return dot_product / (norm_a * norm_b)
        except:
            return 0.0

    def _calculate_importance(
        self, content: str, entry_type: str, tags: List[str]
    ) -> float:
        """Calcular importância inicial de uma entrada."""
        importance = 0.5  # Base

        # Tipo de entrada
        type_weights = {
            "preference": 1.0,
            "fact": 0.8,
            "conversation": 0.6,
            "context": 0.4,
        }
        importance *= type_weights.get(entry_type, 0.5)

        # Comprimento do conteúdo
        if len(content) > 200:
            importance += 0.2
        elif len(content) < 50:
            importance -= 0.1

        # Número de tags
        importance += min(len(tags) * 0.1, 0.3)

        # Palavras-chave importantes
        important_keywords = [
            "preferência",
            "gosto",
            "não gosto",
            "importante",
            "lembrar",
            "sempre",
            "nunca",
            "problema",
            "erro",
        ]

        for keyword in important_keywords:
            if keyword in content.lower():
                importance += 0.1

        return min(importance, 1.0)

    async def _update_user_profile(self, user_id: str, entry: MemoryEntry):
        """Atualizar perfil do usuário."""
        profile = self.user_profiles[user_id]

        # Atualizar estatísticas
        profile["total_interactions"] = profile.get("total_interactions", 0) + 1
        profile["last_interaction"] = entry.timestamp.isoformat()

        # Extrair preferências
        if "gosto" in entry.content.lower() or "prefiro" in entry.content.lower():
            preferences = profile.setdefault("preferences", [])
            preferences.append(
                {
                    "content": entry.content,
                    "timestamp": entry.timestamp.isoformat(),
                    "confidence": entry.importance_score,
                }
            )

            # Manter apenas as 20 preferências mais recentes
            profile["preferences"] = preferences[-20:]

        # Atualizar tópicos de interesse
        if entry.tags:
            interests = profile.setdefault("interests", Counter())
            for tag in entry.tags:
                interests[tag] += 1
            profile["interests"] = dict(interests)

    async def _update_topic_clusters(self):
        """Atualizar clusters de tópicos."""
        if not SKLEARN_AVAILABLE or len(self.entries) < self.min_cluster_size:
            return

        try:
            # Coletar embeddings válidos
            embeddings = []
            entry_ids = []

            for entry_id, entry in self.entries.items():
                if entry.embedding:
                    embeddings.append(entry.embedding)
                    entry_ids.append(entry_id)

            if len(embeddings) < self.min_cluster_size:
                return

            # Clustering K-means
            n_clusters = min(10, len(embeddings) // 5)  # Clusters adaptativos
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Criar/atualizar clusters
            new_clusters = {}
            for i in range(n_clusters):
                cluster_entries = [
                    entry_ids[j] for j, label in enumerate(cluster_labels) if label == i
                ]

                if len(cluster_entries) >= self.min_cluster_size:
                    # Extrair keywords do cluster
                    cluster_tags = []
                    for entry_id in cluster_entries:
                        cluster_tags.extend(self.entries[entry_id].tags)

                    top_keywords = [
                        tag for tag, _ in Counter(cluster_tags).most_common(5)
                    ]

                    cluster_id = f"cluster_{i}_{int(time.time())}"
                    new_clusters[cluster_id] = TopicCluster(
                        id=cluster_id,
                        name=f"Tópico {i+1}",
                        entries=cluster_entries,
                        centroid=kmeans.cluster_centers_[i].tolist(),
                        keywords=top_keywords,
                        created_at=datetime.now(),
                        last_updated=datetime.now(),
                        confidence_score=0.8,
                    )

            self.topic_clusters = new_clusters
            self.metrics["clusters_created"] = len(new_clusters)

            print(f"🔄 Clusters atualizados: {len(new_clusters)} tópicos identificados")

        except Exception as e:
            print(f"❌ Erro no clustering: {str(e)}")

    async def _cleanup_old_entries(self):
        """Limpar entradas antigas de baixa importância."""
        if len(self.entries) <= self.max_entries:
            return

        # Ordenar por importância e idade
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda x: (
                x[1].importance_score
                * (1 - (datetime.now() - x[1].timestamp).days / 365)
            ),
            reverse=False,
        )

        # Remover 10% das entradas menos importantes
        to_remove = int(len(self.entries) * 0.1)
        for entry_id, _ in sorted_entries[:to_remove]:
            # Remover do ChromaDB
            if self.chroma_collection:
                try:
                    self.chroma_collection.delete(ids=[entry_id])
                except:
                    pass

            # Remover da memória
            del self.entries[entry_id]

        print(f"🧹 Limpeza realizada: {to_remove} entradas antigas removidas")

    def _load_persistent_data(self):
        """Carregar dados persistentes."""
        entries_file = self.persist_directory / "entries.pkl"
        profiles_file = self.persist_directory / "profiles.json"
        clusters_file = self.persist_directory / "clusters.pkl"

        # Carregar entradas
        if entries_file.exists():
            with open(entries_file, "rb") as f:
                data = pickle.load(f)
                for entry_data in data:
                    entry = MemoryEntry.from_dict(entry_data)
                    self.entries[entry.id] = entry

        # Carregar perfis
        if profiles_file.exists():
            with open(profiles_file, "r") as f:
                self.user_profiles = json.load(f)

        # Carregar clusters
        if clusters_file.exists():
            with open(clusters_file, "rb") as f:
                self.topic_clusters = pickle.load(f)

    def save_persistent_data(self):
        """Salvar dados persistentes."""
        # Salvar entradas
        entries_data = [entry.to_dict() for entry in self.entries.values()]
        with open(self.persist_directory / "entries.pkl", "wb") as f:
            pickle.dump(entries_data, f)

        # Salvar perfis
        with open(self.persist_directory / "profiles.json", "w") as f:
            json.dump(self.user_profiles, f, indent=2)

        # Salvar clusters
        with open(self.persist_directory / "clusters.pkl", "wb") as f:
            pickle.dump(self.topic_clusters, f)

    def export_user_data(self, user_id: str, format: str = "json") -> Dict:
        """Exportar dados de um usuário."""
        user_entries = [
            entry.to_dict()
            for entry in self.entries.values()
            if entry.user_id == user_id
        ]

        user_profile = self.user_profiles.get(user_id, {})

        export_data = {
            "user_id": user_id,
            "export_date": datetime.now().isoformat(),
            "entries": user_entries,
            "profile": user_profile,
            "total_entries": len(user_entries),
            "date_range": {
                "oldest": min(
                    (entry["timestamp"] for entry in user_entries), default=None
                ),
                "newest": max(
                    (entry["timestamp"] for entry in user_entries), default=None
                ),
            },
        }

        return export_data

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Obter estatísticas da memória."""
        now = datetime.now()

        # Estatísticas por tipo
        type_counts = Counter(entry.entry_type for entry in self.entries.values())

        # Estatísticas temporais
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        recent_counts = {
            "last_day": sum(
                1 for entry in self.entries.values() if entry.timestamp > day_ago
            ),
            "last_week": sum(
                1 for entry in self.entries.values() if entry.timestamp > week_ago
            ),
            "last_month": sum(
                1 for entry in self.entries.values() if entry.timestamp > month_ago
            ),
        }

        # Usuários ativos
        unique_users = len(set(entry.user_id for entry in self.entries.values()))

        return {
            "total_entries": len(self.entries),
            "unique_users": unique_users,
            "topic_clusters": len(self.topic_clusters),
            "entry_types": dict(type_counts),
            "recent_activity": recent_counts,
            "metrics": self.metrics,
            "storage_size": self._calculate_storage_size(),
            "avg_importance": (
                sum(entry.importance_score for entry in self.entries.values())
                / len(self.entries)
                if self.entries
                else 0
            ),
        }

    def _calculate_storage_size(self) -> str:
        """Calcular tamanho de armazenamento."""
        total_size = 0

        for file_path in self.persist_directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        # Converter para formato legível
        for unit in ["B", "KB", "MB", "GB"]:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0

        return f"{total_size:.1f} TB"


async def demonstrate_semantic_memory():
    """Demonstração completa do sistema de memória semântica."""
    print("🧠 DEMONSTRAÇÃO DO SISTEMA DE MEMÓRIA SEMÂNTICA")
    print("=" * 60)

    # Inicializar sistema
    memory_system = AdvancedSemanticMemory()

    # Simular conversações e conhecimentos
    user_id = "demo_user_001"
    session_id = "demo_session_001"

    print("\\n1️⃣ Adicionando conversações e conhecimentos...")

    # Adicionar várias entradas de exemplo
    sample_entries = [
        (
            "Olá! Meu nome é João e eu gosto muito de programação Python",
            "conversation",
            ["apresentação", "python", "programação"],
        ),
        (
            "Prefiro café no período da manhã, nunca chá",
            "preference",
            ["café", "manhã", "bebida"],
        ),
        (
            "Estou trabalhando em um projeto de IA usando machine learning",
            "conversation",
            ["ia", "machine learning", "projeto"],
        ),
        (
            "Não gosto de acordar muito cedo, prefiro trabalhar à tarde",
            "preference",
            ["horário", "trabalho", "tarde"],
        ),
        (
            "Python é minha linguagem favorita para ciência de dados",
            "fact",
            ["python", "ciência de dados", "linguagem"],
        ),
        (
            "Tenho dificuldade com matemática avançada, mas entendo o básico",
            "fact",
            ["matemática", "dificuldade", "conhecimento"],
        ),
        (
            "Gostaria de aprender mais sobre deep learning e redes neurais",
            "preference",
            ["deep learning", "aprendizado", "redes neurais"],
        ),
        (
            "Uso VS Code como editor principal para programação",
            "fact",
            ["vs code", "editor", "ferramenta"],
        ),
        (
            "Moro no Brasil, especificamente em São Paulo",
            "fact",
            ["brasil", "são paulo", "localização"],
        ),
        (
            "Tenho interesse em aplicações de IA para saúde",
            "preference",
            ["ia", "saúde", "aplicações"],
        ),
    ]

    entry_ids = []
    for content, entry_type, tags in sample_entries:
        entry_id = await memory_system.add_memory(
            content=content,
            session_id=session_id,
            user_id=user_id,
            entry_type=entry_type,
            tags=tags,
        )
        entry_ids.append(entry_id)
        await asyncio.sleep(0.1)  # Pequena pausa para simular tempo

    print(f"✅ {len(entry_ids)} entradas adicionadas com sucesso")

    # Demonstrar buscas semânticas
    print("\\n2️⃣ Testando buscas semânticas...")

    queries = [
        "O que João gosta de beber?",
        "Quais são as linguagens de programação preferidas?",
        "Onde ele mora?",
        "Quais são os interesses em inteligência artificial?",
        "Que ferramentas de desenvolvimento usa?",
    ]

    for query in queries:
        print(f"\\n🔍 Buscando: '{query}'")

        results = await memory_system.search_memories(
            query=query, user_id=user_id, limit=3, min_similarity=0.3
        )

        if results:
            for entry, similarity in results:
                print(f"   📄 [{similarity:.2f}] {entry.content}")
                print(f"       🏷️ Tags: {', '.join(entry.tags)}")
        else:
            print("   ❌ Nenhum resultado encontrado")

    # Demonstrar contexto de conversação
    print("\\n3️⃣ Obtendo contexto de conversação...")

    context = await memory_system.get_conversation_context(user_id=user_id)

    print(f"📊 Contexto do usuário {user_id}:")
    print(f"   • Entradas recentes: {context['recent_entries']}")
    print(f"   • Tópicos principais: {', '.join(context['topics'][:5])}")
    print(
        f"   • Preferências identificadas: {len(context['preferences'].get('preferences', []))}"
    )

    if context["summary"]:
        print(f"   📝 Resumo: {context['summary'][:100]}...")

    # Demonstrar tópicos relacionados
    print("\\n4️⃣ Encontrando tópicos relacionados...")

    related_topics = await memory_system.find_related_topics("programação")
    print("🔗 Tópicos relacionados a 'programação':")
    for topic, score in related_topics[:5]:
        print(f"   • {topic} (relevância: {score:.2f})")

    # Estatísticas finais
    print("\\n5️⃣ Estatísticas do sistema...")

    stats = memory_system.get_memory_statistics()
    print("📊 Estatísticas da memória:")
    print(f"   • Total de entradas: {stats['total_entries']}")
    print(f"   • Usuários únicos: {stats['unique_users']}")
    print(f"   • Clusters de tópicos: {stats['topic_clusters']}")
    print(f"   • Tamanho de armazenamento: {stats['storage_size']}")
    print(f"   • Importância média: {stats['avg_importance']:.2f}")
    print(f"   • Tempo médio de embedding: {stats['metrics']['embedding_time']:.3f}s")

    # Salvar dados
    memory_system.save_persistent_data()
    print("\\n💾 Dados salvos persistentemente")

    # Exportar dados do usuário
    print("\\n6️⃣ Exportando dados do usuário...")

    export_data = memory_system.export_user_data(user_id)
    export_file = f"user_export_{user_id}_{int(time.time())}.json"

    with open(export_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"📤 Dados exportados para: {export_file}")

    print("\\n✅ Demonstração concluída com sucesso!")
    print("💡 O sistema mantém memória persistente e aprende com interações")


async def interactive_memory_demo():
    """Demonstração interativa do sistema de memória."""
    print("🤖 DEMO INTERATIVO - SISTEMA DE MEMÓRIA SEMÂNTICA")
    print("=" * 55)

    memory_system = AdvancedSemanticMemory()
    user_id = "interactive_user"
    session_id = f"interactive_session_{int(time.time())}"

    print("\\n💡 Comandos disponíveis:")
    print("   'add [texto]' - Adicionar conhecimento")
    print("   'search [query]' - Buscar na memória")
    print("   'context' - Ver contexto atual")
    print("   'stats' - Ver estatísticas")
    print("   'export' - Exportar dados")
    print("   'quit' - Sair")

    while True:
        try:
            command = input("\\n🎯 Digite um comando: ").strip()

            if command.lower() in ["quit", "sair", "exit"]:
                break

            if command.startswith("add "):
                content = command[4:]
                if len(content) < 3:
                    print("❌ Conteúdo muito curto")
                    continue

                # Detectar tipo de entrada
                entry_type = "conversation"
                tags = []

                if any(
                    word in content.lower()
                    for word in ["gosto", "prefiro", "não gosto"]
                ):
                    entry_type = "preference"
                    tags.append("preferência")
                elif any(
                    word in content.lower() for word in ["sempre", "nunca", "fato"]
                ):
                    entry_type = "fact"
                    tags.append("fato")

                entry_id = await memory_system.add_memory(
                    content=content,
                    session_id=session_id,
                    user_id=user_id,
                    entry_type=entry_type,
                    tags=tags,
                )

                print(f"✅ Conhecimento adicionado: {entry_id}")

            elif command.startswith("search "):
                query = command[7:]
                if len(query) < 2:
                    print("❌ Query muito curta")
                    continue

                results = await memory_system.search_memories(
                    query=query, user_id=user_id, limit=5
                )

                if results:
                    print(f"🔍 Resultados para '{query}':")
                    for i, (entry, similarity) in enumerate(results, 1):
                        print(f"   {i}. [{similarity:.2f}] {entry.content}")
                        print(f"       🏷️ {', '.join(entry.tags)} | {entry.entry_type}")
                else:
                    print("❌ Nenhum resultado encontrado")

            elif command == "context":
                context = await memory_system.get_conversation_context(user_id)
                print("📊 Seu contexto atual:")
                print(f"   • Interações: {context['recent_entries']}")
                print(f"   • Tópicos: {', '.join(context['topics'][:8])}")

                if context["summary"]:
                    print(f"   📝 Resumo: {context['summary'][:150]}...")

            elif command == "stats":
                stats = memory_system.get_memory_statistics()
                print("📊 Estatísticas:")
                print(f"   • Entradas: {stats['total_entries']}")
                print(f"   • Clusters: {stats['topic_clusters']}")
                print(f"   • Última hora: {stats['recent_activity']['last_day']}")

            elif command == "export":
                export_data = memory_system.export_user_data(user_id)
                filename = f"memory_export_{int(time.time())}.json"

                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

                print(f"📤 Dados exportados para: {filename}")

            else:
                print("❌ Comando não reconhecido. Use 'quit' para sair.")

        except KeyboardInterrupt:
            print("\\n🛑 Interrompido pelo usuário")
            break
        except Exception as e:
            print(f"❌ Erro: {str(e)}")

    # Salvar dados ao sair
    memory_system.save_persistent_data()
    print("\\n💾 Dados salvos. Até logo!")


def main():
    """Função principal."""
    print("🧠 SISTEMA AVANÇADO DE MEMÓRIA SEMÂNTICA - GIANNA")
    print("=" * 60)

    mode = input(
        "\\n🎭 Escolha o modo:\\n1. Demonstração automática\\n2. Demo interativo\\nOpção (1-2): "
    ).strip()

    if mode == "2":
        asyncio.run(interactive_memory_demo())
    else:
        asyncio.run(demonstrate_semantic_memory())


if __name__ == "__main__":
    main()
