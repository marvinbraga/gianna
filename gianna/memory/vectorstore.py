"""
Vector store implementations for semantic memory system.

This module provides various vector storage backends including Chroma, FAISS,
and in-memory implementations for different use cases.
"""

import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


class VectorStoreProvider(Enum):
    """Available vector store providers."""

    CHROMA = "chroma"
    FAISS = "faiss"
    IN_MEMORY = "in_memory"


class SearchResult:
    """Container for search results."""

    def __init__(
        self, id: str, content: str, metadata: Dict[str, Any], similarity_score: float
    ):
        self.id = id
        self.content = content
        self.metadata = metadata
        self.similarity_score = similarity_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "similarity_score": self.similarity_score,
        }


class AbstractVectorStore(ABC):
    """Abstract base class for vector store implementations."""

    def __init__(self, collection_name: str, persist_directory: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_directory = (
            Path(persist_directory)
            if persist_directory
            else Path.home() / ".gianna" / "vectorstore"
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add texts with embeddings and metadata to the store."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by their IDs."""
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Get total number of documents in the store."""
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        pass

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a_np = np.array(a)
            b_np = np.array(b)
            return float(
                np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
            )
        except Exception:
            return 0.0


class ChromaVectorStore(AbstractVectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, collection_name: str, persist_directory: Optional[str] = None):
        super().__init__(collection_name, persist_directory)
        self.client = None
        self.collection = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings

            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
        except ImportError:
            logger.error("chromadb not installed. Install with: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add texts to ChromaDB collection."""
        if not self.collection:
            raise ValueError("ChromaDB collection not initialized")

        if not ids:
            ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(texts))]

        try:
            self.collection.add(
                documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
            )
            logger.debug(f"Added {len(texts)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search ChromaDB collection for similar vectors."""
        if not self.collection:
            raise ValueError("ChromaDB collection not initialized")

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=k, where=filter_criteria
            )

            search_results = []
            for i in range(len(results["ids"][0])):
                search_results.append(
                    SearchResult(
                        id=results["ids"][0][i],
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        similarity_score=1.0
                        - results["distances"][0][i],  # Convert distance to similarity
                    )
                )

            return search_results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents from ChromaDB by IDs."""
        if not self.collection:
            raise ValueError("ChromaDB collection not initialized")

        try:
            self.collection.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} documents from ChromaDB")
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")

    def get_count(self) -> int:
        """Get document count from ChromaDB."""
        if not self.collection:
            return 0

        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting ChromaDB count: {e}")
            return 0

    def persist(self) -> None:
        """ChromaDB automatically persists data."""
        pass


class FAISSVectorStore(AbstractVectorStore):
    """FAISS vector store implementation."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
        dimension: int = 384,
    ):
        super().__init__(collection_name, persist_directory)
        self.dimension = dimension
        self.index = None
        self.documents: Dict[str, str] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self._next_index = 0
        self._init_index()
        self._load_data()

    def _init_index(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss

            # Use cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Initialized FAISS index with dimension: {self.dimension}")
        except ImportError:
            logger.error("faiss not installed. Install with: pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise

    def _normalize_vector(self, vector: List[float]) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.reshape(1, -1)

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add texts to FAISS index."""
        if not self.index:
            raise ValueError("FAISS index not initialized")

        if not ids:
            ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(texts))]

        try:
            # Normalize embeddings for cosine similarity
            normalized_embeddings = np.vstack(
                [self._normalize_vector(emb) for emb in embeddings]
            )

            # Add to index
            self.index.add(normalized_embeddings)

            # Store documents and metadata
            for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
                index_id = self._next_index + i
                self.documents[doc_id] = text
                self.metadatas[doc_id] = metadata
                self.id_to_index[doc_id] = index_id
                self.index_to_id[index_id] = doc_id

            self._next_index += len(texts)
            logger.debug(f"Added {len(texts)} documents to FAISS index")
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            raise

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search FAISS index for similar vectors."""
        if not self.index:
            raise ValueError("FAISS index not initialized")

        try:
            # Normalize query vector
            normalized_query = self._normalize_vector(query_embedding)

            # Search index
            similarities, indices = self.index.search(normalized_query, k)

            # Convert results
            search_results = []
            for i in range(len(indices[0])):
                index_id = indices[0][i]
                similarity = float(similarities[0][i])

                if index_id in self.index_to_id:
                    doc_id = self.index_to_id[index_id]

                    # Apply filter criteria
                    if filter_criteria:
                        metadata = self.metadatas.get(doc_id, {})
                        if not all(
                            metadata.get(k) == v for k, v in filter_criteria.items()
                        ):
                            continue

                    search_results.append(
                        SearchResult(
                            id=doc_id,
                            content=self.documents[doc_id],
                            metadata=self.metadatas[doc_id],
                            similarity_score=similarity,
                        )
                    )

            return search_results
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by IDs (FAISS doesn't support deletion, so we mark as deleted)."""
        try:
            for doc_id in ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    del self.metadatas[doc_id]
                    if doc_id in self.id_to_index:
                        index_id = self.id_to_index[doc_id]
                        del self.id_to_index[doc_id]
                        if index_id in self.index_to_id:
                            del self.index_to_id[index_id]

            logger.debug(f"Marked {len(ids)} documents as deleted in FAISS")
        except Exception as e:
            logger.error(f"Error deleting documents from FAISS: {e}")

    def get_count(self) -> int:
        """Get document count."""
        return len(self.documents)

    def persist(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            import faiss

            # Save FAISS index
            index_path = self.persist_directory / f"{self.collection_name}.faiss"
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            metadata_path = (
                self.persist_directory / f"{self.collection_name}_metadata.pkl"
            )
            with open(metadata_path, "wb") as f:
                pickle.dump(
                    {
                        "documents": self.documents,
                        "metadatas": self.metadatas,
                        "id_to_index": self.id_to_index,
                        "index_to_id": self.index_to_id,
                        "next_index": self._next_index,
                    },
                    f,
                )

            logger.debug(f"Persisted FAISS index to {index_path}")
        except Exception as e:
            logger.error(f"Error persisting FAISS index: {e}")

    def _load_data(self) -> None:
        """Load FAISS index and metadata from disk."""
        try:
            import faiss

            index_path = self.persist_directory / f"{self.collection_name}.faiss"
            metadata_path = (
                self.persist_directory / f"{self.collection_name}_metadata.pkl"
            )

            if index_path.exists() and metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))

                # Load metadata
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["documents"]
                    self.metadatas = data["metadatas"]
                    self.id_to_index = data["id_to_index"]
                    self.index_to_id = data["index_to_id"]
                    self._next_index = data["next_index"]

                logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
        except Exception as e:
            logger.warning(f"Failed to load existing FAISS data: {e}")


class InMemoryVectorStore(AbstractVectorStore):
    """Simple in-memory vector store implementation."""

    def __init__(self, collection_name: str, persist_directory: Optional[str] = None):
        super().__init__(collection_name, persist_directory)
        self.documents: Dict[str, str] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}
        self._load_data()

    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add texts to in-memory store."""
        if not ids:
            ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(texts))]

        for text, embedding, metadata, doc_id in zip(texts, embeddings, metadatas, ids):
            self.documents[doc_id] = text
            self.embeddings[doc_id] = embedding
            self.metadatas[doc_id] = metadata

        logger.debug(f"Added {len(texts)} documents to in-memory store")

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search in-memory store for similar vectors."""
        results = []

        for doc_id, doc_embedding in self.embeddings.items():
            # Apply filter criteria
            if filter_criteria:
                metadata = self.metadatas.get(doc_id, {})
                if not all(
                    metadata.get(key) == value for key, value in filter_criteria.items()
                ):
                    continue

            similarity = self.cosine_similarity(query_embedding, doc_embedding)

            results.append(
                SearchResult(
                    id=doc_id,
                    content=self.documents[doc_id],
                    metadata=self.metadatas[doc_id],
                    similarity_score=similarity,
                )
            )

        # Sort by similarity and return top k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:k]

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        for doc_id in ids:
            self.documents.pop(doc_id, None)
            self.embeddings.pop(doc_id, None)
            self.metadatas.pop(doc_id, None)

        logger.debug(f"Deleted {len(ids)} documents from in-memory store")

    def get_count(self) -> int:
        """Get document count."""
        return len(self.documents)

    def persist(self) -> None:
        """Save in-memory store to disk."""
        try:
            data_path = self.persist_directory / f"{self.collection_name}_data.json"

            data = {
                "documents": self.documents,
                "embeddings": self.embeddings,
                "metadatas": self.metadatas,
            }

            with open(data_path, "w") as f:
                json.dump(data, f)

            logger.debug(f"Persisted in-memory store to {data_path}")
        except Exception as e:
            logger.error(f"Error persisting in-memory store: {e}")

    def _load_data(self) -> None:
        """Load in-memory store from disk."""
        try:
            data_path = self.persist_directory / f"{self.collection_name}_data.json"

            if data_path.exists():
                with open(data_path, "r") as f:
                    data = json.load(f)

                self.documents = data.get("documents", {})
                self.embeddings = data.get("embeddings", {})
                self.metadatas = data.get("metadatas", {})

                logger.info(
                    f"Loaded in-memory store with {len(self.documents)} documents"
                )
        except Exception as e:
            logger.warning(f"Failed to load existing in-memory data: {e}")


def create_vector_store(
    provider: VectorStoreProvider,
    collection_name: str,
    persist_directory: Optional[str] = None,
    dimension: int = 384,
) -> AbstractVectorStore:
    """
    Factory function to create vector store providers.

    Args:
        provider: Type of vector store to create
        collection_name: Name of the collection
        persist_directory: Directory for persistence
        dimension: Vector dimension (for FAISS)

    Returns:
        AbstractVectorStore: Configured vector store
    """
    if provider == VectorStoreProvider.CHROMA:
        return ChromaVectorStore(collection_name, persist_directory)
    elif provider == VectorStoreProvider.FAISS:
        return FAISSVectorStore(collection_name, persist_directory, dimension)
    elif provider == VectorStoreProvider.IN_MEMORY:
        return InMemoryVectorStore(collection_name, persist_directory)
    else:
        raise ValueError(f"Unsupported vector store provider: {provider}")


def get_available_vector_stores() -> List[VectorStoreProvider]:
    """
    Get list of available vector store providers.

    Returns:
        List[VectorStoreProvider]: Available providers
    """
    available = [VectorStoreProvider.IN_MEMORY]  # Always available

    # Check ChromaDB
    try:
        import chromadb

        available.append(VectorStoreProvider.CHROMA)
    except ImportError:
        pass

    # Check FAISS
    try:
        import faiss

        available.append(VectorStoreProvider.FAISS)
    except ImportError:
        pass

    return available
