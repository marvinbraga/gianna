"""
Memory System for Gianna AI Assistant

This package provides semantic memory capabilities for the Gianna system,
including embedding-based storage and retrieval of conversation interactions.
"""

from .embeddings import EmbeddingProvider
from .semantic_memory import MemoryConfig, SemanticMemory
from .vectorstore import VectorStoreProvider

__all__ = [
    "SemanticMemory",
    "MemoryConfig",
    "EmbeddingProvider",
    "VectorStoreProvider",
]
