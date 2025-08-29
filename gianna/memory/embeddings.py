"""
Embedding providers for semantic memory system.

This module provides various embedding implementations including OpenAI embeddings
and local alternatives for offline usage.
"""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class EmbeddingProvider(Enum):
    """Available embedding providers."""

    OPENAI = "openai"
    LOCAL_SENTENCE_TRANSFORMERS = "sentence_transformers"
    LOCAL_HUGGINGFACE = "huggingface"


class AbstractEmbedding(ABC):
    """Abstract base class for embedding implementations."""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".gianna" / "embeddings_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_cache: Dict[str, List[float]] = {}
        self._load_cache()

    @abstractmethod
    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for a single text."""
        pass

    def embed_text(self, text: str) -> List[float]:
        """
        Get embedding for text with caching.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector
        """
        # Create cache key from text hash
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            embedding = self._compute_embedding(text)
            self._embedding_cache[cache_key] = embedding
            self._save_cache()
            return embedding
        except Exception as e:
            logger.error(f"Error computing embedding for text: {e}")
            # Return zero vector as fallback
            return [0.0] * self.get_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        return [self.embed_text(text) for text in texts]

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass

    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / f"{self.model_name}_cache.json"
        try:
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    self._embedding_cache = json.load(f)
                logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self._embedding_cache = {}

    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        cache_file = self.cache_dir / f"{self.model_name}_cache.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(self._embedding_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")


class OpenAIEmbedding(AbstractEmbedding):
    """OpenAI embedding implementation."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(model_name, cache_dir)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found, OpenAI embeddings will not work")
            return

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
            logger.info(
                f"Initialized OpenAI embedding client with model: {self.model_name}"
            )
        except ImportError:
            logger.error(
                "OpenAI package not installed. Install with: pip install openai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding using OpenAI API."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        try:
            response = self.client.embeddings.create(input=text, model=self.model_name)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding API error: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for OpenAI models."""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(self.model_name, 1536)


class SentenceTransformerEmbedding(AbstractEmbedding):
    """Local sentence transformers embedding implementation."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None
    ):
        super().__init__(model_name, cache_dir)
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence transformer model: {self.model_name}")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding using sentence transformers."""
        if not self.model:
            raise ValueError("Sentence transformer model not initialized")

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Sentence transformer embedding error: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for sentence transformer models."""
        model_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-MiniLM-L6-v2": 384,
            "paraphrase-mpnet-base-v2": 768,
        }
        return model_dimensions.get(self.model_name, 384)


class HuggingFaceEmbedding(AbstractEmbedding):
    """Local HuggingFace transformers embedding implementation."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(model_name, cache_dir)
        self.tokenizer = None
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize HuggingFace model and tokenizer."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            logger.info(f"Loaded HuggingFace model: {self.model_name} on {self.device}")
        except ImportError:
            logger.error(
                "transformers not installed. Install with: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding using HuggingFace transformers."""
        if not self.model or not self.tokenizer:
            raise ValueError("HuggingFace model not initialized")

        try:
            import torch

            # Tokenize and encode
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for HuggingFace models."""
        if self.model:
            return self.model.config.hidden_size
        return 768  # Default for BERT-like models


def create_embedding_provider(
    provider: EmbeddingProvider = EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS,
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> AbstractEmbedding:
    """
    Factory function to create embedding providers.

    Args:
        provider: Type of embedding provider to create
        model_name: Optional model name override
        cache_dir: Optional cache directory

    Returns:
        AbstractEmbedding: Configured embedding provider
    """
    if provider == EmbeddingProvider.OPENAI:
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbedding(model, cache_dir)

    elif provider == EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS:
        model = model_name or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedding(model, cache_dir)

    elif provider == EmbeddingProvider.LOCAL_HUGGINGFACE:
        model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbedding(model, cache_dir)

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def get_available_providers() -> List[EmbeddingProvider]:
    """
    Get list of available embedding providers based on installed packages.

    Returns:
        List[EmbeddingProvider]: Available providers
    """
    available = []

    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai

            available.append(EmbeddingProvider.OPENAI)
        except ImportError:
            pass

    # Check sentence transformers
    try:
        import sentence_transformers

        available.append(EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS)
    except ImportError:
        pass

    # Check HuggingFace transformers
    try:
        import torch
        import transformers

        available.append(EmbeddingProvider.LOCAL_HUGGINGFACE)
    except ImportError:
        pass

    return available
