"""
Semantic Memory System for Gianna AI Assistant.

This module provides comprehensive semantic memory capabilities including
embedding-based storage, retrieval, clustering, and context management.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from ..core.state import ConversationState, GiannaState
from .embeddings import (
    AbstractEmbedding,
    EmbeddingProvider,
    create_embedding_provider,
    get_available_providers,
)
from .vectorstore import (
    AbstractVectorStore,
    SearchResult,
    VectorStoreProvider,
    create_vector_store,
    get_available_vector_stores,
)


@dataclass
class MemoryConfig:
    """Configuration for semantic memory system."""

    # Embedding configuration
    embedding_provider: EmbeddingProvider = (
        EmbeddingProvider.LOCAL_SENTENCE_TRANSFORMERS
    )
    embedding_model: Optional[str] = None

    # Vector store configuration
    vectorstore_provider: VectorStoreProvider = VectorStoreProvider.IN_MEMORY
    collection_name: str = "gianna_memory"
    persist_directory: Optional[str] = None

    # Search configuration
    similarity_threshold: float = 0.7
    max_search_results: int = 10

    # Context configuration
    context_window_size: int = 5  # Number of interactions to consider for context
    max_context_length: int = 4000  # Max characters in context summary

    # Clustering configuration
    enable_clustering: bool = True
    cluster_similarity_threshold: float = 0.85
    min_cluster_size: int = 3

    # Maintenance configuration
    auto_summarize_threshold: int = 50  # Summarize after N interactions
    cleanup_old_interactions: bool = True
    max_age_days: int = 30


class InteractionMemory(BaseModel):
    """Represents a stored interaction in memory."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime
    user_input: str
    assistant_response: str
    context: str = ""
    interaction_type: str = "conversation"  # conversation, command, system
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    cluster_id: Optional[str] = None
    summary: Optional[str] = None


class ContextSummary(BaseModel):
    """Represents a context summary."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    timestamp: datetime
    summary_text: str
    interaction_count: int
    time_range: Tuple[datetime, datetime]
    key_topics: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticMemory:
    """
    Semantic memory system for Gianna AI Assistant.

    Provides embedding-based storage and retrieval of conversation interactions,
    with support for clustering, context summarization, and pattern detection.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize semantic memory system.

        Args:
            config: Configuration for the memory system
        """
        self.config = config or MemoryConfig()
        self.embedding_provider: Optional[AbstractEmbedding] = None
        self.vector_store: Optional[AbstractVectorStore] = None
        self.interaction_cache: Dict[str, InteractionMemory] = {}
        self.context_summaries: Dict[str, List[ContextSummary]] = {}

        self._init_providers()

        logger.info(
            f"Initialized SemanticMemory with {self.config.embedding_provider.value} embeddings "
            f"and {self.config.vectorstore_provider.value} vector store"
        )

    def _init_providers(self) -> None:
        """Initialize embedding and vector store providers."""
        try:
            # Initialize embedding provider
            available_embedding_providers = get_available_providers()

            if self.config.embedding_provider not in available_embedding_providers:
                logger.warning(
                    f"Preferred embedding provider {self.config.embedding_provider.value} not available"
                )
                if available_embedding_providers:
                    self.config.embedding_provider = available_embedding_providers[0]
                    logger.info(
                        f"Falling back to {self.config.embedding_provider.value}"
                    )
                else:
                    raise ValueError("No embedding providers available")

            self.embedding_provider = create_embedding_provider(
                self.config.embedding_provider,
                self.config.embedding_model,
                str(Path.home() / ".gianna" / "embeddings_cache"),
            )

            # Initialize vector store
            available_vector_stores = get_available_vector_stores()

            if self.config.vectorstore_provider not in available_vector_stores:
                logger.warning(
                    f"Preferred vector store {self.config.vectorstore_provider.value} not available"
                )
                self.config.vectorstore_provider = VectorStoreProvider.IN_MEMORY
                logger.info("Falling back to in-memory vector store")

            # Get embedding dimension
            dimension = self.embedding_provider.get_embedding_dimension()

            self.vector_store = create_vector_store(
                self.config.vectorstore_provider,
                self.config.collection_name,
                self.config.persist_directory
                or str(Path.home() / ".gianna" / "vectorstore"),
                dimension,
            )

        except Exception as e:
            logger.error(f"Failed to initialize memory providers: {e}")
            raise

    def store_interaction(
        self,
        session_id: str,
        user_input: str,
        assistant_response: str,
        context: str = "",
        interaction_type: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store an interaction in semantic memory.

        Args:
            session_id: Session identifier
            user_input: User's input text
            assistant_response: Assistant's response
            context: Additional context information
            interaction_type: Type of interaction
            metadata: Additional metadata

        Returns:
            str: Interaction ID
        """
        try:
            # Create interaction object
            interaction = InteractionMemory(
                session_id=session_id,
                timestamp=datetime.now(),
                user_input=user_input,
                assistant_response=assistant_response,
                context=context,
                interaction_type=interaction_type,
                metadata=metadata or {},
            )

            # Create text for embedding
            text_to_embed = f"User: {user_input}\nAssistant: {assistant_response}"
            if context:
                text_to_embed += f"\nContext: {context}"

            # Generate embedding
            if self.embedding_provider:
                interaction.embedding = self.embedding_provider.embed_text(
                    text_to_embed
                )

            # Store in vector store
            if self.vector_store and interaction.embedding:
                self.vector_store.add_texts(
                    texts=[text_to_embed],
                    embeddings=[interaction.embedding],
                    metadatas=[
                        {
                            "session_id": session_id,
                            "timestamp": interaction.timestamp.isoformat(),
                            "interaction_type": interaction_type,
                            "user_input": user_input,
                            "assistant_response": assistant_response,
                            **interaction.metadata,
                        }
                    ],
                    ids=[interaction.id],
                )

                # Persist vector store
                self.vector_store.persist()

            # Cache interaction
            self.interaction_cache[interaction.id] = interaction

            # Check for clustering
            if self.config.enable_clustering:
                self._update_clusters(interaction)

            # Check for auto-summarization
            session_interactions = self._get_session_interactions(session_id)
            if len(session_interactions) % self.config.auto_summarize_threshold == 0:
                self._create_context_summary(session_id)

            logger.debug(
                f"Stored interaction {interaction.id} for session {session_id}"
            )
            return interaction.id

        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            return ""

    def search_similar_interactions(
        self,
        query: str,
        session_id: Optional[str] = None,
        interaction_type: Optional[str] = None,
        max_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[InteractionMemory]:
        """
        Search for similar interactions using semantic similarity.

        Args:
            query: Search query
            session_id: Optional session filter
            interaction_type: Optional interaction type filter
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity threshold

        Returns:
            List[InteractionMemory]: Similar interactions
        """
        if not self.embedding_provider or not self.vector_store:
            logger.warning("Memory providers not initialized")
            return []

        max_results = max_results or self.config.max_search_results
        similarity_threshold = similarity_threshold or self.config.similarity_threshold

        try:
            # Generate query embedding
            query_embedding = self.embedding_provider.embed_text(query)

            # Build filter criteria
            filter_criteria = {}
            if session_id:
                filter_criteria["session_id"] = session_id
            if interaction_type:
                filter_criteria["interaction_type"] = interaction_type

            # Search vector store
            search_results = self.vector_store.similarity_search(
                query_embedding,
                k=max_results,
                filter_criteria=filter_criteria if filter_criteria else None,
            )

            # Filter by similarity threshold and convert to InteractionMemory
            similar_interactions = []
            for result in search_results:
                if result.similarity_score >= similarity_threshold:
                    # Try to get from cache first
                    if result.id in self.interaction_cache:
                        interaction = self.interaction_cache[result.id]
                    else:
                        # Reconstruct from search result
                        interaction = InteractionMemory(
                            id=result.id,
                            session_id=result.metadata.get("session_id", ""),
                            timestamp=datetime.fromisoformat(
                                result.metadata.get(
                                    "timestamp", datetime.now().isoformat()
                                )
                            ),
                            user_input=result.metadata.get("user_input", ""),
                            assistant_response=result.metadata.get(
                                "assistant_response", ""
                            ),
                            interaction_type=result.metadata.get(
                                "interaction_type", "conversation"
                            ),
                            metadata={
                                k: v
                                for k, v in result.metadata.items()
                                if k
                                not in [
                                    "session_id",
                                    "timestamp",
                                    "user_input",
                                    "assistant_response",
                                    "interaction_type",
                                ]
                            },
                        )
                        self.interaction_cache[result.id] = interaction

                    similar_interactions.append(interaction)

            logger.debug(
                f"Found {len(similar_interactions)} similar interactions for query: {query[:50]}..."
            )
            return similar_interactions

        except Exception as e:
            logger.error(f"Error searching similar interactions: {e}")
            return []

    def get_context_summary(
        self, session_id: str, max_interactions: Optional[int] = None
    ) -> str:
        """
        Get a context summary for a session.

        Args:
            session_id: Session identifier
            max_interactions: Maximum interactions to consider

        Returns:
            str: Context summary
        """
        max_interactions = max_interactions or self.config.context_window_size

        try:
            # Get recent interactions
            recent_interactions = self._get_session_interactions(
                session_id, limit=max_interactions
            )

            if not recent_interactions:
                return "No interaction history available."

            # Check if we have existing summaries
            if (
                session_id in self.context_summaries
                and self.context_summaries[session_id]
            ):
                latest_summary = max(
                    self.context_summaries[session_id], key=lambda x: x.timestamp
                )

                # Find interactions after the summary
                new_interactions = [
                    i
                    for i in recent_interactions
                    if i.timestamp > latest_summary.time_range[1]
                ]

                if len(new_interactions) < max_interactions // 2:
                    # Use existing summary with recent additions
                    summary_text = latest_summary.summary_text
                    if new_interactions:
                        new_summary = self._generate_interaction_summary(
                            new_interactions
                        )
                        summary_text += f"\n\nRecent updates: {new_summary}"
                    return summary_text

            # Generate new summary
            summary = self._generate_interaction_summary(recent_interactions)

            # Store summary if we have many interactions
            if len(recent_interactions) >= self.config.auto_summarize_threshold // 2:
                self._create_context_summary(session_id, recent_interactions, summary)

            return summary

        except Exception as e:
            logger.error(f"Error generating context summary: {e}")
            return "Error generating context summary."

    def detect_patterns(self, session_id: str) -> Dict[str, Any]:
        """
        Detect patterns in user interactions.

        Args:
            session_id: Session identifier

        Returns:
            Dict containing detected patterns
        """
        try:
            interactions = self._get_session_interactions(session_id)

            if len(interactions) < 3:
                return {"message": "Not enough interactions for pattern detection"}

            patterns = {
                "total_interactions": len(interactions),
                "time_span": self._get_session_timespan(interactions),
                "interaction_types": self._analyze_interaction_types(interactions),
                "common_themes": self._extract_common_themes(interactions),
                "user_preferences": self._detect_user_preferences(interactions),
                "conversation_patterns": self._analyze_conversation_patterns(
                    interactions
                ),
                "clusters": self._get_interaction_clusters(session_id),
            }

            return patterns

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {"error": "Failed to detect patterns"}

    def cleanup_old_interactions(self, max_age_days: Optional[int] = None) -> int:
        """
        Clean up old interactions.

        Args:
            max_age_days: Maximum age in days

        Returns:
            int: Number of interactions cleaned up
        """
        if not self.config.cleanup_old_interactions:
            return 0

        max_age_days = max_age_days or self.config.max_age_days
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        try:
            old_interactions = []

            # Find old interactions in cache
            for interaction_id, interaction in self.interaction_cache.items():
                if interaction.timestamp < cutoff_date:
                    old_interactions.append(interaction_id)

            # Remove from cache
            for interaction_id in old_interactions:
                del self.interaction_cache[interaction_id]

            # Remove from vector store
            if self.vector_store and old_interactions:
                self.vector_store.delete_by_ids(old_interactions)
                self.vector_store.persist()

            logger.info(f"Cleaned up {len(old_interactions)} old interactions")
            return len(old_interactions)

        except Exception as e:
            logger.error(f"Error cleaning up old interactions: {e}")
            return 0

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Dict containing memory statistics
        """
        try:
            total_interactions = (
                self.vector_store.get_count() if self.vector_store else 0
            )

            stats = {
                "total_interactions": total_interactions,
                "cached_interactions": len(self.interaction_cache),
                "total_summaries": sum(
                    len(summaries) for summaries in self.context_summaries.values()
                ),
                "embedding_provider": self.config.embedding_provider.value,
                "vector_store_provider": self.config.vectorstore_provider.value,
                "embedding_dimension": (
                    self.embedding_provider.get_embedding_dimension()
                    if self.embedding_provider
                    else 0
                ),
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "average_similarity_threshold": self.config.similarity_threshold,
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": "Failed to get memory statistics"}

    def _get_session_interactions(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[InteractionMemory]:
        """Get interactions for a specific session."""
        # Get from cache first
        session_interactions = [
            interaction
            for interaction in self.interaction_cache.values()
            if interaction.session_id == session_id
        ]

        # Sort by timestamp
        session_interactions.sort(key=lambda x: x.timestamp, reverse=True)

        if limit:
            session_interactions = session_interactions[:limit]

        return session_interactions

    def _generate_interaction_summary(
        self, interactions: List[InteractionMemory]
    ) -> str:
        """Generate a summary of interactions."""
        if not interactions:
            return "No interactions to summarize."

        # Simple extractive summary (can be improved with LLM)
        summary_parts = []

        # Add time range
        start_time = min(i.timestamp for i in interactions)
        end_time = max(i.timestamp for i in interactions)
        summary_parts.append(
            f"Conversation from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}"
        )

        # Add interaction count
        summary_parts.append(f"Total interactions: {len(interactions)}")

        # Add common themes
        themes = self._extract_common_themes(interactions)
        if themes:
            summary_parts.append(f"Main topics: {', '.join(themes[:3])}")

        # Add key interactions
        if len(interactions) > 5:
            key_interactions = interactions[:2] + interactions[-1:]  # First 2 and last
            summary_parts.append("Key interactions:")
            for i, interaction in enumerate(key_interactions):
                summary_parts.append(
                    f"{i+1}. User: {interaction.user_input[:100]}{'...' if len(interaction.user_input) > 100 else ''}"
                )

        return "\n".join(summary_parts)

    def _create_context_summary(
        self,
        session_id: str,
        interactions: Optional[List[InteractionMemory]] = None,
        summary_text: Optional[str] = None,
    ) -> None:
        """Create and store a context summary."""
        if not interactions:
            interactions = self._get_session_interactions(session_id)

        if not interactions:
            return

        if not summary_text:
            summary_text = self._generate_interaction_summary(interactions)

        summary = ContextSummary(
            session_id=session_id,
            timestamp=datetime.now(),
            summary_text=summary_text,
            interaction_count=len(interactions),
            time_range=(
                min(i.timestamp for i in interactions),
                max(i.timestamp for i in interactions),
            ),
            key_topics=self._extract_common_themes(interactions)[:5],
        )

        if session_id not in self.context_summaries:
            self.context_summaries[session_id] = []

        self.context_summaries[session_id].append(summary)

        # Keep only recent summaries
        self.context_summaries[session_id] = sorted(
            self.context_summaries[session_id], key=lambda x: x.timestamp, reverse=True
        )[
            :10
        ]  # Keep last 10 summaries

    def _update_clusters(self, interaction: InteractionMemory) -> None:
        """Update interaction clusters."""
        if not interaction.embedding:
            return

        # Simple clustering based on similarity threshold
        # This is a basic implementation - can be improved with proper clustering algorithms

        similar_interactions = []
        for cached_interaction in self.interaction_cache.values():
            if (
                cached_interaction.session_id == interaction.session_id
                and cached_interaction.embedding
                and cached_interaction.id != interaction.id
            ):

                similarity = self._cosine_similarity(
                    interaction.embedding, cached_interaction.embedding
                )
                if similarity >= self.config.cluster_similarity_threshold:
                    similar_interactions.append((cached_interaction, similarity))

        if similar_interactions:
            # Find the most similar interaction with a cluster
            clustered_interactions = [
                (i, s) for i, s in similar_interactions if i.cluster_id
            ]

            if clustered_interactions:
                # Join existing cluster
                most_similar = max(clustered_interactions, key=lambda x: x[1])
                interaction.cluster_id = most_similar[0].cluster_id
            else:
                # Create new cluster
                cluster_id = str(uuid.uuid4())
                interaction.cluster_id = cluster_id

                # Assign cluster to similar interactions
                for similar_interaction, _ in similar_interactions:
                    similar_interaction.cluster_id = cluster_id

    def _get_interaction_clusters(self, session_id: str) -> Dict[str, List[str]]:
        """Get interaction clusters for a session."""
        clusters = {}

        for interaction in self.interaction_cache.values():
            if interaction.session_id == session_id and interaction.cluster_id:
                if interaction.cluster_id not in clusters:
                    clusters[interaction.cluster_id] = []
                clusters[interaction.cluster_id].append(interaction.id)

        # Filter clusters by minimum size
        filtered_clusters = {
            cluster_id: interaction_ids
            for cluster_id, interaction_ids in clusters.items()
            if len(interaction_ids) >= self.config.min_cluster_size
        }

        return filtered_clusters

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a_np = np.array(a)
            b_np = np.array(b)
            return float(
                np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
            )
        except Exception:
            return 0.0

    def _get_session_timespan(
        self, interactions: List[InteractionMemory]
    ) -> Dict[str, Any]:
        """Calculate session timespan."""
        if not interactions:
            return {}

        start_time = min(i.timestamp for i in interactions)
        end_time = max(i.timestamp for i in interactions)
        duration = end_time - start_time

        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": int(duration.total_seconds() / 60),
            "total_interactions": len(interactions),
        }

    def _analyze_interaction_types(
        self, interactions: List[InteractionMemory]
    ) -> Dict[str, int]:
        """Analyze interaction types distribution."""
        type_counts = {}
        for interaction in interactions:
            type_counts[interaction.interaction_type] = (
                type_counts.get(interaction.interaction_type, 0) + 1
            )
        return type_counts

    def _extract_common_themes(
        self, interactions: List[InteractionMemory]
    ) -> List[str]:
        """Extract common themes from interactions (simple keyword-based)."""
        # Simple keyword extraction - can be improved with NLP
        word_freq = {}

        for interaction in interactions:
            words = (
                (interaction.user_input + " " + interaction.assistant_response)
                .lower()
                .split()
            )
            for word in words:
                if len(word) > 4 and word.isalpha():  # Simple filtering
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Return most frequent words as themes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:10] if count > 1]

    def _detect_user_preferences(
        self, interactions: List[InteractionMemory]
    ) -> Dict[str, Any]:
        """Detect user preferences from interactions."""
        preferences = {
            "interaction_frequency": len(interactions),
            "preferred_interaction_types": self._analyze_interaction_types(
                interactions
            ),
            "average_input_length": np.mean([len(i.user_input) for i in interactions]),
            "session_duration_preference": self._get_session_timespan(interactions).get(
                "duration_minutes", 0
            ),
        }

        return preferences

    def _analyze_conversation_patterns(
        self, interactions: List[InteractionMemory]
    ) -> Dict[str, Any]:
        """Analyze conversation patterns."""
        if len(interactions) < 2:
            return {}

        patterns = {
            "average_response_complexity": np.mean(
                [len(i.assistant_response) for i in interactions]
            ),
            "question_to_statement_ratio": self._calculate_question_ratio(interactions),
            "conversation_continuity": self._analyze_continuity(interactions),
        }

        return patterns

    def _calculate_question_ratio(self, interactions: List[InteractionMemory]) -> float:
        """Calculate the ratio of questions to statements in user inputs."""
        questions = sum(1 for i in interactions if "?" in i.user_input)
        total = len(interactions)
        return questions / total if total > 0 else 0.0

    def _analyze_continuity(self, interactions: List[InteractionMemory]) -> float:
        """Analyze conversation continuity (simple time-based measure)."""
        if len(interactions) < 2:
            return 1.0

        intervals = []
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)

        for i in range(1, len(sorted_interactions)):
            interval = (
                sorted_interactions[i].timestamp - sorted_interactions[i - 1].timestamp
            ).total_seconds()
            intervals.append(interval)

        # Calculate continuity based on consistent timing
        if not intervals:
            return 1.0

        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        # Higher continuity if intervals are consistent
        continuity = 1.0 / (
            1.0 + (std_interval / avg_interval if avg_interval > 0 else 1.0)
        )
        return float(continuity)

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder implementation)."""
        # This would need to be tracked during actual usage
        return 0.85  # Placeholder value
