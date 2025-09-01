"""
Integration between Semantic Memory and Gianna State Management.

This module provides utilities to integrate the semantic memory system
with the existing GiannaState and StateManager infrastructure.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from ..core.state import ConversationState, GiannaState
from ..core.state_manager import StateManager
from .semantic_memory import InteractionMemory, MemoryConfig, SemanticMemory


class MemoryIntegratedStateManager(StateManager):
    """
    Extended StateManager with semantic memory integration.

    This class extends the standard StateManager to automatically
    store conversations in semantic memory and provide enhanced
    context retrieval capabilities.
    """

    def __init__(
        self,
        db_path: str = "gianna_state.db",
        memory_config: Optional[MemoryConfig] = None,
    ):
        """
        Initialize with semantic memory integration.

        Args:
            db_path: Path to SQLite database
            memory_config: Configuration for semantic memory
        """
        super().__init__(db_path)

        # Initialize semantic memory
        self.memory_config = memory_config or MemoryConfig()
        self.semantic_memory: Optional[SemanticMemory] = None

        self._init_semantic_memory()

    def _init_semantic_memory(self) -> None:
        """Initialize semantic memory system."""
        try:
            self.semantic_memory = SemanticMemory(self.memory_config)
            logger.info("Semantic memory integration initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic memory: {e}")
            logger.info("Continuing without semantic memory features")

    def save_state(self, session_id: str, state: GiannaState) -> None:
        """
        Save state with semantic memory integration.

        Args:
            session_id: Session identifier
            state: GiannaState to save
        """
        # Save to standard database first
        super().save_state(session_id, state)

        # Extract and store interactions in semantic memory
        if self.semantic_memory and state["conversation"].messages:
            self._store_conversations_in_memory(session_id, state["conversation"])

    def load_state(self, session_id: str) -> Optional[GiannaState]:
        """
        Load state with enhanced context from semantic memory.

        Args:
            session_id: Session identifier

        Returns:
            Optional[GiannaState]: Loaded state with enhanced context
        """
        state = super().load_state(session_id)

        if state and self.semantic_memory:
            # Enhance context summary with semantic memory
            enhanced_summary = self._get_enhanced_context_summary(
                session_id, state["conversation"]
            )
            state["conversation"].context_summary = enhanced_summary

        return state

    def search_similar_conversations(
        self,
        query: str,
        session_id: Optional[str] = None,
        max_results: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[InteractionMemory]:
        """
        Search for similar conversations across all sessions.

        Args:
            query: Search query
            session_id: Optional session filter
            max_results: Maximum results to return
            similarity_threshold: Similarity threshold

        Returns:
            List of similar interactions
        """
        if not self.semantic_memory:
            logger.warning("Semantic memory not available")
            return []

        return self.semantic_memory.search_similar_interactions(
            query=query,
            session_id=session_id,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
        )

    def get_conversation_patterns(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation patterns for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary of detected patterns
        """
        if not self.semantic_memory:
            return {"error": "Semantic memory not available"}

        return self.semantic_memory.detect_patterns(session_id)

    def get_semantic_context_summary(
        self, session_id: str, max_interactions: int = 10
    ) -> str:
        """
        Get semantic context summary for a session.

        Args:
            session_id: Session identifier
            max_interactions: Maximum interactions to consider

        Returns:
            Context summary string
        """
        if not self.semantic_memory:
            return "Semantic memory context not available"

        return self.semantic_memory.get_context_summary(session_id, max_interactions)

    def cleanup_old_memory(self, max_age_days: int = 30) -> Dict[str, int]:
        """
        Clean up old memory data.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {}

        # Clean up old sessions from database
        db_cleanup = super().cleanup_old_sessions(max_age_days)
        cleanup_stats["database_sessions"] = db_cleanup

        # Clean up old semantic memory
        if self.semantic_memory:
            memory_cleanup = self.semantic_memory.cleanup_old_interactions(max_age_days)
            cleanup_stats["semantic_interactions"] = memory_cleanup

        return cleanup_stats

    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "database_sessions": len(self.get_session_list()),
            "semantic_memory_available": self.semantic_memory is not None,
        }

        if self.semantic_memory:
            semantic_stats = self.semantic_memory.get_memory_stats()
            stats.update(semantic_stats)

        return stats

    def _store_conversations_in_memory(
        self, session_id: str, conversation: ConversationState
    ) -> None:
        """Store conversation messages in semantic memory."""
        if not self.semantic_memory or not conversation.messages:
            return

        # Process messages in pairs (user input + assistant response)
        messages = conversation.messages

        for i in range(0, len(messages) - 1, 2):
            if (
                i + 1 < len(messages)
                and messages[i].get("role") in ["user", "human"]
                and messages[i + 1].get("role") in ["assistant", "ai"]
            ):

                user_msg = messages[i]
                assistant_msg = messages[i + 1]

                try:
                    self.semantic_memory.store_interaction(
                        session_id=session_id,
                        user_input=user_msg.get("content", ""),
                        assistant_response=assistant_msg.get("content", ""),
                        context=conversation.context_summary or "",
                        interaction_type="conversation",
                        metadata={
                            "timestamp": user_msg.get(
                                "timestamp", datetime.now().isoformat()
                            ),
                            "user_preferences": conversation.user_preferences,
                        },
                    )

                except Exception as e:
                    logger.error(f"Error storing interaction in semantic memory: {e}")

    def _get_enhanced_context_summary(
        self, session_id: str, conversation: ConversationState
    ) -> str:
        """Get enhanced context summary using semantic memory."""
        if not self.semantic_memory:
            return conversation.context_summary

        try:
            # Get semantic context summary
            semantic_summary = self.semantic_memory.get_context_summary(session_id)

            # Combine with existing context
            if conversation.context_summary:
                enhanced_summary = f"{conversation.context_summary}\n\nSemantic Context:\n{semantic_summary}"
            else:
                enhanced_summary = semantic_summary

            return enhanced_summary

        except Exception as e:
            logger.error(f"Error generating enhanced context summary: {e}")
            return conversation.context_summary


def create_memory_integrated_manager(
    db_path: str = "gianna_state.db", memory_config: Optional[MemoryConfig] = None
) -> MemoryIntegratedStateManager:
    """
    Factory function to create a memory-integrated state manager.

    Args:
        db_path: Path to SQLite database
        memory_config: Configuration for semantic memory

    Returns:
        MemoryIntegratedStateManager: Configured manager
    """
    return MemoryIntegratedStateManager(db_path, memory_config)


def migrate_conversations_to_semantic_memory(
    state_manager: StateManager,
    semantic_memory: SemanticMemory,
    session_limit: Optional[int] = None,
) -> Dict[str, int]:
    """
    Migrate existing conversations from database to semantic memory.

    Args:
        state_manager: Existing state manager
        semantic_memory: Semantic memory instance
        session_limit: Optional limit on number of sessions to migrate

    Returns:
        Dictionary with migration statistics
    """
    logger.info("Starting migration of conversations to semantic memory")

    stats = {"sessions_processed": 0, "interactions_migrated": 0, "errors": 0}

    try:
        # Get list of sessions
        sessions = state_manager.get_session_list(limit=session_limit or 1000)

        for session_info in sessions:
            session_id = session_info["session_id"]

            try:
                # Load session state
                state = state_manager.load_state(session_id)

                if state and state["conversation"].messages:
                    # Extract and store interactions
                    messages = state["conversation"].messages

                    for i in range(0, len(messages) - 1, 2):
                        if (
                            i + 1 < len(messages)
                            and messages[i].get("role") in ["user", "human"]
                            and messages[i + 1].get("role") in ["assistant", "ai"]
                        ):

                            user_msg = messages[i]
                            assistant_msg = messages[i + 1]

                            semantic_memory.store_interaction(
                                session_id=session_id,
                                user_input=user_msg.get("content", ""),
                                assistant_response=assistant_msg.get("content", ""),
                                context=state["conversation"].context_summary or "",
                                interaction_type="conversation",
                                metadata={
                                    "migrated": True,
                                    "original_timestamp": user_msg.get("timestamp", ""),
                                    "session_created": session_info.get(
                                        "created_at", ""
                                    ),
                                },
                            )

                            stats["interactions_migrated"] += 1

                stats["sessions_processed"] += 1

                if stats["sessions_processed"] % 10 == 0:
                    logger.info(
                        f"Migrated {stats['sessions_processed']} sessions, {stats['interactions_migrated']} interactions"
                    )

            except Exception as e:
                logger.error(f"Error migrating session {session_id}: {e}")
                stats["errors"] += 1

    except Exception as e:
        logger.error(f"Error during migration: {e}")
        stats["errors"] += 1

    logger.info(f"Migration completed: {stats}")
    return stats


# Utility functions for working with memory-integrated workflows


def add_memory_to_workflow_state(
    state: GiannaState,
    session_id: str,
    semantic_memory: SemanticMemory,
    query: Optional[str] = None,
) -> GiannaState:
    """
    Add semantic memory context to a workflow state.

    Args:
        state: Current workflow state
        session_id: Session identifier
        semantic_memory: Semantic memory instance
        query: Optional query for retrieving relevant context

    Returns:
        Enhanced workflow state
    """
    try:
        # Get context summary
        if query:
            # Search for relevant context
            similar_interactions = semantic_memory.search_similar_interactions(
                query=query, session_id=session_id, max_results=3
            )

            if similar_interactions:
                context_parts = ["Relevant previous interactions:"]
                for i, interaction in enumerate(similar_interactions, 1):
                    context_parts.append(f"{i}. User: {interaction.user_input}")
                    context_parts.append(
                        f"   Response: {interaction.assistant_response[:200]}..."
                    )

                additional_context = "\n".join(context_parts)
            else:
                additional_context = "No relevant previous interactions found."
        else:
            # Get general context summary
            additional_context = semantic_memory.get_context_summary(session_id)

        # Add to conversation context
        if state["conversation"].context_summary:
            state["conversation"].context_summary += f"\n\n{additional_context}"
        else:
            state["conversation"].context_summary = additional_context

        # Add memory metadata
        if "memory" not in state["metadata"]:
            state["metadata"]["memory"] = {}

        state["metadata"]["memory"].update(
            {
                "semantic_context_added": True,
                "context_timestamp": datetime.now().isoformat(),
                "query_used": query is not None,
            }
        )

    except Exception as e:
        logger.error(f"Error adding memory context to workflow: {e}")

    return state
