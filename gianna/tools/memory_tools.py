"""
Memory management tools that integrate with Gianna's state system.

This module provides tools for managing conversation history, user preferences,
and persistent state through Gianna's core state management system.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
from loguru import logger
from pydantic import Field

from gianna.core.state import (
    ConversationState,
    GiannaState,
    create_initial_state,
    state_to_dict,
)
from gianna.core.state_manager import StateManager

# Import semantic memory components (with fallback for optional dependency)
try:
    from gianna.memory import MemoryConfig, SemanticMemory
    from gianna.memory.state_integration import MemoryIntegratedStateManager

    SEMANTIC_MEMORY_AVAILABLE = True
except ImportError:
    SEMANTIC_MEMORY_AVAILABLE = False
    SemanticMemory = None
    MemoryConfig = None
    MemoryIntegratedStateManager = None


class MemoryTool(BaseTool):
    """
    Memory management tool for conversation history and persistent state.

    Integrates with Gianna's state management system to provide persistent
    memory across sessions and conversations.
    """

    name: str = "memory_manager"
    description: str = """Manage conversation history and persistent memory.
    Input: JSON with 'action' (store|retrieve|search|clear), 'data', optional 'session_id', 'query'
    Output: JSON with memory operation results and retrieved data"""

    state_manager: Optional[StateManager] = Field(
        default=None, description="State manager instance"
    )
    db_path: str = Field(
        default="gianna_state.db", description="Database path for persistent storage"
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Initialize state manager if not provided
        if self.state_manager is None:
            self.state_manager = StateManager(db_path=self.db_path)

    def _run(self, input_data: str) -> str:
        """
        Manage memory operations with Gianna's state system.

        Args:
            input_data: JSON string with memory action and parameters

        Returns:
            JSON string with memory operation results
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data

            action = data.get("action", "").lower()
            session_id = data.get("session_id", "default")

            if action == "store":
                return self._store_memory(data, session_id)
            elif action == "retrieve":
                return self._retrieve_memory(session_id, data.get("key"))
            elif action == "search":
                return self._search_memory(data.get("query", ""), session_id)
            elif action == "clear":
                return self._clear_memory(session_id, data.get("confirm", False))
            elif action == "list_sessions":
                return self._list_sessions()
            elif action == "get_stats":
                return self._get_memory_stats(session_id)
            else:
                return json.dumps(
                    {
                        "error": f"Unknown action: {action}. Use 'store', 'retrieve', 'search', 'clear', 'list_sessions', or 'get_stats'",
                        "success": False,
                    }
                )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            return json.dumps(
                {"error": f"Invalid JSON input: {str(e)}", "success": False}
            )
        except Exception as e:
            logger.error(f"Memory operation error: {e}")
            return json.dumps(
                {"error": f"Memory operation failed: {str(e)}", "success": False}
            )

    def _store_memory(self, data: dict, session_id: str) -> str:
        """Store data in memory/state system."""
        try:
            memory_data = data.get("data", {})
            memory_type = data.get("type", "general")

            # Get or create state for session
            state = self.state_manager.load_state(session_id)
            if state is None:
                state = create_initial_state(session_id)

            # Store based on memory type
            if memory_type == "conversation":
                # Add to conversation messages
                message = memory_data.get("message", {})
                if message:
                    state["conversation"].messages.append(
                        {
                            "role": message.get("role", "user"),
                            "content": message.get("content", ""),
                            "timestamp": datetime.now().isoformat(),
                            "session_id": session_id,
                        }
                    )

            elif memory_type == "preference":
                # Store user preferences
                preferences = memory_data.get("preferences", {})
                state["conversation"].user_preferences.update(preferences)

            elif memory_type == "context":
                # Update context summary
                context = memory_data.get("context", "")
                if context:
                    state["conversation"].context_summary = context

            else:
                # Store in metadata
                if "custom_data" not in state["metadata"]:
                    state["metadata"]["custom_data"] = {}
                state["metadata"]["custom_data"][memory_type] = memory_data

            # Update timestamp
            state["metadata"]["last_updated"] = datetime.now().isoformat()

            # Save state
            self.state_manager.save_state(session_id, state)

            return json.dumps(
                {
                    "success": True,
                    "action": "store",
                    "session_id": session_id,
                    "memory_type": memory_type,
                    "message": f"Memory stored successfully for session {session_id}",
                    "data_keys": (
                        list(memory_data.keys())
                        if isinstance(memory_data, dict)
                        else ["data"]
                    ),
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to store memory: {str(e)}",
                    "success": False,
                    "session_id": session_id,
                }
            )

    def _retrieve_memory(self, session_id: str, key: Optional[str] = None) -> str:
        """Retrieve memory from state system."""
        try:
            state = self.state_manager.load_state(session_id)

            if state is None:
                return json.dumps(
                    {
                        "success": True,
                        "action": "retrieve",
                        "session_id": session_id,
                        "message": f"No memory found for session {session_id}",
                        "data": None,
                    }
                )

            # Convert state to dict for JSON serialization
            state_dict = state_to_dict(state)

            if key:
                # Retrieve specific key
                if key in state_dict:
                    data = state_dict[key]
                elif key in state_dict.get("metadata", {}).get("custom_data", {}):
                    data = state_dict["metadata"]["custom_data"][key]
                else:
                    data = None

                return json.dumps(
                    {
                        "success": True,
                        "action": "retrieve",
                        "session_id": session_id,
                        "key": key,
                        "data": data,
                        "message": (
                            f"Retrieved {key} for session {session_id}"
                            if data
                            else f"Key {key} not found"
                        ),
                    }
                )
            else:
                # Return entire state
                return json.dumps(
                    {
                        "success": True,
                        "action": "retrieve",
                        "session_id": session_id,
                        "data": state_dict,
                        "message": f"Retrieved full memory for session {session_id}",
                    }
                )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to retrieve memory: {str(e)}",
                    "success": False,
                    "session_id": session_id,
                }
            )

    def _search_memory(self, query: str, session_id: str) -> str:
        """Search through memory/conversation history."""
        try:
            if not query.strip():
                return json.dumps(
                    {"error": "Search query cannot be empty", "success": False}
                )

            state = self.state_manager.load_state(session_id)
            if state is None:
                return json.dumps(
                    {
                        "success": True,
                        "action": "search",
                        "session_id": session_id,
                        "query": query,
                        "results": [],
                        "message": f"No memory found for session {session_id}",
                    }
                )

            results = []
            query_lower = query.lower()

            # Search conversation messages
            for i, msg in enumerate(state["conversation"].messages):
                if query_lower in msg.get("content", "").lower():
                    results.append(
                        {
                            "type": "conversation",
                            "index": i,
                            "role": msg.get("role", ""),
                            "content": msg.get("content", ""),
                            "timestamp": msg.get("timestamp", ""),
                            "match_type": "content",
                        }
                    )

            # Search preferences
            for key, value in state["conversation"].user_preferences.items():
                if query_lower in str(value).lower() or query_lower in key.lower():
                    results.append(
                        {
                            "type": "preference",
                            "key": key,
                            "value": value,
                            "match_type": "preference",
                        }
                    )

            # Search context summary
            if query_lower in state["conversation"].context_summary.lower():
                results.append(
                    {
                        "type": "context",
                        "content": state["conversation"].context_summary,
                        "match_type": "context_summary",
                    }
                )

            return json.dumps(
                {
                    "success": True,
                    "action": "search",
                    "session_id": session_id,
                    "query": query,
                    "results": results,
                    "total_matches": len(results),
                    "message": f"Found {len(results)} matches for '{query}'",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Search failed: {str(e)}",
                    "success": False,
                    "session_id": session_id,
                    "query": query,
                }
            )

    def _clear_memory(self, session_id: str, confirm: bool = False) -> str:
        """Clear memory for a session."""
        try:
            if not confirm:
                return json.dumps(
                    {
                        "error": "Memory clear requires confirmation. Set 'confirm': true",
                        "success": False,
                        "session_id": session_id,
                        "warning": "This action will permanently delete all memory for this session",
                    }
                )

            # Delete state for session
            deleted = self.state_manager.delete_session(session_id)

            return json.dumps(
                {
                    "success": deleted,
                    "action": "clear",
                    "session_id": session_id,
                    "message": (
                        f"Memory cleared for session {session_id}"
                        if deleted
                        else f"No memory found for session {session_id}"
                    ),
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to clear memory: {str(e)}",
                    "success": False,
                    "session_id": session_id,
                }
            )

    def _list_sessions(self) -> str:
        """List all available sessions."""
        try:
            sessions_list = self.state_manager.get_session_list()
            sessions = [s["session_id"] for s in sessions_list]

            session_info = []
            for session_id in sessions:
                state = self.state_manager.load_state(session_id)
                if state:
                    session_info.append(
                        {
                            "session_id": session_id,
                            "message_count": len(state["conversation"].messages),
                            "last_updated": state["metadata"].get("last_updated", ""),
                            "created_at": state["metadata"].get("created_at", ""),
                            "has_preferences": len(
                                state["conversation"].user_preferences
                            )
                            > 0,
                        }
                    )

            return json.dumps(
                {
                    "success": True,
                    "action": "list_sessions",
                    "sessions": session_info,
                    "total_sessions": len(session_info),
                    "message": f"Found {len(session_info)} sessions",
                }
            )

        except Exception as e:
            return json.dumps(
                {"error": f"Failed to list sessions: {str(e)}", "success": False}
            )

    def _get_memory_stats(self, session_id: str) -> str:
        """Get memory statistics for a session."""
        try:
            state = self.state_manager.load_state(session_id)
            if state is None:
                return json.dumps(
                    {
                        "success": True,
                        "action": "get_stats",
                        "session_id": session_id,
                        "message": f"No memory found for session {session_id}",
                        "stats": {},
                    }
                )

            stats = {
                "session_id": session_id,
                "message_count": len(state["conversation"].messages),
                "preference_count": len(state["conversation"].user_preferences),
                "context_length": len(state["conversation"].context_summary),
                "created_at": state["metadata"].get("created_at", ""),
                "last_updated": state["metadata"].get("last_updated", ""),
                "has_context_summary": bool(
                    state["conversation"].context_summary.strip()
                ),
                "audio_mode": state["audio"].current_mode,
                "command_history_count": len(state["commands"].execution_history),
            }

            return json.dumps(
                {
                    "success": True,
                    "action": "get_stats",
                    "session_id": session_id,
                    "stats": stats,
                    "message": f"Memory statistics for session {session_id}",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to get memory stats: {str(e)}",
                    "success": False,
                    "session_id": session_id,
                }
            )

    async def _arun(self, input_data: str) -> str:
        """Async version - delegates to sync version for now."""
        return self._run(input_data)


class SemanticMemoryTool(BaseTool):
    """
    Semantic memory tool with embedding-based similarity search.

    Provides advanced memory capabilities using vector embeddings for
    semantic similarity search and pattern detection across conversations.
    """

    name: str = "semantic_memory"
    description: str = """Advanced semantic memory with similarity search and pattern detection.
    Input: JSON with 'action' (store|search|patterns|context|stats|cleanup), 'data', optional 'session_id', 'query', 'threshold'
    Output: JSON with semantic memory operation results and intelligent insights"""

    semantic_memory: Optional[SemanticMemory] = Field(
        default=None, description="Semantic memory instance"
    )
    memory_config: Optional[MemoryConfig] = Field(
        default=None, description="Memory configuration"
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Check if semantic memory is available
        if not SEMANTIC_MEMORY_AVAILABLE:
            logger.warning(
                "Semantic memory components not available. Install required dependencies."
            )
            return

        # Initialize semantic memory if not provided
        if self.semantic_memory is None:
            config = self.memory_config or MemoryConfig()
            try:
                self.semantic_memory = SemanticMemory(config)
                logger.info("Initialized semantic memory for tool")
            except Exception as e:
                logger.error(f"Failed to initialize semantic memory: {e}")
                self.semantic_memory = None

    def _run(self, input_data: str) -> str:
        """
        Execute semantic memory operations.

        Args:
            input_data: JSON string with action and parameters

        Returns:
            JSON string with semantic memory results
        """
        if not SEMANTIC_MEMORY_AVAILABLE:
            return json.dumps(
                {
                    "error": "Semantic memory not available. Install required dependencies: sentence-transformers, numpy",
                    "success": False,
                    "available_actions": [],
                }
            )

        if not self.semantic_memory:
            return json.dumps(
                {
                    "error": "Semantic memory not initialized",
                    "success": False,
                    "suggestion": "Check embedding providers and vector store availability",
                }
            )

        try:
            # Parse input
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data

            action = data.get("action", "").lower()
            session_id = data.get("session_id", "default")

            if action == "store":
                return self._store_interaction(data, session_id)
            elif action == "search":
                return self._search_similar(data, session_id)
            elif action == "patterns":
                return self._detect_patterns(session_id)
            elif action == "context":
                return self._get_context_summary(data, session_id)
            elif action == "stats":
                return self._get_memory_stats()
            elif action == "cleanup":
                return self._cleanup_memory(data)
            else:
                return json.dumps(
                    {
                        "error": f"Unknown action: {action}. Use 'store', 'search', 'patterns', 'context', 'stats', or 'cleanup'",
                        "success": False,
                        "available_actions": [
                            "store",
                            "search",
                            "patterns",
                            "context",
                            "stats",
                            "cleanup",
                        ],
                    }
                )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            return json.dumps(
                {"error": f"Invalid JSON input: {str(e)}", "success": False}
            )
        except Exception as e:
            logger.error(f"Semantic memory operation error: {e}")
            return json.dumps(
                {
                    "error": f"Semantic memory operation failed: {str(e)}",
                    "success": False,
                }
            )

    def _store_interaction(self, data: dict, session_id: str) -> str:
        """Store interaction in semantic memory."""
        try:
            user_input = data.get("user_input", "")
            assistant_response = data.get("assistant_response", "")
            context = data.get("context", "")
            interaction_type = data.get("interaction_type", "conversation")
            metadata = data.get("metadata", {})

            if not user_input or not assistant_response:
                return json.dumps(
                    {
                        "error": "Both user_input and assistant_response are required",
                        "success": False,
                    }
                )

            interaction_id = self.semantic_memory.store_interaction(
                session_id=session_id,
                user_input=user_input,
                assistant_response=assistant_response,
                context=context,
                interaction_type=interaction_type,
                metadata=metadata,
            )

            if interaction_id:
                return json.dumps(
                    {
                        "success": True,
                        "action": "store",
                        "interaction_id": interaction_id,
                        "session_id": session_id,
                        "message": "Interaction stored in semantic memory successfully",
                        "embedding_generated": True,
                    }
                )
            else:
                return json.dumps(
                    {
                        "success": False,
                        "action": "store",
                        "message": "Failed to store interaction",
                        "session_id": session_id,
                    }
                )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to store interaction: {str(e)}",
                    "success": False,
                    "session_id": session_id,
                }
            )

    def _search_similar(self, data: dict, session_id: str) -> str:
        """Search for semantically similar interactions."""
        try:
            query = data.get("query", "")
            max_results = data.get("max_results", 5)
            similarity_threshold = data.get("similarity_threshold", 0.7)
            interaction_type = data.get("interaction_type")
            global_search = data.get("global_search", False)

            if not query.strip():
                return json.dumps(
                    {"error": "Search query cannot be empty", "success": False}
                )

            # Search with optional session filter
            search_session_id = None if global_search else session_id

            similar_interactions = self.semantic_memory.search_similar_interactions(
                query=query,
                session_id=search_session_id,
                interaction_type=interaction_type,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
            )

            # Convert interactions to serializable format
            results = []
            for interaction in similar_interactions:
                results.append(
                    {
                        "id": interaction.id,
                        "session_id": interaction.session_id,
                        "timestamp": interaction.timestamp.isoformat(),
                        "user_input": interaction.user_input,
                        "assistant_response": interaction.assistant_response,
                        "context": interaction.context,
                        "interaction_type": interaction.interaction_type,
                        "metadata": interaction.metadata,
                        "cluster_id": interaction.cluster_id,
                    }
                )

            return json.dumps(
                {
                    "success": True,
                    "action": "search",
                    "query": query,
                    "session_id": session_id,
                    "global_search": global_search,
                    "results": results,
                    "total_results": len(results),
                    "similarity_threshold": similarity_threshold,
                    "message": f"Found {len(results)} semantically similar interactions",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Search failed: {str(e)}",
                    "success": False,
                    "query": data.get("query", ""),
                    "session_id": session_id,
                }
            )

    def _detect_patterns(self, session_id: str) -> str:
        """Detect patterns in user interactions."""
        try:
            patterns = self.semantic_memory.detect_patterns(session_id)

            return json.dumps(
                {
                    "success": True,
                    "action": "patterns",
                    "session_id": session_id,
                    "patterns": patterns,
                    "message": f"Pattern analysis completed for session {session_id}",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Pattern detection failed: {str(e)}",
                    "success": False,
                    "session_id": session_id,
                }
            )

    def _get_context_summary(self, data: dict, session_id: str) -> str:
        """Get semantic context summary."""
        try:
            max_interactions = data.get("max_interactions", 10)

            context_summary = self.semantic_memory.get_context_summary(
                session_id=session_id, max_interactions=max_interactions
            )

            return json.dumps(
                {
                    "success": True,
                    "action": "context",
                    "session_id": session_id,
                    "context_summary": context_summary,
                    "max_interactions": max_interactions,
                    "message": f"Context summary generated for session {session_id}",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Context summary failed: {str(e)}",
                    "success": False,
                    "session_id": session_id,
                }
            )

    def _get_memory_stats(self) -> str:
        """Get semantic memory statistics."""
        try:
            stats = self.semantic_memory.get_memory_stats()

            return json.dumps(
                {
                    "success": True,
                    "action": "stats",
                    "stats": stats,
                    "message": "Semantic memory statistics retrieved successfully",
                }
            )

        except Exception as e:
            return json.dumps(
                {"error": f"Failed to get memory stats: {str(e)}", "success": False}
            )

    def _cleanup_memory(self, data: dict) -> str:
        """Clean up old interactions from semantic memory."""
        try:
            max_age_days = data.get("max_age_days", 30)
            confirm = data.get("confirm", False)

            if not confirm:
                return json.dumps(
                    {
                        "error": "Memory cleanup requires confirmation. Set 'confirm': true",
                        "success": False,
                        "warning": f"This action will permanently delete interactions older than {max_age_days} days",
                    }
                )

            cleaned_count = self.semantic_memory.cleanup_old_interactions(max_age_days)

            return json.dumps(
                {
                    "success": True,
                    "action": "cleanup",
                    "max_age_days": max_age_days,
                    "cleaned_interactions": cleaned_count,
                    "message": f"Cleaned up {cleaned_count} old interactions",
                }
            )

        except Exception as e:
            return json.dumps({"error": f"Cleanup failed: {str(e)}", "success": False})

    async def _arun(self, input_data: str) -> str:
        """Async version - delegates to sync version for now."""
        return self._run(input_data)


# Factory functions for creating memory tools


def create_memory_tool(db_path: str = "gianna_state.db") -> MemoryTool:
    """
    Create a standard memory tool.

    Args:
        db_path: Database path for state storage

    Returns:
        MemoryTool: Configured memory tool
    """
    return MemoryTool(db_path=db_path)


def create_semantic_memory_tool(
    memory_config: Optional[MemoryConfig] = None,
) -> SemanticMemoryTool:
    """
    Create a semantic memory tool.

    Args:
        memory_config: Optional configuration for semantic memory

    Returns:
        SemanticMemoryTool: Configured semantic memory tool
    """
    if not SEMANTIC_MEMORY_AVAILABLE:
        logger.warning("Semantic memory not available, returning None")
        return None

    return SemanticMemoryTool(memory_config=memory_config)


def get_available_memory_tools() -> List[str]:
    """
    Get list of available memory tools based on dependencies.

    Returns:
        List[str]: Available memory tool names
    """
    tools = ["memory_manager"]  # Always available

    if SEMANTIC_MEMORY_AVAILABLE:
        tools.append("semantic_memory")

    return tools
