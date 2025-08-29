"""
Migration Utilities for Gianna LangGraph Integration

This module provides utilities for migrating between traditional chains
and LangGraph chains, ensuring backward compatibility and seamless transitions.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from ..assistants.models.basics import AbstractBasicChain
from .state import (
    AudioState,
    CommandState,
    ConversationState,
    GiannaState,
    create_initial_state,
)


class ChainMigrationUtility:
    """
    Utility class for migrating between chain types and handling backward compatibility.
    """

    @staticmethod
    def convert_legacy_input(input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert legacy input formats to standardized format.

        Args:
            input_data: Input in various legacy formats

        Returns:
            Standardized input dictionary
        """
        if isinstance(input_data, str):
            return {"input": input_data, "content": input_data}
        elif isinstance(input_data, dict):
            # Ensure both 'input' and 'content' fields exist
            standardized = input_data.copy()
            if "input" not in standardized and "content" in standardized:
                standardized["input"] = standardized["content"]
            elif "content" not in standardized and "input" in standardized:
                standardized["content"] = standardized["input"]
            elif "input" not in standardized and "content" not in standardized:
                # Use the entire dict as content if no recognized fields
                standardized["input"] = str(input_data)
                standardized["content"] = str(input_data)
            return standardized
        else:
            # Convert other types to string
            content = str(input_data)
            return {"input": content, "content": content}

    @staticmethod
    def convert_legacy_output(
        output_data: Union[str, Dict[str, Any], AbstractBasicChain],
    ) -> Dict[str, Any]:
        """
        Convert legacy output formats to standardized format.

        Args:
            output_data: Output in various legacy formats

        Returns:
            Standardized output dictionary
        """
        if isinstance(output_data, AbstractBasicChain):
            # Extract output from chain instance
            return {
                "output": output_data.output,
                "text": output_data.output,
                "chain_type": type(output_data).__name__,
            }
        elif isinstance(output_data, str):
            return {"output": output_data, "text": output_data}
        elif isinstance(output_data, dict):
            # Ensure standard fields exist
            standardized = output_data.copy()
            if "output" not in standardized and "text" in standardized:
                standardized["output"] = standardized["text"]
            elif "text" not in standardized and "output" in standardized:
                standardized["text"] = standardized["output"]
            return standardized
        else:
            content = str(output_data)
            return {"output": content, "text": content}

    @staticmethod
    def create_conversation_from_history(
        messages: List[Dict[str, Any]],
        session_id: str = "",
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> ConversationState:
        """
        Create a ConversationState from message history.

        Args:
            messages: List of conversation messages
            session_id: Session identifier
            user_preferences: Optional user preferences

        Returns:
            ConversationState instance
        """
        # Ensure all messages have required fields
        standardized_messages = []
        for msg in messages:
            standardized_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
            }
            # Add any additional metadata
            for key, value in msg.items():
                if key not in ["role", "content", "timestamp"]:
                    if "metadata" not in standardized_msg:
                        standardized_msg["metadata"] = {}
                    standardized_msg["metadata"][key] = value
            standardized_messages.append(standardized_msg)

        return ConversationState(
            messages=standardized_messages,
            session_id=session_id,
            user_preferences=user_preferences or {},
            context_summary=ChainMigrationUtility._generate_context_summary(
                standardized_messages
            ),
        )

    @staticmethod
    def _generate_context_summary(messages: List[Dict[str, Any]]) -> str:
        """
        Generate a context summary from conversation messages.

        Args:
            messages: List of conversation messages

        Returns:
            Context summary string
        """
        if not messages:
            return ""

        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]

        summary_parts = []

        if user_messages:
            recent_user = (
                user_messages[-1]["content"][:100] + "..."
                if len(user_messages[-1]["content"]) > 100
                else user_messages[-1]["content"]
            )
            summary_parts.append(f"Last user input: {recent_user}")

        if assistant_messages:
            summary_parts.append(f"Assistant responses: {len(assistant_messages)}")

        summary_parts.append(f"Total messages: {len(messages)}")

        return " | ".join(summary_parts)

    @staticmethod
    def migrate_chain_to_langgraph(
        traditional_chain: AbstractBasicChain,
        model_name: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        session_id: str = "",
    ) -> "LangGraphChain":
        """
        Migrate a traditional chain to LangGraph format.

        Args:
            traditional_chain: Existing traditional chain
            model_name: Name of the model to use
            conversation_history: Optional conversation history
            session_id: Session identifier

        Returns:
            LangGraphChain instance
        """
        try:
            from .langgraph_chain import LangGraphChain

            # Extract prompt template from traditional chain
            prompt_template = ""
            if (
                hasattr(traditional_chain, "_prompt_template")
                and traditional_chain._prompt_template
            ):
                prompt_template = traditional_chain._prompt_template.template

            # Try to extract LLM instance
            llm_instance = None
            if hasattr(traditional_chain, "_chain") and traditional_chain._chain:
                chain = traditional_chain._get_chain()
                if hasattr(chain, "llm"):
                    llm_instance = chain.llm
                elif hasattr(chain, "model"):
                    llm_instance = chain.model

            # Create LangGraph chain
            langgraph_chain = LangGraphChain(
                model_name=model_name,
                prompt=prompt_template,
                llm_instance=llm_instance,
                enable_state_management=True,
            )

            # If conversation history provided, populate it
            if conversation_history and session_id:
                # This would require saving to state manager
                # For now, we'll set it in the next invocation
                pass

            logger.info(
                f"Successfully migrated traditional chain to LangGraph for model: {model_name}"
            )
            return langgraph_chain

        except ImportError:
            logger.error("LangGraph not available for migration")
            raise
        except Exception as e:
            logger.error(f"Error migrating chain to LangGraph: {e}")
            raise

    @staticmethod
    def create_state_from_chain_output(
        chain_output: Union[str, Dict[str, Any], AbstractBasicChain],
        user_input: Union[str, Dict[str, Any]],
        session_id: str = "",
        model_name: str = "",
    ) -> GiannaState:
        """
        Create a GiannaState from traditional chain input/output.

        Args:
            chain_output: Output from traditional chain
            user_input: Original user input
            session_id: Session identifier
            model_name: Name of the model used

        Returns:
            GiannaState instance
        """
        # Convert input and output to standard formats
        standardized_input = ChainMigrationUtility.convert_legacy_input(user_input)
        standardized_output = ChainMigrationUtility.convert_legacy_output(chain_output)

        # Create conversation messages
        messages = [
            {
                "role": "user",
                "content": standardized_input["content"],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    k: v for k, v in standardized_input.items() if k != "content"
                },
            },
            {
                "role": "assistant",
                "content": standardized_output["output"],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "model": model_name,
                    **{
                        k: v
                        for k, v in standardized_output.items()
                        if k not in ["output", "text"]
                    },
                },
            },
        ]

        # Create conversation state
        conversation = ConversationState(
            messages=messages,
            session_id=session_id or "migrated_session",
            user_preferences={},
            context_summary=ChainMigrationUtility._generate_context_summary(messages),
        )

        # Create complete state
        state = GiannaState(
            conversation=conversation,
            audio=AudioState(),
            commands=CommandState(),
            metadata={
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "system": "gianna",
                "migration": {
                    "from": "traditional_chain",
                    "model": model_name,
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )

        return state

    @staticmethod
    def extract_conversation_history(state: GiannaState) -> List[Dict[str, Any]]:
        """
        Extract conversation history in legacy format from GiannaState.

        Args:
            state: GiannaState instance

        Returns:
            List of conversation messages in legacy format
        """
        messages = state["conversation"].messages
        legacy_messages = []

        for msg in messages:
            legacy_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }

            # Add metadata as top-level fields for backward compatibility
            if "metadata" in msg:
                legacy_msg.update(msg["metadata"])

            # Add timestamp if present
            if "timestamp" in msg:
                legacy_msg["timestamp"] = msg["timestamp"]

            legacy_messages.append(legacy_msg)

        return legacy_messages


class BackwardCompatibilityWrapper:
    """
    Wrapper to ensure backward compatibility for existing code using traditional chains.
    """

    def __init__(self, chain: Union[AbstractBasicChain, "LangGraphChain"]):
        """
        Initialize wrapper with either traditional or LangGraph chain.

        Args:
            chain: Chain instance to wrap
        """
        self._chain = chain
        self._is_langgraph = hasattr(chain, "has_state_management")

    def invoke(
        self, input_data: Union[str, Dict[str, Any]], **kwargs
    ) -> Union[Dict[str, Any], AbstractBasicChain]:
        """
        Invoke the wrapped chain with backward compatibility.

        Args:
            input_data: Input data
            **kwargs: Additional arguments

        Returns:
            Compatible output format
        """
        if self._is_langgraph:
            # LangGraph chain returns dict directly
            return self._chain.invoke(input_data, **kwargs)
        else:
            # Traditional chain returns self, need to extract output
            result = self._chain.invoke(input_data, **kwargs)
            return ChainMigrationUtility.convert_legacy_output(result)

    def process(self, input_data: Union[str, Dict[str, Any]], **kwargs):
        """
        Process method for backward compatibility.

        Args:
            input_data: Input data
            **kwargs: Additional arguments

        Returns:
            Self for chaining
        """
        self.invoke(input_data, **kwargs)
        return self

    @property
    def output(self) -> str:
        """Get the output from the wrapped chain."""
        return getattr(self._chain, "output", "")

    @property
    def is_langgraph(self) -> bool:
        """Check if this is wrapping a LangGraph chain."""
        return self._is_langgraph

    def get_conversation_history(self, session_id: str = None) -> List[Dict[str, Any]]:
        """
        Get conversation history if available.

        Args:
            session_id: Optional session ID

        Returns:
            List of conversation messages or empty list
        """
        if self._is_langgraph and hasattr(self._chain, "get_conversation_history"):
            return self._chain.get_conversation_history(session_id)
        return []


def ensure_backward_compatibility(
    chain: Union[AbstractBasicChain, "LangGraphChain"],
) -> BackwardCompatibilityWrapper:
    """
    Ensure a chain has backward compatibility features.

    Args:
        chain: Chain to wrap for compatibility

    Returns:
        BackwardCompatibilityWrapper instance
    """
    return BackwardCompatibilityWrapper(chain)


def detect_chain_type(chain: Any) -> str:
    """
    Detect the type of chain being used.

    Args:
        chain: Chain instance to analyze

    Returns:
        String indicating chain type
    """
    if hasattr(chain, "has_state_management"):
        return "langgraph"
    elif isinstance(chain, AbstractBasicChain):
        return "traditional"
    else:
        return "unknown"


def get_migration_recommendations(current_usage: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get recommendations for migrating to LangGraph based on current usage patterns.

    Args:
        current_usage: Dictionary describing current usage patterns

    Returns:
        Dictionary with migration recommendations
    """
    recommendations = {
        "should_migrate": False,
        "reasons": [],
        "benefits": [],
        "considerations": [],
        "migration_complexity": "low",
    }

    # Analyze usage patterns
    if current_usage.get("maintains_conversation_history", False):
        recommendations["should_migrate"] = True
        recommendations["reasons"].append("Conversation history management")
        recommendations["benefits"].append("Persistent conversation history")

    if current_usage.get("multiple_sessions", False):
        recommendations["should_migrate"] = True
        recommendations["reasons"].append("Multi-session support needed")
        recommendations["benefits"].append("Session isolation and persistence")

    if current_usage.get("complex_workflows", False):
        recommendations["should_migrate"] = True
        recommendations["reasons"].append("Complex workflow requirements")
        recommendations["benefits"].append("Structured workflow management")
        recommendations["migration_complexity"] = "medium"

    if current_usage.get("audio_integration", False):
        recommendations["benefits"].append("Enhanced audio state management")

    if current_usage.get("command_execution", False):
        recommendations["benefits"].append("Command execution tracking")

    # Add general benefits
    recommendations["benefits"].extend(
        [
            "Enhanced error handling",
            "Better debugging capabilities",
            "Future-proof architecture",
        ]
    )

    # Add considerations
    recommendations["considerations"] = [
        "Requires LangGraph dependency",
        "Slightly more complex initialization",
        "New concepts to learn (StateGraph, checkpointers)",
    ]

    return recommendations
