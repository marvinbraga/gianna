"""
Core State Management for Gianna AI Assistant

This module defines the state schemas used throughout the Gianna system,
including conversation state, audio state, command state, and the main GiannaState.
All states use Pydantic for validation and type safety.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ConversationState(BaseModel):
    """
    Manages conversation history and user preferences.

    Attributes:
        messages: List of conversation messages with role, content, and metadata
        session_id: Unique identifier for the conversation session
        user_preferences: Dictionary of user-specific preferences and settings
        context_summary: Summary of the current conversation context
    """

    messages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of conversation messages with role and content",
    )
    session_id: str = Field(default="", description="Unique session identifier")
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict, description="User-specific preferences and settings"
    )
    context_summary: str = Field(
        default="", description="Summary of the current conversation context"
    )


class AudioState(BaseModel):
    """
    Manages audio system state and configuration.

    Attributes:
        current_mode: Current audio processing mode
        voice_settings: Audio configuration and voice parameters
        speech_type: Type of speech synthesis engine to use
        language: Language code for speech processing
    """

    current_mode: str = Field(
        default="idle",
        description="Current audio mode: idle, listening, speaking, processing",
    )
    voice_settings: Dict[str, Any] = Field(
        default_factory=dict, description="Voice configuration and audio parameters"
    )
    speech_type: str = Field(
        default="google", description="Speech synthesis engine type"
    )
    language: str = Field(
        default="pt-br", description="Language code for speech processing"
    )


class CommandState(BaseModel):
    """
    Manages command execution history and pending operations.

    Attributes:
        execution_history: History of executed commands with results and metadata
        pending_operations: List of operations waiting to be executed
    """

    execution_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of executed commands with results"
    )
    pending_operations: List[str] = Field(
        default_factory=list, description="List of pending operations to execute"
    )


class GiannaState(TypedDict):
    """
    Main state container for the Gianna AI Assistant.

    This TypedDict defines the complete state structure that flows through
    the LangGraph workflows. It combines all subsystem states and metadata.

    Fields:
        conversation: Conversation and user interaction state
        audio: Audio processing and voice system state
        commands: Command execution and operation state
        metadata: Additional system metadata and configuration
    """

    conversation: ConversationState
    audio: AudioState
    commands: CommandState
    metadata: Dict[str, Any]


def create_initial_state(session_id: str = "") -> GiannaState:
    """
    Create an initial GiannaState with default values.

    Args:
        session_id: Optional session identifier

    Returns:
        GiannaState: Initialized state with default values
    """
    return GiannaState(
        conversation=ConversationState(session_id=session_id),
        audio=AudioState(),
        commands=CommandState(),
        metadata={
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "system": "gianna",
        },
    )


def validate_state(state: Dict[str, Any]) -> GiannaState:
    """
    Validate and convert a dictionary to a proper GiannaState.

    Args:
        state: Dictionary containing state data

    Returns:
        GiannaState: Validated and properly typed state

    Raises:
        ValidationError: If state data is invalid
    """
    # Validate each component
    conversation = ConversationState.model_validate(state.get("conversation", {}))
    audio = AudioState.model_validate(state.get("audio", {}))
    commands = CommandState.model_validate(state.get("commands", {}))

    return GiannaState(
        conversation=conversation,
        audio=audio,
        commands=commands,
        metadata=state.get("metadata", {}),
    )


def state_to_dict(state: GiannaState) -> Dict[str, Any]:
    """
    Convert GiannaState to a serializable dictionary.

    Args:
        state: GiannaState to convert

    Returns:
        Dict[str, Any]: Serializable dictionary representation
    """
    return {
        "conversation": state["conversation"].model_dump(),
        "audio": state["audio"].model_dump(),
        "commands": state["commands"].model_dump(),
        "metadata": state["metadata"],
    }
