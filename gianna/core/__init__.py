"""
Gianna Core State Management Module

This module provides the foundation for state management in the Gianna AI Assistant.
It includes state schemas, persistence management, and integration with LangGraph workflows.

Key Components:
- State Schemas: Pydantic models for type-safe state management
- StateManager: SQLite-based persistence with LangGraph integration
- Utility Functions: State validation, conversion, and initialization helpers

Example Usage:
    from gianna.core import StateManager, GiannaState, create_initial_state

    # Initialize state manager
    state_manager = StateManager("my_assistant.db")

    # Create a new session
    session_id = state_manager.create_session()

    # Create initial state
    state = create_initial_state(session_id)

    # Save state
    state_manager.save_state(session_id, state)

    # Load state
    loaded_state = state_manager.load_state(session_id)
"""

from .state import (
    AudioState,
    CommandState,
    ConversationState,
    GiannaState,
    create_initial_state,
    state_to_dict,
    validate_state,
)
from .state_manager import StateManager

# LangGraph integration (optional)
try:
    from .langgraph_chain import (
        LANGGRAPH_AVAILABLE,
        LangGraphChain,
        create_langgraph_chain,
    )
    from .langgraph_factory import (
        create_compatible_chain,
        get_enhanced_chain_instance,
        get_langgraph_capabilities,
        register_langgraph_chains,
    )
    from .migration_utils import (
        BackwardCompatibilityWrapper,
        ChainMigrationUtility,
        detect_chain_type,
        ensure_backward_compatibility,
        get_migration_recommendations,
    )

    LANGGRAPH_INTEGRATION = True
except ImportError:
    LANGGRAPH_INTEGRATION = False
    LANGGRAPH_AVAILABLE = False

__all__ = [
    # State schemas
    "GiannaState",
    "ConversationState",
    "AudioState",
    "CommandState",
    # State utilities
    "create_initial_state",
    "validate_state",
    "state_to_dict",
    # State management
    "StateManager",
    # LangGraph availability flag
    "LANGGRAPH_AVAILABLE",
    "LANGGRAPH_INTEGRATION",
]

# Add LangGraph components to __all__ if available
if LANGGRAPH_INTEGRATION:
    __all__.extend(
        [
            # LangGraph chains
            "LangGraphChain",
            "create_langgraph_chain",
            # Enhanced factory functions
            "register_langgraph_chains",
            "get_enhanced_chain_instance",
            "create_compatible_chain",
            "get_langgraph_capabilities",
            # Migration utilities
            "ChainMigrationUtility",
            "BackwardCompatibilityWrapper",
            "ensure_backward_compatibility",
            "detect_chain_type",
            "get_migration_recommendations",
        ]
    )

__version__ = "1.0.0"
__author__ = "Gianna Development Team"
__description__ = "Core state management system for Gianna AI Assistant"
