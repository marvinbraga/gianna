# Gianna Core State Management System

This module implements the Core State Management system for FASE 1 of the Gianna AI Assistant project, as specified in `PHASE_SPECIFICATIONS.md`.

## Overview

The core module provides a robust state management foundation with:

- **Type-safe state schemas** using Pydantic
- **Persistent storage** with SQLite integration
- **LangGraph compatibility** for workflow orchestration
- **Session management** with user preferences
- **Error handling and validation**

## Components

### 1. State Schemas (`state.py`)

#### `ConversationState`
Manages conversation history and user preferences:
- `messages`: List of conversation messages with metadata
- `session_id`: Unique session identifier
- `user_preferences`: User-specific settings
- `context_summary`: Current conversation context

#### `AudioState`
Manages audio system state:
- `current_mode`: Audio processing mode (idle, listening, speaking, processing)
- `voice_settings`: Audio configuration parameters
- `speech_type`: Speech synthesis engine type
- `language`: Language code for speech processing

#### `CommandState`
Manages command execution:
- `execution_history`: History of executed commands with results
- `pending_operations`: Operations waiting to be executed

#### `GiannaState` (TypedDict)
Main state container combining all subsystems:
```python
{
    "conversation": ConversationState,
    "audio": AudioState,
    "commands": CommandState,
    "metadata": Dict[str, Any]
}
```

### 2. State Manager (`state_manager.py`)

The `StateManager` class provides:

#### Core Functionality
- **Session Creation**: `create_session(user_preferences=None)`
- **State Persistence**: `save_state(session_id, state)` / `load_state(session_id)`
- **LangGraph Integration**: `get_config(session_id)` + checkpointer access
- **Session Management**: List, delete, and cleanup operations

#### Database Schema
- `user_sessions`: Session metadata and preferences
- `conversation_messages`: Message history with timestamps
- `audio_history`: Audio state changes over time
- `command_history`: Command execution logs

#### Error Handling
- Graceful fallback to memory checkpointer
- Comprehensive error logging
- Database connection management
- Transaction safety

### 3. Utility Functions (`state.py`)

- `create_initial_state(session_id)`: Create default state
- `validate_state(dict)`: Validate and convert dictionary to GiannaState
- `state_to_dict(state)`: Serialize state for storage

## Usage Examples

### Basic Usage

```python
from gianna.core import StateManager, create_initial_state

# Initialize state manager
state_manager = StateManager("assistant.db")

# Create new session
session_id = state_manager.create_session({
    "language": "pt-br",
    "voice_preference": "female"
})

# Create and use state
state = create_initial_state(session_id)
state["conversation"].messages.append({
    "role": "user",
    "content": "Hello, Gianna!"
})

# Persist state
state_manager.save_state(session_id, state)

# Load state later
loaded_state = state_manager.load_state(session_id)
```

### LangGraph Integration

```python
from langgraph.graph import StateGraph
from gianna.core import StateManager, GiannaState

state_manager = StateManager()
session_id = state_manager.create_session()

# Use with LangGraph workflow
graph = StateGraph(GiannaState)
# ... add nodes and edges ...
compiled_graph = graph.compile(checkpointer=state_manager.checkpointer)

# Execute with session config
config = state_manager.get_config(session_id)
result = compiled_graph.invoke(state, config)
```

## Features Implemented

✅ **Pydantic State Validation**: Type-safe state schemas with validation
✅ **SQLite Persistence**: Full database integration with proper schema
✅ **Session Management**: Create, load, save, delete, and list sessions
✅ **LangGraph Compatibility**: Checkpointer integration for workflows
✅ **Error Handling**: Comprehensive error handling and logging
✅ **Type Hints**: Full type annotations throughout
✅ **Documentation**: Detailed docstrings and examples
✅ **Memory Fallback**: Graceful fallback when SQLite checkpointer unavailable

## Database Schema

The system creates the following tables:

### `user_sessions`
- `session_id` (TEXT PRIMARY KEY)
- `created_at` (TIMESTAMP)
- `last_activity` (TIMESTAMP)
- `session_data` (JSON)
- `user_preferences` (JSON)
- `context_summary` (TEXT)

### `conversation_messages`
- `id` (INTEGER PRIMARY KEY)
- `session_id` (TEXT, FK)
- `role` (TEXT)
- `content` (TEXT)
- `timestamp` (TIMESTAMP)
- `metadata` (JSON)

### `audio_history`
- `id` (INTEGER PRIMARY KEY)
- `session_id` (TEXT, FK)
- `mode` (TEXT)
- `speech_type` (TEXT)
- `language` (TEXT)
- `timestamp` (TIMESTAMP)
- `settings` (JSON)

### `command_history`
- `id` (INTEGER PRIMARY KEY)
- `session_id` (TEXT, FK)
- `command` (TEXT)
- `result` (JSON)
- `timestamp` (TIMESTAMP)
- `success` (BOOLEAN)
- `execution_time` (REAL)

## Testing

Run the demo to verify functionality:

```bash
PYTHONPATH=/path/to/gianna python examples/core_state_demo.py
```

## Next Steps (FASE 2)

The core state management system is ready for:

1. **LangGraph Migration Layer** - Replace existing chains with LangGraph workflows
2. **ReAct Agent Integration** - Use state with specialized agents
3. **Tool Integration** - Command and audio tool integration
4. **Multi-Agent Coordination** - Agent orchestration using shared state

This implementation fully satisfies the requirements specified in lines 7-65 of `PHASE_SPECIFICATIONS.md`.
