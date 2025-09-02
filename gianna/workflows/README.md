# Gianna Workflow Templates

This module provides basic workflow templates for FASE 1 implementation using LangGraph StateGraph. The templates offer reusable patterns for common AI assistant operations while integrating seamlessly with Gianna's core systems.

## Overview

The workflow templates system provides:

- **StateGraph Integration**: Full LangGraph StateGraph compatibility
- **State Management**: Integration with Gianna's core state management system
- **Error Handling**: Comprehensive error recovery mechanisms
- **Extensibility**: Support for custom nodes and edges
- **Audio Integration**: Built-in support for voice interactions
- **Command Safety**: Safe command execution with validation

## Available Templates

### 1. Conversation Workflow (`create_conversation_workflow`)

A basic template for text-based conversations with the following flow:

```
Input Validation → Context Processing → Response Generation → Output Formatting
```

**Features:**
- Message validation and processing
- Conversation context management
- LLM response generation
- State persistence support
- Error recovery

**Usage:**
```python
from gianna.workflows import create_conversation_workflow
from gianna.core.state import create_initial_state

# Create workflow
workflow = create_conversation_workflow(
    name="my_chat",
    enable_state_management=True,
    enable_error_recovery=True
)

# Setup state
session_id = "session_123"
state = create_initial_state(session_id)
state["conversation"].messages.append({
    "role": "user",
    "content": "Hello!",
    "timestamp": "unique_id"
})

# Execute
config = {"configurable": {"thread_id": session_id}}
result = workflow.invoke(state, config)
```

### 2. Command Workflow (`create_command_workflow`)

A template for safe command execution with validation:

```
Command Parsing → Safety Validation → Execution → Output Capture → Response Format
```

**Features:**
- Command parsing and analysis
- Safety validation (prevents dangerous commands)
- Secure command execution
- Output capture and processing
- Execution history tracking

**Usage:**
```python
from gianna.workflows import create_command_workflow

# Create workflow with safety validation
workflow = create_command_workflow(
    name="safe_executor",
    enable_safety_validation=True,
    enable_error_recovery=True
)

# Setup command state
state = create_initial_state("cmd_session")
state["conversation"].messages.append({
    "role": "user",
    "content": "ls -la",  # Safe command
    "timestamp": "unique_id"
})

# Execute
result = workflow.invoke(state, config)
```

### 3. Voice Workflow (`create_voice_workflow`)

A complete voice interaction template combining STT, processing, and TTS:

```
Audio Input → Speech-to-Text → Intent Processing → Response Generation → Text-to-Speech → Audio Output
```

**Features:**
- Audio input processing
- Speech-to-text conversion
- Intent analysis
- Voice response generation
- Text-to-speech synthesis
- Audio output management

**Usage:**
```python
from gianna.workflows import create_voice_workflow

# Create voice workflow
workflow = create_voice_workflow(
    name="voice_assistant",
    enable_async_processing=True,
    enable_error_recovery=True
)

# Setup voice state
state = create_initial_state("voice_session")
state["audio"].speech_type = "google"
state["audio"].language = "en-US"
state["audio"].current_mode = "listening"

# Execute
result = workflow.invoke(state, config)
```

## Custom Workflows

You can extend existing templates or create completely custom workflows:

### Adding Custom Nodes

```python
from gianna.workflows.templates import WorkflowBuilder, WorkflowConfig, WorkflowType

def my_custom_node(state: GiannaState) -> GiannaState:
    # Custom processing logic
    state["metadata"]["custom_processed"] = True
    return state

# Create custom workflow
config = WorkflowConfig(
    workflow_type=WorkflowType.CONVERSATION,
    name="custom_workflow",
    custom_nodes={"my_node": my_custom_node},
    custom_edges=[("process_conversation", "my_node"), ("my_node", "generate_response")]
)

builder = WorkflowBuilder(config)
workflow = builder.compile()
```

### Custom Workflow Type

```python
config = WorkflowConfig(
    workflow_type=WorkflowType.CUSTOM,
    name="fully_custom",
    custom_nodes={
        "start_node": start_processing,
        "middle_node": middle_processing,
        "end_node": end_processing
    },
    custom_edges=[
        ("start_node", "middle_node"),
        ("middle_node", "end_node")
    ]
)
```

## State Management

All workflows integrate with Gianna's state management system:

### GiannaState Structure

```python
{
    "conversation": ConversationState,  # Messages, session data, preferences
    "audio": AudioState,               # Audio settings, current mode
    "commands": CommandState,          # Command history, pending operations
    "metadata": dict                   # Processing metadata, errors, etc.
}
```

### Session Persistence

```python
# Enable state persistence
workflow = create_conversation_workflow(enable_state_management=True)

# Use consistent session ID for persistence
session_id = "persistent_session"
config = {"configurable": {"thread_id": session_id}}

# State is automatically saved and loaded
result = workflow.invoke(state, config)
```

## Error Handling

Workflows include comprehensive error handling:

### Error Recovery

```python
# Enable error recovery
workflow = create_conversation_workflow(enable_error_recovery=True)

# Errors are automatically caught and handled
# Error messages are added to conversation
# Workflows continue execution when possible
```

### Error Types

- **WorkflowError**: Base workflow exception
- **WorkflowStateError**: State validation/management errors
- **Processing Errors**: Node-specific processing failures
- **Integration Errors**: External system integration issues

## Integration Points

### Audio System Integration

Workflows integrate with Gianna's audio system:

```python
# Audio configuration in state
state["audio"] = AudioState(
    current_mode="listening",
    speech_type="google",
    language="en-US",
    voice_settings={
        "speed": 1.0,
        "pitch": 0.0,
        "volume": 0.8
    }
)
```

### Command System Integration

Command workflows integrate with the command system:

```python
# Command execution integrates with:
# - Command safety validation
# - Command history tracking
# - Output capture and formatting
```

### Model Integration

Workflows can integrate with any LLM model:

```python
# Workflows are model-agnostic
# Integrate with existing model factory
# Support for streaming and async processing
```

## Configuration Options

### WorkflowConfig Parameters

- `workflow_type`: Type of workflow (CONVERSATION, COMMAND, VOICE, CUSTOM)
- `name`: Workflow identifier
- `description`: Workflow description
- `enable_state_management`: Enable state persistence (default: True)
- `enable_error_recovery`: Enable error recovery nodes (default: True)
- `enable_async_processing`: Enable async processing (default: False)
- `custom_nodes`: Dictionary of custom node functions
- `custom_edges`: List of custom edge definitions
- `metadata`: Additional configuration metadata

### Template Function Parameters

Each template function accepts:
- `name`: Workflow name
- `enable_state_management`: State persistence toggle
- `enable_error_recovery`: Error recovery toggle
- Template-specific options (e.g., `enable_safety_validation` for commands)
- `custom_nodes`: Custom node functions
- `**kwargs`: Additional metadata

## Performance Considerations

### State Size Management

- Keep conversation history reasonable (implement truncation)
- Use context summaries for long conversations
- Clean up metadata after processing

### Resource Usage

- Enable async processing for I/O-heavy operations
- Use appropriate timeout values
- Monitor memory usage for long-running sessions

### Scaling

- Workflows are stateless except for persistence
- Multiple workflows can run concurrently
- State management handles concurrent access

## Testing

Run the examples to test functionality:

```python
from gianna.workflows.examples import run_all_examples

# Run all example workflows
results = run_all_examples()
```

Individual examples:

```python
from gianna.workflows.examples import (
    example_basic_conversation,
    example_command_execution,
    example_voice_interaction,
    example_custom_workflow
)

# Run specific examples
result = example_basic_conversation()
```

## Migration Guide

### From AbstractBasicChain

Existing code using AbstractBasicChain can migrate to workflows:

```python
# Old approach
from gianna.assistants.models.factory_method import get_chain_instance
chain = get_chain_instance("gpt35", "You are helpful")
result = chain.invoke({"input": "Hello"})

# New workflow approach
from gianna.workflows import create_conversation_workflow
from gianna.core.state import create_initial_state

workflow = create_conversation_workflow()
state = create_initial_state()
state["conversation"].messages.append({
    "role": "user",
    "content": "Hello",
    "timestamp": "unique_id"
})
result = workflow.invoke(state, {"configurable": {"thread_id": "session"}})
```

### Backward Compatibility

The LangGraphChain provides backward compatibility while workflows offer enhanced functionality.

## Future Extensions

The template system is designed for extension:

- Additional workflow types (search, analysis, etc.)
- Enhanced audio processing workflows
- Multi-agent collaboration workflows
- Streaming and real-time processing
- Advanced error recovery strategies
- Performance optimization templates

## Dependencies

- LangGraph: StateGraph workflow orchestration
- Loguru: Logging system
- Pydantic: State validation
- UUID: Unique identifier generation
- Gianna Core: State management and integration points

## Examples

See `examples.py` for comprehensive usage examples covering:

1. Basic conversation workflows
2. Command execution with safety validation
3. Voice interaction workflows
4. Custom workflow creation
5. State persistence
6. Error recovery
7. Integration patterns

Run examples with:

```bash
python -m gianna.workflows.examples
```
