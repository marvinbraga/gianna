"""
Workflow Templates Module for Gianna AI Assistant

This module provides comprehensive workflow templates using LangGraph StateGraph.
It includes core workflow patterns for voice interactions, streaming processing,
and general conversation management.

The workflow templates integrate with:
- Core State Management system (GiannaState)
- Audio system (TTS/STT, Voice processing, VAD)
- Command system (Shell commands, Speech commands)
- ReAct Agents for intelligent processing
- LangGraph StateGraph for workflow orchestration
- Real-time streaming capabilities

Available Workflow Templates:
- create_conversation_workflow: Basic conversation template
- create_command_workflow: Command execution template
- create_voice_workflow: Complete voice interaction template

Voice Workflow Implementations:
- VoiceInteractionWorkflow: Complete voice pipeline with ReAct agents
- StreamingVoiceWorkflow: Real-time streaming voice interactions with VAD
"""

from .templates import (
    WorkflowBuilder,
    WorkflowConfig,
    WorkflowError,
    WorkflowStateError,
    create_command_workflow,
    create_conversation_workflow,
    create_voice_workflow,
)

# Voice workflow implementations
from .voice_interaction import (
    VoiceInteractionError,
    VoiceInteractionWorkflow,
    VoiceWorkflowConfig,
    create_simple_voice_workflow,
    create_voice_interaction_workflow,
)
from .voice_streaming import (
    StreamingEvent,
    StreamingState,
    StreamingVoiceWorkflow,
    StreamingWorkflowConfig,
    StreamingWorkflowError,
    create_simple_streaming_workflow,
    create_streaming_voice_workflow,
)

# Examples are available but not exported by default
# Import explicitly: from gianna.workflows.examples import run_all_examples

__all__ = [
    # Basic templates
    "create_conversation_workflow",
    "create_command_workflow",
    "create_voice_workflow",
    "WorkflowConfig",
    "WorkflowBuilder",
    "WorkflowError",
    "WorkflowStateError",
    # Voice Interaction Workflow
    "VoiceInteractionWorkflow",
    "VoiceWorkflowConfig",
    "VoiceInteractionError",
    "create_voice_interaction_workflow",
    "create_simple_voice_workflow",
    # Streaming Voice Workflow
    "StreamingVoiceWorkflow",
    "StreamingWorkflowConfig",
    "StreamingWorkflowError",
    "StreamingState",
    "StreamingEvent",
    "create_streaming_voice_workflow",
    "create_simple_streaming_workflow",
]

__version__ = "1.0.0"
__author__ = "Gianna AI Team"
__description__ = "Basic workflow templates for LangGraph integration"
