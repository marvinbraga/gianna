"""
Core Workflow Templates for Gianna AI Assistant

This module provides reusable workflow templates using LangGraph StateGraph.
Each template defines a complete workflow with nodes, edges, and state transitions
that can be customized for specific use cases.

Templates include:
1. Basic conversation workflow for text interactions
2. Command execution workflow for system commands
3. Complete voice interaction workflow combining audio and conversation
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    START = None
    SqliteSaver = None

from loguru import logger

from ..core.state import (
    AudioState,
    CommandState,
    ConversationState,
    GiannaState,
    create_initial_state,
)
from ..core.state_manager import StateManager


class WorkflowError(Exception):
    """Base exception for workflow-related errors."""

    pass


class WorkflowStateError(WorkflowError):
    """Exception for workflow state management errors."""

    pass


class WorkflowType(Enum):
    """Available workflow types."""

    CONVERSATION = "conversation"
    COMMAND = "command"
    VOICE = "voice"
    CUSTOM = "custom"


@dataclass
class WorkflowConfig:
    """Configuration for workflow creation and execution."""

    workflow_type: WorkflowType
    name: str = ""
    description: str = ""
    enable_state_management: bool = True
    enable_error_recovery: bool = True
    enable_async_processing: bool = False
    custom_nodes: Dict[str, Callable] = field(default_factory=dict)
    custom_edges: List[tuple] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowBuilder:
    """Builder class for creating and managing workflows."""

    def __init__(self, config: WorkflowConfig):
        """
        Initialize workflow builder with configuration.

        Args:
            config: WorkflowConfig instance with workflow settings
        """
        if not LANGGRAPH_AVAILABLE:
            raise WorkflowError(
                "LangGraph is not available. Please install langgraph package."
            )

        self.config = config
        self.state_manager = None
        self._graph = None
        self._compiled_graph = None

        if config.enable_state_management:
            self.state_manager = StateManager()

        logger.info(
            f"WorkflowBuilder initialized for type: {config.workflow_type.value}"
        )

    def build_graph(self) -> StateGraph:
        """
        Build the StateGraph based on configuration.

        Returns:
            StateGraph: Configured but not compiled graph
        """
        if not LANGGRAPH_AVAILABLE:
            raise WorkflowError("LangGraph not available")

        graph = StateGraph(GiannaState)

        # Add workflow-specific nodes based on type
        if self.config.workflow_type == WorkflowType.CONVERSATION:
            self._add_conversation_nodes(graph)
        elif self.config.workflow_type == WorkflowType.COMMAND:
            self._add_command_nodes(graph)
        elif self.config.workflow_type == WorkflowType.VOICE:
            self._add_voice_nodes(graph)
        else:
            # Custom workflow - add custom nodes
            for node_name, node_func in self.config.custom_nodes.items():
                graph.add_node(node_name, node_func)

        # Add custom nodes if provided
        for node_name, node_func in self.config.custom_nodes.items():
            if not graph.get_node(node_name):  # Avoid duplicates
                graph.add_node(node_name, node_func)

        # Set up edges
        self._setup_edges(graph)

        self._graph = graph
        return graph

    def _add_conversation_nodes(self, graph: StateGraph):
        """Add conversation workflow nodes."""
        graph.add_node("validate_input", self._validate_conversation_input)
        graph.add_node("process_conversation", self._process_conversation)
        graph.add_node("generate_response", self._generate_conversation_response)
        graph.add_node("format_output", self._format_conversation_output)
        if self.config.enable_error_recovery:
            graph.add_node("error_recovery", self._handle_error_recovery)

    def _add_command_nodes(self, graph: StateGraph):
        """Add command workflow nodes."""
        graph.add_node("parse_command", self._parse_command_input)
        graph.add_node("validate_command", self._validate_command_safety)
        graph.add_node("execute_command", self._execute_command)
        graph.add_node("capture_output", self._capture_command_output)
        graph.add_node("format_response", self._format_command_response)
        if self.config.enable_error_recovery:
            graph.add_node("error_recovery", self._handle_error_recovery)

    def _add_voice_nodes(self, graph: StateGraph):
        """Add voice workflow nodes."""
        graph.add_node("audio_input", self._process_audio_input)
        graph.add_node("speech_to_text", self._convert_speech_to_text)
        graph.add_node("process_intent", self._process_voice_intent)
        graph.add_node("generate_response", self._generate_voice_response)
        graph.add_node("text_to_speech", self._convert_text_to_speech)
        graph.add_node("audio_output", self._process_audio_output)
        if self.config.enable_error_recovery:
            graph.add_node("error_recovery", self._handle_error_recovery)

    def _setup_edges(self, graph: StateGraph):
        """Set up workflow edges based on type."""
        if self.config.workflow_type == WorkflowType.CONVERSATION:
            self._setup_conversation_edges(graph)
        elif self.config.workflow_type == WorkflowType.COMMAND:
            self._setup_command_edges(graph)
        elif self.config.workflow_type == WorkflowType.VOICE:
            self._setup_voice_edges(graph)

        # Add custom edges
        for edge in self.config.custom_edges:
            if len(edge) == 2:
                graph.add_edge(edge[0], edge[1])
            elif len(edge) == 3:
                # Conditional edge
                graph.add_conditional_edges(edge[0], edge[1], edge[2])

    def _setup_conversation_edges(self, graph: StateGraph):
        """Set up conversation workflow edges."""
        graph.set_entry_point("validate_input")
        graph.add_edge("validate_input", "process_conversation")
        graph.add_edge("process_conversation", "generate_response")
        graph.add_edge("generate_response", "format_output")
        graph.add_edge("format_output", END)

        if self.config.enable_error_recovery:
            # Add error recovery edges
            graph.add_conditional_edges(
                "validate_input",
                self._should_recover_from_error,
                {"error": "error_recovery", "continue": "process_conversation"},
            )
            graph.add_edge("error_recovery", END)

    def _setup_command_edges(self, graph: StateGraph):
        """Set up command workflow edges."""
        graph.set_entry_point("parse_command")
        graph.add_edge("parse_command", "validate_command")
        graph.add_conditional_edges(
            "validate_command",
            self._is_command_safe,
            {"safe": "execute_command", "unsafe": END},
        )
        graph.add_edge("execute_command", "capture_output")
        graph.add_edge("capture_output", "format_response")
        graph.add_edge("format_response", END)

        if self.config.enable_error_recovery:
            graph.add_conditional_edges(
                "execute_command",
                self._should_recover_from_error,
                {"error": "error_recovery", "continue": "capture_output"},
            )
            graph.add_edge("error_recovery", END)

    def _setup_voice_edges(self, graph: StateGraph):
        """Set up voice workflow edges."""
        graph.set_entry_point("audio_input")
        graph.add_edge("audio_input", "speech_to_text")
        graph.add_edge("speech_to_text", "process_intent")
        graph.add_edge("process_intent", "generate_response")
        graph.add_edge("generate_response", "text_to_speech")
        graph.add_edge("text_to_speech", "audio_output")
        graph.add_edge("audio_output", END)

        if self.config.enable_error_recovery:
            # Add error recovery at key points
            graph.add_conditional_edges(
                "speech_to_text",
                self._should_recover_from_error,
                {"error": "error_recovery", "continue": "process_intent"},
            )
            graph.add_conditional_edges(
                "text_to_speech",
                self._should_recover_from_error,
                {"error": "error_recovery", "continue": "audio_output"},
            )
            graph.add_edge("error_recovery", END)

    def compile(self) -> Any:
        """
        Compile the workflow graph.

        Returns:
            Compiled graph ready for execution
        """
        if not self._graph:
            self.build_graph()

        if self.state_manager:
            self._compiled_graph = self._graph.compile(
                checkpointer=self.state_manager.checkpointer
            )
        else:
            self._compiled_graph = self._graph.compile()

        logger.info(f"Workflow compiled: {self.config.workflow_type.value}")
        return self._compiled_graph

    # Node implementation methods for conversation workflow
    def _validate_conversation_input(self, state: GiannaState) -> GiannaState:
        """Validate conversation input."""
        try:
            messages = state["conversation"].messages
            if not messages:
                raise WorkflowStateError("No messages to process")

            last_message = messages[-1]
            if not last_message.get("content"):
                raise WorkflowStateError("Empty message content")

            state["metadata"]["validation"] = "passed"
            state["metadata"]["processing_stage"] = "input_validated"

            logger.debug("Conversation input validation passed")
            return state

        except Exception as e:
            logger.error(f"Input validation error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "validation_error"
            return state

    def _process_conversation(self, state: GiannaState) -> GiannaState:
        """Process conversation context and prepare for response generation."""
        try:
            # Update conversation context
            messages = state["conversation"].messages
            recent_messages = messages[-5:] if len(messages) > 5 else messages

            # Create simple context summary
            context_parts = []
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]  # Truncate for summary
                context_parts.append(f"{role}: {content}")

            state["conversation"].context_summary = " | ".join(context_parts)
            state["metadata"]["processing_stage"] = "conversation_processed"
            state["metadata"]["context_length"] = len(recent_messages)

            logger.debug("Conversation processing completed")
            return state

        except Exception as e:
            logger.error(f"Conversation processing error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "processing_error"
            return state

    def _generate_conversation_response(self, state: GiannaState) -> GiannaState:
        """Generate response using configured LLM."""
        try:
            # This is a placeholder - actual LLM integration would happen here
            # For now, create a simple echo response
            last_message = state["conversation"].messages[-1]
            user_input = last_message.get("content", "")

            # Simple response generation (would be replaced by actual LLM)
            response_content = f"I received: {user_input}"

            # Add assistant response
            response_message = {
                "role": "assistant",
                "content": response_content,
                "timestamp": str(uuid4()),
                "metadata": {
                    "workflow_type": self.config.workflow_type.value,
                    "generated_at": "workflow_template",
                },
            }

            state["conversation"].messages.append(response_message)
            state["metadata"]["processing_stage"] = "response_generated"
            state["metadata"]["response_length"] = len(response_content)

            logger.debug("Response generation completed")
            return state

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "generation_error"
            # Add error response
            error_message = {
                "role": "assistant",
                "content": f"Error generating response: {str(e)}",
                "timestamp": str(uuid4()),
                "metadata": {"error": True},
            }
            state["conversation"].messages.append(error_message)
            return state

    def _format_conversation_output(self, state: GiannaState) -> GiannaState:
        """Format final conversation output."""
        try:
            state["metadata"]["processing_stage"] = "output_formatted"
            state["metadata"]["workflow_completed"] = True

            # Save state if state management is enabled
            if self.state_manager and state["conversation"].session_id:
                self.state_manager.save_state(state["conversation"].session_id, state)

            logger.debug("Conversation output formatting completed")
            return state

        except Exception as e:
            logger.error(f"Output formatting error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "formatting_error"
            return state

    # Node implementation methods for command workflow
    def _parse_command_input(self, state: GiannaState) -> GiannaState:
        """Parse and analyze command input."""
        try:
            messages = state["conversation"].messages
            if not messages:
                raise WorkflowStateError("No command to parse")

            last_message = messages[-1]
            command_text = last_message.get("content", "")

            # Simple command parsing (would be enhanced with actual parsing)
            parsed_command = {
                "raw_command": command_text,
                "command_type": "shell",  # Default type
                "arguments": command_text.split()[1:] if command_text.split() else [],
                "is_safe": True,  # Default - would be determined by safety analysis
                "timestamp": str(uuid4()),
            }

            # Add to command history
            state["commands"].pending_operations.append(command_text)
            state["metadata"]["parsed_command"] = parsed_command
            state["metadata"]["processing_stage"] = "command_parsed"

            logger.debug(f"Command parsed: {command_text[:50]}...")
            return state

        except Exception as e:
            logger.error(f"Command parsing error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "parsing_error"
            return state

    def _validate_command_safety(self, state: GiannaState) -> GiannaState:
        """Validate command for safety and security."""
        try:
            parsed_command = state["metadata"].get("parsed_command", {})
            raw_command = parsed_command.get("raw_command", "")

            # Simple safety check (would be enhanced with proper validation)
            dangerous_patterns = ["rm -rf", "del /f", "format", "mkfs", "sudo su"]
            is_safe = not any(
                pattern in raw_command.lower() for pattern in dangerous_patterns
            )

            parsed_command["is_safe"] = is_safe
            state["metadata"]["parsed_command"] = parsed_command
            state["metadata"]["processing_stage"] = "command_validated"
            state["metadata"]["safety_check"] = "passed" if is_safe else "failed"

            if not is_safe:
                logger.warning(f"Unsafe command detected: {raw_command[:50]}...")
            else:
                logger.debug("Command safety validation passed")

            return state

        except Exception as e:
            logger.error(f"Command validation error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "validation_error"
            return state

    def _execute_command(self, state: GiannaState) -> GiannaState:
        """Execute the validated command."""
        try:
            parsed_command = state["metadata"].get("parsed_command", {})
            raw_command = parsed_command.get("raw_command", "")

            # This is a placeholder - actual command execution would use the command system
            # For safety in this template, we simulate execution
            execution_result = {
                "command": raw_command,
                "exit_code": 0,
                "stdout": f"Simulated output for: {raw_command}",
                "stderr": "",
                "execution_time": "0.1s",
                "timestamp": str(uuid4()),
            }

            # Add to execution history
            state["commands"].execution_history.append(execution_result)

            # Remove from pending operations
            if raw_command in state["commands"].pending_operations:
                state["commands"].pending_operations.remove(raw_command)

            state["metadata"]["execution_result"] = execution_result
            state["metadata"]["processing_stage"] = "command_executed"

            logger.debug(f"Command executed: {raw_command[:50]}...")
            return state

        except Exception as e:
            logger.error(f"Command execution error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "execution_error"
            return state

    def _capture_command_output(self, state: GiannaState) -> GiannaState:
        """Capture and process command output."""
        try:
            execution_result = state["metadata"].get("execution_result", {})
            stdout = execution_result.get("stdout", "")
            stderr = execution_result.get("stderr", "")

            # Process output for response
            output_summary = {
                "has_output": bool(stdout or stderr),
                "output_length": len(stdout) + len(stderr),
                "exit_code": execution_result.get("exit_code", -1),
                "success": execution_result.get("exit_code", -1) == 0,
            }

            state["metadata"]["output_summary"] = output_summary
            state["metadata"]["processing_stage"] = "output_captured"

            logger.debug("Command output captured")
            return state

        except Exception as e:
            logger.error(f"Output capture error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "capture_error"
            return state

    def _format_command_response(self, state: GiannaState) -> GiannaState:
        """Format command execution response for user."""
        try:
            execution_result = state["metadata"].get("execution_result", {})
            output_summary = state["metadata"].get("output_summary", {})

            # Create response message
            if output_summary.get("success", False):
                stdout = execution_result.get("stdout", "")
                response_content = f"Command executed successfully:\n{stdout}"
            else:
                stderr = execution_result.get("stderr", "No error details")
                response_content = f"Command failed:\n{stderr}"

            response_message = {
                "role": "assistant",
                "content": response_content,
                "timestamp": str(uuid4()),
                "metadata": {
                    "workflow_type": self.config.workflow_type.value,
                    "command_execution": execution_result,
                    "output_summary": output_summary,
                },
            }

            state["conversation"].messages.append(response_message)
            state["metadata"]["processing_stage"] = "response_formatted"

            logger.debug("Command response formatted")
            return state

        except Exception as e:
            logger.error(f"Response formatting error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "formatting_error"
            return state

    # Node implementation methods for voice workflow
    def _process_audio_input(self, state: GiannaState) -> GiannaState:
        """Process incoming audio input."""
        try:
            # Set audio mode to processing
            state["audio"].current_mode = "processing"
            state["metadata"]["processing_stage"] = "audio_input_processed"

            # Placeholder for actual audio processing
            # Would integrate with actual audio input system
            audio_metadata = {
                "format": "wav",
                "sample_rate": 16000,
                "channels": 1,
                "duration": "estimated_from_audio",
                "timestamp": str(uuid4()),
            }

            state["metadata"]["audio_input"] = audio_metadata

            logger.debug("Audio input processing completed")
            return state

        except Exception as e:
            logger.error(f"Audio input processing error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "audio_input_error"
            return state

    def _convert_speech_to_text(self, state: GiannaState) -> GiannaState:
        """Convert speech to text using STT system."""
        try:
            # This would integrate with actual STT system
            # For now, simulate STT conversion
            simulated_text = "Hello, this is simulated speech-to-text output"

            # Add user message from speech
            speech_message = {
                "role": "user",
                "content": simulated_text,
                "timestamp": str(uuid4()),
                "metadata": {
                    "source": "speech",
                    "stt_engine": state["audio"].speech_type,
                    "language": state["audio"].language,
                    "confidence": 0.95,  # Simulated confidence
                },
            }

            state["conversation"].messages.append(speech_message)
            state["metadata"]["stt_result"] = simulated_text
            state["metadata"]["processing_stage"] = "speech_converted"

            logger.debug("Speech-to-text conversion completed")
            return state

        except Exception as e:
            logger.error(f"Speech-to-text error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "stt_error"
            return state

    def _process_voice_intent(self, state: GiannaState) -> GiannaState:
        """Process intent from voice input."""
        try:
            # Analyze the converted text for intent
            stt_result = state["metadata"].get("stt_result", "")

            # Simple intent classification (would be enhanced with actual NLP)
            intent_analysis = {
                "text": stt_result,
                "intent": "conversation",  # Default intent
                "confidence": 0.8,
                "entities": [],  # Would extract entities
                "requires_response": True,
            }

            state["metadata"]["intent_analysis"] = intent_analysis
            state["metadata"]["processing_stage"] = "intent_processed"

            logger.debug("Voice intent processing completed")
            return state

        except Exception as e:
            logger.error(f"Intent processing error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "intent_error"
            return state

    def _generate_voice_response(self, state: GiannaState) -> GiannaState:
        """Generate response for voice interaction."""
        try:
            intent_analysis = state["metadata"].get("intent_analysis", {})
            user_text = intent_analysis.get("text", "")

            # Generate response (would use actual LLM)
            response_text = f"I heard you say: {user_text}. How can I help you?"

            # Add assistant response
            voice_response = {
                "role": "assistant",
                "content": response_text,
                "timestamp": str(uuid4()),
                "metadata": {
                    "workflow_type": "voice",
                    "intent_based": True,
                    "requires_tts": True,
                },
            }

            state["conversation"].messages.append(voice_response)
            state["metadata"]["voice_response"] = response_text
            state["metadata"]["processing_stage"] = "voice_response_generated"

            logger.debug("Voice response generation completed")
            return state

        except Exception as e:
            logger.error(f"Voice response generation error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "voice_generation_error"
            return state

    def _convert_text_to_speech(self, state: GiannaState) -> GiannaState:
        """Convert response text to speech."""
        try:
            response_text = state["metadata"].get("voice_response", "")

            # This would integrate with actual TTS system
            # For now, simulate TTS conversion
            tts_metadata = {
                "text": response_text,
                "voice_type": state["audio"].speech_type,
                "language": state["audio"].language,
                "audio_format": "mp3",
                "estimated_duration": len(response_text) * 0.05,  # Rough estimate
                "timestamp": str(uuid4()),
            }

            state["metadata"]["tts_result"] = tts_metadata
            state["metadata"]["processing_stage"] = "text_converted_to_speech"

            # Update audio state
            state["audio"].current_mode = "speaking"

            logger.debug("Text-to-speech conversion completed")
            return state

        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "tts_error"
            return state

    def _process_audio_output(self, state: GiannaState) -> GiannaState:
        """Process and output audio response."""
        try:
            tts_result = state["metadata"].get("tts_result", {})

            # This would integrate with actual audio output system
            # For now, simulate audio output processing
            output_metadata = {
                "audio_played": True,
                "playback_duration": tts_result.get("estimated_duration", 0),
                "output_device": "default",
                "volume_level": 0.8,
                "timestamp": str(uuid4()),
            }

            state["metadata"]["audio_output"] = output_metadata
            state["metadata"]["processing_stage"] = "audio_output_completed"

            # Reset audio state
            state["audio"].current_mode = "idle"

            # Save state if enabled
            if self.state_manager and state["conversation"].session_id:
                self.state_manager.save_state(state["conversation"].session_id, state)

            logger.debug("Audio output processing completed")
            return state

        except Exception as e:
            logger.error(f"Audio output error: {e}")
            state["metadata"]["error"] = str(e)
            state["metadata"]["processing_stage"] = "audio_output_error"
            return state

    # Utility methods for conditional edges
    def _should_recover_from_error(self, state: GiannaState) -> str:
        """Determine if error recovery should be attempted."""
        error = state["metadata"].get("error")
        if error and self.config.enable_error_recovery:
            return "error"
        return "continue"

    def _is_command_safe(self, state: GiannaState) -> str:
        """Check if command is safe to execute."""
        parsed_command = state["metadata"].get("parsed_command", {})
        is_safe = parsed_command.get("is_safe", False)
        return "safe" if is_safe else "unsafe"

    def _handle_error_recovery(self, state: GiannaState) -> GiannaState:
        """Handle error recovery for workflows."""
        try:
            error = state["metadata"].get("error", "Unknown error")

            # Create error response
            error_response = {
                "role": "assistant",
                "content": f"I encountered an error: {error}. Please try again.",
                "timestamp": str(uuid4()),
                "metadata": {"error_recovery": True, "original_error": error},
            }

            state["conversation"].messages.append(error_response)
            state["metadata"]["processing_stage"] = "error_recovered"
            state["metadata"]["error_handled"] = True

            # Reset states for recovery
            state["audio"].current_mode = "idle"

            logger.info(f"Error recovery completed for: {error}")
            return state

        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {recovery_error}")
            state["metadata"]["recovery_error"] = str(recovery_error)
            return state


# Template factory functions
def create_conversation_workflow(
    name: str = "conversation_workflow",
    enable_state_management: bool = True,
    enable_error_recovery: bool = True,
    custom_nodes: Optional[Dict[str, Callable]] = None,
    **kwargs,
) -> Any:
    """
    Create a basic conversation workflow template.

    This workflow handles text-based conversation with the following flow:
    1. Validate input message
    2. Process conversation context
    3. Generate response using LLM
    4. Format and return output

    Args:
        name: Workflow name for identification
        enable_state_management: Whether to enable state persistence
        enable_error_recovery: Whether to enable error recovery nodes
        custom_nodes: Optional custom nodes to add to the workflow
        **kwargs: Additional configuration options

    Returns:
        Compiled workflow graph ready for execution

    Example:
        >>> workflow = create_conversation_workflow(
        ...     name="my_chat_workflow",
        ...     enable_state_management=True
        ... )
        >>> result = workflow.invoke(state, config={"configurable": {"thread_id": "session1"}})
    """
    config = WorkflowConfig(
        workflow_type=WorkflowType.CONVERSATION,
        name=name,
        description="Basic conversation workflow for text interactions",
        enable_state_management=enable_state_management,
        enable_error_recovery=enable_error_recovery,
        custom_nodes=custom_nodes or {},
        metadata=kwargs,
    )

    builder = WorkflowBuilder(config)
    return builder.compile()


def create_command_workflow(
    name: str = "command_workflow",
    enable_state_management: bool = True,
    enable_error_recovery: bool = True,
    enable_safety_validation: bool = True,
    custom_nodes: Optional[Dict[str, Callable]] = None,
    **kwargs,
) -> Any:
    """
    Create a command execution workflow template.

    This workflow handles command execution with the following flow:
    1. Parse command input
    2. Validate command safety
    3. Execute command (if safe)
    4. Capture command output
    5. Format response

    Args:
        name: Workflow name for identification
        enable_state_management: Whether to enable state persistence
        enable_error_recovery: Whether to enable error recovery nodes
        enable_safety_validation: Whether to validate command safety
        custom_nodes: Optional custom nodes to add to the workflow
        **kwargs: Additional configuration options

    Returns:
        Compiled workflow graph ready for execution

    Example:
        >>> workflow = create_command_workflow(
        ...     name="my_command_workflow",
        ...     enable_safety_validation=True
        ... )
        >>> result = workflow.invoke(state, config={"configurable": {"thread_id": "session1"}})
    """
    config = WorkflowConfig(
        workflow_type=WorkflowType.COMMAND,
        name=name,
        description="Command execution workflow with safety validation",
        enable_state_management=enable_state_management,
        enable_error_recovery=enable_error_recovery,
        custom_nodes=custom_nodes or {},
        metadata={"enable_safety_validation": enable_safety_validation, **kwargs},
    )

    builder = WorkflowBuilder(config)
    return builder.compile()


def create_voice_workflow(
    name: str = "voice_workflow",
    enable_state_management: bool = True,
    enable_error_recovery: bool = True,
    enable_async_processing: bool = False,
    custom_nodes: Optional[Dict[str, Callable]] = None,
    **kwargs,
) -> Any:
    """
    Create a complete voice interaction workflow template.

    This workflow handles voice interactions with the following flow:
    1. Process audio input
    2. Convert speech to text (STT)
    3. Process voice intent
    4. Generate response
    5. Convert text to speech (TTS)
    6. Output audio response

    Args:
        name: Workflow name for identification
        enable_state_management: Whether to enable state persistence
        enable_error_recovery: Whether to enable error recovery nodes
        enable_async_processing: Whether to enable async processing
        custom_nodes: Optional custom nodes to add to the workflow
        **kwargs: Additional configuration options

    Returns:
        Compiled workflow graph ready for execution

    Example:
        >>> workflow = create_voice_workflow(
        ...     name="my_voice_workflow",
        ...     enable_async_processing=True
        ... )
        >>> result = workflow.invoke(state, config={"configurable": {"thread_id": "session1"}})
    """
    config = WorkflowConfig(
        workflow_type=WorkflowType.VOICE,
        name=name,
        description="Complete voice interaction workflow with STT/TTS integration",
        enable_state_management=enable_state_management,
        enable_error_recovery=enable_error_recovery,
        enable_async_processing=enable_async_processing,
        custom_nodes=custom_nodes or {},
        metadata=kwargs,
    )

    builder = WorkflowBuilder(config)
    return builder.compile()
