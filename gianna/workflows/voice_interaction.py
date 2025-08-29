"""
Voice Interaction Workflow for Gianna AI Assistant

This module implements a comprehensive LangGraph workflow for voice interactions,
providing a complete pipeline from voice input to voice output with integrated
ReAct agent processing, state management, and checkpointing capabilities.

The workflow includes the following nodes:
1. voice_input: Capture and validate audio input
2. stt_processing: Convert speech to text using STT systems
3. agent_processing: Process user intent using ReAct agents
4. tts_synthesis: Convert response text to speech
5. voice_output: Output synthesized audio and update state

Key features:
- Full LangGraph StateGraph implementation
- Integration with GiannaState and StateManager
- Support for checkpointing and session persistence
- ReAct agent integration for intelligent processing
- Error handling and recovery mechanisms
- Asynchronous processing capabilities
"""

import asyncio
import tempfile
import wave
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np

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

from ..agents.react_agents import AudioAgent, ConversationAgent
from ..assistants.audio.stt.factory_method import speech_to_text
from ..assistants.audio.tts.factory_method import text_to_speech
from ..core.state import (
    AudioState,
    CommandState,
    ConversationState,
    GiannaState,
    create_initial_state,
)
from ..core.state_manager import StateManager

# Try to import ReAct agents if available
try:
    from ..agents.react_agents import LANGGRAPH_AVAILABLE as REACT_AVAILABLE
except ImportError:
    REACT_AVAILABLE = False


class VoiceInteractionError(Exception):
    """Base exception for voice interaction errors."""

    pass


class VoiceProcessingState(Enum):
    """Voice processing states."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING_STT = "processing_stt"
    PROCESSING_AGENT = "processing_agent"
    SYNTHESIZING = "synthesizing"
    SPEAKING = "speaking"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class VoiceWorkflowConfig:
    """Configuration for voice interaction workflow."""

    name: str = "voice_interaction_workflow"
    description: str = (
        "Complete voice interaction workflow with ReAct agent integration"
    )

    # Audio configuration
    audio_format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024

    # STT configuration
    stt_engine: str = "whisper"
    stt_local: bool = False
    stt_language: str = "pt-br"

    # TTS configuration
    tts_engine: str = "google"
    tts_language: str = "pt-br"
    tts_voice: str = "default"

    # Agent configuration
    use_react_agents: bool = True
    default_agent: str = "conversation"
    fallback_to_simple: bool = True

    # Processing configuration
    enable_async: bool = True
    enable_error_recovery: bool = True
    enable_checkpointing: bool = True
    max_audio_duration: int = 60  # seconds

    # Callback configuration
    enable_callbacks: bool = True
    callback_timeout: float = 1.0

    metadata: Dict[str, Any] = field(default_factory=dict)


class VoiceInteractionWorkflow:
    """
    Complete voice interaction workflow using LangGraph.

    This class provides a full pipeline for voice interactions, from audio
    input to audio output, with integrated ReAct agent processing and
    state management.
    """

    def __init__(
        self,
        config: Optional[VoiceWorkflowConfig] = None,
        state_manager: Optional[StateManager] = None,
        llm_instance: Optional[Any] = None,
    ):
        """
        Initialize the voice interaction workflow.

        Args:
            config: Voice workflow configuration
            state_manager: State manager for persistence
            llm_instance: LLM instance for agent processing
        """
        if not LANGGRAPH_AVAILABLE:
            raise VoiceInteractionError(
                "LangGraph is not available. Please install langgraph package."
            )

        self.config = config or VoiceWorkflowConfig()
        self.state_manager = state_manager or StateManager()
        self.llm_instance = llm_instance

        # Initialize workflow components
        self._graph = None
        self._compiled_graph = None
        self._agents = {}

        # Callbacks
        self.callbacks = {
            "on_voice_input": None,
            "on_stt_start": None,
            "on_stt_complete": None,
            "on_agent_start": None,
            "on_agent_complete": None,
            "on_tts_start": None,
            "on_tts_complete": None,
            "on_voice_output": None,
            "on_error": None,
        }

        self._initialize_agents()
        self._build_workflow()

        logger.info(f"VoiceInteractionWorkflow initialized: {self.config.name}")

    def _initialize_agents(self):
        """Initialize ReAct agents if available."""
        if not self.config.use_react_agents or not REACT_AVAILABLE:
            logger.info(
                "ReAct agents not available or disabled, using fallback processing"
            )
            return

        if not self.llm_instance:
            logger.warning(
                "No LLM instance provided, ReAct agents will not be initialized"
            )
            return

        try:
            # Initialize conversation agent for general interactions
            self._agents["conversation"] = ConversationAgent(self.llm_instance)

            # Initialize audio agent for audio-specific operations
            self._agents["audio"] = AudioAgent(self.llm_instance)

            logger.info(f"Initialized {len(self._agents)} ReAct agents")

        except Exception as e:
            logger.error(f"Failed to initialize ReAct agents: {e}")
            if not self.config.fallback_to_simple:
                raise VoiceInteractionError(f"Agent initialization failed: {e}")

    def _build_workflow(self):
        """Build the LangGraph workflow."""
        try:
            # Create StateGraph
            self._graph = StateGraph(GiannaState)

            # Add workflow nodes
            self._graph.add_node("voice_input", self._voice_input_node)
            self._graph.add_node("stt_processing", self._stt_processing_node)
            self._graph.add_node("agent_processing", self._agent_processing_node)
            self._graph.add_node("tts_synthesis", self._tts_synthesis_node)
            self._graph.add_node("voice_output", self._voice_output_node)

            # Add error recovery node if enabled
            if self.config.enable_error_recovery:
                self._graph.add_node("error_recovery", self._error_recovery_node)

            # Set up edges
            self._setup_edges()

            logger.info("Voice interaction workflow graph built successfully")

        except Exception as e:
            logger.error(f"Failed to build workflow graph: {e}")
            raise VoiceInteractionError(f"Graph building failed: {e}")

    def _setup_edges(self):
        """Set up workflow edges and routing logic."""
        # Main workflow path
        self._graph.set_entry_point("voice_input")
        self._graph.add_edge("voice_input", "stt_processing")
        self._graph.add_edge("stt_processing", "agent_processing")
        self._graph.add_edge("agent_processing", "tts_synthesis")
        self._graph.add_edge("tts_synthesis", "voice_output")
        self._graph.add_edge("voice_output", END)

        # Error recovery edges if enabled
        if self.config.enable_error_recovery:
            # Add conditional edges for error handling
            self._graph.add_conditional_edges(
                "stt_processing",
                self._should_recover_from_error,
                {"error": "error_recovery", "continue": "agent_processing"},
            )

            self._graph.add_conditional_edges(
                "agent_processing",
                self._should_recover_from_error,
                {"error": "error_recovery", "continue": "tts_synthesis"},
            )

            self._graph.add_conditional_edges(
                "tts_synthesis",
                self._should_recover_from_error,
                {"error": "error_recovery", "continue": "voice_output"},
            )

            self._graph.add_edge("error_recovery", END)

    def compile(self) -> Any:
        """
        Compile the workflow graph for execution.

        Returns:
            Compiled LangGraph workflow ready for execution
        """
        if not self._graph:
            raise VoiceInteractionError("Workflow graph not built")

        try:
            if self.config.enable_checkpointing:
                # Use StateManager's checkpointer for state persistence
                self._compiled_graph = self._graph.compile(
                    checkpointer=self.state_manager.checkpointer
                )
            else:
                self._compiled_graph = self._graph.compile()

            logger.info("Voice interaction workflow compiled successfully")
            return self._compiled_graph

        except Exception as e:
            logger.error(f"Failed to compile workflow: {e}")
            raise VoiceInteractionError(f"Compilation failed: {e}")

    # Workflow Node Implementations

    def _voice_input_node(self, state: GiannaState) -> GiannaState:
        """
        Process voice input and prepare audio data.

        This node handles the initial voice input, validates the audio data,
        and prepares it for STT processing.
        """
        try:
            logger.debug("Processing voice input")

            # Trigger callback
            self._trigger_callback("on_voice_input", state)

            # Update audio state
            state["audio"].current_mode = VoiceProcessingState.LISTENING.value

            # Validate audio input (placeholder - actual audio validation would be here)
            audio_metadata = state["metadata"].get("audio_input", {})

            if not audio_metadata:
                # If no audio metadata provided, create placeholder
                audio_metadata = {
                    "format": self.config.audio_format,
                    "sample_rate": self.config.sample_rate,
                    "channels": self.config.channels,
                    "duration": 0,  # Would be calculated from actual audio
                    "timestamp": datetime.now().isoformat(),
                    "source": "microphone",
                }

            # Store audio metadata
            state["metadata"]["audio_input"] = audio_metadata
            state["metadata"]["processing_stage"] = "voice_input_processed"
            state["metadata"]["workflow_step"] = 1

            logger.debug("Voice input processing completed")
            return state

        except Exception as e:
            logger.error(f"Voice input processing error: {e}")
            return self._handle_node_error(state, "voice_input", e)

    def _stt_processing_node(self, state: GiannaState) -> GiannaState:
        """
        Convert speech to text using STT system.

        This node processes the audio input and converts it to text using
        the configured STT engine.
        """
        try:
            logger.debug("Starting STT processing")

            # Trigger callback
            self._trigger_callback("on_stt_start", state)

            # Update processing state
            state["audio"].current_mode = VoiceProcessingState.PROCESSING_STT.value

            # Get audio data from metadata
            audio_metadata = state["metadata"].get("audio_input", {})
            audio_path = audio_metadata.get("file_path")

            # If no audio path provided, create a placeholder
            if not audio_path:
                # In a real implementation, this would be the actual audio data
                # For now, we'll simulate STT processing
                transcript = self._simulate_stt_processing(state)
            else:
                # Use actual STT processing
                transcript = self._process_actual_stt(audio_path)

            if not transcript:
                raise VoiceInteractionError("STT processing returned empty result")

            # Create user message from speech
            speech_message = {
                "role": "user",
                "content": transcript,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "source": "speech",
                    "stt_engine": self.config.stt_engine,
                    "language": self.config.stt_language,
                    "confidence": audio_metadata.get("stt_confidence", 0.9),
                    "audio_duration": audio_metadata.get("duration", 0),
                },
            }

            # Add message to conversation
            state["conversation"].messages.append(speech_message)

            # Store STT result
            state["metadata"]["stt_result"] = {
                "transcript": transcript,
                "confidence": audio_metadata.get("stt_confidence", 0.9),
                "processing_time": "calculated_in_real_implementation",
                "engine": self.config.stt_engine,
            }

            state["metadata"]["processing_stage"] = "stt_completed"
            state["metadata"]["workflow_step"] = 2

            # Trigger callback
            self._trigger_callback("on_stt_complete", state, transcript)

            logger.debug(f"STT processing completed: {transcript[:50]}...")
            return state

        except Exception as e:
            logger.error(f"STT processing error: {e}")
            return self._handle_node_error(state, "stt_processing", e)

    def _agent_processing_node(self, state: GiannaState) -> GiannaState:
        """
        Process user intent using ReAct agents.

        This node analyzes the transcribed text and generates an appropriate
        response using the configured ReAct agents.
        """
        try:
            logger.debug("Starting agent processing")

            # Trigger callback
            self._trigger_callback("on_agent_start", state)

            # Update processing state
            state["audio"].current_mode = VoiceProcessingState.PROCESSING_AGENT.value

            # Get the user's message
            if not state["conversation"].messages:
                raise VoiceInteractionError("No user message to process")

            last_message = state["conversation"].messages[-1]
            user_input = last_message.get("content", "")

            if not user_input:
                raise VoiceInteractionError("Empty user input")

            # Choose appropriate agent
            agent = self._select_agent_for_input(user_input, state)

            # Process with agent
            response = self._process_with_agent(agent, user_input, state)

            if not response:
                # Fallback response
                response = self._generate_fallback_response(user_input, state)

            # Create assistant response message
            assistant_message = {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "agent_used": agent.name if agent else "fallback",
                    "workflow_type": "voice_interaction",
                    "processing_method": "react_agent" if agent else "fallback",
                    "requires_tts": True,
                },
            }

            # Add response to conversation
            state["conversation"].messages.append(assistant_message)

            # Store agent processing result
            state["metadata"]["agent_result"] = {
                "response": response,
                "agent_used": agent.name if agent else "fallback",
                "processing_successful": True,
                "response_length": len(response),
            }

            state["metadata"]["processing_stage"] = "agent_completed"
            state["metadata"]["workflow_step"] = 3

            # Trigger callback
            self._trigger_callback("on_agent_complete", state, response)

            logger.debug(f"Agent processing completed: {response[:50]}...")
            return state

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return self._handle_node_error(state, "agent_processing", e)

    def _tts_synthesis_node(self, state: GiannaState) -> GiannaState:
        """
        Convert response text to speech.

        This node takes the agent's response and converts it to speech
        using the configured TTS engine.
        """
        try:
            logger.debug("Starting TTS synthesis")

            # Trigger callback
            self._trigger_callback("on_tts_start", state)

            # Update processing state
            state["audio"].current_mode = VoiceProcessingState.SYNTHESIZING.value

            # Get response text
            if not state["conversation"].messages:
                raise VoiceInteractionError("No response to synthesize")

            last_message = state["conversation"].messages[-1]
            if last_message.get("role") != "assistant":
                raise VoiceInteractionError("Last message is not an assistant response")

            response_text = last_message.get("content", "")
            if not response_text:
                raise VoiceInteractionError("Empty response text")

            # Generate speech using TTS
            tts_result = self._synthesize_speech(response_text, state)

            # Store TTS result
            state["metadata"]["tts_result"] = tts_result
            state["metadata"]["processing_stage"] = "tts_completed"
            state["metadata"]["workflow_step"] = 4

            # Trigger callback
            self._trigger_callback("on_tts_complete", state, tts_result)

            logger.debug("TTS synthesis completed")
            return state

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return self._handle_node_error(state, "tts_synthesis", e)

    def _voice_output_node(self, state: GiannaState) -> GiannaState:
        """
        Output synthesized audio and complete the workflow.

        This node handles the final audio output and workflow completion.
        """
        try:
            logger.debug("Processing voice output")

            # Trigger callback
            self._trigger_callback("on_voice_output", state)

            # Update processing state
            state["audio"].current_mode = VoiceProcessingState.SPEAKING.value

            # Get TTS result
            tts_result = state["metadata"].get("tts_result", {})

            # Process audio output (placeholder for actual playback)
            output_result = self._process_audio_output(tts_result, state)

            # Store output result
            state["metadata"]["voice_output"] = output_result
            state["metadata"]["processing_stage"] = "voice_output_completed"
            state["metadata"]["workflow_step"] = 5
            state["metadata"]["workflow_completed"] = True

            # Reset audio state to idle
            state["audio"].current_mode = VoiceProcessingState.IDLE.value

            # Save state if state management is enabled
            if self.config.enable_checkpointing and state["conversation"].session_id:
                try:
                    self.state_manager.save_state(
                        state["conversation"].session_id, state
                    )
                except Exception as e:
                    logger.warning(f"Failed to save state: {e}")

            logger.debug("Voice output processing completed")
            return state

        except Exception as e:
            logger.error(f"Voice output processing error: {e}")
            return self._handle_node_error(state, "voice_output", e)

    def _error_recovery_node(self, state: GiannaState) -> GiannaState:
        """
        Handle error recovery for the workflow.

        This node provides error recovery mechanisms when issues occur
        during workflow execution.
        """
        try:
            logger.debug("Starting error recovery")

            error = state["metadata"].get("error", "Unknown error")
            error_node = state["metadata"].get("error_node", "unknown")

            # Create error response
            error_response = {
                "role": "assistant",
                "content": f"Desculpe, ocorreu um erro durante o processamento de voz: {error}. Por favor, tente novamente.",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "error_recovery": True,
                    "original_error": error,
                    "error_node": error_node,
                    "requires_tts": True,
                },
            }

            # Add error response to conversation
            state["conversation"].messages.append(error_response)

            # Generate TTS for error message if possible
            try:
                error_tts = self._synthesize_speech(error_response["content"], state)
                state["metadata"]["error_tts_result"] = error_tts
            except Exception as tts_error:
                logger.warning(f"Failed to synthesize error message: {tts_error}")

            # Update metadata
            state["metadata"]["processing_stage"] = "error_recovered"
            state["metadata"]["error_handled"] = True
            state["metadata"]["recovery_timestamp"] = datetime.now().isoformat()

            # Reset audio state
            state["audio"].current_mode = VoiceProcessingState.ERROR.value

            # Trigger error callback
            self._trigger_callback("on_error", error, error_node)

            logger.info(f"Error recovery completed for: {error}")
            return state

        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {recovery_error}")
            state["metadata"]["recovery_error"] = str(recovery_error)
            state["audio"].current_mode = VoiceProcessingState.ERROR.value
            return state

    # Helper Methods

    def _simulate_stt_processing(self, state: GiannaState) -> str:
        """Simulate STT processing for testing purposes."""
        # This would be replaced with actual STT processing
        return "Olá, como você está hoje?"

    def _process_actual_stt(self, audio_path: str) -> str:
        """Process actual STT using the configured engine."""
        try:
            # Use the existing STT factory method
            from pathlib import Path

            audio_dir = str(Path(audio_path).parent)
            audio_format = Path(audio_path).suffix[1:]  # Remove the dot

            documents = speech_to_text(
                audio_dir, filetype=audio_format, local=self.config.stt_local
            )

            if documents and len(documents) > 0:
                transcript = " ".join([doc.page_content for doc in documents]).strip()
                return transcript
            else:
                return ""

        except Exception as e:
            logger.error(f"STT processing failed: {e}")
            return ""

    def _select_agent_for_input(
        self, user_input: str, state: GiannaState
    ) -> Optional[Any]:
        """Select appropriate agent based on user input."""
        if not self._agents:
            return None

        # Simple agent selection logic - could be enhanced
        user_input_lower = user_input.lower()

        # Check for audio-related keywords
        audio_keywords = ["áudio", "voz", "som", "volume", "música", "tocar"]
        if any(keyword in user_input_lower for keyword in audio_keywords):
            return self._agents.get("audio")

        # Default to conversation agent
        return self._agents.get("conversation")

    def _process_with_agent(
        self, agent: Any, user_input: str, state: GiannaState
    ) -> str:
        """Process input with the selected agent."""
        if not agent:
            return ""

        try:
            # Prepare input for agent
            agent_input = {"input": user_input, "state": state}

            # Execute agent
            result = agent.execute(agent_input, state)

            # Extract response from agent result
            if isinstance(result, dict):
                return result.get("content", result.get("response", ""))
            elif hasattr(result, "content"):
                return result.content
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Agent processing failed: {e}")
            return ""

    def _generate_fallback_response(self, user_input: str, state: GiannaState) -> str:
        """Generate a fallback response when agents are not available."""
        # Simple fallback response
        return f"Recebi sua mensagem: '{user_input}'. Como posso ajudá-lo?"

    def _synthesize_speech(self, text: str, state: GiannaState) -> Dict[str, Any]:
        """Synthesize speech from text using TTS engine."""
        try:
            # Use the existing TTS factory method
            audio_file = text_to_speech(
                text=text,
                speech_type=self.config.tts_engine,
                lang=self.config.tts_language,
                voice=self.config.tts_voice,
            )

            # Return TTS metadata
            return {
                "text": text,
                "audio_file": audio_file,
                "engine": self.config.tts_engine,
                "language": self.config.tts_language,
                "voice": self.config.tts_voice,
                "synthesis_time": "calculated_in_real_implementation",
                "estimated_duration": len(text) * 0.05,  # Rough estimate
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return {
                "text": text,
                "error": str(e),
                "synthesis_failed": True,
                "timestamp": datetime.now().isoformat(),
            }

    def _process_audio_output(
        self, tts_result: Dict[str, Any], state: GiannaState
    ) -> Dict[str, Any]:
        """Process audio output (placeholder for actual playback)."""
        # This would integrate with actual audio playback system
        return {
            "audio_played": not tts_result.get("synthesis_failed", False),
            "playback_duration": tts_result.get("estimated_duration", 0),
            "output_device": "default",
            "volume_level": 0.8,
            "playback_successful": True,
            "timestamp": datetime.now().isoformat(),
        }

    def _should_recover_from_error(self, state: GiannaState) -> str:
        """Determine if error recovery should be attempted."""
        error = state["metadata"].get("error")
        if error and self.config.enable_error_recovery:
            return "error"
        return "continue"

    def _handle_node_error(
        self, state: GiannaState, node_name: str, error: Exception
    ) -> GiannaState:
        """Handle errors that occur in workflow nodes."""
        state["metadata"]["error"] = str(error)
        state["metadata"]["error_node"] = node_name
        state["metadata"]["error_timestamp"] = datetime.now().isoformat()
        state["audio"].current_mode = VoiceProcessingState.ERROR.value
        return state

    def _trigger_callback(self, callback_name: str, *args, **kwargs):
        """Safely trigger callbacks with timeout protection."""
        if not self.config.enable_callbacks:
            return

        callback = self.callbacks.get(callback_name)
        if callback and callable(callback):
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Handle async callbacks
                    asyncio.create_task(
                        asyncio.wait_for(
                            callback(*args, **kwargs),
                            timeout=self.config.callback_timeout,
                        )
                    )
                else:
                    # Handle sync callbacks
                    callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback {callback_name} failed: {e}")

    # Public Interface Methods

    def set_callback(self, event_name: str, callback: Callable):
        """
        Set a callback function for workflow events.

        Args:
            event_name: Name of the event to listen for
            callback: Callback function to execute
        """
        if event_name in self.callbacks:
            self.callbacks[event_name] = callback
            logger.debug(f"Callback set for event: {event_name}")
        else:
            logger.warning(f"Unknown callback event: {event_name}")

    def execute(
        self,
        initial_state: Optional[GiannaState] = None,
        session_id: Optional[str] = None,
        audio_data: Optional[Dict[str, Any]] = None,
    ) -> GiannaState:
        """
        Execute the voice interaction workflow.

        Args:
            initial_state: Initial state for the workflow
            session_id: Session ID for state persistence
            audio_data: Audio data for processing

        Returns:
            Final state after workflow execution
        """
        if not self._compiled_graph:
            self.compile()

        try:
            # Prepare initial state
            if initial_state is None:
                initial_state = create_initial_state(session_id or str(uuid4()))

            # Add audio data if provided
            if audio_data:
                initial_state["metadata"]["audio_input"] = audio_data

            # Configure execution
            config = {}
            if session_id:
                config = self.state_manager.get_config(session_id)

            # Execute workflow
            result = self._compiled_graph.invoke(initial_state, config=config)

            logger.info("Voice interaction workflow executed successfully")
            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise VoiceInteractionError(f"Execution failed: {e}")

    async def execute_async(
        self,
        initial_state: Optional[GiannaState] = None,
        session_id: Optional[str] = None,
        audio_data: Optional[Dict[str, Any]] = None,
    ) -> GiannaState:
        """
        Execute the voice interaction workflow asynchronously.

        Args:
            initial_state: Initial state for the workflow
            session_id: Session ID for state persistence
            audio_data: Audio data for processing

        Returns:
            Final state after workflow execution
        """
        if not self.config.enable_async:
            raise VoiceInteractionError("Async execution not enabled in configuration")

        # Run the synchronous execution in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.execute, initial_state, session_id, audio_data
        )

    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow configuration and state.

        Returns:
            Dict containing workflow information
        """
        return {
            "name": self.config.name,
            "description": self.config.description,
            "config": {
                "stt_engine": self.config.stt_engine,
                "tts_engine": self.config.tts_engine,
                "use_react_agents": self.config.use_react_agents,
                "enable_async": self.config.enable_async,
                "enable_checkpointing": self.config.enable_checkpointing,
                "enable_error_recovery": self.config.enable_error_recovery,
            },
            "agents": {
                "available": list(self._agents.keys()),
                "count": len(self._agents),
            },
            "graph": {
                "compiled": self._compiled_graph is not None,
                "nodes": [
                    "voice_input",
                    "stt_processing",
                    "agent_processing",
                    "tts_synthesis",
                    "voice_output",
                ]
                + (["error_recovery"] if self.config.enable_error_recovery else []),
            },
        }


# Factory Functions


def create_voice_interaction_workflow(
    config: Optional[VoiceWorkflowConfig] = None,
    state_manager: Optional[StateManager] = None,
    llm_instance: Optional[Any] = None,
) -> VoiceInteractionWorkflow:
    """
    Factory function to create a voice interaction workflow.

    Args:
        config: Voice workflow configuration
        state_manager: State manager for persistence
        llm_instance: LLM instance for agent processing

    Returns:
        Configured VoiceInteractionWorkflow instance
    """
    workflow = VoiceInteractionWorkflow(
        config=config, state_manager=state_manager, llm_instance=llm_instance
    )
    workflow.compile()
    return workflow


def create_simple_voice_workflow(
    stt_engine: str = "whisper",
    tts_engine: str = "google",
    language: str = "pt-br",
    enable_agents: bool = True,
) -> VoiceInteractionWorkflow:
    """
    Create a simple voice interaction workflow with minimal configuration.

    Args:
        stt_engine: STT engine to use
        tts_engine: TTS engine to use
        language: Language for processing
        enable_agents: Whether to enable ReAct agents

    Returns:
        Configured VoiceInteractionWorkflow instance
    """
    config = VoiceWorkflowConfig(
        name="simple_voice_workflow",
        stt_engine=stt_engine,
        tts_engine=tts_engine,
        stt_language=language,
        tts_language=language,
        use_react_agents=enable_agents,
    )

    return create_voice_interaction_workflow(config=config)
