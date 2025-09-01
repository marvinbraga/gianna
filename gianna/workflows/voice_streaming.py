"""
Streaming Voice Workflow for Gianna AI Assistant

This module implements a real-time streaming voice workflow using LangGraph,
integrating with the existing VAD (Voice Activity Detection) and streaming
pipeline components to provide continuous voice interaction capabilities.

The streaming workflow manages the following states:
- idle: System ready for voice input
- listening: Actively listening for voice input
- processing: Processing captured voice and generating response
- speaking: Playing back synthesized response

Key features:
- Real-time voice activity detection integration
- Continuous streaming audio processing
- LangGraph-based state management
- Integration with existing StreamingVoicePipeline
- Asynchronous event-driven architecture
- ReAct agent integration for intelligent responses
- Multiple session support with state persistence
"""

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
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

from ..agents.react_agents import AudioAgent, ConversationAgent
from ..assistants.audio.vad import VoiceActivityDetector
from ..audio.streaming import AudioBuffer, PipelineState, StreamingVoicePipeline
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


class StreamingWorkflowError(Exception):
    """Base exception for streaming workflow errors."""

    pass


class StreamingState(Enum):
    """Streaming workflow states."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class StreamingEvent(Enum):
    """Streaming workflow events."""

    START_LISTENING = "start_listening"
    VOICE_DETECTED = "voice_detected"
    SPEECH_COMPLETED = "speech_completed"
    PROCESSING_STARTED = "processing_started"
    RESPONSE_GENERATED = "response_generated"
    TTS_STARTED = "tts_started"
    SPEAKING_STARTED = "speaking_started"
    SPEAKING_COMPLETED = "speaking_completed"
    ERROR_OCCURRED = "error_occurred"
    WORKFLOW_STOPPED = "workflow_stopped"


@dataclass
class StreamingWorkflowConfig:
    """Configuration for streaming voice workflow."""

    name: str = "streaming_voice_workflow"
    description: str = "Real-time streaming voice workflow with VAD integration"

    # Streaming configuration
    enable_continuous_listening: bool = True
    enable_interruption_detection: bool = True
    interruption_threshold: float = 0.3
    silence_timeout: float = 2.0
    max_session_duration: int = 3600  # seconds

    # Audio pipeline configuration
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    buffer_size: int = 1000

    # VAD configuration
    vad_threshold: float = 0.02
    min_silence_duration: float = 1.0
    voice_sensitivity: float = 0.5

    # STT configuration
    stt_engine: str = "whisper"
    stt_language: str = "pt-br"
    stt_local: bool = False

    # TTS configuration
    tts_engine: str = "google"
    tts_language: str = "pt-br"
    tts_voice: str = "default"

    # Agent configuration
    use_react_agents: bool = True
    agent_timeout: float = 10.0
    fallback_to_simple: bool = True

    # State management
    enable_state_persistence: bool = True
    checkpoint_interval: int = 10  # seconds

    # Event configuration
    enable_events: bool = True
    event_queue_size: int = 100
    event_timeout: float = 0.1

    # Performance configuration
    enable_async_processing: bool = True
    max_concurrent_sessions: int = 5
    processing_timeout: float = 30.0

    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingVoiceWorkflow:
    """
    Real-time streaming voice workflow using LangGraph and VAD integration.

    This class provides continuous voice interaction capabilities with
    real-time processing, state management, and event-driven architecture.
    """

    def __init__(
        self,
        config: Optional[StreamingWorkflowConfig] = None,
        state_manager: Optional[StateManager] = None,
        llm_instance: Optional[Any] = None,
    ):
        """
        Initialize the streaming voice workflow.

        Args:
            config: Streaming workflow configuration
            state_manager: State manager for persistence
            llm_instance: LLM instance for agent processing
        """
        if not LANGGRAPH_AVAILABLE:
            raise StreamingWorkflowError(
                "LangGraph is not available. Please install langgraph package."
            )

        self.config = config or StreamingWorkflowConfig()
        self.state_manager = state_manager or StateManager()
        self.llm_instance = llm_instance

        # Initialize workflow components
        self._graph = None
        self._compiled_graph = None
        self._agents = {}
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

        # Streaming components
        self._pipeline: Optional[StreamingVoicePipeline] = None
        self._vad: Optional[VoiceActivityDetector] = None

        # Threading and async
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._workflow_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._stop_requested = False

        # Event system
        self._event_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.event_queue_size
        )
        self._event_handlers: Dict[StreamingEvent, List[Callable]] = {}

        # State tracking
        self._current_state = StreamingState.IDLE
        self._state_lock = asyncio.Lock()

        # Session management
        self._session_states: Dict[str, GiannaState] = {}
        self._session_locks: Dict[str, asyncio.Lock] = {}

        self._initialize_components()
        self._build_streaming_workflow()

        logger.info(f"StreamingVoiceWorkflow initialized: {self.config.name}")

    def _initialize_components(self):
        """Initialize workflow components."""
        try:
            # Initialize ReAct agents if available
            self._initialize_agents()

            # Initialize VAD
            self._initialize_vad()

            # Initialize event handlers
            self._initialize_event_handlers()

            logger.info("Streaming workflow components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise StreamingWorkflowError(f"Component initialization failed: {e}")

    def _initialize_agents(self):
        """Initialize ReAct agents if available."""
        if not self.config.use_react_agents or not REACT_AVAILABLE:
            logger.info("ReAct agents not available or disabled")
            return

        if not self.llm_instance:
            logger.warning(
                "No LLM instance provided, ReAct agents will not be initialized"
            )
            return

        try:
            # Initialize conversation agent
            self._agents["conversation"] = ConversationAgent(self.llm_instance)

            # Initialize audio agent
            self._agents["audio"] = AudioAgent(self.llm_instance)

            logger.info(f"Initialized {len(self._agents)} ReAct agents for streaming")

        except Exception as e:
            logger.error(f"Failed to initialize ReAct agents: {e}")
            if not self.config.fallback_to_simple:
                raise

    def _initialize_vad(self):
        """Initialize Voice Activity Detection."""
        try:
            self._vad = VoiceActivityDetector(
                threshold=self.config.vad_threshold,
                min_silence_duration=self.config.min_silence_duration,
                sample_rate=self.config.sample_rate,
                chunk_size=self.config.chunk_size,
                speech_start_callback=self._on_vad_speech_start,
                speech_end_callback=self._on_vad_speech_end,
            )

            logger.info("VAD initialized for streaming workflow")

        except Exception as e:
            logger.error(f"Failed to initialize VAD: {e}")
            raise StreamingWorkflowError(f"VAD initialization failed: {e}")

    def _initialize_event_handlers(self):
        """Initialize default event handlers."""
        for event in StreamingEvent:
            self._event_handlers[event] = []

        # Register default handlers
        self.add_event_handler(
            StreamingEvent.VOICE_DETECTED, self._handle_voice_detected
        )
        self.add_event_handler(
            StreamingEvent.SPEECH_COMPLETED, self._handle_speech_completed
        )
        self.add_event_handler(
            StreamingEvent.RESPONSE_GENERATED, self._handle_response_generated
        )
        self.add_event_handler(
            StreamingEvent.ERROR_OCCURRED, self._handle_error_occurred
        )

    def _build_streaming_workflow(self):
        """Build the LangGraph streaming workflow."""
        try:
            # Create StateGraph
            self._graph = StateGraph(GiannaState)

            # Add streaming nodes
            self._graph.add_node("idle_state", self._idle_state_node)
            self._graph.add_node("listening_state", self._listening_state_node)
            self._graph.add_node("processing_state", self._processing_state_node)
            self._graph.add_node("speaking_state", self._speaking_state_node)
            self._graph.add_node("error_state", self._error_state_node)

            # Set up streaming edges
            self._setup_streaming_edges()

            logger.info("Streaming workflow graph built successfully")

        except Exception as e:
            logger.error(f"Failed to build streaming workflow: {e}")
            raise StreamingWorkflowError(f"Streaming workflow build failed: {e}")

    def _setup_streaming_edges(self):
        """Set up streaming workflow edges."""
        # Start in idle state
        self._graph.set_entry_point("idle_state")

        # Main state transitions
        self._graph.add_conditional_edges(
            "idle_state",
            self._route_from_idle,
            {"listening": "listening_state", "stopped": END, "error": "error_state"},
        )

        self._graph.add_conditional_edges(
            "listening_state",
            self._route_from_listening,
            {
                "processing": "processing_state",
                "idle": "idle_state",
                "error": "error_state",
            },
        )

        self._graph.add_conditional_edges(
            "processing_state",
            self._route_from_processing,
            {
                "speaking": "speaking_state",
                "listening": "listening_state",
                "error": "error_state",
            },
        )

        self._graph.add_conditional_edges(
            "speaking_state",
            self._route_from_speaking,
            {
                "listening": "listening_state",
                "idle": "idle_state",
                "error": "error_state",
            },
        )

        # Error state transitions
        self._graph.add_conditional_edges(
            "error_state",
            self._route_from_error,
            {"idle": "idle_state", "stopped": END},
        )

    def compile(self) -> Any:
        """Compile the streaming workflow graph."""
        if not self._graph:
            raise StreamingWorkflowError("Streaming workflow graph not built")

        try:
            if self.config.enable_state_persistence:
                self._compiled_graph = self._graph.compile(
                    checkpointer=self.state_manager.checkpointer
                )
            else:
                self._compiled_graph = self._graph.compile()

            logger.info("Streaming workflow compiled successfully")
            return self._compiled_graph

        except Exception as e:
            logger.error(f"Failed to compile streaming workflow: {e}")
            raise StreamingWorkflowError(f"Compilation failed: {e}")

    # Streaming Workflow Node Implementations

    def _idle_state_node(self, state: GiannaState) -> GiannaState:
        """Handle idle state - system ready for voice input."""
        try:
            logger.debug("Processing idle state")

            # Update audio state
            state["audio"].current_mode = StreamingState.IDLE.value

            # Check if we should start listening
            if self.config.enable_continuous_listening and self._is_running:
                state["metadata"]["next_action"] = "start_listening"
            else:
                state["metadata"]["next_action"] = "stay_idle"

            state["metadata"]["processing_stage"] = "idle_processed"
            state["metadata"]["timestamp"] = datetime.now().isoformat()

            # Emit event
            asyncio.create_task(
                self._emit_event(
                    StreamingEvent.START_LISTENING,
                    {"state": "idle", "session_id": state["conversation"].session_id},
                )
            )

            return state

        except Exception as e:
            logger.error(f"Idle state processing error: {e}")
            return self._handle_streaming_error(state, "idle_state", e)

    def _listening_state_node(self, state: GiannaState) -> GiannaState:
        """Handle listening state - actively listening for voice input."""
        try:
            logger.debug("Processing listening state")

            # Update audio state
            state["audio"].current_mode = StreamingState.LISTENING.value

            # Initialize streaming pipeline if needed
            if not self._pipeline:
                self._initialize_streaming_pipeline(state)

            # Check for voice activity
            voice_activity = state["metadata"].get("voice_activity", {})

            if voice_activity.get("speech_detected", False):
                state["metadata"]["next_action"] = "process_speech"
                state["metadata"]["captured_audio"] = voice_activity.get(
                    "audio_data", {}
                )
            elif voice_activity.get("silence_timeout", False):
                state["metadata"]["next_action"] = "return_to_idle"
            else:
                state["metadata"]["next_action"] = "continue_listening"

            state["metadata"]["processing_stage"] = "listening_processed"
            state["metadata"]["timestamp"] = datetime.now().isoformat()

            return state

        except Exception as e:
            logger.error(f"Listening state processing error: {e}")
            return self._handle_streaming_error(state, "listening_state", e)

    def _processing_state_node(self, state: GiannaState) -> GiannaState:
        """Handle processing state - processing captured speech."""
        try:
            logger.debug("Processing state - converting speech and generating response")

            # Update audio state
            state["audio"].current_mode = StreamingState.PROCESSING.value

            # Emit processing started event
            asyncio.create_task(
                self._emit_event(
                    StreamingEvent.PROCESSING_STARTED,
                    {"session_id": state["conversation"].session_id},
                )
            )

            # Get captured audio data
            audio_data = state["metadata"].get("captured_audio", {})

            # Process STT
            transcript = self._process_streaming_stt(audio_data, state)

            if not transcript:
                # No valid transcript, return to listening
                state["metadata"]["next_action"] = "return_to_listening"
                state["metadata"]["processing_result"] = "no_transcript"
                return state

            # Add user message
            user_message = {
                "role": "user",
                "content": transcript,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "source": "streaming_voice",
                    "stt_engine": self.config.stt_engine,
                    "confidence": audio_data.get("confidence", 0.9),
                },
            }
            state["conversation"].messages.append(user_message)

            # Process with agent
            response = self._process_with_streaming_agent(transcript, state)

            if response:
                # Add assistant response
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "source": "streaming_workflow",
                        "agent_used": "streaming_agent",
                        "requires_tts": True,
                    },
                }
                state["conversation"].messages.append(assistant_message)

                # Prepare for TTS
                state["metadata"]["tts_text"] = response
                state["metadata"]["next_action"] = "synthesize_and_speak"

                # Emit response generated event
                asyncio.create_task(
                    self._emit_event(
                        StreamingEvent.RESPONSE_GENERATED,
                        {
                            "response": response,
                            "session_id": state["conversation"].session_id,
                        },
                    )
                )
            else:
                # No response generated, return to listening
                state["metadata"]["next_action"] = "return_to_listening"

            state["metadata"]["processing_stage"] = "processing_completed"
            state["metadata"]["timestamp"] = datetime.now().isoformat()

            return state

        except Exception as e:
            logger.error(f"Processing state error: {e}")
            return self._handle_streaming_error(state, "processing_state", e)

    def _speaking_state_node(self, state: GiannaState) -> GiannaState:
        """Handle speaking state - outputting synthesized response."""
        try:
            logger.debug("Processing speaking state")

            # Update audio state
            state["audio"].current_mode = StreamingState.SPEAKING.value

            # Emit speaking started event
            asyncio.create_task(
                self._emit_event(
                    StreamingEvent.SPEAKING_STARTED,
                    {"session_id": state["conversation"].session_id},
                )
            )

            # Get text for TTS
            tts_text = state["metadata"].get("tts_text", "")

            if tts_text:
                # Synthesize and play speech
                tts_result = self._process_streaming_tts(tts_text, state)
                state["metadata"]["tts_result"] = tts_result

                # Play audio (integrated with streaming pipeline)
                self._play_streaming_audio(tts_result, state)

            # Determine next action
            if self.config.enable_continuous_listening:
                state["metadata"]["next_action"] = "return_to_listening"
            else:
                state["metadata"]["next_action"] = "return_to_idle"

            state["metadata"]["processing_stage"] = "speaking_completed"
            state["metadata"]["timestamp"] = datetime.now().isoformat()

            # Emit speaking completed event
            asyncio.create_task(
                self._emit_event(
                    StreamingEvent.SPEAKING_COMPLETED,
                    {"session_id": state["conversation"].session_id},
                )
            )

            return state

        except Exception as e:
            logger.error(f"Speaking state error: {e}")
            return self._handle_streaming_error(state, "speaking_state", e)

    def _error_state_node(self, state: GiannaState) -> GiannaState:
        """Handle error state - recover from errors."""
        try:
            logger.debug("Processing error state")

            # Update audio state
            state["audio"].current_mode = StreamingState.ERROR.value

            error = state["metadata"].get("error", "Unknown error")
            error_node = state["metadata"].get("error_node", "unknown")

            # Create error response
            error_response = {
                "role": "assistant",
                "content": "Desculpe, ocorreu um erro durante o processamento. Continuarei escutando.",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "error_recovery": True,
                    "original_error": error,
                    "error_node": error_node,
                },
            }

            state["conversation"].messages.append(error_response)

            # Determine recovery action
            if self.config.enable_continuous_listening:
                state["metadata"]["next_action"] = "recover_to_idle"
            else:
                state["metadata"]["next_action"] = "stop_workflow"

            state["metadata"]["processing_stage"] = "error_handled"
            state["metadata"]["error_handled"] = True
            state["metadata"]["timestamp"] = datetime.now().isoformat()

            # Emit error event
            asyncio.create_task(
                self._emit_event(
                    StreamingEvent.ERROR_OCCURRED,
                    {
                        "error": error,
                        "node": error_node,
                        "session_id": state["conversation"].session_id,
                    },
                )
            )

            return state

        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {recovery_error}")
            state["metadata"]["recovery_failed"] = True
            state["metadata"]["recovery_error"] = str(recovery_error)
            state["metadata"]["next_action"] = "stop_workflow"
            return state

    # Routing Functions for Conditional Edges

    def _route_from_idle(self, state: GiannaState) -> str:
        """Route from idle state based on next action."""
        next_action = state["metadata"].get("next_action", "stay_idle")

        if self._stop_requested:
            return "stopped"
        elif state["metadata"].get("error"):
            return "error"
        elif next_action == "start_listening":
            return "listening"
        else:
            return "stopped"  # If not starting listening, end workflow

    def _route_from_listening(self, state: GiannaState) -> str:
        """Route from listening state based on voice activity."""
        next_action = state["metadata"].get("next_action", "continue_listening")

        if state["metadata"].get("error"):
            return "error"
        elif next_action == "process_speech":
            return "processing"
        elif next_action == "return_to_idle":
            return "idle"
        else:
            # Continue listening - this would loop in a real streaming scenario
            return "idle"

    def _route_from_processing(self, state: GiannaState) -> str:
        """Route from processing state based on processing result."""
        next_action = state["metadata"].get("next_action", "return_to_listening")

        if state["metadata"].get("error"):
            return "error"
        elif next_action == "synthesize_and_speak":
            return "speaking"
        else:
            return "listening"

    def _route_from_speaking(self, state: GiannaState) -> str:
        """Route from speaking state based on continuation settings."""
        next_action = state["metadata"].get("next_action", "return_to_idle")

        if state["metadata"].get("error"):
            return "error"
        elif next_action == "return_to_listening":
            return "listening"
        else:
            return "idle"

    def _route_from_error(self, state: GiannaState) -> str:
        """Route from error state based on recovery action."""
        next_action = state["metadata"].get("next_action", "stop_workflow")

        if next_action == "recover_to_idle":
            return "idle"
        else:
            return "stopped"

    # Streaming Processing Methods

    def _initialize_streaming_pipeline(self, state: GiannaState):
        """Initialize the streaming voice pipeline."""
        try:
            if self._pipeline:
                return  # Already initialized

            self._pipeline = StreamingVoicePipeline(
                model_name="gpt35",  # Could be configurable
                system_prompt="Você é um assistente de voz em tempo real. Seja conciso e útil.",
                sample_rate=self.config.sample_rate,
                chunk_size=self.config.chunk_size,
                channels=self.config.channels,
                vad_threshold=self.config.vad_threshold,
                min_silence_duration=self.config.min_silence_duration,
                buffer_max_chunks=self.config.buffer_size,
                tts_type=self.config.tts_engine,
                tts_language=self.config.tts_language,
                # Callbacks
                on_listening=lambda: self._on_pipeline_listening(),
                on_speech_detected=lambda transcript: self._on_pipeline_speech(
                    transcript
                ),
                on_processing=lambda: self._on_pipeline_processing(),
                on_response=lambda response: self._on_pipeline_response(response),
                on_speaking=lambda: self._on_pipeline_speaking(),
                on_error=lambda error: self._on_pipeline_error(error),
            )

            logger.info("Streaming pipeline initialized")

        except Exception as e:
            logger.error(f"Failed to initialize streaming pipeline: {e}")
            raise

    def _process_streaming_stt(
        self, audio_data: Dict[str, Any], state: GiannaState
    ) -> str:
        """Process STT for streaming audio."""
        try:
            # In a real implementation, this would process the actual audio data
            # For now, simulate STT processing
            return audio_data.get("transcript", "")

        except Exception as e:
            logger.error(f"Streaming STT processing failed: {e}")
            return ""

    def _process_with_streaming_agent(self, text: str, state: GiannaState) -> str:
        """Process input with streaming-optimized agent."""
        try:
            # Select appropriate agent
            agent = self._select_streaming_agent(text, state)

            if agent:
                # Process with ReAct agent
                result = agent.execute({"input": text}, state)

                if isinstance(result, dict):
                    return result.get("content", result.get("response", ""))
                elif hasattr(result, "content"):
                    return result.content
                else:
                    return str(result)
            else:
                # Fallback response
                return f"Entendi: {text}. Como posso ajudar?"

        except Exception as e:
            logger.error(f"Streaming agent processing failed: {e}")
            return "Desculpe, ocorreu um erro ao processar sua mensagem."

    def _select_streaming_agent(self, text: str, state: GiannaState) -> Optional[Any]:
        """Select appropriate agent for streaming processing."""
        if not self._agents:
            return None

        # Simple selection logic - could be enhanced
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in ["áudio", "voz", "som", "música"]):
            return self._agents.get("audio")
        else:
            return self._agents.get("conversation")

    def _process_streaming_tts(self, text: str, state: GiannaState) -> Dict[str, Any]:
        """Process TTS for streaming output."""
        try:
            # Use existing TTS system
            from ..assistants.audio.tts.factory_method import text_to_speech

            audio_file = text_to_speech(
                text=text,
                speech_type=self.config.tts_engine,
                lang=self.config.tts_language,
                voice=self.config.tts_voice,
            )

            return {
                "text": text,
                "audio_file": audio_file,
                "engine": self.config.tts_engine,
                "language": self.config.tts_language,
                "synthesis_time": time.time(),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Streaming TTS processing failed: {e}")
            return {"text": text, "error": str(e), "success": False}

    def _play_streaming_audio(self, tts_result: Dict[str, Any], state: GiannaState):
        """Play synthesized audio through streaming pipeline."""
        try:
            if tts_result.get("success", False):
                # In a real implementation, this would integrate with the audio output system
                logger.debug(
                    f"Playing streamed audio: {tts_result.get('text', '')[:50]}..."
                )
            else:
                logger.warning("TTS synthesis failed, cannot play audio")

        except Exception as e:
            logger.error(f"Streaming audio playback failed: {e}")

    # VAD Callback Handlers

    def _on_vad_speech_start(self):
        """Handle VAD speech start detection."""
        logger.debug("VAD detected speech start")
        asyncio.create_task(
            self._emit_event(StreamingEvent.VOICE_DETECTED, {"event": "speech_start"})
        )

    def _on_vad_speech_end(self):
        """Handle VAD speech end detection."""
        logger.debug("VAD detected speech end")
        asyncio.create_task(
            self._emit_event(StreamingEvent.SPEECH_COMPLETED, {"event": "speech_end"})
        )

    # Pipeline Callback Handlers

    def _on_pipeline_listening(self):
        """Handle pipeline listening callback."""
        logger.debug("Pipeline listening state")

    def _on_pipeline_speech(self, transcript: str):
        """Handle pipeline speech detection."""
        logger.debug(f"Pipeline detected speech: {transcript[:50]}...")

    def _on_pipeline_processing(self):
        """Handle pipeline processing callback."""
        logger.debug("Pipeline processing")

    def _on_pipeline_response(self, response: str):
        """Handle pipeline response callback."""
        logger.debug(f"Pipeline response: {response[:50]}...")

    def _on_pipeline_speaking(self):
        """Handle pipeline speaking callback."""
        logger.debug("Pipeline speaking")

    def _on_pipeline_error(self, error: Exception):
        """Handle pipeline error callback."""
        logger.error(f"Pipeline error: {error}")
        asyncio.create_task(
            self._emit_event(
                StreamingEvent.ERROR_OCCURRED,
                {"error": str(error), "source": "pipeline"},
            )
        )

    # Event System

    async def _emit_event(self, event: StreamingEvent, data: Dict[str, Any] = None):
        """Emit a workflow event."""
        if not self.config.enable_events:
            return

        try:
            event_data = {
                "event": event.value,
                "timestamp": datetime.now().isoformat(),
                "data": data or {},
            }

            await asyncio.wait_for(
                self._event_queue.put(event_data), timeout=self.config.event_timeout
            )

            # Call registered handlers
            handlers = self._event_handlers.get(event, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data)
                    else:
                        handler(event_data)
                except Exception as e:
                    logger.warning(f"Event handler failed for {event.value}: {e}")

        except asyncio.TimeoutError:
            logger.warning(f"Event queue timeout for {event.value}")
        except Exception as e:
            logger.error(f"Failed to emit event {event.value}: {e}")

    def add_event_handler(self, event: StreamingEvent, handler: Callable):
        """Add an event handler for a specific event."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []

        self._event_handlers[event].append(handler)
        logger.debug(f"Added event handler for {event.value}")

    def remove_event_handler(self, event: StreamingEvent, handler: Callable):
        """Remove an event handler."""
        if event in self._event_handlers:
            try:
                self._event_handlers[event].remove(handler)
                logger.debug(f"Removed event handler for {event.value}")
            except ValueError:
                logger.warning(f"Handler not found for {event.value}")

    # Default Event Handlers

    async def _handle_voice_detected(self, event_data: Dict[str, Any]):
        """Handle voice detected event."""
        logger.debug("Handling voice detected event")

    async def _handle_speech_completed(self, event_data: Dict[str, Any]):
        """Handle speech completed event."""
        logger.debug("Handling speech completed event")

    async def _handle_response_generated(self, event_data: Dict[str, Any]):
        """Handle response generated event."""
        response = event_data.get("data", {}).get("response", "")
        logger.debug(f"Handling response generated: {response[:50]}...")

    async def _handle_error_occurred(self, event_data: Dict[str, Any]):
        """Handle error occurred event."""
        error = event_data.get("data", {}).get("error", "Unknown error")
        logger.warning(f"Handling error event: {error}")

    # Utility Methods

    def _handle_streaming_error(
        self, state: GiannaState, node_name: str, error: Exception
    ) -> GiannaState:
        """Handle errors in streaming workflow nodes."""
        state["metadata"]["error"] = str(error)
        state["metadata"]["error_node"] = node_name
        state["metadata"]["error_timestamp"] = datetime.now().isoformat()
        state["audio"].current_mode = StreamingState.ERROR.value
        return state

    async def _get_session_state(self, session_id: str) -> GiannaState:
        """Get or create session state."""
        if session_id not in self._session_states:
            # Create new session state
            self._session_states[session_id] = create_initial_state(session_id)
            self._session_locks[session_id] = asyncio.Lock()

        return self._session_states[session_id]

    async def _update_session_state(self, session_id: str, state: GiannaState):
        """Update session state."""
        async with self._session_locks.get(session_id, asyncio.Lock()):
            self._session_states[session_id] = state

            # Save to persistent storage if enabled
            if self.config.enable_state_persistence:
                try:
                    self.state_manager.save_state(session_id, state)
                except Exception as e:
                    logger.warning(f"Failed to save session state: {e}")

    # Public Interface

    async def start_streaming(self, session_id: Optional[str] = None) -> str:
        """
        Start the streaming voice workflow.

        Args:
            session_id: Optional session ID for state persistence

        Returns:
            Session ID for the started workflow
        """
        if not self._compiled_graph:
            self.compile()

        if self._is_running:
            raise StreamingWorkflowError("Streaming workflow is already running")

        try:
            # Create or get session
            session_id = session_id or str(uuid4())
            initial_state = await self._get_session_state(session_id)

            # Set running flags
            self._is_running = True
            self._stop_requested = False

            # Get event loop
            self._event_loop = asyncio.get_event_loop()

            # Start workflow task
            config = (
                self.state_manager.get_config(session_id)
                if self.config.enable_state_persistence
                else {}
            )

            async def workflow_runner():
                try:
                    result = self._compiled_graph.invoke(initial_state, config=config)
                    await self._update_session_state(session_id, result)
                    return result
                except Exception as e:
                    logger.error(f"Workflow execution failed: {e}")
                    raise

            self._workflow_task = asyncio.create_task(workflow_runner())

            logger.info(f"Streaming workflow started for session: {session_id}")
            return session_id

        except Exception as e:
            self._is_running = False
            logger.error(f"Failed to start streaming workflow: {e}")
            raise StreamingWorkflowError(f"Start failed: {e}")

    async def stop_streaming(self, session_id: Optional[str] = None):
        """Stop the streaming voice workflow."""
        try:
            self._stop_requested = True

            if self._workflow_task and not self._workflow_task.done():
                self._workflow_task.cancel()
                try:
                    await self._workflow_task
                except asyncio.CancelledError:
                    pass

            if self._pipeline:
                await self._pipeline.stop_listening()
                self._pipeline = None

            self._is_running = False

            # Emit stop event
            await self._emit_event(
                StreamingEvent.WORKFLOW_STOPPED, {"session_id": session_id}
            )

            logger.info("Streaming workflow stopped")

        except Exception as e:
            logger.error(f"Error stopping streaming workflow: {e}")

    async def pause_streaming(self, session_id: Optional[str] = None):
        """Pause the streaming workflow."""
        try:
            if self._pipeline:
                self._pipeline.pause_listening()

            # Update state
            if session_id:
                state = await self._get_session_state(session_id)
                state["audio"].current_mode = StreamingState.PAUSED.value
                await self._update_session_state(session_id, state)

            logger.info("Streaming workflow paused")

        except Exception as e:
            logger.error(f"Error pausing streaming workflow: {e}")

    async def resume_streaming(self, session_id: Optional[str] = None):
        """Resume the streaming workflow."""
        try:
            if self._pipeline:
                self._pipeline.resume_listening()

            # Update state
            if session_id:
                state = await self._get_session_state(session_id)
                state["audio"].current_mode = StreamingState.LISTENING.value
                await self._update_session_state(session_id, state)

            logger.info("Streaming workflow resumed")

        except Exception as e:
            logger.error(f"Error resuming streaming workflow: {e}")

    async def get_event_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Get an async generator for workflow events."""
        while self._is_running or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=self.config.event_timeout
                )
                yield event
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error getting event: {e}")
                break

    def get_streaming_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current streaming workflow status."""
        return {
            "is_running": self._is_running,
            "current_state": self._current_state.value,
            "active_sessions": len(self._session_states),
            "config": {
                "continuous_listening": self.config.enable_continuous_listening,
                "vad_threshold": self.config.vad_threshold,
                "stt_engine": self.config.stt_engine,
                "tts_engine": self.config.tts_engine,
            },
            "pipeline": {
                "initialized": self._pipeline is not None,
                "status": (
                    self._pipeline.get_pipeline_status() if self._pipeline else None
                ),
            },
            "events": {
                "queue_size": self._event_queue.qsize(),
                "handlers_count": sum(
                    len(handlers) for handlers in self._event_handlers.values()
                ),
            },
        }


# Factory Functions


def create_streaming_voice_workflow(
    config: Optional[StreamingWorkflowConfig] = None,
    state_manager: Optional[StateManager] = None,
    llm_instance: Optional[Any] = None,
) -> StreamingVoiceWorkflow:
    """
    Factory function to create a streaming voice workflow.

    Args:
        config: Streaming workflow configuration
        state_manager: State manager for persistence
        llm_instance: LLM instance for agent processing

    Returns:
        Configured StreamingVoiceWorkflow instance
    """
    workflow = StreamingVoiceWorkflow(
        config=config, state_manager=state_manager, llm_instance=llm_instance
    )
    workflow.compile()
    return workflow


def create_simple_streaming_workflow(
    enable_continuous: bool = True, vad_threshold: float = 0.02, language: str = "pt-br"
) -> StreamingVoiceWorkflow:
    """
    Create a simple streaming voice workflow with minimal configuration.

    Args:
        enable_continuous: Whether to enable continuous listening
        vad_threshold: Voice activity detection threshold
        language: Language for STT/TTS processing

    Returns:
        Configured StreamingVoiceWorkflow instance
    """
    config = StreamingWorkflowConfig(
        name="simple_streaming_workflow",
        enable_continuous_listening=enable_continuous,
        vad_threshold=vad_threshold,
        stt_language=language,
        tts_language=language,
    )

    return create_streaming_voice_workflow(config=config)
