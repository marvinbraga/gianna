"""
Streaming Audio Pipeline for Gianna Assistant

This module provides real-time voice streaming capabilities with integrated
Voice Activity Detection, Speech-to-Text, LLM processing, and Text-to-Speech.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np
import pyaudio

from gianna.assistants.audio.stt.factory_method import speech_to_text
from gianna.assistants.audio.tts.factory_method import text_to_speech
from gianna.assistants.models.factory_method import get_chain_instance

# Import new VAD system with fallback to legacy
try:
    from gianna.audio.vad import BaseVAD, VADConfig, create_vad

    _NEW_VAD_AVAILABLE = True
except ImportError:
    _NEW_VAD_AVAILABLE = False

# Fallback to legacy VAD if new system is not available
if not _NEW_VAD_AVAILABLE:
    try:
        from gianna.assistants.audio.vad import VoiceActivityDetector

        _LEGACY_VAD_AVAILABLE = True
    except ImportError:
        _LEGACY_VAD_AVAILABLE = False

        # Create a minimal VAD stub
        class VoiceActivityDetector:
            def __init__(self, **kwargs):
                logger.warning("No VAD system available - using stub implementation")
                self.threshold = kwargs.get("threshold", 0.02)

            def process_stream(self, audio_chunk):
                return {
                    "is_speaking": False,
                    "is_voice_active": False,
                    "energy": 0.0,
                    "threshold": self.threshold,
                    "state_changed": False,
                    "event_type": None,
                    "timestamp": time.time(),
                }

            def set_threshold(self, threshold):
                self.threshold = threshold

            def set_min_silence_duration(self, duration):
                pass


logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline state enumeration."""

    STOPPED = "stopped"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


class AudioBuffer:
    """Thread-safe circular audio buffer for streaming."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize the audio buffer.

        Args:
            max_size (int): Maximum number of chunks to store
        """
        self.max_size = max_size
        self.buffer: Deque[np.ndarray] = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self._total_samples = 0

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Add audio chunk to buffer."""
        with self.lock:
            self.buffer.append(chunk)
            self._total_samples += len(chunk)

    def get_chunks(self, count: Optional[int] = None) -> List[np.ndarray]:
        """Get chunks from buffer."""
        with self.lock:
            if count is None:
                return list(self.buffer)
            return list(self.buffer)[-count:]

    def clear(self) -> None:
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()
            self._total_samples = 0

    def get_combined_audio(self) -> np.ndarray:
        """Get all buffer chunks combined as single array."""
        with self.lock:
            if not self.buffer:
                return np.array([], dtype=np.int16)
            return np.concatenate(list(self.buffer))

    @property
    def size(self) -> int:
        """Get number of chunks in buffer."""
        with self.lock:
            return len(self.buffer)

    @property
    def total_samples(self) -> int:
        """Get total number of audio samples."""
        with self.lock:
            return self._total_samples


class StreamingVoicePipeline:
    """
    Real-time voice streaming pipeline with VAD, STT, LLM, and TTS integration.

    Pipeline flow:
    Audio Input → VAD → Buffer → STT → LLM → TTS → Audio Output
    """

    def __init__(
        self,
        model_name: str = "gpt35",
        system_prompt: str = (
            "You are a helpful voice assistant. Provide concise responses."
        ),
        # Audio parameters
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        # VAD parameters - unified interface
        vad_algorithm: str = "energy",  # New parameter for VAD algorithm selection
        vad_threshold: float = 0.02,
        min_silence_duration: float = 1.0,
        min_speech_duration: float = 0.1,  # New parameter
        vad_config: Optional[
            VADConfig
        ] = None,  # New parameter for advanced configuration
        # Legacy VAD parameters (for backward compatibility)
        vad_instance: Optional = None,  # Allow passing custom VAD instance
        # Buffer parameters
        buffer_max_chunks: int = 1000,
        # TTS parameters
        tts_type: str = "google",
        tts_language: str = "en",
        tts_voice: str = "default",
        # Callbacks
        on_listening: Optional[Callable[[], None]] = None,
        on_speech_detected: Optional[Callable[[str], None]] = None,
        on_processing: Optional[Callable[[], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_speaking: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        **kwargs,
    ):
        """
        Initialize the streaming voice pipeline.

        Args:
            model_name (str): LLM model name for processing
            system_prompt (str): System prompt for the LLM
            sample_rate (int): Audio sample rate in Hz
            chunk_size (int): Audio chunk size for processing
            channels (int): Number of audio channels (1 for mono)
            vad_threshold (float): Voice activity detection threshold
            min_silence_duration (float): Minimum silence duration to end speech
            buffer_max_chunks (int): Maximum chunks in audio buffer
            tts_type (str): Text-to-speech engine type
            tts_language (str): TTS language code
            tts_voice (str): TTS voice identifier
            on_listening (Callable): Callback when starting to listen
            on_speech_detected (Callable): Callback when speech is detected
            on_processing (Callable): Callback when processing LLM request
            on_response (Callable): Callback when LLM response is received
            on_speaking (Callable): Callback when starting TTS playback
            on_error (Callable): Callback for error handling
        """
        # Audio configuration
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_format = pyaudio.paInt16

        # Pipeline components - Initialize VAD with new factory system
        self.vad = self._create_vad_instance(
            vad_algorithm=vad_algorithm,
            vad_threshold=vad_threshold,
            min_silence_duration=min_silence_duration,
            min_speech_duration=min_speech_duration,
            vad_config=vad_config,
            vad_instance=vad_instance,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            **kwargs,
        )

        self.audio_buffer = AudioBuffer(max_size=buffer_max_chunks)

        # LLM chain
        self.llm_chain = get_chain_instance(model_name, system_prompt)

        # TTS configuration
        self.tts_type = tts_type
        self.tts_language = tts_language
        self.tts_voice = tts_voice

        # State management
        self.state = PipelineState.STOPPED
        self.state_lock = threading.RLock()

        # Threading
        self.audio_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # PyAudio instance
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.audio_stream: Optional[pyaudio.Stream] = None

        # Processing queues
        self.speech_queue: Queue = Queue()
        self.response_queue: Queue = Queue()

        # Control flags
        self._running = False
        self._stop_requested = False

        # Callbacks
        self.on_listening = on_listening
        self.on_speech_detected = on_speech_detected
        self.on_processing = on_processing
        self.on_response = on_response
        self.on_speaking = on_speaking
        self.on_error = on_error

        # Session management
        self.session_id = str(uuid.uuid4())

        # Statistics
        self.stats = {
            "chunks_processed": 0,
            "speech_detected_count": 0,
            "llm_requests": 0,
            "tts_synthesized": 0,
            "errors": 0,
            "session_start_time": None,
            "last_activity_time": None,
        }

        logger.info(
            f"StreamingVoicePipeline initialized with session {self.session_id}"
        )

    def _create_vad_instance(
        self,
        vad_algorithm: str,
        vad_threshold: float,
        min_silence_duration: float,
        min_speech_duration: float,
        vad_config: Optional,
        vad_instance: Optional,
        sample_rate: int,
        chunk_size: int,
        **kwargs,
    ):
        """
        Create VAD instance using new factory system with fallback to legacy.

        Args:
            vad_algorithm (str): VAD algorithm to use
            vad_threshold (float): Detection threshold
            min_silence_duration (float): Minimum silence duration
            min_speech_duration (float): Minimum speech duration
            vad_config: Advanced VAD configuration
            vad_instance: Pre-configured VAD instance
            sample_rate (int): Audio sample rate
            chunk_size (int): Audio chunk size
            **kwargs: Additional parameters

        Returns:
            VAD instance (new or legacy)
        """
        # If a custom VAD instance is provided, use it
        if vad_instance is not None:
            logger.info("Using provided VAD instance")
            return vad_instance

        # Try to use new VAD factory system
        if _NEW_VAD_AVAILABLE:
            try:
                # Create VAD config
                if vad_config is None:
                    vad_config = VADConfig(
                        threshold=vad_threshold,
                        min_silence_duration=min_silence_duration,
                        min_speech_duration=min_speech_duration,
                        sample_rate=sample_rate,
                        chunk_size=chunk_size,
                    )

                # Create VAD instance using factory
                vad = create_vad(vad_algorithm, config=vad_config, **kwargs)

                # Set up callbacks for the new VAD system
                vad.set_speech_start_callback(lambda result: self._on_speech_start())
                vad.set_speech_end_callback(lambda result: self._on_speech_end())

                logger.info(f"Created {vad_algorithm} VAD using new factory system")
                return vad

            except Exception as e:
                logger.warning(f"Failed to create new VAD system: {e}")
                logger.info("Falling back to legacy VAD system")

        # Fallback to legacy VAD system
        if _LEGACY_VAD_AVAILABLE or hasattr(self, "VoiceActivityDetector"):
            logger.info("Using legacy VAD system")
            return VoiceActivityDetector(
                threshold=vad_threshold,
                min_silence_duration=min_silence_duration,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                speech_start_callback=self._on_speech_start,
                speech_end_callback=self._on_speech_end,
            )
        else:
            # Use stub implementation
            logger.warning("No VAD system available - using minimal stub")
            return VoiceActivityDetector(
                threshold=vad_threshold,
                min_silence_duration=min_silence_duration,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
            )

    def _process_vad_chunk(self, audio_chunk):
        """
        Process audio chunk with VAD, handling both new and legacy interfaces.

        Args:
            audio_chunk: Audio data to process

        Returns:
            Dictionary with VAD results in a normalized format
        """
        try:
            # Check if we're using the new VAD system
            if _NEW_VAD_AVAILABLE and isinstance(self.vad, BaseVAD):
                # New VAD system - returns VADResult object
                result = self.vad.process_stream(audio_chunk)

                # Convert to dictionary format for compatibility
                return {
                    "is_speaking": result.is_speaking,
                    "is_voice_active": result.is_voice_active,
                    "energy": result.energy_level,
                    "threshold": result.threshold_used,
                    "confidence": result.confidence,
                    "state_changed": result.state_changed,
                    "event_type": (
                        result.event_type.value if result.event_type else None
                    ),
                    "timestamp": result.timestamp,
                    "processing_time": result.processing_time,
                }
            else:
                # Legacy VAD system - returns dictionary
                result = self.vad.process_stream(audio_chunk)

                # Ensure all expected keys are present
                normalized_result = {
                    "is_speaking": result.get("is_speaking", False),
                    "is_voice_active": result.get("is_voice_active", False),
                    "energy": result.get("energy", 0.0),
                    "threshold": result.get("threshold", 0.02),
                    "confidence": result.get("confidence", 0.0),
                    "state_changed": result.get("state_changed", False),
                    "event_type": result.get("event_type"),
                    "timestamp": result.get("timestamp", time.time()),
                    "processing_time": result.get("processing_time", 0.0),
                }

                return normalized_result

        except Exception as e:
            logger.error(f"Error processing VAD chunk: {e}")
            # Return safe defaults
            return {
                "is_speaking": False,
                "is_voice_active": False,
                "energy": 0.0,
                "threshold": 0.02,
                "confidence": 0.0,
                "state_changed": False,
                "event_type": None,
                "timestamp": time.time(),
                "processing_time": 0.0,
            }

    def _set_state(self, new_state: PipelineState) -> None:
        """Thread-safe state update."""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            logger.debug(f"Pipeline state: {old_state} → {new_state}")

    def _trigger_callback(self, callback: Optional[Callable], *args, **kwargs) -> None:
        """Safely trigger callback with error handling."""
        if callback:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback {callback.__name__}: {e}")
                self._handle_error(e)

    def _handle_error(self, error: Exception) -> None:
        """Handle pipeline errors."""
        logger.error(f"Pipeline error: {error}")
        self.stats["errors"] += 1
        self._set_state(PipelineState.ERROR)
        self._trigger_callback(self.on_error, error)

    def _on_speech_start(self) -> None:
        """Callback when VAD detects speech start."""
        logger.info("Speech started - clearing buffer and preparing to capture")
        self.audio_buffer.clear()
        self._set_state(PipelineState.LISTENING)
        self._trigger_callback(self.on_listening)

    def _on_speech_end(self) -> None:
        """Callback when VAD detects speech end."""
        logger.info("Speech ended - processing captured audio")
        self.stats["speech_detected_count"] += 1

        # Get captured audio
        audio_data = self.audio_buffer.get_combined_audio()

        if len(audio_data) > 0:
            # Queue for STT processing
            self.speech_queue.put(audio_data)
            self._set_state(PipelineState.PROCESSING)
        else:
            logger.warning("No audio data captured")
            self._set_state(PipelineState.LISTENING)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback for real-time audio processing."""
        if status:
            logger.warning(f"Audio stream status: {status}")

        try:
            # Convert to numpy array
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)

            # Process with VAD (handle both new and legacy interfaces)
            vad_result = self._process_vad_chunk(audio_chunk)
            self.stats["chunks_processed"] += 1
            self.stats["last_activity_time"] = time.time()

            # Add to buffer if we're in a speech segment
            if vad_result.get("is_speaking", False) or vad_result.get(
                "is_voice_active", False
            ):
                self.audio_buffer.add_chunk(audio_chunk)

        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            self._handle_error(e)

        return (None, pyaudio.paContinue)

    def _setup_audio_stream(self) -> None:
        """Setup PyAudio stream for real-time capture."""
        try:
            self.audio_interface = pyaudio.PyAudio()

            self.audio_stream = self.audio_interface.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
                start=False,
            )

            logger.info("Audio stream setup complete")

        except Exception as e:
            logger.error(f"Failed to setup audio stream: {e}")
            raise

    def _cleanup_audio_stream(self) -> None:
        """Cleanup PyAudio resources."""
        try:
            if self.audio_stream:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None

            if self.audio_interface:
                self.audio_interface.terminate()
                self.audio_interface = None

            logger.info("Audio stream cleanup complete")

        except Exception as e:
            logger.error(f"Error during audio cleanup: {e}")

    def _process_speech_to_text(self, audio_data: np.ndarray) -> Optional[str]:
        """Process audio data to text using STT."""
        try:
            # Save audio to temporary file for STT processing
            temp_file = Path(
                f"/tmp/gianna_temp_{self.session_id}_{int(time.time())}.wav"
            )

            # Create a simple WAV file writer
            import wave

            with wave.open(str(temp_file), "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            # Use existing STT system
            documents = speech_to_text(
                str(temp_file.parent), filetype="wav", local=False
            )

            # Clean up temp file
            temp_file.unlink(missing_ok=True)

            if documents and len(documents) > 0:
                transcript = " ".join([doc.page_content for doc in documents]).strip()
                logger.info(f"STT result: {transcript}")
                return transcript
            else:
                logger.warning("STT returned no results")
                return None

        except Exception as e:
            logger.error(f"STT processing error: {e}")
            self._handle_error(e)
            return None

    def _process_llm_request(self, text: str) -> Optional[str]:
        """Process text through LLM and return response."""
        try:
            self.stats["llm_requests"] += 1
            logger.info(f"Processing LLM request: {text}")

            # Use the existing LLM chain
            response = self.llm_chain.invoke({"input": text})

            # Extract text from response
            if hasattr(response, "content"):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            elif isinstance(response, dict) and "output" in response:
                response_text = response["output"]
            else:
                response_text = str(response)

            logger.info(f"LLM response: {response_text}")
            return response_text.strip()

        except Exception as e:
            logger.error(f"LLM processing error: {e}")
            self._handle_error(e)
            return None

    def _synthesize_speech(self, text: str) -> None:
        """Synthesize and play speech response."""
        try:
            self.stats["tts_synthesized"] += 1
            logger.info(f"Synthesizing speech: {text}")

            # Use existing TTS system
            text_to_speech(
                text=text,
                speech_type=self.tts_type,
                lang=self.tts_language,
                voice=self.tts_voice,
            )

            logger.info("Speech synthesis complete")

        except Exception as e:
            logger.error(f"TTS processing error: {e}")
            self._handle_error(e)

    def _processing_worker(self) -> None:
        """Background worker for processing speech and generating responses."""
        logger.info("Processing worker started")

        while self._running:
            try:
                # Check for speech to process
                try:
                    audio_data = self.speech_queue.get(timeout=0.1)

                    self._trigger_callback(self.on_processing)

                    # Step 1: Speech to Text
                    transcript = self._process_speech_to_text(audio_data)

                    if transcript:
                        self._trigger_callback(self.on_speech_detected, transcript)

                        # Step 2: LLM Processing
                        response = self._process_llm_request(transcript)

                        if response:
                            self._trigger_callback(self.on_response, response)

                            # Queue for TTS
                            self.response_queue.put(response)

                    # Reset to listening state
                    self._set_state(PipelineState.LISTENING)

                except Empty:
                    # No speech to process, check for TTS
                    pass

                # Check for TTS to process
                try:
                    response_text = self.response_queue.get(timeout=0.1)

                    self._set_state(PipelineState.SPEAKING)
                    self._trigger_callback(self.on_speaking)

                    # Step 3: Text to Speech
                    self._synthesize_speech(response_text)

                    # Back to listening
                    self._set_state(PipelineState.LISTENING)

                except Empty:
                    pass

            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
                self._handle_error(e)
                time.sleep(0.1)  # Brief pause to prevent tight error loop

        logger.info("Processing worker stopped")

    async def start_listening(self) -> None:
        """Start the streaming voice pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return

        try:
            logger.info("Starting streaming voice pipeline")
            self.stats["session_start_time"] = time.time()

            # Setup audio stream
            self._setup_audio_stream()

            # Start processing worker
            self._running = True
            self._stop_requested = False

            self.processing_thread = threading.Thread(
                target=self._processing_worker, name=f"GiannaPipeline-{self.session_id}"
            )
            self.processing_thread.start()

            # Start audio stream
            if self.audio_stream:
                self.audio_stream.start_stream()
                logger.info("Audio stream started")

            self._set_state(PipelineState.LISTENING)
            logger.info("Streaming pipeline is now active")

        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            await self.stop_listening()
            raise

    async def stop_listening(self) -> None:
        """Stop the streaming voice pipeline."""
        logger.info("Stopping streaming voice pipeline")

        # Signal stop
        self._running = False
        self._stop_requested = True

        # Stop audio stream
        self._cleanup_audio_stream()

        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        # Cleanup executor
        self.executor.shutdown(wait=True)

        self._set_state(PipelineState.STOPPED)
        logger.info("Streaming pipeline stopped")

    def pause_listening(self) -> None:
        """Temporarily pause audio capture."""
        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
            self._set_state(PipelineState.STOPPED)
            logger.info("Audio capture paused")

    def resume_listening(self) -> None:
        """Resume audio capture."""
        if self.audio_stream and not self.audio_stream.is_active():
            self.audio_stream.start_stream()
            self._set_state(PipelineState.LISTENING)
            logger.info("Audio capture resumed")

    def update_vad_settings(
        self,
        threshold: Optional[float] = None,
        min_silence_duration: Optional[float] = None,
    ) -> None:
        """Update VAD settings during runtime, handling both new and legacy VAD systems."""
        try:
            # Handle new VAD system
            if _NEW_VAD_AVAILABLE and isinstance(self.vad, BaseVAD):
                if threshold is not None:
                    self.vad.set_threshold(threshold)

                # For new VAD system, config updates might be different
                if min_silence_duration is not None:
                    # Update configuration
                    new_config = VADConfig(
                        threshold=(
                            threshold
                            if threshold is not None
                            else self.vad.config.threshold
                        ),
                        min_silence_duration=min_silence_duration,
                        min_speech_duration=self.vad.config.min_speech_duration,
                        sample_rate=self.vad.config.sample_rate,
                        chunk_size=self.vad.config.chunk_size,
                        channels=self.vad.config.channels,
                        algorithm_params=self.vad.config.algorithm_params,
                        enable_callbacks=self.vad.config.enable_callbacks,
                        callback_timeout=self.vad.config.callback_timeout,
                        buffer_size=self.vad.config.buffer_size,
                        max_history_length=self.vad.config.max_history_length,
                    )
                    self.vad.update_config(new_config)

                logger.info(
                    f"VAD settings updated: threshold={self.vad.config.threshold}, "
                    f"min_silence={self.vad.config.min_silence_duration}s"
                )
            else:
                # Handle legacy VAD system
                if threshold is not None:
                    self.vad.set_threshold(threshold)

                if min_silence_duration is not None:
                    self.vad.set_min_silence_duration(min_silence_duration)

                # Get current settings based on available attributes
                current_threshold = getattr(self.vad, "threshold", threshold or 0.02)
                current_min_silence = getattr(
                    self.vad, "min_silence_duration", min_silence_duration or 1.0
                )

                logger.info(
                    f"VAD settings updated: threshold={current_threshold}, "
                    f"min_silence={current_min_silence}s"
                )

        except Exception as e:
            logger.error(f"Failed to update VAD settings: {e}")
            # Continue operation with current settings

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics, handling both VAD systems."""
        try:
            # Get VAD statistics based on system type
            if _NEW_VAD_AVAILABLE and isinstance(self.vad, BaseVAD):
                vad_stats = self.vad.statistics.to_dict()
                vad_info = {
                    "algorithm": self.vad.algorithm.algorithm_id,
                    "is_initialized": self.vad.is_initialized,
                    "is_listening": self.vad.is_listening,
                    "current_state": self.vad.state.value,
                }
            else:
                # Legacy VAD system
                vad_stats = (
                    self.vad.get_statistics()
                    if hasattr(self.vad, "get_statistics")
                    else {}
                )
                vad_info = {
                    "algorithm": "legacy",
                    "threshold": getattr(self.vad, "threshold", 0.02),
                    "min_silence_duration": getattr(
                        self.vad, "min_silence_duration", 1.0
                    ),
                }

        except Exception as e:
            logger.error(f"Error getting VAD statistics: {e}")
            vad_stats = {}
            vad_info = {"error": str(e)}

        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "is_running": self._running,
            "stats": self.stats.copy(),
            "vad_stats": vad_stats,
            "vad_info": vad_info,
            "buffer_size": self.audio_buffer.size,
            "buffer_samples": self.audio_buffer.total_samples,
            "speech_queue_size": self.speech_queue.qsize(),
            "response_queue_size": self.response_queue.qsize(),
        }

    async def process_text_input(self, text: str) -> Optional[str]:
        """Process text input directly (bypass STT)."""
        try:
            logger.info(f"Processing direct text input: {text}")

            self._trigger_callback(self.on_speech_detected, text)
            self._trigger_callback(self.on_processing)

            response = self._process_llm_request(text)

            if response:
                self._trigger_callback(self.on_response, response)

                # Optionally synthesize speech
                if self.state != PipelineState.STOPPED:
                    self._set_state(PipelineState.SPEAKING)
                    self._trigger_callback(self.on_speaking)
                    self._synthesize_speech(response)

                    if self._running:
                        self._set_state(PipelineState.LISTENING)

            return response

        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            self._handle_error(e)
            return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._running:
            asyncio.run(self.stop_listening())

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if self._running:
                asyncio.run(self.stop_listening())
        except Exception:
            pass


# Example usage and convenience functions
async def create_voice_assistant(
    model_name: str = "gpt35",
    system_prompt: str = "You are a helpful voice assistant.",
    vad_algorithm: str = "energy",
    vad_preset: str = "balanced",
    **kwargs,
) -> StreamingVoicePipeline:
    """
    Create and configure a voice assistant pipeline with advanced VAD support.

    Args:
        model_name (str): LLM model to use
        system_prompt (str): System prompt for the assistant
        vad_algorithm (str): VAD algorithm to use ("energy", "spectral", "webrtc", "silero", "adaptive")
        vad_preset (str): VAD configuration preset ("fast", "balanced", "accurate")
        **kwargs: Additional configuration options

    Returns:
        StreamingVoicePipeline: Configured pipeline instance

    Example:
        # Create with default energy VAD
        pipeline = await create_voice_assistant()

        # Create with advanced VAD algorithm
        pipeline = await create_voice_assistant(
            vad_algorithm="webrtc",
            vad_preset="accurate"
        )

        # Create with custom VAD configuration
        from gianna.audio.vad import create_vad_config
        vad_config = create_vad_config("energy", preset="fast", threshold=0.03)
        pipeline = await create_voice_assistant(vad_config=vad_config)
    """
    # Create VAD configuration if using new system
    if _NEW_VAD_AVAILABLE and "vad_config" not in kwargs:
        try:
            from gianna.audio.vad import create_vad_config

            vad_config = create_vad_config(vad_algorithm, preset=vad_preset)
            kwargs["vad_config"] = vad_config
        except Exception as e:
            logger.warning(f"Failed to create VAD config, using defaults: {e}")

    # Set VAD algorithm if not already specified
    if "vad_algorithm" not in kwargs:
        kwargs["vad_algorithm"] = vad_algorithm

    return StreamingVoicePipeline(
        model_name=model_name, system_prompt=system_prompt, **kwargs
    )


async def create_streaming_voice_assistant(
    model_name: str = "gpt35",
    system_prompt: str = "You are a helpful voice assistant.",
    vad_algorithm: str = "energy",
    sample_rate: int = 16000,
    chunk_size: int = 1024,
    **kwargs,
) -> StreamingVoicePipeline:
    """
    Create a voice assistant optimized for streaming applications.

    Args:
        model_name (str): LLM model to use
        system_prompt (str): System prompt for the assistant
        vad_algorithm (str): VAD algorithm optimized for streaming
        sample_rate (int): Audio sample rate
        chunk_size (int): Audio chunk size for processing
        **kwargs: Additional configuration options

    Returns:
        StreamingVoicePipeline: Configured streaming pipeline instance
    """
    # Streaming-optimized configuration
    streaming_config = {
        "sample_rate": sample_rate,
        "chunk_size": chunk_size,
        "vad_algorithm": vad_algorithm,
        "min_silence_duration": 0.5,  # Shorter silence for faster response
        "min_speech_duration": 0.1,  # Shorter minimum speech duration
        "buffer_max_chunks": chunk_size * 8,  # Larger buffer for streaming
    }

    # Apply streaming optimizations and user overrides
    streaming_config.update(kwargs)

    return StreamingVoicePipeline(
        model_name=model_name, system_prompt=system_prompt, **streaming_config
    )


# Example implementation
if __name__ == "__main__":

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        # Callback functions
        def on_listening():
            print("👂 Listening...")

        def on_speech_detected(transcript):
            print(f"🎤 Speech: {transcript}")

        def on_processing():
            print("🧠 Processing...")

        def on_response(response):
            print(f"💬 Response: {response}")

        def on_speaking():
            print("🔊 Speaking...")

        def on_error(error):
            print(f"❌ Error: {error}")

        # Create pipeline
        pipeline = StreamingVoicePipeline(
            model_name="gpt35",
            system_prompt="You are a helpful voice assistant. Keep responses concise.",
            vad_threshold=0.02,
            min_silence_duration=1.0,
            on_listening=on_listening,
            on_speech_detected=on_speech_detected,
            on_processing=on_processing,
            on_response=on_response,
            on_speaking=on_speaking,
            on_error=on_error,
        )

        try:
            print("Starting voice assistant...")
            print(
                "Speak naturally - the assistant will respond when you finish talking"
            )
            print("Press Ctrl+C to stop")

            await pipeline.start_listening()

            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
                status = pipeline.get_pipeline_status()
                if status["state"] == "error":
                    print("Pipeline encountered an error")
                    break

        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await pipeline.stop_listening()
            print("Voice assistant stopped")

    # Run the example
    asyncio.run(main())
