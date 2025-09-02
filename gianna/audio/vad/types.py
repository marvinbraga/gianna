"""
Voice Activity Detection (VAD) types and enumerations.

This module defines common types, enums, and data structures used across
the VAD system for consistent type annotations and data handling.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

import numpy as np


class VADAlgorithm(Enum):
    """
    Enumeration of supported VAD algorithms.
    """

    ENERGY = ("energy", "Energy-based VAD using RMS calculation")
    SPECTRAL = ("spectral", "Spectral-based VAD using frequency analysis")
    ML = ("ml", "Machine learning-based VAD")
    WEBRTC = ("webrtc", "WebRTC VAD algorithm")
    SILERO = ("silero", "Silero VAD neural network")
    ADAPTIVE = ("adaptive", "Adaptive multi-algorithm fusion VAD")

    def __init__(self, algorithm_id: str, description: str):
        """
        Initialize VAD algorithm enum.

        Args:
            algorithm_id (str): Unique identifier for the algorithm.
            description (str): Human-readable description of the algorithm.
        """
        self.algorithm_id = algorithm_id
        self.description = description


class VADEventType(Enum):
    """
    Enumeration of VAD event types.
    """

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    SILENCE_START = "silence_start"
    SILENCE_END = "silence_end"
    THRESHOLD_CHANGED = "threshold_changed"
    ERROR = "error"


class VADState(Enum):
    """
    Enumeration of VAD processing states.
    """

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    SILENT = "silent"
    ERROR = "error"


@dataclass
class VADConfig:
    """
    Configuration class for VAD parameters.

    This dataclass provides a structured way to pass configuration
    parameters to VAD implementations with validation.
    """

    # Core detection parameters
    threshold: float = 0.02
    min_silence_duration: float = 1.0
    min_speech_duration: float = 0.1

    # Audio parameters
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1

    # Algorithm-specific parameters
    algorithm_params: Dict[str, Any] = field(default_factory=dict)

    # Callback settings
    enable_callbacks: bool = True
    callback_timeout: float = 5.0

    # Performance settings
    buffer_size: int = 4096
    max_history_length: int = 100

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        if self.min_silence_duration < 0.0:
            raise ValueError("Minimum silence duration must be non-negative")

        if self.min_speech_duration < 0.0:
            raise ValueError("Minimum speech duration must be non-negative")

        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")

        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        if self.channels <= 0:
            raise ValueError("Number of channels must be positive")

        if self.callback_timeout <= 0:
            raise ValueError("Callback timeout must be positive")

        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")

        if self.max_history_length <= 0:
            raise ValueError("Max history length must be positive")


@dataclass
class VADResult:
    """
    Result class for VAD processing operations.

    Contains all relevant information about a single VAD analysis,
    including detection results, energy levels, and state information.
    """

    # Detection results
    is_voice_active: bool
    is_speaking: bool
    confidence: float

    # Energy and signal analysis
    energy_level: float
    threshold_used: float
    signal_to_noise_ratio: Optional[float] = None

    # Timing information
    timestamp: float = field(default_factory=time.time)
    processing_time: float = 0.0

    # State information
    state_changed: bool = False
    event_type: Optional[VADEventType] = None
    previous_state: Optional[VADState] = None
    current_state: Optional[VADState] = None

    # Additional context
    silence_duration: float = 0.0
    speech_duration: float = 0.0
    time_since_last_activity: float = 0.0

    # Metadata
    chunk_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert VAD result to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the result.
        """
        return {
            "is_voice_active": self.is_voice_active,
            "is_speaking": self.is_speaking,
            "confidence": self.confidence,
            "energy_level": self.energy_level,
            "threshold_used": self.threshold_used,
            "signal_to_noise_ratio": self.signal_to_noise_ratio,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "state_changed": self.state_changed,
            "event_type": self.event_type.value if self.event_type else None,
            "previous_state": (
                self.previous_state.value if self.previous_state else None
            ),
            "current_state": self.current_state.value if self.current_state else None,
            "silence_duration": self.silence_duration,
            "speech_duration": self.speech_duration,
            "time_since_last_activity": self.time_since_last_activity,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }


@dataclass
class VADStatistics:
    """
    Statistics class for VAD performance monitoring.

    Tracks various metrics about VAD processing performance
    and detection accuracy over time.
    """

    # Processing statistics
    total_chunks_processed: int = 0
    speech_chunks_detected: int = 0
    silence_chunks_detected: int = 0

    # Accuracy metrics
    speech_detection_ratio: float = 0.0
    silence_detection_ratio: float = 0.0

    # Performance metrics
    average_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float("inf")

    # Energy statistics
    average_energy_level: float = 0.0
    max_energy_level: float = 0.0
    min_energy_level: float = float("inf")

    # State transition counts
    speech_start_events: int = 0
    speech_end_events: int = 0

    # Timing statistics
    total_speech_duration: float = 0.0
    total_silence_duration: float = 0.0

    # Error tracking
    error_count: int = 0
    last_error_time: Optional[float] = None

    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def update_chunk_stats(self, result: VADResult) -> None:
        """
        Update statistics with a new VAD result.

        Args:
            result (VADResult): VAD processing result to incorporate.
        """
        with self._lock:
            self.total_chunks_processed += 1

            # Update detection counts
            if result.is_voice_active:
                self.speech_chunks_detected += 1
            else:
                self.silence_chunks_detected += 1

            # Update ratios
            if self.total_chunks_processed > 0:
                self.speech_detection_ratio = (
                    self.speech_chunks_detected / self.total_chunks_processed
                )
                self.silence_detection_ratio = (
                    self.silence_chunks_detected / self.total_chunks_processed
                )

            # Update processing time statistics
            if result.processing_time > 0:
                self._update_processing_time_stats(result.processing_time)

            # Update energy statistics
            self._update_energy_stats(result.energy_level)

            # Update event counts
            if result.event_type == VADEventType.SPEECH_START:
                self.speech_start_events += 1
            elif result.event_type == VADEventType.SPEECH_END:
                self.speech_end_events += 1

    def _update_processing_time_stats(self, processing_time: float) -> None:
        """Update processing time statistics."""
        self.max_processing_time = max(self.max_processing_time, processing_time)
        self.min_processing_time = min(self.min_processing_time, processing_time)

        # Calculate running average
        total_time = self.average_processing_time * (self.total_chunks_processed - 1)
        self.average_processing_time = (
            total_time + processing_time
        ) / self.total_chunks_processed

    def _update_energy_stats(self, energy_level: float) -> None:
        """Update energy level statistics."""
        self.max_energy_level = max(self.max_energy_level, energy_level)
        self.min_energy_level = min(self.min_energy_level, energy_level)

        # Calculate running average
        total_energy = self.average_energy_level * (self.total_chunks_processed - 1)
        self.average_energy_level = (
            total_energy + energy_level
        ) / self.total_chunks_processed

    def record_error(self) -> None:
        """Record an error occurrence."""
        with self._lock:
            self.error_count += 1
            self.last_error_time = time.time()

    def reset(self) -> None:
        """Reset all statistics to initial values."""
        with self._lock:
            self.__init__()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of statistics.
        """
        with self._lock:
            return {
                "total_chunks_processed": self.total_chunks_processed,
                "speech_chunks_detected": self.speech_chunks_detected,
                "silence_chunks_detected": self.silence_chunks_detected,
                "speech_detection_ratio": self.speech_detection_ratio,
                "silence_detection_ratio": self.silence_detection_ratio,
                "average_processing_time": self.average_processing_time,
                "max_processing_time": self.max_processing_time,
                "min_processing_time": self.min_processing_time,
                "average_energy_level": self.average_energy_level,
                "max_energy_level": self.max_energy_level,
                "min_energy_level": self.min_energy_level,
                "speech_start_events": self.speech_start_events,
                "speech_end_events": self.speech_end_events,
                "total_speech_duration": self.total_speech_duration,
                "total_silence_duration": self.total_silence_duration,
                "error_count": self.error_count,
                "last_error_time": self.last_error_time,
            }


# Type aliases for common use cases
AudioChunk = Union[np.ndarray, bytes]
VADCallback = Callable[[VADResult], None]
EventCallback = Callable[[VADEventType, Dict[str, Any]], None]
