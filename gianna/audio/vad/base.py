"""
Base Voice Activity Detection (VAD) abstract interface.

This module provides the abstract base class and interface that all VAD
implementations must follow, ensuring consistency and interoperability
across different VAD algorithms.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .types import (
    AudioChunk,
    EventCallback,
    VADAlgorithm,
    VADCallback,
    VADConfig,
    VADEventType,
    VADResult,
    VADState,
    VADStatistics,
)

logger = logging.getLogger(__name__)


class BaseVAD(ABC):
    """
    Abstract base class for Voice Activity Detection implementations.

    This class defines the standard interface that all VAD algorithms
    must implement, providing a consistent API for voice activity detection
    across different algorithms and implementations.

    All VAD implementations should inherit from this class and implement
    the required abstract methods while maintaining thread safety and
    proper resource management.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        """
        Initialize the base VAD detector.

        Args:
            config (Optional[VADConfig]): Configuration for the VAD detector.
                If None, default configuration will be used.
        """
        self._config = config or VADConfig()
        self._statistics = VADStatistics()
        self._state = VADState.IDLE
        self._is_initialized = False
        self._is_listening = False

        # Thread safety
        self._lock = threading.RLock()
        self._callback_lock = threading.RLock()

        # Callbacks
        self._speech_start_callback: Optional[VADCallback] = None
        self._speech_end_callback: Optional[VADCallback] = None
        self._event_callbacks: Dict[VADEventType, List[EventCallback]] = {
            event_type: [] for event_type in VADEventType
        }

        # Internal state tracking
        self._last_result: Optional[VADResult] = None
        self._previous_state = VADState.IDLE
        self._state_change_time = time.time()

        logger.info(
            f"Initialized {self.__class__.__name__} with config: {self._config}"
        )

    @property
    def config(self) -> VADConfig:
        """
        Get the current VAD configuration.

        Returns:
            VADConfig: Current configuration settings.
        """
        return self._config

    @property
    def statistics(self) -> VADStatistics:
        """
        Get the current VAD processing statistics.

        Returns:
            VADStatistics: Current processing statistics.
        """
        return self._statistics

    @property
    def state(self) -> VADState:
        """
        Get the current VAD processing state.

        Returns:
            VADState: Current processing state.
        """
        with self._lock:
            return self._state

    @property
    def is_initialized(self) -> bool:
        """
        Check if the VAD detector is initialized.

        Returns:
            bool: True if initialized, False otherwise.
        """
        return self._is_initialized

    @property
    def is_listening(self) -> bool:
        """
        Check if the VAD detector is currently listening.

        Returns:
            bool: True if listening, False otherwise.
        """
        return self._is_listening

    @property
    @abstractmethod
    def algorithm(self) -> VADAlgorithm:
        """
        Get the VAD algorithm type.

        Returns:
            VADAlgorithm: The algorithm type used by this implementation.
        """
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the VAD detector.

        This method should perform any necessary setup for the VAD algorithm,
        including model loading, parameter validation, and resource allocation.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources used by the VAD detector.

        This method should release any allocated resources, close files,
        and perform proper cleanup when the VAD detector is no longer needed.
        """
        pass

    @abstractmethod
    def detect_activity(self, audio_chunk: AudioChunk) -> VADResult:
        """
        Detect voice activity in a single audio chunk.

        This is the core method that analyzes an audio chunk and determines
        whether voice activity is present. Implementations should be thread-safe
        and efficient for real-time processing.

        Args:
            audio_chunk (AudioChunk): Audio data to analyze (numpy array or bytes).

        Returns:
            VADResult: Detailed result of the voice activity detection.
        """
        pass

    @abstractmethod
    def calculate_energy(self, audio_chunk: AudioChunk) -> float:
        """
        Calculate the energy level of an audio chunk.

        This method should compute a normalized energy value for the given
        audio chunk, typically used as input to the voice activity detection
        algorithm.

        Args:
            audio_chunk (AudioChunk): Audio data to analyze.

        Returns:
            float: Normalized energy level (typically 0.0-1.0).
        """
        pass

    def process_stream(self, audio_chunk: AudioChunk) -> VADResult:
        """
        Process audio chunk in streaming mode with state management.

        This method provides higher-level processing that includes state
        management, event generation, and callback execution. It's built
        on top of the detect_activity method.

        Args:
            audio_chunk (AudioChunk): Audio data chunk to process.

        Returns:
            VADResult: Processing result with state information.
        """
        start_time = time.time()

        try:
            with self._lock:
                # Detect activity using algorithm-specific implementation
                result = self.detect_activity(audio_chunk)
                result.processing_time = time.time() - start_time

                # Update state and generate events
                self._update_state(result)

                # Update statistics
                self._statistics.update_chunk_stats(result)

                # Store last result
                self._last_result = result

                # Execute callbacks if enabled
                if self._config.enable_callbacks:
                    self._execute_callbacks(result)

                return result

        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")
            self._statistics.record_error()

            # Return error result
            error_result = VADResult(
                is_voice_active=False,
                is_speaking=False,
                confidence=0.0,
                energy_level=0.0,
                threshold_used=self._config.threshold,
                processing_time=time.time() - start_time,
                event_type=VADEventType.ERROR,
                current_state=VADState.ERROR,
                metadata={"error": str(e)},
            )
            return error_result

    def start_listening(self) -> bool:
        """
        Start the VAD listening mode.

        Returns:
            bool: True if listening was started successfully.
        """
        with self._lock:
            if not self._is_initialized:
                if not self.initialize():
                    logger.error("Failed to initialize VAD before starting listening")
                    return False

            self._is_listening = True
            self._state = VADState.LISTENING
            logger.info("VAD listening started")
            return True

    def stop_listening(self) -> None:
        """Stop the VAD listening mode."""
        with self._lock:
            self._is_listening = False
            self._state = VADState.IDLE
            logger.info("VAD listening stopped")

    def set_threshold(self, threshold: float) -> None:
        """
        Update the voice activity threshold.

        Args:
            threshold (float): New threshold value (0.0-1.0).

        Raises:
            ValueError: If threshold is not in valid range.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        with self._lock:
            old_threshold = self._config.threshold
            self._config.threshold = threshold

            # Generate threshold change event
            event_data = {
                "old_threshold": old_threshold,
                "new_threshold": threshold,
                "timestamp": time.time(),
            }
            self._trigger_event(VADEventType.THRESHOLD_CHANGED, event_data)

            logger.info(f"VAD threshold updated: {old_threshold} -> {threshold}")

    def update_config(self, new_config: VADConfig) -> None:
        """
        Update the VAD configuration.

        Args:
            new_config (VADConfig): New configuration to apply.
        """
        with self._lock:
            old_config = self._config
            self._config = new_config
            logger.info(f"VAD configuration updated from {old_config} to {new_config}")

    def reset_state(self) -> None:
        """Reset the VAD state to initial conditions."""
        with self._lock:
            self._state = VADState.IDLE
            self._previous_state = VADState.IDLE
            self._state_change_time = time.time()
            self._last_result = None
            logger.info("VAD state reset")

    def reset_statistics(self) -> None:
        """Reset all processing statistics."""
        self._statistics.reset()
        logger.info("VAD statistics reset")

    def set_speech_start_callback(self, callback: Optional[VADCallback]) -> None:
        """
        Set the callback for speech start events.

        Args:
            callback (Optional[VADCallback]): Callback function to execute
                when speech starts. Set to None to remove callback.
        """
        with self._callback_lock:
            self._speech_start_callback = callback
            logger.debug(f"Speech start callback {'set' if callback else 'removed'}")

    def set_speech_end_callback(self, callback: Optional[VADCallback]) -> None:
        """
        Set the callback for speech end events.

        Args:
            callback (Optional[VADCallback]): Callback function to execute
                when speech ends. Set to None to remove callback.
        """
        with self._callback_lock:
            self._speech_end_callback = callback
            logger.debug(f"Speech end callback {'set' if callback else 'removed'}")

    def add_event_callback(
        self, event_type: VADEventType, callback: EventCallback
    ) -> None:
        """
        Add an event-specific callback.

        Args:
            event_type (VADEventType): Type of event to listen for.
            callback (EventCallback): Callback function to execute.
        """
        with self._callback_lock:
            self._event_callbacks[event_type].append(callback)
            logger.debug(f"Added callback for {event_type.value} events")

    def remove_event_callback(
        self, event_type: VADEventType, callback: EventCallback
    ) -> bool:
        """
        Remove an event-specific callback.

        Args:
            event_type (VADEventType): Type of event.
            callback (EventCallback): Callback function to remove.

        Returns:
            bool: True if callback was removed, False if not found.
        """
        with self._callback_lock:
            try:
                self._event_callbacks[event_type].remove(callback)
                logger.debug(f"Removed callback for {event_type.value} events")
                return True
            except ValueError:
                logger.warning(f"Callback not found for {event_type.value} events")
                return False

    def get_last_result(self) -> Optional[VADResult]:
        """
        Get the last VAD processing result.

        Returns:
            Optional[VADResult]: Last processing result, or None if no
                processing has been performed yet.
        """
        return self._last_result

    def _update_state(self, result: VADResult) -> None:
        """
        Update internal state based on VAD result.

        Args:
            result (VADResult): VAD processing result.
        """
        current_time = time.time()

        # Store previous state
        result.previous_state = self._state

        # Determine new state based on detection result
        if result.is_voice_active:
            new_state = VADState.SPEAKING
        else:
            new_state = VADState.SILENT

        # Check for state changes
        if new_state != self._state:
            result.state_changed = True
            self._previous_state = self._state
            self._state = new_state
            self._state_change_time = current_time

            # Determine event type
            if new_state == VADState.SPEAKING:
                result.event_type = VADEventType.SPEECH_START
            elif new_state == VADState.SILENT:
                result.event_type = VADEventType.SPEECH_END

        result.current_state = self._state
        result.is_speaking = self._state == VADState.SPEAKING

    def _execute_callbacks(self, result: VADResult) -> None:
        """
        Execute registered callbacks based on VAD result.

        Args:
            result (VADResult): VAD processing result.
        """
        try:
            with self._callback_lock:
                # Execute legacy callbacks
                if (
                    result.event_type == VADEventType.SPEECH_START
                    and self._speech_start_callback
                ):
                    self._execute_callback_safely(self._speech_start_callback, result)

                elif (
                    result.event_type == VADEventType.SPEECH_END
                    and self._speech_end_callback
                ):
                    self._execute_callback_safely(self._speech_end_callback, result)

                # Execute event-specific callbacks
                if result.event_type:
                    callbacks = self._event_callbacks.get(result.event_type, [])
                    for callback in callbacks:
                        event_data = result.to_dict()
                        self._execute_event_callback_safely(
                            callback, result.event_type, event_data
                        )

        except Exception as e:
            logger.error(f"Error executing callbacks: {e}")

    def _execute_callback_safely(
        self, callback: VADCallback, result: VADResult
    ) -> None:
        """
        Execute a VAD callback with error handling.

        Args:
            callback (VADCallback): Callback function to execute.
            result (VADResult): VAD result to pass to callback.
        """
        try:
            callback(result)
        except Exception as e:
            logger.error(f"Error in VAD callback: {e}")

    def _execute_event_callback_safely(
        self,
        callback: EventCallback,
        event_type: VADEventType,
        event_data: Dict[str, Any],
    ) -> None:
        """
        Execute an event callback with error handling.

        Args:
            callback (EventCallback): Callback function to execute.
            event_type (VADEventType): Type of event.
            event_data (Dict[str, Any]): Event data to pass to callback.
        """
        try:
            callback(event_type, event_data)
        except Exception as e:
            logger.error(f"Error in event callback for {event_type.value}: {e}")

    def _trigger_event(
        self, event_type: VADEventType, event_data: Dict[str, Any]
    ) -> None:
        """
        Trigger an event and execute associated callbacks.

        Args:
            event_type (VADEventType): Type of event to trigger.
            event_data (Dict[str, Any]): Data associated with the event.
        """
        try:
            callbacks = self._event_callbacks.get(event_type, [])
            for callback in callbacks:
                self._execute_event_callback_safely(callback, event_type, event_data)
        except Exception as e:
            logger.error(f"Error triggering {event_type.value} event: {e}")

    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized and not self.initialize():
            raise RuntimeError("Failed to initialize VAD detector")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __repr__(self) -> str:
        """String representation of the VAD detector."""
        return (
            f"{self.__class__.__name__}("
            f"algorithm={self.algorithm.algorithm_id}, "
            f"state={self.state.value}, "
            f"threshold={self.config.threshold}, "
            f"initialized={self.is_initialized})"
        )
