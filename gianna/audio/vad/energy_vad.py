"""
Energy-based Voice Activity Detection implementation.

This module provides an energy-based VAD implementation using RMS (Root Mean Square)
energy analysis for detecting voice activity in audio streams. It migrates and
enhances the original VoiceActivityDetector implementation with improved
architecture and additional features.
"""

import logging
import time
from typing import Union

import numpy as np

from .base import BaseVAD
from .types import AudioChunk, VADAlgorithm, VADConfig, VADResult, VADState

logger = logging.getLogger(__name__)


class EnergyVAD(BaseVAD):
    """
    Energy-based Voice Activity Detection using RMS energy analysis.

    This VAD implementation analyzes the RMS (Root Mean Square) energy of
    audio chunks to determine voice activity. It provides configurable
    thresholds, minimum silence duration, and comprehensive state management.

    The algorithm works by:
    1. Computing RMS energy of each audio chunk
    2. Comparing energy against configured threshold
    3. Managing state transitions with minimum duration requirements
    4. Providing callbacks and detailed result information
    """

    def __init__(self, config: VADConfig = None):
        """
        Initialize the Energy-based VAD detector.

        Args:
            config (VADConfig, optional): Configuration for the VAD detector.
                If None, default configuration will be used.
        """
        super().__init__(config)

        # Energy-specific state tracking
        self._silence_start_time = None
        self._speech_start_time = None
        self._last_activity_time = time.time()

        # Energy history for adaptive processing
        self._energy_history = []
        self._max_history_size = self._config.max_history_length

        # Noise floor estimation
        self._noise_floor = 0.0
        self._noise_samples = []
        self._noise_adaptation_rate = 0.01

        logger.info(
            f"EnergyVAD initialized: threshold={self._config.threshold}, "
            f"min_silence={self._config.min_silence_duration}s, "
            f"sample_rate={self._config.sample_rate}Hz"
        )

    @property
    def algorithm(self) -> VADAlgorithm:
        """
        Get the VAD algorithm type.

        Returns:
            VADAlgorithm: Always returns VADAlgorithm.ENERGY for this implementation.
        """
        return VADAlgorithm.ENERGY

    @property
    def noise_floor(self) -> float:
        """
        Get the current estimated noise floor.

        Returns:
            float: Current noise floor estimate.
        """
        return self._noise_floor

    @property
    def energy_history(self) -> list:
        """
        Get the recent energy history.

        Returns:
            list: Recent energy values (up to max_history_length).
        """
        return self._energy_history.copy()

    def initialize(self) -> bool:
        """
        Initialize the Energy VAD detector.

        Performs basic validation and setup for energy-based detection.

        Returns:
            bool: True if initialization was successful.
        """
        try:
            # Validate configuration for energy-based VAD
            if not 0.0 <= self._config.threshold <= 1.0:
                logger.error(f"Invalid threshold: {self._config.threshold}")
                return False

            if self._config.min_silence_duration < 0.0:
                logger.error(
                    f"Invalid min_silence_duration: {self._config.min_silence_duration}"
                )
                return False

            # Initialize state
            self._silence_start_time = None
            self._speech_start_time = None
            self._last_activity_time = time.time()
            self._energy_history.clear()
            self._noise_samples.clear()
            self._noise_floor = 0.0

            self._is_initialized = True
            logger.info("EnergyVAD initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize EnergyVAD: {e}")
            return False

    def cleanup(self) -> None:
        """
        Clean up resources used by the Energy VAD detector.

        Clears energy history and resets internal state.
        """
        with self._lock:
            self._energy_history.clear()
            self._noise_samples.clear()
            self._is_initialized = False
            logger.info("EnergyVAD cleanup completed")

    def calculate_energy(self, audio_chunk: AudioChunk) -> float:
        """
        Calculate RMS (Root Mean Square) energy of an audio chunk.

        This method converts the audio chunk to the appropriate format and
        computes the normalized RMS energy value used for voice detection.

        Args:
            audio_chunk (AudioChunk): Audio data as numpy array or bytes.

        Returns:
            float: Normalized RMS energy value (0.0-1.0).
        """
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_chunk, bytes):
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            else:
                audio_data = audio_chunk.astype(np.int16)

            # Handle empty chunks
            if len(audio_data) == 0:
                return 0.0

            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

            # Normalize to 0-1 range (assuming 16-bit audio)
            normalized_rms = rms / 32768.0

            # Clamp to valid range
            return min(max(normalized_rms, 0.0), 1.0)

        except Exception as e:
            logger.warning(f"Error calculating RMS energy: {e}")
            return 0.0

    def detect_activity(self, audio_chunk: AudioChunk) -> VADResult:
        """
        Detect voice activity in a single audio chunk using energy analysis.

        This method implements the core energy-based voice activity detection
        algorithm, comparing computed energy against the configured threshold.

        Args:
            audio_chunk (AudioChunk): Audio data to analyze.

        Returns:
            VADResult: Detailed result of the voice activity detection.
        """
        start_time = time.time()

        try:
            # Calculate energy
            energy = self.calculate_energy(audio_chunk)

            # Update energy history
            self._update_energy_history(energy)

            # Update noise floor estimation
            self._update_noise_floor(energy)

            # Calculate signal-to-noise ratio if possible
            snr = self._calculate_snr(energy) if self._noise_floor > 0 else None

            # Determine if voice is active
            is_voice_active = energy > self._config.threshold

            # Calculate confidence based on energy relative to threshold
            if is_voice_active:
                confidence = min(energy / self._config.threshold, 1.0)
            else:
                # Confidence decreases as energy gets closer to threshold
                if self._config.threshold > 0:
                    confidence = max(0.0, 1.0 - (energy / self._config.threshold))
                else:
                    confidence = 1.0

            # Create result
            result = VADResult(
                is_voice_active=is_voice_active,
                is_speaking=False,  # Will be updated by process_stream
                confidence=confidence,
                energy_level=energy,
                threshold_used=self._config.threshold,
                signal_to_noise_ratio=snr,
                timestamp=time.time(),
                processing_time=time.time() - start_time,
                metadata={
                    "noise_floor": self._noise_floor,
                    "energy_history_length": len(self._energy_history),
                },
            )

            logger.debug(
                f"EnergyVAD: energy={energy:.4f}, threshold={self._config.threshold:.4f}, "
                f"active={is_voice_active}, confidence={confidence:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in energy-based VAD detection: {e}")
            return VADResult(
                is_voice_active=False,
                is_speaking=False,
                confidence=0.0,
                energy_level=0.0,
                threshold_used=self._config.threshold,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)},
            )

    def process_stream(self, audio_chunk: AudioChunk) -> VADResult:
        """
        Process audio chunk in streaming mode with enhanced state management.

        This method extends the base streaming functionality with energy-specific
        state management including minimum duration requirements and timing logic.

        Args:
            audio_chunk (AudioChunk): Audio data chunk to process.

        Returns:
            VADResult: Processing result with detailed state information.
        """
        with self._lock:
            # Get basic detection result
            result = self.detect_activity(audio_chunk)
            current_time = time.time()

            # Enhanced state management for energy-based detection
            self._manage_energy_state(result, current_time)

            # Update statistics
            self._statistics.update_chunk_stats(result)

            # Store last result
            self._last_result = result

            # Execute callbacks if enabled
            if self._config.enable_callbacks:
                self._execute_callbacks(result)

            return result

    def _manage_energy_state(self, result: VADResult, current_time: float) -> None:
        """
        Manage energy-specific state transitions and timing logic.

        Args:
            result (VADResult): Current VAD result to update.
            current_time (float): Current timestamp.
        """
        # Store previous state
        result.previous_state = self._state

        if result.is_voice_active:
            # Voice activity detected
            self._last_activity_time = current_time
            self._silence_start_time = None

            # Check for speech start
            if self._state != VADState.SPEAKING:
                # Check minimum speech duration if configured
                if self._config.min_speech_duration > 0:
                    if self._speech_start_time is None:
                        self._speech_start_time = current_time

                    speech_duration = current_time - self._speech_start_time
                    if speech_duration >= self._config.min_speech_duration:
                        self._state = VADState.SPEAKING
                        result.state_changed = True
                        result.event_type = self._get_event_type_from_state(
                            VADState.SPEAKING
                        )
                        self._speech_start_time = None
                else:
                    # No minimum duration requirement
                    self._state = VADState.SPEAKING
                    result.state_changed = True
                    result.event_type = self._get_event_type_from_state(
                        VADState.SPEAKING
                    )

        else:
            # Silence detected
            self._speech_start_time = None

            if self._state == VADState.SPEAKING:
                # Start silence timer if not already started
                if self._silence_start_time is None:
                    self._silence_start_time = current_time

                # Check if silence duration threshold is met
                silence_duration = current_time - self._silence_start_time
                if silence_duration >= self._config.min_silence_duration:
                    self._state = VADState.SILENT
                    result.state_changed = True
                    result.event_type = self._get_event_type_from_state(VADState.SILENT)
                    self._silence_start_time = None
            else:
                # Already in silence or other state
                if self._state != VADState.SILENT:
                    self._state = VADState.SILENT

        # Update result with current state information
        result.current_state = self._state
        result.is_speaking = self._state == VADState.SPEAKING

        # Calculate durations
        if self._silence_start_time:
            result.silence_duration = current_time - self._silence_start_time
        else:
            result.silence_duration = 0.0

        if self._speech_start_time:
            result.speech_duration = current_time - self._speech_start_time
        else:
            result.speech_duration = 0.0

        result.time_since_last_activity = current_time - self._last_activity_time

    def _get_event_type_from_state(self, state: VADState):
        """Get the appropriate event type for a state transition."""
        from .types import VADEventType

        if state == VADState.SPEAKING:
            return VADEventType.SPEECH_START
        elif state == VADState.SILENT:
            return VADEventType.SPEECH_END
        else:
            return None

    def _update_energy_history(self, energy: float) -> None:
        """
        Update the energy history buffer.

        Args:
            energy (float): New energy value to add.
        """
        self._energy_history.append(energy)

        # Trim history if too long
        if len(self._energy_history) > self._max_history_size:
            self._energy_history.pop(0)

    def _update_noise_floor(self, energy: float) -> None:
        """
        Update noise floor estimation using adaptive filtering.

        Args:
            energy (float): Current energy level.
        """
        # Only update noise floor during silence periods
        if not (energy > self._config.threshold):
            if len(self._noise_samples) < 50:  # Initial collection phase
                self._noise_samples.append(energy)
                if len(self._noise_samples) == 50:
                    self._noise_floor = np.mean(self._noise_samples)
            else:
                # Adaptive update
                self._noise_floor = (
                    1 - self._noise_adaptation_rate
                ) * self._noise_floor + self._noise_adaptation_rate * energy

    def _calculate_snr(self, energy: float) -> float:
        """
        Calculate Signal-to-Noise Ratio.

        Args:
            energy (float): Current signal energy.

        Returns:
            float: Signal-to-noise ratio in dB.
        """
        if self._noise_floor > 0:
            snr_linear = energy / self._noise_floor
            if snr_linear > 0:
                return 20 * np.log10(snr_linear)
        return 0.0

    def get_adaptive_threshold(self) -> float:
        """
        Calculate an adaptive threshold based on noise floor.

        Returns:
            float: Suggested adaptive threshold value.
        """
        if self._noise_floor > 0:
            # Set threshold as multiple of noise floor
            adaptive_threshold = self._noise_floor * 3.0  # 3x noise floor
            return min(max(adaptive_threshold, 0.01), 0.5)  # Clamp to reasonable range
        return self._config.threshold

    def set_adaptive_threshold(self, multiplier: float = 3.0) -> None:
        """
        Set threshold adaptively based on current noise floor.

        Args:
            multiplier (float): Multiplier for noise floor to determine threshold.
        """
        if self._noise_floor > 0:
            new_threshold = self._noise_floor * multiplier
            new_threshold = min(max(new_threshold, 0.01), 0.5)  # Clamp
            self.set_threshold(new_threshold)
            logger.info(
                f"Set adaptive threshold: {new_threshold:.4f} (noise_floor={self._noise_floor:.4f})"
            )
        else:
            logger.warning("Cannot set adaptive threshold: noise floor not established")

    def get_energy_statistics(self) -> dict:
        """
        Get detailed energy-related statistics.

        Returns:
            dict: Dictionary containing energy statistics.
        """
        if not self._energy_history:
            return {
                "mean_energy": 0.0,
                "max_energy": 0.0,
                "min_energy": 0.0,
                "std_energy": 0.0,
                "noise_floor": self._noise_floor,
                "history_length": 0,
            }

        history_array = np.array(self._energy_history)
        return {
            "mean_energy": float(np.mean(history_array)),
            "max_energy": float(np.max(history_array)),
            "min_energy": float(np.min(history_array)),
            "std_energy": float(np.std(history_array)),
            "noise_floor": self._noise_floor,
            "history_length": len(self._energy_history),
        }

    def __repr__(self) -> str:
        """String representation of the Energy VAD detector."""
        return (
            f"EnergyVAD("
            f"threshold={self._config.threshold:.3f}, "
            f"min_silence={self._config.min_silence_duration}s, "
            f"state={self.state.value}, "
            f"noise_floor={self._noise_floor:.4f}, "
            f"initialized={self.is_initialized})"
        )


# Legacy compatibility class that wraps EnergyVAD
class VoiceActivityDetector(EnergyVAD):
    """
    Legacy compatibility wrapper for the original VoiceActivityDetector.

    This class provides backward compatibility with the original
    VoiceActivityDetector interface while using the new EnergyVAD
    implementation underneath.

    Deprecated: Use EnergyVAD directly for new implementations.
    """

    def __init__(
        self,
        threshold: float = 0.02,
        min_silence_duration: float = 1.0,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        speech_start_callback=None,
        speech_end_callback=None,
    ):
        """
        Initialize with legacy interface.

        Args:
            threshold (float): Energy threshold for voice detection.
            min_silence_duration (float): Minimum silence duration in seconds.
            sample_rate (int): Audio sample rate in Hz.
            chunk_size (int): Size of audio chunks for processing.
            speech_start_callback: Callback when speech starts.
            speech_end_callback: Callback when speech ends.
        """
        # Create config from legacy parameters
        config = VADConfig(
            threshold=threshold,
            min_silence_duration=min_silence_duration,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
        )

        super().__init__(config)

        # Set legacy callbacks
        if speech_start_callback:
            self.set_speech_start_callback(lambda result: speech_start_callback())

        if speech_end_callback:
            self.set_speech_end_callback(lambda result: speech_end_callback())

        # Auto-initialize for legacy compatibility
        self.initialize()

        logger.warning(
            "VoiceActivityDetector is deprecated. Use EnergyVAD directly for new code."
        )

    def calculate_rms_energy(self, audio_chunk: Union[np.ndarray, bytes]) -> float:
        """
        Legacy method name for calculate_energy.

        Args:
            audio_chunk: Audio data as numpy array or bytes.

        Returns:
            float: Normalized RMS energy value.
        """
        return self.calculate_energy(audio_chunk)

    def get_statistics(self) -> dict:
        """
        Legacy method for getting statistics in old format.

        Returns:
            dict: Statistics in legacy format.
        """
        modern_stats = self.statistics.to_dict()

        # Convert to legacy format
        return {
            "total_chunks": modern_stats["total_chunks_processed"],
            "speech_chunks": modern_stats["speech_chunks_detected"],
            "silence_chunks": modern_stats["silence_chunks_detected"],
            "speech_ratio": modern_stats["speech_detection_ratio"],
            "silence_ratio": modern_stats["silence_detection_ratio"],
            "current_threshold": self.config.threshold,
            "min_silence_duration": self.config.min_silence_duration,
            "is_speaking": self.state == VADState.SPEAKING,
        }

    def set_min_silence_duration(self, duration: float) -> None:
        """
        Legacy method for updating minimum silence duration.

        Args:
            duration (float): New minimum silence duration in seconds.
        """
        if duration < 0.0:
            raise ValueError("Silence duration must be non-negative")

        self._config.min_silence_duration = duration
        logger.info(f"Updated min silence duration to {duration}s")


def create_vad_detector(
    threshold: float = 0.02,
    min_silence_duration: float = 1.0,
    sample_rate: int = 16000,
    speech_start_callback=None,
    speech_end_callback=None,
) -> VoiceActivityDetector:
    """
    Create a VoiceActivityDetector instance with common defaults.

    This is a legacy convenience function for backward compatibility.
    For new code, use EnergyVAD constructor directly.

    Args:
        threshold (float): Energy threshold for voice detection.
        min_silence_duration (float): Minimum silence duration in seconds.
        sample_rate (int): Audio sample rate in Hz.
        speech_start_callback: Callback when speech starts.
        speech_end_callback: Callback when speech ends.

    Returns:
        VoiceActivityDetector: Configured legacy VAD instance.
    """
    return VoiceActivityDetector(
        threshold=threshold,
        min_silence_duration=min_silence_duration,
        sample_rate=sample_rate,
        speech_start_callback=speech_start_callback,
        speech_end_callback=speech_end_callback,
    )
