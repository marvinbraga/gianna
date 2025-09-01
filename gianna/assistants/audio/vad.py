"""
Voice Activity Detection (VAD) module for Gianna assistant.

This module provides real-time voice activity detection using energy-based
thresholding with configurable parameters for different audio environments.
"""

import logging
import threading
import time
from typing import Any, Callable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Voice Activity Detection system using RMS energy analysis.

    This detector analyzes audio chunks to determine when voice activity
    starts and ends, providing callbacks for real-time processing.
    """

    def __init__(
        self,
        threshold: float = 0.02,
        min_silence_duration: float = 1.0,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        speech_start_callback: Optional[Callable[[], None]] = None,
        speech_end_callback: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the Voice Activity Detector.

        Args:
            threshold (float): Energy threshold for voice detection (0.0-1.0).
            min_silence_duration (float): Minimum silence duration in seconds
                before considering speech ended.
            sample_rate (int): Audio sample rate in Hz.
            chunk_size (int): Size of audio chunks for processing.
            speech_start_callback (Optional[Callable]): Callback when speech starts.
            speech_end_callback (Optional[Callable]): Callback when speech ends.
        """
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.speech_start_callback = speech_start_callback
        self.speech_end_callback = speech_end_callback

        # State tracking
        self._is_speaking = False
        self._silence_start_time: Optional[float] = None
        self._last_activity_time = time.time()
        self._lock = threading.RLock()

        # Statistics
        self._total_chunks = 0
        self._speech_chunks = 0
        self._silence_chunks = 0

        logger.info(
            f"VAD initialized: threshold={threshold}, "
            f"min_silence={min_silence_duration}s, "
            f"sample_rate={sample_rate}Hz"
        )

    def calculate_rms_energy(self, audio_chunk: Union[np.ndarray, bytes]) -> float:
        """
        Calculate RMS (Root Mean Square) energy of an audio chunk.

        Args:
            audio_chunk: Audio data as numpy array or bytes.

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

            return min(normalized_rms, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating RMS energy: {e}")
            return 0.0

    def detect_activity(
        self, audio_chunk: Union[np.ndarray, bytes]
    ) -> tuple[bool, float]:
        """
        Detect voice activity in a single audio chunk.

        Args:
            audio_chunk: Audio data to analyze.

        Returns:
            tuple: (is_voice_active, energy_level)
        """
        with self._lock:
            self._total_chunks += 1

            # Calculate energy
            energy = self.calculate_rms_energy(audio_chunk)

            # Determine if voice is active
            is_voice_active = energy > self.threshold

            # Update statistics
            if is_voice_active:
                self._speech_chunks += 1
            else:
                self._silence_chunks += 1

            logger.debug(
                f"VAD: energy={energy:.4f}, threshold={self.threshold:.4f}, "
                f"active={is_voice_active}"
            )

            return is_voice_active, energy

    def process_stream(self, audio_chunk: Union[np.ndarray, bytes]) -> dict[str, Any]:
        """
        Process audio chunk in streaming mode with state management.

        Args:
            audio_chunk: Audio data chunk to process.

        Returns:
            dict: Processing result with state information.
        """
        with self._lock:
            is_voice_active, energy = self.detect_activity(audio_chunk)
            current_time = time.time()

            # State transition logic
            state_changed = False
            event_type = None

            if is_voice_active:
                self._last_activity_time = current_time
                self._silence_start_time = None

                # Speech start detection
                if not self._is_speaking:
                    self._is_speaking = True
                    state_changed = True
                    event_type = "speech_start"

                    # Trigger callback
                    if self.speech_start_callback:
                        try:
                            self.speech_start_callback()
                        except Exception as e:
                            logger.error(f"Error in speech_start_callback: {e}")

                    logger.info("Speech started")

            else:  # Silence detected
                if self._is_speaking:
                    # Start silence timer if not already started
                    if self._silence_start_time is None:
                        self._silence_start_time = current_time

                    # Check if silence duration threshold is met
                    silence_duration = current_time - self._silence_start_time
                    if silence_duration >= self.min_silence_duration:
                        self._is_speaking = False
                        state_changed = True
                        event_type = "speech_end"
                        self._silence_start_time = None

                        # Trigger callback
                        if self.speech_end_callback:
                            try:
                                self.speech_end_callback()
                            except Exception as e:
                                logger.error(f"Error in speech_end_callback: {e}")

                        logger.info(
                            f"Speech ended after {silence_duration:.2f}s silence"
                        )

            # Build result
            result = {
                "is_speaking": self._is_speaking,
                "is_voice_active": is_voice_active,
                "energy": energy,
                "threshold": self.threshold,
                "state_changed": state_changed,
                "event_type": event_type,
                "timestamp": current_time,
                "silence_duration": (
                    current_time - self._silence_start_time
                    if self._silence_start_time
                    else 0.0
                ),
                "time_since_last_activity": current_time - self._last_activity_time,
            }

            return result

    def set_threshold(self, threshold: float) -> None:
        """
        Update the voice activity threshold.

        Args:
            threshold (float): New threshold value (0.0-1.0).
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        with self._lock:
            old_threshold = self.threshold
            self.threshold = threshold
            logger.info(f"VAD threshold updated: {old_threshold} -> {threshold}")

    def set_min_silence_duration(self, duration: float) -> None:
        """
        Update the minimum silence duration.

        Args:
            duration (float): New minimum silence duration in seconds.
        """
        if duration < 0.0:
            raise ValueError("Silence duration must be non-negative")

        with self._lock:
            old_duration = self.min_silence_duration
            self.min_silence_duration = duration
            logger.info(
                f"VAD min silence duration updated: {old_duration}s -> {duration}s"
            )

    def reset_state(self) -> None:
        """Reset the VAD state to initial conditions."""
        with self._lock:
            self._is_speaking = False
            self._silence_start_time = None
            self._last_activity_time = time.time()
            logger.info("VAD state reset")

    def get_statistics(self) -> dict[str, Any]:
        """
        Get VAD processing statistics.

        Returns:
            dict: Statistics about processed audio chunks.
        """
        with self._lock:
            if self._total_chunks == 0:
                return {
                    "total_chunks": 0,
                    "speech_chunks": 0,
                    "silence_chunks": 0,
                    "speech_ratio": 0.0,
                    "silence_ratio": 0.0,
                    "current_threshold": self.threshold,
                    "min_silence_duration": self.min_silence_duration,
                    "is_speaking": self._is_speaking,
                }

            speech_ratio = self._speech_chunks / self._total_chunks
            silence_ratio = self._silence_chunks / self._total_chunks

            return {
                "total_chunks": self._total_chunks,
                "speech_chunks": self._speech_chunks,
                "silence_chunks": self._silence_chunks,
                "speech_ratio": speech_ratio,
                "silence_ratio": silence_ratio,
                "current_threshold": self.threshold,
                "min_silence_duration": self.min_silence_duration,
                "is_speaking": self._is_speaking,
            }

    def __repr__(self) -> str:
        """String representation of the VAD instance."""
        return (
            f"VoiceActivityDetector("
            f"threshold={self.threshold}, "
            f"min_silence={self.min_silence_duration}s, "
            f"sample_rate={self.sample_rate}Hz, "
            f"is_speaking={self._is_speaking})"
        )


# Convenience function for simple VAD usage
def create_vad_detector(
    threshold: float = 0.02,
    min_silence_duration: float = 1.0,
    sample_rate: int = 16000,
    speech_start_callback: Optional[Callable[[], None]] = None,
    speech_end_callback: Optional[Callable[[], None]] = None,
) -> VoiceActivityDetector:
    """
    Create a VoiceActivityDetector instance with common defaults.

    Args:
        threshold (float): Energy threshold for voice detection.
        min_silence_duration (float): Minimum silence duration in seconds.
        sample_rate (int): Audio sample rate in Hz.
        speech_start_callback (Optional[Callable]): Callback when speech starts.
        speech_end_callback (Optional[Callable]): Callback when speech ends.

    Returns:
        VoiceActivityDetector: Configured VAD instance.
    """
    return VoiceActivityDetector(
        threshold=threshold,
        min_silence_duration=min_silence_duration,
        sample_rate=sample_rate,
        speech_start_callback=speech_start_callback,
        speech_end_callback=speech_end_callback,
    )


# Example usage
if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create VAD detector
    def on_speech_start():
        print("🎤 Speech detected!")

    def on_speech_end():
        print("🔇 Speech ended")

    vad = create_vad_detector(
        threshold=0.02,
        min_silence_duration=1.5,
        speech_start_callback=on_speech_start,
        speech_end_callback=on_speech_end,
    )

    print(f"VAD Demo: {vad}")
    print("VAD system ready. Processing would happen here in a real application.")

    # Simulate some processing
    fake_audio = np.random.randn(1024).astype(np.int16) * 1000  # Low energy
    result = vad.process_stream(fake_audio)
    print(f"Processed chunk: {result}")

    # Show statistics
    stats = vad.get_statistics()
    print(f"Statistics: {stats}")
