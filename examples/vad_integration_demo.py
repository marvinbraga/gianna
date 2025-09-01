"""
Voice Activity Detection Integration Demo

This example demonstrates how to integrate the VAD system with
Gianna's audio recording and processing pipeline.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

from gianna.assistants.audio.vad import VoiceActivityDetector, create_vad_detector


class VADRecordingSession:
    """
    Example integration of VAD with recording session management.

    This class demonstrates how to use VAD to automatically start/stop
    recording based on voice activity detection.
    """

    def __init__(
        self,
        vad_threshold: float = 0.02,
        min_silence_duration: float = 2.0,
        sample_rate: int = 16000,
    ):
        """
        Initialize VAD recording session.

        Args:
            vad_threshold (float): Voice activity threshold
            min_silence_duration (float): Silence duration before stopping
            sample_rate (int): Audio sample rate
        """
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_buffer = []
        self.session_active = False
        self._lock = threading.Lock()

        # Initialize VAD with callbacks
        self.vad = create_vad_detector(
            threshold=vad_threshold,
            min_silence_duration=min_silence_duration,
            sample_rate=sample_rate,
            speech_start_callback=self._on_speech_start,
            speech_end_callback=self._on_speech_end,
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("VAD Recording Session initialized")

    def _on_speech_start(self) -> None:
        """Callback for when speech starts."""
        with self._lock:
            if not self.is_recording and self.session_active:
                self.is_recording = True
                self.audio_buffer = []
                self.logger.info("🎤 Started recording due to voice activity")

    def _on_speech_end(self) -> None:
        """Callback for when speech ends."""
        with self._lock:
            if self.is_recording:
                self.is_recording = False
                audio_length = len(self.audio_buffer)
                self.logger.info(
                    f"🔇 Stopped recording - captured {audio_length} chunks"
                )

                # Here you would typically process the recorded audio
                self._process_recorded_audio()

    def _process_recorded_audio(self) -> None:
        """Process the recorded audio buffer."""
        if not self.audio_buffer:
            self.logger.warning("No audio data to process")
            return

        # Simulate audio processing
        total_samples = sum(len(chunk) for chunk in self.audio_buffer)
        duration = total_samples / self.sample_rate

        self.logger.info(f"Processing {duration:.2f}s of audio...")

        # Here you would typically:
        # 1. Convert audio buffer to proper format
        # 2. Save to file if needed
        # 3. Send to speech-to-text
        # 4. Process the transcription

        # Simulated processing delay
        time.sleep(0.5)
        self.logger.info("Audio processing completed")

    def start_session(self) -> None:
        """Start the VAD recording session."""
        with self._lock:
            self.session_active = True
            self.vad.reset_state()
            self.logger.info("VAD session started - listening for voice activity")

    def stop_session(self) -> None:
        """Stop the VAD recording session."""
        with self._lock:
            self.session_active = False
            if self.is_recording:
                self.is_recording = False
                self._process_recorded_audio()
            self.logger.info("VAD session stopped")

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Process an incoming audio chunk with VAD.

        Args:
            audio_chunk: Audio data chunk

        Returns:
            dict: VAD processing result
        """
        # Process with VAD
        vad_result = self.vad.process_stream(audio_chunk)

        # If we're recording, add to buffer
        with self._lock:
            if self.is_recording:
                self.audio_buffer.append(audio_chunk.copy())

        return vad_result

    def get_session_stats(self) -> dict:
        """Get session statistics."""
        vad_stats = self.vad.get_statistics()

        with self._lock:
            session_stats = {
                "session_active": self.session_active,
                "is_recording": self.is_recording,
                "buffer_chunks": len(self.audio_buffer),
                "vad_stats": vad_stats,
            }

        return session_stats


def run_vad_demo():
    """Run a comprehensive VAD demonstration."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting VAD Integration Demo")

    # Create VAD recording session
    session = VADRecordingSession(
        vad_threshold=0.03,
        min_silence_duration=1.5,
        sample_rate=16000,
    )

    # Start session
    session.start_session()

    # Simulate audio stream processing
    chunk_size = 1024
    duration_seconds = 10
    chunks_per_second = 16000 // chunk_size
    total_chunks = duration_seconds * chunks_per_second

    logger.info(f"Simulating {duration_seconds}s of audio processing...")

    try:
        for i in range(total_chunks):
            # Simulate different types of audio
            if 2 <= i / chunks_per_second <= 4:  # Speech from 2-4 seconds
                # High energy - simulate speech
                audio_chunk = np.random.randn(chunk_size).astype(np.int16) * 8000
            elif 6 <= i / chunks_per_second <= 7.5:  # Speech from 6-7.5 seconds
                # High energy - another speech segment
                audio_chunk = np.random.randn(chunk_size).astype(np.int16) * 10000
            else:
                # Low energy - simulate silence/background noise
                audio_chunk = np.random.randn(chunk_size).astype(np.int16) * 500

            # Process chunk
            result = session.process_audio_chunk(audio_chunk)

            # Log significant events
            if result.get("state_changed"):
                event = result.get("event_type", "unknown")
                energy = result.get("energy", 0)
                logger.info(f"State change: {event} (energy: {energy:.4f})")

            # Simulate real-time processing delay
            time.sleep(chunk_size / 16000)  # Realistic chunk timing

        # Show final statistics
        stats = session.get_session_stats()
        logger.info(f"Final session statistics: {stats}")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")

    finally:
        # Clean shutdown
        session.stop_session()
        logger.info("VAD Integration Demo completed")


def test_vad_thresholds():
    """Test different VAD threshold settings."""
    print("\n=== VAD Threshold Testing ===")

    # Test different threshold values
    thresholds = [0.01, 0.02, 0.05, 0.1]
    test_audio_chunks = [
        np.random.randn(1024).astype(np.int16) * 500,  # Background noise
        np.random.randn(1024).astype(np.int16) * 2000,  # Quiet speech
        np.random.randn(1024).astype(np.int16) * 8000,  # Normal speech
        np.random.randn(1024).astype(np.int16) * 15000,  # Loud speech
    ]

    audio_types = ["Background noise", "Quiet speech", "Normal speech", "Loud speech"]

    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        vad = create_vad_detector(threshold=threshold)

        for audio_chunk, audio_type in zip(test_audio_chunks, audio_types):
            is_active, energy = vad.detect_activity(audio_chunk)
            status = "ACTIVE" if is_active else "SILENT"
            print(f"  {audio_type:15} - Energy: {energy:.4f} - Status: {status}")


if __name__ == "__main__":
    print("Gianna VAD Integration Demo")
    print("==========================")

    # Run threshold testing
    test_vad_thresholds()

    print("\nStarting full integration demo...")
    print("Press Ctrl+C to interrupt")

    # Run full demo
    run_vad_demo()
