"""
WebRTC Voice Activity Detection implementation.

This module provides a VAD implementation using the WebRTC VAD algorithm,
which is the industry standard for real-time voice communication applications.
The WebRTC VAD provides excellent balance between accuracy and performance.

Key Features:
- Industry-standard algorithm used in Google Chrome and many VoIP applications
- Multiple aggressiveness levels for different use cases
- Optimized for real-time processing with low latency
- Robust performance across various audio conditions
- Optional dependency handling with energy-based fallback

Performance Characteristics:
- Accuracy: Good-Very Good (85-95% typical)
- CPU Usage: Low-Medium (highly optimized C implementation)
- Memory Usage: Low (~1-5MB)
- Latency: Very Low (1-5ms typical)
- Best Use Cases: Real-time communication, streaming, low-latency requirements
"""

import logging
import time
from typing import Any, Dict, Optional, Union

import numpy as np

from .base import BaseVAD
from .types import AudioChunk, VADAlgorithm, VADConfig, VADResult, VADState

logger = logging.getLogger(__name__)

# Optional dependency handling
try:
    import webrtcvad

    _WEBRTC_VAD_AVAILABLE = True
except ImportError:
    webrtcvad = None
    _WEBRTC_VAD_AVAILABLE = False
    logger.warning(
        "webrtcvad library not available. WebRtcVAD will use fallback implementation."
    )


class WebRtcVADConfig(VADConfig):
    """
    Configuration class specific to WebRTC VAD.

    Extends the base VADConfig with WebRTC-specific parameters for
    aggressiveness levels, frame processing, and performance optimization.
    """

    def __init__(
        self,
        # Base VAD parameters
        threshold: float = 0.5,  # Not directly used by WebRTC, kept for compatibility
        min_silence_duration: float = 0.5,
        min_speech_duration: float = 0.1,
        sample_rate: int = 16000,
        chunk_size: int = 320,  # 20ms at 16kHz (WebRTC standard)
        channels: int = 1,
        enable_callbacks: bool = True,
        callback_timeout: float = 5.0,
        buffer_size: int = 4096,
        max_history_length: int = 100,
        # WebRTC-specific parameters
        aggressiveness: int = 2,  # 0-3, higher = more aggressive filtering
        frame_duration_ms: int = 20,  # 10, 20, or 30ms
        use_optimized_frames: bool = True,
        confidence_smoothing: bool = True,
        smoothing_window_size: int = 5,
        **kwargs,
    ):
        """
        Initialize WebRTC VAD configuration.

        Args:
            threshold (float): Compatibility threshold (not used by WebRTC core).
            min_silence_duration (float): Minimum silence duration in seconds.
            min_speech_duration (float): Minimum speech duration in seconds.
            sample_rate (int): Audio sample rate (8000, 16000, 32000, or 48000 Hz).
            chunk_size (int): Size of audio chunks for processing.
            channels (int): Number of audio channels (must be 1).
            enable_callbacks (bool): Enable callback execution.
            callback_timeout (float): Callback execution timeout.
            buffer_size (int): Internal buffer size.
            max_history_length (int): Maximum history length for statistics.
            aggressiveness (int): WebRTC aggressiveness mode (0-3).
            frame_duration_ms (int): Frame duration in milliseconds (10, 20, or 30).
            use_optimized_frames (bool): Use frame sizes optimized for WebRTC.
            confidence_smoothing (bool): Apply smoothing to confidence values.
            smoothing_window_size (int): Window size for confidence smoothing.
            **kwargs: Additional parameters.
        """
        # Validate WebRTC-specific constraints
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(
                "WebRTC VAD supports only 8kHz, 16kHz, 32kHz, and 48kHz sample rates"
            )

        if channels != 1:
            raise ValueError("WebRTC VAD only supports mono audio (channels=1)")

        if not 0 <= aggressiveness <= 3:
            raise ValueError("WebRTC aggressiveness must be between 0 and 3")

        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError("WebRTC frame duration must be 10, 20, or 30 milliseconds")

        # Calculate optimal chunk size if using optimized frames
        if use_optimized_frames:
            frame_samples = (sample_rate * frame_duration_ms) // 1000
            if chunk_size != frame_samples:
                logger.info(
                    f"Adjusting chunk_size from {chunk_size} to {frame_samples} for optimal WebRTC processing"
                )
                chunk_size = frame_samples

        super().__init__(
            threshold=threshold,
            min_silence_duration=min_silence_duration,
            min_speech_duration=min_speech_duration,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            channels=channels,
            enable_callbacks=enable_callbacks,
            callback_timeout=callback_timeout,
            buffer_size=buffer_size,
            max_history_length=max_history_length,
            algorithm_params=kwargs,
        )

        # Store WebRTC-specific parameters
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.use_optimized_frames = use_optimized_frames
        self.confidence_smoothing = confidence_smoothing
        self.smoothing_window_size = smoothing_window_size


class WebRtcVAD(BaseVAD):
    """
    WebRTC Voice Activity Detection implementation.

    This VAD implementation uses the Google WebRTC VAD algorithm, which is the
    industry standard for real-time voice communication. It provides excellent
    balance between accuracy, performance, and low latency.

    Features:
    - Industry-standard algorithm from Google WebRTC project
    - Multiple aggressiveness levels (0=least aggressive, 3=most aggressive)
    - Optimized for real-time processing with minimal latency
    - Frame-based processing with standard durations (10/20/30ms)
    - Confidence smoothing for stable detection
    - Graceful fallback when webrtcvad library unavailable

    Performance Characteristics:
    - Accuracy: 85-95% (good to very good)
    - CPU Usage: Low (highly optimized C implementation)
    - Memory: Very Low (~1-5MB)
    - Latency: Excellent (<5ms typical)
    - GPU Acceleration: No (CPU-optimized)

    Best Use Cases:
    - Real-time voice communication (VoIP, conferencing)
    - Streaming applications requiring low latency
    - Resource-constrained environments
    - Applications where consistent performance is critical
    """

    def __init__(self, config: Optional[Union[VADConfig, WebRtcVADConfig]] = None):
        """
        Initialize WebRTC VAD detector.

        Args:
            config: Configuration object. If VADConfig is provided, it will be
                   converted to WebRtcVADConfig with default WebRTC parameters.
        """
        # Convert to WebRtcVADConfig if needed
        if config is None:
            config = WebRtcVADConfig()
        elif isinstance(config, VADConfig) and not isinstance(config, WebRtcVADConfig):
            # Convert base config to WebRTC config
            config = WebRtcVADConfig(
                threshold=config.threshold,
                min_silence_duration=config.min_silence_duration,
                min_speech_duration=config.min_speech_duration,
                sample_rate=config.sample_rate,
                chunk_size=config.chunk_size,
                channels=config.channels,
                enable_callbacks=config.enable_callbacks,
                callback_timeout=config.callback_timeout,
                buffer_size=config.buffer_size,
                max_history_length=config.max_history_length,
                **config.algorithm_params,
            )

        super().__init__(config)

        # WebRTC-specific attributes
        self._webrtc_vad = None
        self._fallback_to_energy = False

        # Frame processing
        self._frame_size = (
            self._config.sample_rate * self._config.frame_duration_ms
        ) // 1000
        self._audio_buffer = bytearray()

        # Confidence smoothing
        self._confidence_window = []
        self._smoothed_confidence = 0.0

        # Performance tracking
        self._frame_times = []
        self._total_frames_processed = 0

        logger.info(
            f"WebRtcVAD initialized with aggressiveness={self._config.aggressiveness}, "
            f"frame_duration={self._config.frame_duration_ms}ms, "
            f"webrtcvad_available={_WEBRTC_VAD_AVAILABLE}"
        )

    @property
    def algorithm(self) -> VADAlgorithm:
        """Get the VAD algorithm type."""
        return VADAlgorithm.WEBRTC

    @property
    def webrtc_info(self) -> Dict[str, Any]:
        """
        Get information about the WebRTC VAD configuration.

        Returns:
            Dict containing WebRTC-specific configuration and status.
        """
        return {
            "aggressiveness": self._config.aggressiveness,
            "frame_duration_ms": self._config.frame_duration_ms,
            "frame_size_samples": self._frame_size,
            "sample_rate": self._config.sample_rate,
            "webrtcvad_available": _WEBRTC_VAD_AVAILABLE,
            "webrtc_initialized": self._webrtc_vad is not None,
            "fallback_mode": self._fallback_to_energy,
            "confidence_smoothing": self._config.confidence_smoothing,
            "smoothing_window_size": self._config.smoothing_window_size,
        }

    @property
    def performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the WebRTC VAD.

        Returns:
            Dict containing processing times and performance metrics.
        """
        if not self._frame_times:
            return {
                "average_frame_time": 0.0,
                "min_frame_time": 0.0,
                "max_frame_time": 0.0,
                "total_frames": self._total_frames_processed,
                "frames_per_second": 0.0,
            }

        return {
            "average_frame_time": np.mean(self._frame_times),
            "min_frame_time": np.min(self._frame_times),
            "max_frame_time": np.max(self._frame_times),
            "total_frames": self._total_frames_processed,
            "frames_per_second": (
                1000.0 / self._config.frame_duration_ms
                if self._config.frame_duration_ms > 0
                else 0.0
            ),
        }

    def initialize(self) -> bool:
        """
        Initialize the WebRTC VAD detector.

        This method creates the WebRTC VAD instance with the specified
        aggressiveness level. If the webrtcvad library is not available,
        it falls back to energy-based detection.

        Returns:
            bool: True if initialization was successful.
        """
        try:
            if not _WEBRTC_VAD_AVAILABLE:
                logger.warning(
                    "webrtcvad library not available. Falling back to energy-based VAD."
                )
                self._fallback_to_energy = True
                self._is_initialized = True
                return True

            # Create WebRTC VAD instance
            self._webrtc_vad = webrtcvad.Vad(self._config.aggressiveness)

            # Clear processing state
            self._audio_buffer.clear()
            self._confidence_window.clear()
            self._frame_times.clear()
            self._total_frames_processed = 0
            self._smoothed_confidence = 0.0

            self._is_initialized = True

            logger.info(
                f"WebRTC VAD initialized successfully with aggressiveness={self._config.aggressiveness}, "
                f"sample_rate={self._config.sample_rate}Hz, "
                f"frame_duration={self._config.frame_duration_ms}ms"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize WebRTC VAD: {e}")
            # Try fallback to energy-based detection
            logger.warning("Falling back to energy-based detection")
            self._fallback_to_energy = True
            self._is_initialized = True
            return True

    def cleanup(self) -> None:
        """Clean up resources used by the WebRTC VAD detector."""
        with self._lock:
            self._webrtc_vad = None
            self._audio_buffer.clear()
            self._confidence_window.clear()
            self._frame_times.clear()
            self._is_initialized = False
            logger.info("WebRTC VAD cleanup completed")

    def calculate_energy(self, audio_chunk: AudioChunk) -> float:
        """
        Calculate energy level of audio chunk (fallback method).

        This method is used when falling back to energy-based detection
        or as a supplementary measure alongside WebRTC detection.

        Args:
            audio_chunk: Audio data to analyze.

        Returns:
            float: Normalized energy level (0.0-1.0).
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio_chunk, bytes):
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            else:
                audio_data = audio_chunk.astype(np.int16)

            if len(audio_data) == 0:
                return 0.0

            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

            # Normalize to 0-1 range (assuming 16-bit audio)
            normalized_rms = rms / 32768.0

            return min(max(normalized_rms, 0.0), 1.0)

        except Exception as e:
            logger.warning(f"Error calculating energy: {e}")
            return 0.0

    def detect_activity(self, audio_chunk: AudioChunk) -> VADResult:
        """
        Detect voice activity using WebRTC VAD algorithm.

        This method performs frame-based voice activity detection using the
        WebRTC algorithm. If the library is unavailable, it falls back to
        energy-based detection.

        Args:
            audio_chunk: Audio data to analyze.

        Returns:
            VADResult: Detailed result of voice activity detection.
        """
        start_time = time.time()

        try:
            # Fallback to energy-based detection if needed
            if self._fallback_to_energy or self._webrtc_vad is None:
                return self._detect_activity_fallback(audio_chunk, start_time)

            # Convert audio chunk to bytes if needed
            if isinstance(audio_chunk, np.ndarray):
                audio_bytes = audio_chunk.astype(np.int16).tobytes()
            else:
                audio_bytes = audio_chunk

            # Add to buffer for frame-based processing
            self._audio_buffer.extend(audio_bytes)

            # Process frames
            frame_results = []
            frame_size_bytes = self._frame_size * 2  # 2 bytes per 16-bit sample

            while len(self._audio_buffer) >= frame_size_bytes:
                # Extract frame
                frame_bytes = bytes(self._audio_buffer[:frame_size_bytes])
                self._audio_buffer = self._audio_buffer[frame_size_bytes:]

                # Process frame with WebRTC VAD
                frame_start = time.time()
                is_speech = self._webrtc_vad.is_speech(
                    frame_bytes, self._config.sample_rate
                )
                frame_time = time.time() - frame_start

                self._frame_times.append(frame_time * 1000)  # Convert to milliseconds
                self._total_frames_processed += 1

                # Convert boolean to confidence value
                confidence = 1.0 if is_speech else 0.0
                frame_results.append(confidence)

                # Keep only recent frame times
                if len(self._frame_times) > 1000:
                    self._frame_times = self._frame_times[-500:]

            # Calculate overall confidence for this chunk
            if frame_results:
                # Use average of frame results as chunk confidence
                chunk_confidence = np.mean(frame_results)
            else:
                # No complete frames processed, use previous confidence
                chunk_confidence = self._smoothed_confidence

            # Apply confidence smoothing if enabled
            if self._config.confidence_smoothing:
                self._confidence_window.append(chunk_confidence)
                if len(self._confidence_window) > self._config.smoothing_window_size:
                    self._confidence_window.pop(0)

                # Calculate smoothed confidence
                self._smoothed_confidence = np.mean(self._confidence_window)
                final_confidence = self._smoothed_confidence
            else:
                final_confidence = chunk_confidence

            # Determine voice activity (WebRTC gives binary result, use threshold for smoothed)
            if self._config.confidence_smoothing:
                is_voice_active = final_confidence > 0.5
            else:
                is_voice_active = chunk_confidence > 0.5

            # Calculate energy for supplementary information
            energy = self.calculate_energy(audio_chunk)

            # Create result
            result = VADResult(
                is_voice_active=is_voice_active,
                is_speaking=False,  # Will be updated by process_stream
                confidence=final_confidence,
                energy_level=energy,
                threshold_used=0.5,  # WebRTC uses internal thresholding
                timestamp=time.time(),
                processing_time=time.time() - start_time,
                metadata={
                    "webrtc_confidence": chunk_confidence,
                    "smoothed_confidence": self._smoothed_confidence,
                    "frames_processed": len(frame_results),
                    "aggressiveness": self._config.aggressiveness,
                    "frame_duration_ms": self._config.frame_duration_ms,
                    "energy_supplement": energy,
                    "buffer_size": len(self._audio_buffer),
                },
            )

            logger.debug(
                f"WebRTC VAD: confidence={final_confidence:.3f}, "
                f"active={is_voice_active}, frames={len(frame_results)}, "
                f"aggressiveness={self._config.aggressiveness}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in WebRTC VAD detection: {e}")
            return self._create_error_result(start_time, str(e))

    def _detect_activity_fallback(
        self, audio_chunk: AudioChunk, start_time: float
    ) -> VADResult:
        """
        Fallback voice activity detection using energy-based method.

        Args:
            audio_chunk: Audio data to analyze.
            start_time: Processing start time.

        Returns:
            VADResult: Energy-based detection result.
        """
        try:
            energy = self.calculate_energy(audio_chunk)
            is_voice_active = energy > self._config.threshold
            confidence = energy  # Use energy as confidence measure

            return VADResult(
                is_voice_active=is_voice_active,
                is_speaking=False,
                confidence=confidence,
                energy_level=energy,
                threshold_used=self._config.threshold,
                timestamp=time.time(),
                processing_time=time.time() - start_time,
                metadata={"fallback_mode": True, "energy_based": True},
            )

        except Exception as e:
            return self._create_error_result(
                start_time, f"Fallback detection error: {e}"
            )

    def _create_error_result(self, start_time: float, error_msg: str) -> VADResult:
        """Create an error VAD result."""
        return VADResult(
            is_voice_active=False,
            is_speaking=False,
            confidence=0.0,
            energy_level=0.0,
            threshold_used=0.5,
            timestamp=time.time(),
            processing_time=time.time() - start_time,
            metadata={"error": error_msg},
        )

    def set_aggressiveness(self, aggressiveness: int) -> None:
        """
        Update the WebRTC VAD aggressiveness level.

        Args:
            aggressiveness (int): New aggressiveness level (0-3).
                                0 = Least aggressive (more permissive)
                                3 = Most aggressive (more restrictive)

        Raises:
            ValueError: If aggressiveness is not in valid range.
        """
        if not 0 <= aggressiveness <= 3:
            raise ValueError("WebRTC aggressiveness must be between 0 and 3")

        old_aggressiveness = self._config.aggressiveness
        self._config.aggressiveness = aggressiveness

        # Update WebRTC VAD instance if available
        if self._webrtc_vad is not None:
            try:
                self._webrtc_vad.set_mode(aggressiveness)
                logger.info(
                    f"Updated WebRTC aggressiveness: {old_aggressiveness} -> {aggressiveness}"
                )
            except Exception as e:
                logger.error(f"Failed to update WebRTC aggressiveness: {e}")
                # Revert on failure
                self._config.aggressiveness = old_aggressiveness
                raise
        else:
            logger.info(
                f"WebRTC aggressiveness updated to {aggressiveness} (will take effect on next initialization)"
            )

    def get_frame_benchmark(self, test_duration: float = 10.0) -> Dict[str, Any]:
        """
        Run a benchmark test on the WebRTC VAD.

        Args:
            test_duration: Duration of test in seconds.

        Returns:
            Dict containing benchmark results.
        """
        if not self._is_initialized or self._fallback_to_energy:
            return {"error": "WebRTC VAD not available for benchmarking"}

        # Generate test audio
        sample_rate = self._config.sample_rate
        num_samples = int(test_duration * sample_rate)
        test_audio = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)

        # Split into frames for WebRTC processing
        frame_samples = self._frame_size
        frames = [
            test_audio[i : i + frame_samples]
            for i in range(0, len(test_audio), frame_samples)
        ]

        # Filter complete frames only
        frames = [frame for frame in frames if len(frame) == frame_samples]

        # Run benchmark
        start_time = time.time()
        frame_times = []

        for frame in frames:
            frame_bytes = frame.tobytes()
            frame_start = time.time()

            if self._webrtc_vad is not None:
                self._webrtc_vad.is_speech(frame_bytes, sample_rate)

            frame_times.append(time.time() - frame_start)

        total_time = time.time() - start_time

        return {
            "test_duration": test_duration,
            "total_frames": len(frames),
            "frame_duration_ms": self._config.frame_duration_ms,
            "total_processing_time": total_time,
            "average_frame_time": np.mean(frame_times) if frame_times else 0.0,
            "max_frame_time": np.max(frame_times) if frame_times else 0.0,
            "min_frame_time": np.min(frame_times) if frame_times else 0.0,
            "real_time_factor": total_time / test_duration,
            "frames_per_second": len(frames) / total_time if total_time > 0 else 0.0,
            "aggressiveness": self._config.aggressiveness,
            "webrtc_info": self.webrtc_info,
        }

    def reset_confidence_smoothing(self) -> None:
        """Reset the confidence smoothing window."""
        with self._lock:
            self._confidence_window.clear()
            self._smoothed_confidence = 0.0
            logger.info("Confidence smoothing window reset")

    def get_confidence_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about confidence values and smoothing.

        Returns:
            Dict containing confidence statistics.
        """
        if not self._confidence_window:
            return {
                "window_size": len(self._confidence_window),
                "smoothed_confidence": self._smoothed_confidence,
                "confidence_smoothing_enabled": self._config.confidence_smoothing,
                "smoothing_window_size": self._config.smoothing_window_size,
            }

        window_array = np.array(self._confidence_window)
        return {
            "window_size": len(self._confidence_window),
            "mean_confidence": float(np.mean(window_array)),
            "max_confidence": float(np.max(window_array)),
            "min_confidence": float(np.min(window_array)),
            "std_confidence": float(np.std(window_array)),
            "smoothed_confidence": self._smoothed_confidence,
            "confidence_smoothing_enabled": self._config.confidence_smoothing,
            "smoothing_window_size": self._config.smoothing_window_size,
        }

    def __repr__(self) -> str:
        """String representation of the WebRTC VAD detector."""
        return (
            f"WebRtcVAD("
            f"aggressiveness={self._config.aggressiveness}, "
            f"frame_duration={self._config.frame_duration_ms}ms, "
            f"sample_rate={self._config.sample_rate}Hz, "
            f"state={self.state.value}, "
            f"fallback={self._fallback_to_energy}, "
            f"initialized={self.is_initialized})"
        )
