"""
Silero Voice Activity Detection implementation.

This module provides a deep learning-based VAD implementation using the Silero VAD model.
Silero VAD is a neural network-based approach that provides high accuracy voice activity
detection using pre-trained models optimized for various audio conditions.

Key Features:
- Deep learning-based detection with high accuracy
- Pre-trained models optimized for different languages and conditions
- GPU acceleration support when available
- Configurable confidence thresholds and model parameters
- Optional dependency handling with graceful fallbacks

Performance Characteristics:
- Accuracy: Very High (95-98% typical)
- CPU Usage: Medium-High (neural network inference)
- Memory Usage: Medium (model parameters ~20MB)
- Latency: Medium (10-50ms typical)
- Best Use Cases: High-accuracy requirements, batch processing
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
    import torch
    import torchaudio

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    torchaudio = None
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. SileroVAD will use fallback implementation.")


class SileroVADConfig(VADConfig):
    """
    Configuration class specific to Silero VAD.

    Extends the base VADConfig with Silero-specific parameters for model
    selection, inference optimization, and confidence tuning.
    """

    def __init__(
        self,
        # Base VAD parameters
        threshold: float = 0.5,
        min_silence_duration: float = 0.5,
        min_speech_duration: float = 0.1,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        enable_callbacks: bool = True,
        callback_timeout: float = 5.0,
        buffer_size: int = 4096,
        max_history_length: int = 100,
        # Silero-specific parameters
        model_name: str = "silero_vad",
        device: str = "auto",  # "cpu", "cuda", "auto"
        force_reload: bool = False,
        onnx_model: bool = False,
        use_jit: bool = True,
        window_size_samples: int = 1536,  # 96ms at 16kHz
        speech_threshold: float = 0.5,
        silence_threshold: float = 0.35,
        min_silence_samples: int = 0,
        min_speech_samples: int = 0,
        **kwargs,
    ):
        """
        Initialize Silero VAD configuration.

        Args:
            threshold (float): Base threshold for voice activity detection.
            min_silence_duration (float): Minimum silence duration in seconds.
            min_speech_duration (float): Minimum speech duration in seconds.
            sample_rate (int): Audio sample rate in Hz (must be 16000 or 8000).
            chunk_size (int): Size of audio chunks for processing.
            channels (int): Number of audio channels (must be 1).
            enable_callbacks (bool): Enable callback execution.
            callback_timeout (float): Callback execution timeout.
            buffer_size (int): Internal buffer size.
            max_history_length (int): Maximum history length for statistics.
            model_name (str): Silero model name to use.
            device (str): Device for inference ("cpu", "cuda", "auto").
            force_reload (bool): Force model reload even if cached.
            onnx_model (bool): Use ONNX version of the model.
            use_jit (bool): Use TorchScript JIT compilation.
            window_size_samples (int): Window size in samples for processing.
            speech_threshold (float): Threshold for speech detection.
            silence_threshold (float): Threshold for silence detection.
            min_silence_samples (int): Minimum silence duration in samples.
            min_speech_samples (int): Minimum speech duration in samples.
            **kwargs: Additional parameters.
        """
        # Validate Silero-specific constraints
        if sample_rate not in [8000, 16000]:
            raise ValueError("Silero VAD only supports 8kHz and 16kHz sample rates")

        if channels != 1:
            raise ValueError("Silero VAD only supports mono audio (channels=1)")

        if not 0.0 <= speech_threshold <= 1.0:
            raise ValueError("Speech threshold must be between 0.0 and 1.0")

        if not 0.0 <= silence_threshold <= 1.0:
            raise ValueError("Silence threshold must be between 0.0 and 1.0")

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

        # Store Silero-specific parameters
        self.model_name = model_name
        self.device = device
        self.force_reload = force_reload
        self.onnx_model = onnx_model
        self.use_jit = use_jit
        self.window_size_samples = window_size_samples
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.min_silence_samples = min_silence_samples
        self.min_speech_samples = min_speech_samples


class SileroVAD(BaseVAD):
    """
    Silero Voice Activity Detection using deep learning models.

    This VAD implementation uses the Silero VAD neural network model to provide
    high-accuracy voice activity detection. The model is pre-trained on diverse
    audio datasets and provides robust performance across various audio conditions.

    Features:
    - High accuracy neural network-based detection
    - GPU acceleration when available
    - Multiple pre-trained model options
    - Configurable confidence thresholds
    - Batch processing support
    - Graceful fallback when PyTorch unavailable

    Performance Characteristics:
    - Accuracy: 95-98% (excellent)
    - CPU Usage: Medium-High (neural inference)
    - Memory: ~20-50MB (model parameters)
    - Latency: 10-50ms per chunk
    - GPU Acceleration: Yes (CUDA)

    Best Use Cases:
    - High-accuracy voice detection required
    - Batch processing of audio files
    - Applications where accuracy is more important than speed
    - Multi-language voice detection
    """

    def __init__(self, config: Optional[Union[VADConfig, SileroVADConfig]] = None):
        """
        Initialize Silero VAD detector.

        Args:
            config: Configuration object. If VADConfig is provided, it will be
                   converted to SileroVADConfig with default Silero parameters.
        """
        # Convert to SileroVADConfig if needed
        if config is None:
            config = SileroVADConfig()
        elif isinstance(config, VADConfig) and not isinstance(config, SileroVADConfig):
            # Convert base config to Silero config
            config = SileroVADConfig(
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

        # Silero-specific attributes
        self._model = None
        self._device = None
        self._model_sample_rate = None
        self._fallback_to_energy = False

        # Processing state
        self._audio_buffer = []
        self._confidence_history = []
        self._last_confidence = 0.0

        # Performance metrics
        self._inference_times = []
        self._model_load_time = 0.0

        logger.info(
            f"SileroVAD initialized with model={self._config.model_name}, "
            f"device={self._config.device}, torch_available={_TORCH_AVAILABLE}"
        )

    @property
    def algorithm(self) -> VADAlgorithm:
        """Get the VAD algorithm type."""
        return VADAlgorithm.SILERO

    @property
    def model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict containing model information including device, sample rate, etc.
        """
        return {
            "model_name": self._config.model_name,
            "device": str(self._device) if self._device else None,
            "sample_rate": self._model_sample_rate,
            "torch_available": _TORCH_AVAILABLE,
            "model_loaded": self._model is not None,
            "fallback_mode": self._fallback_to_energy,
            "load_time": self._model_load_time,
        }

    @property
    def performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the Silero model.

        Returns:
            Dict containing inference times and performance metrics.
        """
        if not self._inference_times:
            return {
                "average_inference_time": 0.0,
                "min_inference_time": 0.0,
                "max_inference_time": 0.0,
                "total_inferences": 0,
            }

        return {
            "average_inference_time": np.mean(self._inference_times),
            "min_inference_time": np.min(self._inference_times),
            "max_inference_time": np.max(self._inference_times),
            "total_inferences": len(self._inference_times),
            "model_load_time": self._model_load_time,
        }

    def initialize(self) -> bool:
        """
        Initialize the Silero VAD detector.

        This method loads the Silero model and prepares it for inference.
        If PyTorch is not available, it falls back to energy-based detection.

        Returns:
            bool: True if initialization was successful.
        """
        try:
            start_time = time.time()

            if not _TORCH_AVAILABLE:
                logger.warning(
                    "PyTorch not available. Falling back to energy-based VAD."
                )
                self._fallback_to_energy = True
                self._is_initialized = True
                return True

            # Determine device
            self._device = self._determine_device()
            logger.info(f"Using device: {self._device}")

            # Load Silero model
            logger.info(f"Loading Silero model: {self._config.model_name}")
            self._model = self._load_silero_model()

            if self._model is None:
                logger.error("Failed to load Silero model")
                return False

            # Move model to device
            self._model.to(self._device)
            self._model.eval()

            # Set model sample rate
            self._model_sample_rate = self._config.sample_rate

            # Clear processing state
            self._audio_buffer.clear()
            self._confidence_history.clear()
            self._inference_times.clear()

            self._model_load_time = time.time() - start_time
            self._is_initialized = True

            logger.info(
                f"Silero VAD initialized successfully in {self._model_load_time:.3f}s "
                f"on device {self._device}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            # Try fallback to energy-based detection
            logger.warning("Falling back to energy-based detection")
            self._fallback_to_energy = True
            self._is_initialized = True
            return True

    def cleanup(self) -> None:
        """Clean up resources used by the Silero VAD detector."""
        with self._lock:
            if self._model is not None:
                # Clear model from memory
                del self._model
                self._model = None

                # Clear CUDA cache if using GPU
                if _TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Clear buffers
            self._audio_buffer.clear()
            self._confidence_history.clear()
            self._inference_times.clear()

            self._is_initialized = False
            logger.info("Silero VAD cleanup completed")

    def calculate_energy(self, audio_chunk: AudioChunk) -> float:
        """
        Calculate energy level of audio chunk (fallback method).

        This method is used when falling back to energy-based detection
        or as a supplementary measure alongside neural network inference.

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
        Detect voice activity using Silero neural network model.

        This method performs the core voice activity detection using the
        Silero model. If the model is unavailable, it falls back to
        energy-based detection.

        Args:
            audio_chunk: Audio data to analyze.

        Returns:
            VADResult: Detailed result of voice activity detection.
        """
        start_time = time.time()

        try:
            # Fallback to energy-based detection if needed
            if self._fallback_to_energy or self._model is None:
                return self._detect_activity_fallback(audio_chunk, start_time)

            # Convert audio chunk to tensor
            audio_tensor = self._prepare_audio_tensor(audio_chunk)
            if audio_tensor is None:
                return self._create_error_result(
                    start_time, "Failed to prepare audio tensor"
                )

            # Perform inference
            inference_start = time.time()
            with torch.no_grad():
                confidence = self._model(audio_tensor, self._config.sample_rate).item()

            inference_time = time.time() - inference_start
            self._inference_times.append(inference_time)

            # Keep only recent inference times
            if len(self._inference_times) > 1000:
                self._inference_times = self._inference_times[-500:]

            # Update confidence history
            self._confidence_history.append(confidence)
            if len(self._confidence_history) > self._config.max_history_length:
                self._confidence_history.pop(0)

            self._last_confidence = confidence

            # Determine voice activity
            is_voice_active = confidence > self._config.speech_threshold

            # Calculate energy for supplementary information
            energy = self.calculate_energy(audio_chunk)

            # Create result
            result = VADResult(
                is_voice_active=is_voice_active,
                is_speaking=False,  # Will be updated by process_stream
                confidence=confidence,
                energy_level=energy,
                threshold_used=self._config.speech_threshold,
                timestamp=time.time(),
                processing_time=time.time() - start_time,
                metadata={
                    "model_confidence": confidence,
                    "inference_time": inference_time,
                    "device": str(self._device),
                    "confidence_history_length": len(self._confidence_history),
                    "energy_supplement": energy,
                },
            )

            logger.debug(
                f"Silero VAD: confidence={confidence:.3f}, "
                f"threshold={self._config.speech_threshold:.3f}, "
                f"active={is_voice_active}, inference_time={inference_time:.3f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Error in Silero VAD detection: {e}")
            return self._create_error_result(start_time, str(e))

    def _determine_device(self):
        """Determine the best device for model inference."""
        if not _TORCH_AVAILABLE or torch is None:
            return None

        if self._config.device == "cpu":
            return torch.device("cpu")
        elif self._config.device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("CUDA requested but not available, using CPU")
                return torch.device("cpu")
        else:  # auto
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")

    def _load_silero_model(self):
        """Load the Silero VAD model."""
        try:
            # Use torch.hub to load the model
            model, utils = torch.hub.load(
                repo_or_dir="silero-ai/vad",
                model="silero_vad",
                force_reload=self._config.force_reload,
                onnx=self._config.onnx_model,
            )

            if self._config.use_jit and not self._config.onnx_model:
                # Use TorchScript for optimization
                model = torch.jit.script(model)

            return model

        except Exception as e:
            logger.error(f"Failed to load Silero model: {e}")
            return None

    def _prepare_audio_tensor(self, audio_chunk: AudioChunk):
        """
        Prepare audio chunk as tensor for model inference.

        Args:
            audio_chunk: Raw audio data.

        Returns:
            Prepared audio tensor or None.
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio_chunk, bytes):
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            else:
                audio_data = audio_chunk.astype(np.int16)

            if len(audio_data) == 0:
                return None

            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_float).float()

            # Ensure proper shape (add batch dimension if needed)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Move to device
            audio_tensor = audio_tensor.to(self._device)

            return audio_tensor

        except Exception as e:
            logger.error(f"Error preparing audio tensor: {e}")
            return None

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
            threshold_used=self._config.speech_threshold,
            timestamp=time.time(),
            processing_time=time.time() - start_time,
            metadata={"error": error_msg},
        )

    def get_model_benchmark(self, test_duration: float = 10.0) -> Dict[str, Any]:
        """
        Run a benchmark test on the Silero model.

        Args:
            test_duration: Duration of test in seconds.

        Returns:
            Dict containing benchmark results.
        """
        if not self._is_initialized or self._fallback_to_energy:
            return {"error": "Model not available for benchmarking"}

        # Generate test audio
        sample_rate = self._config.sample_rate
        num_samples = int(test_duration * sample_rate)
        test_audio = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)

        # Split into chunks
        chunk_size = self._config.chunk_size
        chunks = [
            test_audio[i : i + chunk_size]
            for i in range(0, len(test_audio), chunk_size)
        ]

        # Run benchmark
        start_time = time.time()
        inference_times = []

        for chunk in chunks:
            chunk_start = time.time()
            self.detect_activity(chunk)
            inference_times.append(time.time() - chunk_start)

        total_time = time.time() - start_time

        return {
            "test_duration": test_duration,
            "total_chunks": len(chunks),
            "total_processing_time": total_time,
            "average_chunk_time": np.mean(inference_times),
            "max_chunk_time": np.max(inference_times),
            "min_chunk_time": np.min(inference_times),
            "real_time_factor": total_time / test_duration,
            "chunks_per_second": len(chunks) / total_time,
            "device": str(self._device),
            "model_info": self.model_info,
        }

    def update_thresholds(
        self,
        speech_threshold: Optional[float] = None,
        silence_threshold: Optional[float] = None,
    ) -> None:
        """
        Update Silero-specific thresholds.

        Args:
            speech_threshold: New speech detection threshold.
            silence_threshold: New silence detection threshold.
        """
        if speech_threshold is not None:
            if not 0.0 <= speech_threshold <= 1.0:
                raise ValueError("Speech threshold must be between 0.0 and 1.0")
            self._config.speech_threshold = speech_threshold
            logger.info(f"Updated speech threshold to {speech_threshold}")

        if silence_threshold is not None:
            if not 0.0 <= silence_threshold <= 1.0:
                raise ValueError("Silence threshold must be between 0.0 and 1.0")
            self._config.silence_threshold = silence_threshold
            logger.info(f"Updated silence threshold to {silence_threshold}")

    def get_confidence_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about model confidence values.

        Returns:
            Dict containing confidence statistics.
        """
        if not self._confidence_history:
            return {
                "mean_confidence": 0.0,
                "max_confidence": 0.0,
                "min_confidence": 0.0,
                "std_confidence": 0.0,
                "last_confidence": self._last_confidence,
                "history_length": 0,
            }

        conf_array = np.array(self._confidence_history)
        return {
            "mean_confidence": float(np.mean(conf_array)),
            "max_confidence": float(np.max(conf_array)),
            "min_confidence": float(np.min(conf_array)),
            "std_confidence": float(np.std(conf_array)),
            "last_confidence": self._last_confidence,
            "history_length": len(self._confidence_history),
            "speech_threshold": self._config.speech_threshold,
            "silence_threshold": self._config.silence_threshold,
        }

    def __repr__(self) -> str:
        """String representation of the Silero VAD detector."""
        return (
            f"SileroVAD("
            f"model={self._config.model_name}, "
            f"device={self._device}, "
            f"speech_threshold={self._config.speech_threshold:.3f}, "
            f"state={self.state.value}, "
            f"fallback={self._fallback_to_energy}, "
            f"initialized={self.is_initialized})"
        )
