"""
Voice Activity Detection (VAD) Module for Gianna

This module provides a comprehensive Voice Activity Detection system with
multiple advanced algorithms, extensive configuration options, and a clean
factory-based architecture following the project's design patterns.

Key Features:
- Multiple VAD algorithms (Energy, Spectral, WebRTC, Silero, Adaptive)
- Thread-safe implementations with comprehensive state management
- Configurable parameters and adaptive thresholds
- Real-time streaming support with callback system
- Optional dependency handling with graceful fallbacks
- Performance benchmarking and monitoring capabilities
- Multi-algorithm fusion with voting strategies
- Extensive audio processing utilities
- Legacy compatibility with existing VoiceActivityDetector

Main Components:
- BaseVAD: Abstract interface for all VAD implementations
- EnergyVAD: Energy-based VAD using RMS analysis
- SpectralVAD: Frequency-domain analysis with spectral features
- WebRtcVAD: Industry-standard WebRTC algorithm
- SileroVAD: Deep learning-based neural network VAD
- AdaptiveVAD: Multi-algorithm fusion with voting strategies
- Algorithm-specific configuration classes
- Factory functions: Easy creation of VAD instances
- Utilities: Audio processing and signal analysis tools

Example Usage:
    # Create a basic energy-based VAD
    vad = create_vad("energy", threshold=0.03, min_silence_duration=1.5)

    # Create a high-accuracy neural network VAD
    vad = create_vad("silero", speech_threshold=0.6)

    # Create an adaptive VAD combining multiple algorithms
    vad = create_vad("adaptive", algorithms=["webrtc", "spectral", "silero"])

    # Process audio chunk
    result = vad.process_stream(audio_chunk)

    # Check for voice activity
    if result.is_voice_active:
        print(f"Voice detected! Confidence: {result.confidence:.3f}")

    # Advanced configuration example
    from gianna.audio.vad import SpectralVADConfig
    config = SpectralVADConfig(
        threshold=0.3,
        fft_size=2048,
        use_spectral_flux=True,
        noise_floor_adaptation=True
    )
    vad = SpectralVAD(config)

Supported VAD Algorithms:
- "energy": Energy-based VAD using RMS energy analysis (always available)
- "spectral": Frequency-domain analysis with multiple spectral features
- "webrtc": Industry-standard WebRTC algorithm (requires webrtcvad)
- "silero": Deep learning-based neural network VAD (requires torch/torchaudio)
- "adaptive": Multi-algorithm fusion with configurable voting strategies

Performance Characteristics:
- Energy: Fast, low CPU, good for quiet environments
- Spectral: Medium CPU, excellent for noisy/music environments
- WebRTC: Very fast, industry standard, balanced accuracy/performance
- Silero: High CPU, highest accuracy, best for challenging conditions
- Adaptive: Variable CPU, excellent accuracy by combining multiple algorithms
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

# Setup logger first to avoid NameError issues
logger = logging.getLogger(__name__)

# Core VAD components
from .base import BaseVAD
from .energy_vad import EnergyVAD, VoiceActivityDetector, create_vad_detector
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

# Advanced VAD algorithms with optional dependency handling
try:
    from .spectral_vad import SpectralVAD, SpectralVADConfig

    _SPECTRAL_AVAILABLE = True
except ImportError:
    SpectralVAD = None
    SpectralVADConfig = None
    _SPECTRAL_AVAILABLE = False

try:
    from .webrtc_vad import WebRtcVAD, WebRtcVADConfig

    _WEBRTC_AVAILABLE = True
except ImportError:
    WebRtcVAD = None
    WebRtcVADConfig = None
    _WEBRTC_AVAILABLE = False

try:
    from .silero_vad import SileroVAD, SileroVADConfig

    _SILERO_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # Logger not yet defined, so we'll skip logging here
    SileroVAD = None
    SileroVADConfig = None
    _SILERO_AVAILABLE = False

try:
    from .adaptive_vad import (
        AdaptationMode,
        AdaptiveVAD,
        AdaptiveVADConfig,
        VotingStrategy,
    )

    _ADAPTIVE_AVAILABLE = True
except (ImportError, AttributeError):
    AdaptiveVAD = None
    AdaptiveVADConfig = None
    VotingStrategy = None
    AdaptationMode = None
    _ADAPTIVE_AVAILABLE = False

# Utilities - with graceful import handling
try:
    from .utils import (
        apply_bandpass_filter,
        calculate_audio_statistics,
        calculate_spectral_features,
        calculate_zero_crossing_rate,
        compute_mfcc_features,
        compute_spectral_centroid,
        convert_audio_format,
        detect_clipping,
        estimate_noise_floor,
        normalize_audio_chunk,
        preemphasis_filter,
        resample_audio,
        validate_audio_chunk,
    )

    _UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"VAD utilities not available: {e}")
    _UTILS_AVAILABLE = False

    # Create stub functions for missing utilities
    def validate_audio_chunk(*args, **kwargs):
        """Stub for missing validate_audio_chunk utility."""
        return True

    def convert_audio_format(*args, **kwargs):
        """Stub for missing convert_audio_format utility."""
        return args[0] if args else None

    def normalize_audio_chunk(chunk, *args, **kwargs):
        """Stub for missing normalize_audio_chunk utility."""
        return chunk

    # Create stubs for other utilities
    def resample_audio(chunk, *args, **kwargs):
        return chunk

    def detect_clipping(*args, **kwargs):
        return False

    def calculate_audio_statistics(*args, **kwargs):
        return {}

    def preemphasis_filter(chunk, *args, **kwargs):
        return chunk

    def apply_bandpass_filter(chunk, *args, **kwargs):
        return chunk

    def calculate_zero_crossing_rate(*args, **kwargs):
        return 0.0

    def compute_spectral_centroid(*args, **kwargs):
        return 0.0

    def calculate_spectral_features(*args, **kwargs):
        return {}

    def compute_mfcc_features(*args, **kwargs):
        return []

    def estimate_noise_floor(*args, **kwargs):
        return 0.0


# Calibration and metrics system - with graceful import handling
try:
    from .calibration import (
        CalibrationConfig,
        CalibrationResult,
        EnvironmentDetector,
        EnvironmentType,
        GroundTruthData,
        OptimizationMethod,
        VadCalibrator,
    )

    _CALIBRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"VAD calibration system not available: {e}")
    VadCalibrator = None
    EnvironmentDetector = None
    EnvironmentType = None
    OptimizationMethod = None
    CalibrationConfig = None
    CalibrationResult = None
    GroundTruthData = None
    _CALIBRATION_AVAILABLE = False

try:
    from .metrics import (
        AlertLevel,
        MetricThreshold,
        MetricType,
        PerformanceAlert,
        PerformanceMonitor,
        PerformanceReport,
        VadMetrics,
    )

    _METRICS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"VAD metrics system not available: {e}")
    VadMetrics = None
    PerformanceMonitor = None
    PerformanceReport = None
    PerformanceAlert = None
    MetricType = None
    AlertLevel = None
    MetricThreshold = None
    _METRICS_AVAILABLE = False

try:
    from .benchmark import (
        BenchmarkCategory,
        BenchmarkConfig,
        BenchmarkDataset,
        BenchmarkResult,
        DatasetGenerator,
        VadBenchmark,
    )

    _BENCHMARK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"VAD benchmarking system not available: {e}")
    VadBenchmark = None
    BenchmarkConfig = None
    BenchmarkResult = None
    BenchmarkDataset = None
    BenchmarkCategory = None
    DatasetGenerator = None
    _BENCHMARK_AVAILABLE = False

try:
    from .realtime_monitor import (
        AdaptationTrigger,
        MonitoringConfig,
        MonitoringSnapshot,
        MonitoringState,
        RealtimeMonitor,
    )

    _REALTIME_MONITOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"VAD real-time monitoring not available: {e}")
    RealtimeMonitor = None
    MonitoringConfig = None
    MonitoringSnapshot = None
    MonitoringState = None
    AdaptationTrigger = None
    _REALTIME_MONITOR_AVAILABLE = False

# Logger already defined at top of module

# Build VAD Algorithm Registry with available algorithms
_VAD_REGISTRY: Dict[str, Type[BaseVAD]] = {
    "energy": EnergyVAD,  # Always available
}

# Add optional algorithms if dependencies are available
if _SPECTRAL_AVAILABLE and SpectralVAD is not None:
    _VAD_REGISTRY["spectral"] = SpectralVAD

if _WEBRTC_AVAILABLE and WebRtcVAD is not None:
    _VAD_REGISTRY["webrtc"] = WebRtcVAD

if _SILERO_AVAILABLE and SileroVAD is not None:
    _VAD_REGISTRY["silero"] = SileroVAD

if _ADAPTIVE_AVAILABLE and AdaptiveVAD is not None:
    _VAD_REGISTRY["adaptive"] = AdaptiveVAD

# Version info
__version__ = "1.0.0"
__author__ = "Gianna Development Team"

# Build public API list with available components
__all__ = [
    # Core classes (always available)
    "BaseVAD",
    "EnergyVAD",
    "VADConfig",
    "VADResult",
    "VADStatistics",
    # Enums and types
    "VADAlgorithm",
    "VADState",
    "VADEventType",
    "AudioChunk",
    "VADCallback",
    "EventCallback",
    # Factory functions
    "create_vad",
    "create_vad_config",
    "create_streaming_vad",
    "create_vad_pipeline",
    "create_monitored_vad",
    "create_benchmarking_suite",
    "create_production_vad",
    "get_available_algorithms",
    "register_vad_algorithm",
    "get_vad_registry",
    "is_algorithm_available",
    "get_algorithm_info",
    # Legacy compatibility
    "VoiceActivityDetector",
    "create_vad_detector",
    # Convenience functions
    "create_energy_vad",
    "create_realtime_vad",
    # Utilities
    "validate_audio_chunk",
    "convert_audio_format",
    "normalize_audio_chunk",
    "resample_audio",
    "detect_clipping",
    "calculate_audio_statistics",
    "preemphasis_filter",
    "apply_bandpass_filter",
    "calculate_zero_crossing_rate",
    "compute_spectral_centroid",
    "calculate_spectral_features",
    "compute_mfcc_features",
    "estimate_noise_floor",
]

# Add calibration and metrics system exports if available
if _CALIBRATION_AVAILABLE:
    __all__.extend(
        [
            "VadCalibrator",
            "EnvironmentDetector",
            "EnvironmentType",
            "OptimizationMethod",
            "CalibrationConfig",
            "CalibrationResult",
            "GroundTruthData",
        ]
    )

if _METRICS_AVAILABLE:
    __all__.extend(
        [
            "VadMetrics",
            "PerformanceMonitor",
            "PerformanceReport",
            "PerformanceAlert",
            "MetricType",
            "AlertLevel",
            "MetricThreshold",
        ]
    )

if _BENCHMARK_AVAILABLE:
    __all__.extend(
        [
            "VadBenchmark",
            "BenchmarkConfig",
            "BenchmarkResult",
            "BenchmarkDataset",
            "BenchmarkCategory",
            "DatasetGenerator",
        ]
    )

if _REALTIME_MONITOR_AVAILABLE:
    __all__.extend(
        [
            "RealtimeMonitor",
            "MonitoringConfig",
            "MonitoringSnapshot",
            "MonitoringState",
            "AdaptationTrigger",
        ]
    )

# Add optional algorithm classes to exports if available
if _SPECTRAL_AVAILABLE and SpectralVAD is not None:
    __all__.extend(["SpectralVAD", "SpectralVADConfig", "create_spectral_vad"])

if _WEBRTC_AVAILABLE and WebRtcVAD is not None:
    __all__.extend(["WebRtcVAD", "WebRtcVADConfig", "create_webrtc_vad"])

if _SILERO_AVAILABLE and SileroVAD is not None:
    __all__.extend(["SileroVAD", "SileroVADConfig", "create_silero_vad"])

if _ADAPTIVE_AVAILABLE and AdaptiveVAD is not None:
    __all__.extend(
        [
            "AdaptiveVAD",
            "AdaptiveVADConfig",
            "VotingStrategy",
            "AdaptationMode",
            "create_adaptive_vad",
        ]
    )

# Add legacy compatibility functions
__all__.extend(["get_vad_instance", "get_vad_detector"])


def create_vad(
    algorithm: str = "energy", config: Optional[VADConfig] = None, **kwargs
) -> BaseVAD:
    """
    Factory function to create VAD detectors.

    This is the main entry point for creating VAD instances. It provides
    a clean, extensible interface that follows the project's factory pattern.

    Args:
        algorithm (str): VAD algorithm to use ("energy", "spectral", etc.).
        config (Optional[VADConfig]): Complete configuration object.
        **kwargs: Configuration parameters (merged with config if provided).

    Returns:
        BaseVAD: Configured VAD detector instance.

    Raises:
        ValueError: If algorithm is not supported or configuration is invalid.
        RuntimeError: If VAD initialization fails.

    Example:
        # Simple creation
        vad = create_vad("energy", threshold=0.03)

        # With full configuration
        config = VADConfig(threshold=0.025, min_silence_duration=1.5)
        vad = create_vad("energy", config=config)

        # Override config parameters
        vad = create_vad("energy", config=config, threshold=0.04)
    """
    # Normalize algorithm name
    algorithm = algorithm.lower().strip()

    # Check if algorithm is supported
    if algorithm not in _VAD_REGISTRY:
        available = ", ".join(_VAD_REGISTRY.keys())
        raise ValueError(
            f"Unsupported VAD algorithm: '{algorithm}'. "
            f"Available algorithms: {available}"
        )

    try:
        # Merge configuration
        if config is not None:
            # Create a copy to avoid modifying the original
            merged_config = VADConfig(
                threshold=kwargs.get("threshold", config.threshold),
                min_silence_duration=kwargs.get(
                    "min_silence_duration", config.min_silence_duration
                ),
                min_speech_duration=kwargs.get(
                    "min_speech_duration", config.min_speech_duration
                ),
                sample_rate=kwargs.get("sample_rate", config.sample_rate),
                chunk_size=kwargs.get("chunk_size", config.chunk_size),
                channels=kwargs.get("channels", config.channels),
                algorithm_params=kwargs.get(
                    "algorithm_params", config.algorithm_params.copy()
                ),
                enable_callbacks=kwargs.get(
                    "enable_callbacks", config.enable_callbacks
                ),
                callback_timeout=kwargs.get(
                    "callback_timeout", config.callback_timeout
                ),
                buffer_size=kwargs.get("buffer_size", config.buffer_size),
                max_history_length=kwargs.get(
                    "max_history_length", config.max_history_length
                ),
            )
        else:
            # Create new configuration from kwargs
            merged_config = VADConfig(**kwargs)

        # Get VAD class and create instance
        VADClass = _VAD_REGISTRY[algorithm]
        vad_instance = VADClass(config=merged_config)

        # Initialize if not already done
        if not vad_instance.is_initialized:
            if not vad_instance.initialize():
                raise RuntimeError(f"Failed to initialize {algorithm} VAD instance")

        logger.info(
            f"Created {algorithm} VAD with config: "
            f"threshold={merged_config.threshold}, "
            f"min_silence={merged_config.min_silence_duration}s"
        )

        return vad_instance

    except Exception as e:
        logger.error(f"Failed to create {algorithm} VAD: {e}")
        raise RuntimeError(f"Failed to create {algorithm} VAD: {e}") from e


def get_available_algorithms() -> list[str]:
    """
    Get list of available VAD algorithms.

    Returns:
        list[str]: List of supported algorithm names.
    """
    return list(_VAD_REGISTRY.keys())


def register_vad_algorithm(name: str, vad_class: Type[BaseVAD]) -> None:
    """
    Register a new VAD algorithm.

    This allows extending the VAD system with custom implementations
    while maintaining the same factory interface.

    Args:
        name (str): Algorithm name (will be normalized to lowercase).
        vad_class (Type[BaseVAD]): VAD implementation class.

    Raises:
        ValueError: If the class doesn't inherit from BaseVAD.
        RuntimeWarning: If algorithm name is already registered.
    """
    # Validate input
    if not issubclass(vad_class, BaseVAD):
        raise ValueError(f"VAD class must inherit from BaseVAD, got {vad_class}")

    # Normalize name
    name = name.lower().strip()

    # Check for existing registration
    if name in _VAD_REGISTRY:
        logger.warning(f"Overriding existing VAD algorithm: {name}")

    # Register
    _VAD_REGISTRY[name] = vad_class
    logger.info(f"Registered VAD algorithm: {name} -> {vad_class.__name__}")


def create_energy_vad(
    threshold: float = 0.02,
    min_silence_duration: float = 1.0,
    min_speech_duration: float = 0.1,
    sample_rate: int = 16000,
    adaptive_threshold: bool = False,
    **kwargs,
) -> EnergyVAD:
    """
    Convenience function to create Energy-based VAD with common parameters.

    Args:
        threshold (float): Energy threshold for voice detection.
        min_silence_duration (float): Minimum silence duration in seconds.
        min_speech_duration (float): Minimum speech duration in seconds.
        sample_rate (int): Audio sample rate in Hz.
        adaptive_threshold (bool): Enable adaptive threshold based on noise floor.
        **kwargs: Additional configuration parameters.

    Returns:
        EnergyVAD: Configured energy-based VAD instance.
    """
    config = VADConfig(
        threshold=threshold,
        min_silence_duration=min_silence_duration,
        min_speech_duration=min_speech_duration,
        sample_rate=sample_rate,
        **kwargs,
    )

    vad = EnergyVAD(config)

    # Enable adaptive threshold if requested
    if adaptive_threshold:
        logger.info("Adaptive threshold will be enabled after noise floor estimation")

    return vad


def create_realtime_vad(
    algorithm: str = "energy",
    latency_mode: str = "low",  # "low", "balanced", "high_quality"
    **kwargs,
) -> BaseVAD:
    """
    Create a VAD optimized for real-time processing.

    Args:
        algorithm (str): VAD algorithm to use.
        latency_mode (str): Optimization mode for latency vs quality.
        **kwargs: Additional configuration parameters.

    Returns:
        BaseVAD: VAD instance optimized for real-time use.
    """
    # Preset configurations for different latency modes
    latency_presets = {
        "low": {
            "chunk_size": 512,
            "min_silence_duration": 0.3,
            "min_speech_duration": 0.05,
            "buffer_size": 2048,
            "max_history_length": 50,
        },
        "balanced": {
            "chunk_size": 1024,
            "min_silence_duration": 0.5,
            "min_speech_duration": 0.1,
            "buffer_size": 4096,
            "max_history_length": 100,
        },
        "high_quality": {
            "chunk_size": 2048,
            "min_silence_duration": 1.0,
            "min_speech_duration": 0.2,
            "buffer_size": 8192,
            "max_history_length": 200,
        },
    }

    if latency_mode not in latency_presets:
        raise ValueError(f"Unknown latency mode: {latency_mode}")

    # Merge preset with user parameters
    preset = latency_presets[latency_mode]
    merged_kwargs = {**preset, **kwargs}

    logger.info(f"Creating real-time VAD with {latency_mode} latency mode")
    return create_vad(algorithm, **merged_kwargs)


# Advanced algorithm-specific factory functions
def create_spectral_vad(
    threshold: float = 0.3,
    fft_size: int = 1024,
    use_spectral_flux: bool = True,
    noise_floor_adaptation: bool = True,
    **kwargs,
) -> BaseVAD:
    """
    Convenience function to create Spectral VAD with common parameters.

    Args:
        threshold (float): Voice activity threshold.
        fft_size (int): FFT window size for spectral analysis.
        use_spectral_flux (bool): Enable spectral flux feature.
        noise_floor_adaptation (bool): Enable adaptive noise floor.
        **kwargs: Additional configuration parameters.

    Returns:
        BaseVAD: Configured spectral VAD instance.

    Raises:
        RuntimeError: If spectral VAD is not available.
    """
    if not _SPECTRAL_AVAILABLE or SpectralVAD is None:
        raise RuntimeError("Spectral VAD not available. scipy is required.")

    return create_vad(
        "spectral",
        threshold=threshold,
        fft_size=fft_size,
        use_spectral_flux=use_spectral_flux,
        noise_floor_adaptation=noise_floor_adaptation,
        **kwargs,
    )


def create_webrtc_vad(
    aggressiveness: int = 2,
    frame_duration_ms: int = 20,
    confidence_smoothing: bool = True,
    **kwargs,
) -> BaseVAD:
    """
    Convenience function to create WebRTC VAD with common parameters.

    Args:
        aggressiveness (int): WebRTC aggressiveness level (0-3).
        frame_duration_ms (int): Frame duration in milliseconds (10, 20, or 30).
        confidence_smoothing (bool): Enable confidence smoothing.
        **kwargs: Additional configuration parameters.

    Returns:
        BaseVAD: Configured WebRTC VAD instance.

    Raises:
        RuntimeError: If WebRTC VAD is not available.
    """
    if not _WEBRTC_AVAILABLE or WebRtcVAD is None:
        raise RuntimeError("WebRTC VAD not available. webrtcvad library is required.")

    return create_vad(
        "webrtc",
        aggressiveness=aggressiveness,
        frame_duration_ms=frame_duration_ms,
        confidence_smoothing=confidence_smoothing,
        **kwargs,
    )


def create_silero_vad(
    speech_threshold: float = 0.5,
    device: str = "auto",
    model_name: str = "silero_vad",
    **kwargs,
) -> BaseVAD:
    """
    Convenience function to create Silero VAD with common parameters.

    Args:
        speech_threshold (float): Speech detection threshold.
        device (str): Device for inference ("cpu", "cuda", "auto").
        model_name (str): Silero model name to use.
        **kwargs: Additional configuration parameters.

    Returns:
        BaseVAD: Configured Silero VAD instance.

    Raises:
        RuntimeError: If Silero VAD is not available.
    """
    if not _SILERO_AVAILABLE or SileroVAD is None:
        raise RuntimeError(
            "Silero VAD not available. torch and torchaudio are required."
        )

    return create_vad(
        "silero",
        speech_threshold=speech_threshold,
        device=device,
        model_name=model_name,
        **kwargs,
    )


def create_adaptive_vad(
    algorithms: list = None,
    voting_strategy: str = "weighted",
    adaptation_mode: str = "hybrid",
    **kwargs,
) -> BaseVAD:
    """
    Convenience function to create Adaptive VAD with common parameters.

    Args:
        algorithms (list): List of algorithms to combine.
        voting_strategy (str): Strategy for combining results.
        adaptation_mode (str): How to adapt algorithm weights.
        **kwargs: Additional configuration parameters.

    Returns:
        BaseVAD: Configured Adaptive VAD instance.

    Raises:
        RuntimeError: If Adaptive VAD is not available.
    """
    if not _ADAPTIVE_AVAILABLE or AdaptiveVAD is None:
        raise RuntimeError("Adaptive VAD not available.")

    # Default to available algorithms if not specified
    if algorithms is None:
        algorithms = []
        if _WEBRTC_AVAILABLE:
            algorithms.append("webrtc")
        if _SPECTRAL_AVAILABLE:
            algorithms.append("spectral")
        if _SILERO_AVAILABLE:
            algorithms.append("silero")

        # Ensure at least energy is included
        if not algorithms:
            algorithms = ["energy"]
        elif "energy" not in algorithms:
            algorithms.append("energy")

    return create_vad(
        "adaptive",
        algorithms=algorithms,
        voting_strategy=voting_strategy,
        adaptation_mode=adaptation_mode,
        **kwargs,
    )


# Configuration management functions
def create_vad_config(algorithm: str, preset: str = "balanced", **kwargs) -> VADConfig:
    """
    Create a VAD configuration with preset optimizations.

    Args:
        algorithm (str): Target VAD algorithm.
        preset (str): Preset configuration ("fast", "balanced", "accurate").
        **kwargs: Override specific configuration parameters.

    Returns:
        VADConfig: Configured VAD configuration object.
    """
    # Preset configurations optimized for different use cases
    presets = {
        "fast": {
            "threshold": 0.03,
            "min_silence_duration": 0.3,
            "min_speech_duration": 0.05,
            "chunk_size": 512,
            "buffer_size": 2048,
            "max_history_length": 50,
        },
        "balanced": {
            "threshold": 0.02,
            "min_silence_duration": 1.0,
            "min_speech_duration": 0.1,
            "chunk_size": 1024,
            "buffer_size": 4096,
            "max_history_length": 100,
        },
        "accurate": {
            "threshold": 0.015,
            "min_silence_duration": 1.5,
            "min_speech_duration": 0.2,
            "chunk_size": 2048,
            "buffer_size": 8192,
            "max_history_length": 200,
        },
    }

    # Algorithm-specific optimizations
    algorithm_optimizations = {
        "energy": {"algorithm_params": {"adaptive_threshold": True}},
        "spectral": {"algorithm_params": {"fft_size": 1024, "use_spectral_flux": True}},
        "webrtc": {"algorithm_params": {"aggressiveness": 2, "frame_duration_ms": 20}},
        "silero": {"algorithm_params": {"speech_threshold": 0.5, "device": "auto"}},
        "adaptive": {
            "algorithm_params": {
                "voting_strategy": "weighted",
                "adaptation_mode": "hybrid",
            }
        },
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    # Start with preset configuration
    config_params = presets[preset].copy()

    # Add algorithm-specific optimizations
    if algorithm in algorithm_optimizations:
        algo_params = algorithm_optimizations[algorithm]
        config_params.update(algo_params)

        # Merge algorithm_params if both exist
        if "algorithm_params" in config_params and "algorithm_params" in kwargs:
            merged_algo_params = config_params["algorithm_params"].copy()
            merged_algo_params.update(kwargs["algorithm_params"])
            kwargs["algorithm_params"] = merged_algo_params

    # Apply user overrides
    config_params.update(kwargs)

    logger.info(f"Created {preset} configuration for {algorithm} VAD")
    return VADConfig(**config_params)


def create_streaming_vad(
    algorithm: str = "energy",
    sample_rate: int = 16000,
    chunk_size: int = 1024,
    **kwargs,
) -> BaseVAD:
    """
    Create a VAD optimized for streaming applications.

    Args:
        algorithm (str): VAD algorithm to use.
        sample_rate (int): Audio sample rate.
        chunk_size (int): Processing chunk size.
        **kwargs: Additional configuration parameters.

    Returns:
        BaseVAD: VAD instance optimized for streaming.
    """
    streaming_config = {
        "sample_rate": sample_rate,
        "chunk_size": chunk_size,
        "min_silence_duration": 0.5,
        "min_speech_duration": 0.1,
        "enable_callbacks": True,
        "buffer_size": chunk_size * 4,
        "max_history_length": 100,
    }

    # Merge with user parameters
    streaming_config.update(kwargs)

    logger.info(f"Creating streaming {algorithm} VAD")
    return create_vad(algorithm, **streaming_config)


# Legacy compatibility functions - ensure backward compatibility
def get_vad_instance(*args, **kwargs):
    """
    Legacy function for backward compatibility.

    Deprecated: Use create_vad() instead.
    """
    import warnings

    warnings.warn(
        "get_vad_instance() is deprecated and will be removed in v2.0. "
        "Use create_vad() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning("get_vad_instance() is deprecated. Use create_vad() instead.")
    return create_vad(*args, **kwargs)


def get_vad_detector(*args, **kwargs):
    """
    Alternative legacy function name for backward compatibility.

    Deprecated: Use create_vad() instead.
    """
    import warnings

    warnings.warn(
        "get_vad_detector() is deprecated and will be removed in v2.0. "
        "Use create_vad() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning("get_vad_detector() is deprecated. Use create_vad() instead.")
    return create_vad(*args, **kwargs)


# Initialize logging for the module
def _setup_logging():
    """Setup module-specific logging configuration."""
    # Only setup if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


# Module initialization
_setup_logging()
logger.info(f"Gianna VAD module initialized (v{__version__})")

# Log available algorithms and their status
available_algorithms = get_available_algorithms()
logger.info(f"Available algorithms: {', '.join(available_algorithms)}")

# Log optional dependency status
dependency_status = []
if _SPECTRAL_AVAILABLE:
    dependency_status.append("spectral (scipy available)")
else:
    dependency_status.append("spectral (scipy required)")

if _WEBRTC_AVAILABLE:
    dependency_status.append("webrtc (webrtcvad available)")
else:
    dependency_status.append("webrtc (webrtcvad required)")

if _SILERO_AVAILABLE:
    dependency_status.append("silero (torch/torchaudio available)")
else:
    dependency_status.append("silero (torch/torchaudio required)")

if _ADAPTIVE_AVAILABLE:
    dependency_status.append("adaptive (available)")
else:
    dependency_status.append("adaptive (unavailable)")

logger.debug(f"Algorithm dependency status: {', '.join(dependency_status)}")

# Log advanced system status
advanced_systems = []
if _CALIBRATION_AVAILABLE:
    advanced_systems.append("calibration system")
if _METRICS_AVAILABLE:
    advanced_systems.append("metrics system")
if _BENCHMARK_AVAILABLE:
    advanced_systems.append("benchmarking system")
if _REALTIME_MONITOR_AVAILABLE:
    advanced_systems.append("real-time monitoring")

if advanced_systems:
    logger.info(f"Advanced systems available: {', '.join(advanced_systems)}")
else:
    logger.info("Advanced systems require scipy and sklearn dependencies")


# Expose key classes at module level for convenience
VAD = BaseVAD  # Alias for the abstract base class
EnergyBasedVAD = EnergyVAD  # More descriptive alias


# Factory registry management functions
def get_vad_registry() -> Dict[str, Type[BaseVAD]]:
    """
    Get a copy of the current VAD registry.

    Returns:
        Dict[str, Type[BaseVAD]]: Copy of the VAD algorithm registry.
    """
    return _VAD_REGISTRY.copy()


def is_algorithm_available(algorithm: str) -> bool:
    """
    Check if a VAD algorithm is available.

    Args:
        algorithm (str): Algorithm name to check.

    Returns:
        bool: True if algorithm is available, False otherwise.
    """
    return algorithm.lower().strip() in _VAD_REGISTRY


def get_algorithm_info(algorithm: str = None) -> Dict[str, Any]:
    """
    Get information about VAD algorithms.

    Args:
        algorithm (str, optional): Specific algorithm to get info for.
                                 If None, returns info for all algorithms.

    Returns:
        Dict[str, Any]: Algorithm information including availability and features.
    """

    def get_single_algorithm_info(algo_name: str) -> Dict[str, Any]:
        if algo_name not in _VAD_REGISTRY:
            return {"available": False, "reason": "Algorithm not registered"}

        algo_class = _VAD_REGISTRY[algo_name]
        info = {
            "available": True,
            "class_name": algo_class.__name__,
            "module": algo_class.__module__,
        }

        # Add algorithm-specific information
        if hasattr(algo_class, "get_algorithm_info"):
            try:
                info.update(algo_class.get_algorithm_info())
            except Exception as e:
                logger.warning(f"Failed to get info for {algo_name}: {e}")

        return info

    if algorithm is not None:
        return get_single_algorithm_info(algorithm.lower().strip())

    # Return info for all algorithms
    all_algorithms = ["energy", "spectral", "webrtc", "silero", "adaptive"]
    return {algo: get_single_algorithm_info(algo) for algo in all_algorithms}


# Advanced factory functions
def create_vad_pipeline(
    algorithms: list = None, fallback_algorithm: str = "energy", **kwargs
) -> BaseVAD:
    """
    Create a VAD pipeline with fallback support.

    Args:
        algorithms (list): Preferred algorithms in order of priority.
        fallback_algorithm (str): Algorithm to use if others fail.
        **kwargs: Configuration parameters.

    Returns:
        BaseVAD: VAD instance with the best available algorithm.
    """
    if algorithms is None:
        algorithms = ["silero", "webrtc", "spectral", "energy"]

    # Try algorithms in order of preference
    for algorithm in algorithms:
        if is_algorithm_available(algorithm):
            try:
                logger.info(f"Using {algorithm} VAD as primary algorithm")
                return create_vad(algorithm, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to create {algorithm} VAD: {e}")
                continue

    # Use fallback algorithm
    logger.warning(f"Using fallback algorithm: {fallback_algorithm}")
    return create_vad(fallback_algorithm, **kwargs)


def create_monitored_vad(
    algorithm: str = "energy",
    enable_calibration: bool = True,
    enable_realtime_monitor: bool = True,
    **kwargs,
) -> Tuple[BaseVAD, Optional["RealtimeMonitor"]]:
    """
    Create a VAD with integrated monitoring and calibration.

    Args:
        algorithm (str): VAD algorithm to use.
        enable_calibration (bool): Enable automatic calibration.
        enable_realtime_monitor (bool): Enable real-time monitoring.
        **kwargs: VAD configuration parameters.

    Returns:
        Tuple of (vad_instance, monitor_instance) or (vad_instance, None).
    """
    # Create base VAD
    vad = create_vad(algorithm, **kwargs)

    # Create monitor if available and requested
    if enable_realtime_monitor and _REALTIME_MONITOR_AVAILABLE:
        monitor_config = MonitoringConfig(
            enable_auto_adaptation=enable_calibration,
            **{k: v for k, v in kwargs.items() if k.startswith("monitoring_")},
        )
        monitor = RealtimeMonitor(vad, monitor_config)
        logger.info(f"Created monitored {algorithm} VAD with real-time adaptation")
        return vad, monitor

    logger.info(f"Created {algorithm} VAD (monitoring not available)")
    return vad, None


def create_benchmarking_suite(
    algorithms: List[str] = None, categories: List[str] = None, **kwargs
) -> Optional["VadBenchmark"]:
    """
    Create a comprehensive VAD benchmarking suite.

    Args:
        algorithms (List[str]): Algorithms to benchmark.
        categories (List[str]): Benchmark categories to include.
        **kwargs: Benchmark configuration parameters.

    Returns:
        VadBenchmark instance if available, None otherwise.
    """
    if not _BENCHMARK_AVAILABLE:
        logger.warning("Benchmarking system not available")
        return None

    # Default configurations
    if algorithms is None:
        algorithms = get_available_algorithms()

    if categories is None:
        categories = ["accuracy", "performance", "robustness"]

    # Convert category strings to enums
    category_enums = []
    for cat in categories:
        try:
            category_enums.append(BenchmarkCategory(cat))
        except ValueError:
            logger.warning(f"Unknown benchmark category: {cat}")

    # Create benchmark configuration
    config = BenchmarkConfig(categories=category_enums, **kwargs)

    benchmark = VadBenchmark(config)
    logger.info(f"Created benchmarking suite for algorithms: {', '.join(algorithms)}")
    return benchmark


def create_production_vad(
    algorithm: str = "energy", environment: str = "auto", **kwargs
) -> Tuple[BaseVAD, Optional["RealtimeMonitor"], Optional["VadCalibrator"]]:
    """
    Create a production-ready VAD with full monitoring and calibration.

    Args:
        algorithm (str): VAD algorithm to use.
        environment (str): Target environment ("auto", "office", "noisy", etc.).
        **kwargs: Configuration parameters.

    Returns:
        Tuple of (vad_instance, monitor, calibrator).
    """
    # Environment-specific configurations
    env_configs = {
        "studio": {"threshold": 0.015, "min_silence_duration": 1.5},
        "office": {"threshold": 0.025, "min_silence_duration": 1.0},
        "noisy": {"threshold": 0.04, "min_silence_duration": 0.5},
        "car": {"threshold": 0.05, "min_silence_duration": 0.3},
    }

    # Apply environment-specific configuration
    if environment in env_configs:
        env_config = env_configs[environment]
        kwargs = {**env_config, **kwargs}
        logger.info(f"Applied {environment} environment configuration")

    # Create VAD with monitoring
    vad, monitor = create_monitored_vad(
        algorithm=algorithm,
        enable_calibration=True,
        enable_realtime_monitor=True,
        **kwargs,
    )

    # Create calibrator if available
    calibrator = None
    if _CALIBRATION_AVAILABLE:
        calibrator = VadCalibrator()
        logger.info("Created production VAD with full calibration support")

    return vad, monitor, calibrator


# Module metadata
__title__ = "Gianna VAD"
__description__ = "Voice Activity Detection system for Gianna assistant"
__url__ = "https://github.com/yourusername/gianna"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Gianna Development Team"
