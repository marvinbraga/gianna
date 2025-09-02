"""
Gianna Audio Processing Module

This module provides comprehensive audio processing capabilities including:
- Real-time streaming voice pipeline
- Voice Activity Detection (VAD) with multiple algorithms
- Audio buffering and management
- Advanced VAD factory patterns and configurations

Main Components:
- StreamingVoicePipeline: Complete voice conversation pipeline
- AudioBuffer: Thread-safe audio buffering
- PipelineState: State management for streaming operations
- VAD System: Comprehensive voice activity detection with multiple algorithms

VAD Algorithms:
- Energy-based VAD (always available)
- Spectral VAD (requires scipy)
- WebRTC VAD (requires webrtcvad)
- Silero VAD (requires torch/torchaudio)
- Adaptive VAD (combines multiple algorithms)

Example:
    # Create a streaming voice pipeline
    pipeline = create_voice_assistant(model_name="gpt35")

    # Create a VAD detector
    from gianna.audio.vad import create_vad
    vad = create_vad("energy", threshold=0.03)

    # Create with configuration
    from gianna.audio.vad import create_vad_config
    config = create_vad_config("energy", preset="balanced")
    vad = create_vad("energy", config=config)
"""

from gianna.audio.streaming import (
    AudioBuffer,
    PipelineState,
    StreamingVoicePipeline,
    create_streaming_voice_assistant,
    create_voice_assistant,
)

# Import VAD system with fallback handling
try:
    from gianna.audio.vad import (  # Core VAD classes; Factory functions; Utility functions; Legacy compatibility
        BaseVAD,
        EnergyVAD,
        VADConfig,
        VADResult,
        VADStatistics,
        VoiceActivityDetector,
        create_realtime_vad,
        create_streaming_vad,
        create_vad,
        create_vad_config,
        create_vad_detector,
        create_vad_pipeline,
        get_algorithm_info,
        get_available_algorithms,
        is_algorithm_available,
    )

    _VAD_AVAILABLE = True

    # Add VAD exports to __all__
    _VAD_EXPORTS = [
        # Core VAD classes
        "BaseVAD",
        "EnergyVAD",
        "VADConfig",
        "VADResult",
        "VADStatistics",
        # Factory functions
        "create_vad",
        "create_vad_config",
        "create_streaming_vad",
        "create_realtime_vad",
        "create_vad_pipeline",
        # Utility functions
        "get_available_algorithms",
        "is_algorithm_available",
        "get_algorithm_info",
        # Legacy compatibility
        "VoiceActivityDetector",
        "create_vad_detector",
    ]

    # Try to import optional VAD algorithms
    try:
        from gianna.audio.vad import SpectralVAD, SpectralVADConfig, create_spectral_vad

        _VAD_EXPORTS.extend(["SpectralVAD", "SpectralVADConfig", "create_spectral_vad"])
    except ImportError:
        pass

    try:
        from gianna.audio.vad import WebRtcVAD, WebRtcVADConfig, create_webrtc_vad

        _VAD_EXPORTS.extend(["WebRtcVAD", "WebRtcVADConfig", "create_webrtc_vad"])
    except ImportError:
        pass

    try:
        from gianna.audio.vad import SileroVAD, SileroVADConfig, create_silero_vad

        _VAD_EXPORTS.extend(["SileroVAD", "SileroVADConfig", "create_silero_vad"])
    except ImportError:
        pass

    try:
        from gianna.audio.vad import AdaptiveVAD, AdaptiveVADConfig, create_adaptive_vad

        _VAD_EXPORTS.extend(["AdaptiveVAD", "AdaptiveVADConfig", "create_adaptive_vad"])
    except ImportError:
        pass

except ImportError as e:
    import warnings

    warnings.warn(f"VAD system not available: {e}", ImportWarning)
    _VAD_AVAILABLE = False
    _VAD_EXPORTS = []

__all__ = [
    # Core streaming components
    "StreamingVoicePipeline",
    "AudioBuffer",
    "PipelineState",
    "create_voice_assistant",
    "create_streaming_voice_assistant",
] + _VAD_EXPORTS

# Version info
__version__ = "1.0.0"
