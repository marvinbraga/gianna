"""
Gianna Audio Processing Module

This module provides comprehensive audio processing capabilities including:
- Real-time streaming voice pipeline
- Voice Activity Detection (VAD)
- Audio buffering and management

Main Components:
- StreamingVoicePipeline: Complete voice conversation pipeline
- AudioBuffer: Thread-safe audio buffering
- PipelineState: State management for streaming operations
"""

from gianna.audio.streaming import (
    AudioBuffer,
    PipelineState,
    StreamingVoicePipeline,
    create_voice_assistant,
)

__all__ = [
    "StreamingVoicePipeline",
    "AudioBuffer",
    "PipelineState",
    "create_voice_assistant",
]

# Version info
__version__ = "1.0.0"
