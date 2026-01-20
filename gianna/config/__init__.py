"""
Configuration management for Gianna.

This module provides configuration loading, validation, and management
for the Gianna voice assistant framework, with special support for
Voice Activity Detection (VAD) configurations.
"""

from .loader import ConfigLoader
from .settings import (
    AgentSettings,
    AudioSettings,
    CacheBackend,
    CacheSettings,
    EmbeddingProvider,
    LLMProvider,
    LLMSettings,
    LogFormat,
    MemorySettings,
    MonitoringSettings,
    PerformanceSettings,
    SecuritySettings,
    Settings,
    STTProvider,
    TTSProvider,
    VADAlgorithmType,
    VectorStoreProvider,
    generate_config_template,
    get_settings,
    reset_settings,
    validate_config,
)
from .vad_config import VADConfigManager
from .validator import ConfigValidator

__all__ = [
    # Existing exports
    "ConfigLoader",
    "ConfigValidator",
    "VADConfigManager",
    # Main settings
    "Settings",
    "get_settings",
    "reset_settings",
    "validate_config",
    "generate_config_template",
    # Sub-settings
    "LLMSettings",
    "AudioSettings",
    "MemorySettings",
    "CacheSettings",
    "SecuritySettings",
    "MonitoringSettings",
    "PerformanceSettings",
    "AgentSettings",
    # Enums
    "LLMProvider",
    "TTSProvider",
    "STTProvider",
    "VADAlgorithmType",
    "EmbeddingProvider",
    "VectorStoreProvider",
    "CacheBackend",
    "LogFormat",
]
