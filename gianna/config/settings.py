"""
Centralized Configuration System for Gianna

This module provides a unified configuration system with:
- Hierarchical configuration (defaults → env → config file → CLI args)
- Pydantic validation
- Environment variable support
- YAML/JSON config file loading
- Runtime configuration updates
- Configuration export/import

Usage:
    >>> from gianna.config.settings import get_settings, Settings
    >>>
    >>> # Get global settings (singleton)
    >>> settings = get_settings()
    >>>
    >>> # Access configuration
    >>> print(settings.llm.default_provider)
    >>> print(settings.audio.sample_rate)
    >>>
    >>> # Override from environment
    >>> # GIANNA_LLM__DEFAULT_PROVIDER=anthropic python main.py
"""

import json
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union

import yaml
from loguru import logger
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# LLM Configuration
# =============================================================================


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    NVIDIA = "nvidia"
    XAI = "xai"
    COHERE = "cohere"
    OLLAMA = "ollama"


class LLMSettings(BaseModel):
    """LLM-related configuration."""

    default_provider: LLMProvider = LLMProvider.OPENAI
    default_model: str = "gpt-4"

    # API Keys (use environment variables)
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    google_api_key: Optional[SecretStr] = None
    groq_api_key: Optional[SecretStr] = None
    nvidia_api_key: Optional[SecretStr] = None
    xai_api_key: Optional[SecretStr] = None
    cohere_api_key: Optional[SecretStr] = None

    # Model settings
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    timeout_seconds: float = Field(default=30.0, ge=1.0)

    # Retry settings
    max_retries: int = Field(default=3, ge=0)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1)

    # Ollama settings (for local models)
    ollama_base_url: str = "http://localhost:11434"


# =============================================================================
# Audio Configuration
# =============================================================================


class TTSProvider(str, Enum):
    """Supported TTS providers."""

    GOOGLE = "google"
    ELEVENLABS = "elevenlabs"
    LOCAL = "local"


class STTProvider(str, Enum):
    """Supported STT providers."""

    WHISPER_LOCAL = "whisper_local"
    WHISPER_API = "whisper_api"
    GOOGLE = "google"


class VADAlgorithmType(str, Enum):
    """VAD algorithm types."""

    ENERGY = "energy"
    SPECTRAL = "spectral"
    ML = "ml"
    WEBRTC = "webrtc"
    SILERO = "silero"
    ADAPTIVE = "adaptive"


class AudioSettings(BaseModel):
    """Audio-related configuration."""

    # General audio settings
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    chunk_size: int = Field(default=1024, ge=256, le=8192)

    # TTS settings
    tts_provider: TTSProvider = TTSProvider.GOOGLE
    tts_language: str = "pt-BR"
    tts_voice: Optional[str] = None
    elevenlabs_api_key: Optional[SecretStr] = None

    # STT settings
    stt_provider: STTProvider = STTProvider.WHISPER_LOCAL
    stt_language: str = "pt"
    whisper_model: str = "base"

    # VAD settings
    vad_algorithm: VADAlgorithmType = VADAlgorithmType.ADAPTIVE
    vad_threshold: float = Field(default=0.02, ge=0.0, le=1.0)
    vad_min_silence_duration: float = Field(default=1.0, ge=0.1)
    vad_min_speech_duration: float = Field(default=0.1, ge=0.01)

    # Memory pool settings
    enable_memory_pool: bool = True
    memory_pool_size: int = Field(default=32, ge=8, le=256)


# =============================================================================
# Memory Configuration
# =============================================================================


class EmbeddingProvider(str, Enum):
    """Embedding providers."""

    LOCAL = "local"
    OPENAI = "openai"
    GOOGLE = "google"


class VectorStoreProvider(str, Enum):
    """Vector store providers."""

    IN_MEMORY = "in_memory"
    CHROMADB = "chromadb"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"


class MemorySettings(BaseModel):
    """Memory and storage configuration."""

    # Embedding settings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store settings
    vectorstore_provider: VectorStoreProvider = VectorStoreProvider.IN_MEMORY
    collection_name: str = "gianna_memory"
    persist_directory: Optional[str] = None

    # Search settings
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_search_results: int = Field(default=10, ge=1, le=100)
    context_window_size: int = Field(default=5, ge=1, le=50)

    # Clustering
    enable_clustering: bool = True
    cluster_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


# =============================================================================
# Cache Configuration
# =============================================================================


class CacheBackend(str, Enum):
    """Cache backends."""

    MEMORY = "memory"
    REDIS = "redis"
    SQLITE = "sqlite"


class CacheSettings(BaseModel):
    """Cache configuration."""

    # General settings
    enabled: bool = True
    backend: CacheBackend = CacheBackend.MEMORY
    default_ttl_seconds: int = Field(default=3600, ge=60)

    # Memory cache settings
    memory_max_size: int = Field(default=1000, ge=100)

    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    redis_prefix: str = "gianna:"

    # SQLite settings
    sqlite_path: Optional[str] = None

    # Compression
    enable_compression: bool = True
    compression_threshold: int = Field(default=1024, ge=0)


# =============================================================================
# Security Configuration
# =============================================================================


class SecuritySettings(BaseModel):
    """Security-related configuration."""

    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = Field(default=60, ge=1)
    requests_per_hour: int = Field(default=1000, ge=1)

    # Input validation
    max_input_length: int = Field(default=10000, ge=100)
    enable_input_sanitization: bool = True

    # API security
    enable_api_key_validation: bool = True
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])


# =============================================================================
# Monitoring Configuration
# =============================================================================


class LogFormat(str, Enum):
    """Log formats."""

    TEXT = "text"
    JSON = "json"


class MonitoringSettings(BaseModel):
    """Monitoring and observability configuration."""

    # Logging
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    log_file: Optional[str] = None
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"

    # Metrics
    enable_metrics: bool = True
    metrics_port: int = Field(default=9090, ge=1024, le=65535)

    # Tracing
    enable_tracing: bool = False
    tracing_endpoint: Optional[str] = None

    # Health checks
    health_check_interval_seconds: int = Field(default=30, ge=5)


# =============================================================================
# Performance Configuration
# =============================================================================


class PerformanceSettings(BaseModel):
    """Performance optimization configuration."""

    # Async settings
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100)
    task_timeout_seconds: float = Field(default=60.0, ge=1.0)

    # Circuit breaker
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_timeout_seconds: float = Field(default=60.0, ge=5.0)

    # Connection pooling
    enable_connection_pooling: bool = True
    pool_size: int = Field(default=10, ge=1, le=100)

    # Lazy loading
    enable_lazy_loading: bool = True


# =============================================================================
# Agent Configuration
# =============================================================================


class AgentSettings(BaseModel):
    """Agent-related configuration."""

    # ReAct agent settings
    max_iterations: int = Field(default=10, ge=1, le=50)
    timeout_seconds: int = Field(default=300, ge=30)
    verbose: bool = True

    # Orchestration
    default_execution_mode: str = "sequential"  # sequential, parallel, hybrid
    max_concurrent_agents: int = Field(default=5, ge=1)

    # Safety
    enable_safety_checks: bool = True
    enable_error_recovery: bool = True


# =============================================================================
# Main Settings Class
# =============================================================================


class Settings(BaseSettings):
    """
    Main settings class that aggregates all configuration sections.

    Configuration is loaded in order of precedence (lowest to highest):
    1. Default values
    2. Config file (YAML/JSON)
    3. Environment variables
    4. Explicit overrides
    """

    model_config = SettingsConfigDict(
        env_prefix="GIANNA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Application metadata
    app_name: str = "Gianna"
    app_version: str = "0.1.4"
    environment: str = Field(default="development")  # development, staging, production
    debug: bool = False

    # Configuration sections
    llm: LLMSettings = Field(default_factory=LLMSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Settings":
        """Load settings from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Settings":
        """Load settings from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        return cls(**config_data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Export settings to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling SecretStr
        data = self._to_safe_dict()

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def to_json(self, path: Union[str, Path]) -> None:
        """Export settings to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_safe_dict()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _to_safe_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, masking secrets."""
        data = self.model_dump()

        # Mask secret values
        def mask_secrets(d: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    result[key] = mask_secrets(value)
                elif "key" in key.lower() or "secret" in key.lower() or "password" in key.lower():
                    result[key] = "***MASKED***" if value else None
                else:
                    result[key] = value
            return result

        return mask_secrets(data)

    def validate_for_production(self) -> List[str]:
        """Validate settings for production deployment."""
        issues = []

        # Check required API keys
        if self.llm.default_provider == LLMProvider.OPENAI and not self.llm.openai_api_key:
            issues.append("OpenAI API key is required when using OpenAI provider")

        if self.llm.default_provider == LLMProvider.ANTHROPIC and not self.llm.anthropic_api_key:
            issues.append("Anthropic API key is required when using Anthropic provider")

        # Check security settings
        if self.environment == "production":
            if self.debug:
                issues.append("Debug mode should be disabled in production")
            if "*" in self.security.allowed_origins:
                issues.append("Wildcard CORS origins should not be used in production")

        # Check monitoring
        if self.environment == "production" and not self.monitoring.log_file:
            issues.append("Log file should be configured in production")

        return issues


# =============================================================================
# Global Settings Management
# =============================================================================

_settings: Optional[Settings] = None


def get_settings(
    config_path: Optional[Union[str, Path]] = None,
    reload: bool = False,
) -> Settings:
    """
    Get the global settings instance.

    Args:
        config_path: Optional path to config file
        reload: Force reload settings

    Returns:
        Settings instance
    """
    global _settings

    if _settings is None or reload:
        if config_path:
            path = Path(config_path)
            if path.suffix in [".yaml", ".yml"]:
                _settings = Settings.from_yaml(path)
            elif path.suffix == ".json":
                _settings = Settings.from_json(path)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        else:
            # Try to find config file
            for config_name in ["gianna.yaml", "gianna.yml", "gianna.json", "config/gianna.yaml"]:
                config_file = Path(config_name)
                if config_file.exists():
                    logger.info(f"Loading config from {config_file}")
                    return get_settings(config_file, reload=True)

            # Use defaults with environment variables
            _settings = Settings()

    return _settings


def reset_settings() -> None:
    """Reset global settings to None (for testing)."""
    global _settings
    _settings = None


# =============================================================================
# Configuration Utilities
# =============================================================================


def generate_config_template(path: Union[str, Path] = "gianna.example.yaml") -> None:
    """Generate a configuration template file."""
    settings = Settings()
    settings.to_yaml(path)
    logger.info(f"Generated config template at {path}")


def validate_config(path: Union[str, Path]) -> List[str]:
    """Validate a configuration file."""
    try:
        settings = get_settings(path)
        return settings.validate_for_production()
    except Exception as e:
        return [f"Configuration error: {e}"]
