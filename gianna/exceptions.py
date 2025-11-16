"""
Custom exceptions for Gianna system.

This module defines all custom exception classes used throughout the Gianna
codebase to provide clear error handling and better debugging.
"""


class GiannaError(Exception):
    """Base exception for all Gianna-related errors."""

    pass


# Performance and Optimization Errors
class PerformanceError(GiannaError):
    """Base class for performance-related errors."""

    pass


class CacheError(PerformanceError):
    """Cache-related errors."""

    pass


class CacheConnectionError(CacheError):
    """Error connecting to cache backend (e.g., Redis)."""

    pass


class CacheSerializationError(CacheError):
    """Error serializing/deserializing cache data."""

    pass


class ConnectionPoolError(PerformanceError):
    """Connection pool management errors."""

    pass


# Learning and Adaptation Errors
class LearningError(GiannaError):
    """Base class for learning system errors."""

    pass


class FeatureExtractionError(LearningError):
    """Error extracting features for learning."""

    pass


class ModelUpdateError(LearningError):
    """Error updating learning model."""

    pass


class CorrelationAnalysisError(LearningError):
    """Error during correlation analysis."""

    pass


class AdaptationError(LearningError):
    """Error during response adaptation."""

    pass


# Audio Processing Errors
class AudioError(GiannaError):
    """Base class for audio-related errors."""

    pass


class VADError(AudioError):
    """Voice Activity Detection errors."""

    pass


class AudioRecordingError(AudioError):
    """Audio recording errors."""

    pass


# Command Execution Errors
class CommandError(GiannaError):
    """Base class for command execution errors."""

    pass


class CommandExecutionError(CommandError):
    """Error executing a command."""

    pass


class CommandValidationError(CommandError):
    """Error validating a command."""

    pass


# Memory and Context Errors
class MemoryError(GiannaError):
    """Base class for memory system errors."""

    pass


class ContextRetrievalError(MemoryError):
    """Error retrieving context from memory."""

    pass


class ContextStorageError(MemoryError):
    """Error storing context in memory."""

    pass


# Routing and Coordination Errors
class RoutingError(GiannaError):
    """Base class for routing errors."""

    pass


class AgentNotFoundError(RoutingError):
    """Requested agent not found."""

    pass


class InvalidRoutingRulesError(RoutingError):
    """Invalid routing rules configuration."""

    pass


# Configuration Errors
class ConfigurationError(GiannaError):
    """Base class for configuration errors."""

    pass


class InvalidConfigError(ConfigurationError):
    """Invalid configuration provided."""

    pass


class MissingConfigError(ConfigurationError):
    """Required configuration missing."""

    pass


# LLM Provider Errors
class LLMError(GiannaError):
    """Base class for LLM provider errors."""

    pass


class LLMConnectionError(LLMError):
    """Error connecting to LLM provider."""

    pass


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""

    pass


class LLMResponseError(LLMError):
    """Error processing LLM response."""

    pass
