"""
Gianna Learning and Adaptation System.

This module provides intelligent learning capabilities that adapt to user preferences
and behavior patterns over time. It includes user preference learning, pattern analysis,
and adaptive response generation.

Key Components:
- UserPreferenceLearner: Main class for learning user preferences
- PatternAnalyzer: Analyzes usage patterns and behaviors
- AdaptationEngine: Core adaptation algorithms and personalization
- Integration with GiannaState for persistence
"""

from .adaptation_engine import AdaptationEngine, AdaptationStrategy, LearningMode
from .pattern_analysis import InteractionPattern, PatternAnalyzer, UsagePattern
from .state_integration import LearningStateManager
from .user_adaptation import InteractionContext, PreferenceType, UserPreferenceLearner

__all__ = [
    "UserPreferenceLearner",
    "InteractionContext",
    "PreferenceType",
    "PatternAnalyzer",
    "UsagePattern",
    "InteractionPattern",
    "AdaptationEngine",
    "AdaptationStrategy",
    "LearningMode",
    "LearningStateManager",
]
