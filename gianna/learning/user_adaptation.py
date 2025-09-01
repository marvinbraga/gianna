"""
User Preference Learning and Adaptation System.

This module implements intelligent user preference learning that adapts response
styles and behaviors based on user interaction patterns.
"""

import json
import logging
import re
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PreferenceType(Enum):
    """Types of user preferences that can be learned."""

    RESPONSE_LENGTH = "response_length"  # brief, detailed, comprehensive
    COMMUNICATION_STYLE = "communication_style"  # formal, casual, technical
    TOPIC_INTEREST = "topic_interest"  # specific domains of interest
    INTERACTION_MODE = "interaction_mode"  # voice, text, mixed
    DETAIL_LEVEL = "detail_level"  # high, medium, low
    FEEDBACK_PREFERENCE = "feedback_preference"  # immediate, summary, minimal
    ERROR_HANDLING = "error_handling"  # verbose, concise, auto-retry
    PROACTIVITY = "proactivity"  # high, medium, low


class ResponseStyle(Enum):
    """Different response styles for adaptation."""

    BRIEF = "brief"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"


@dataclass
class UserPreference:
    """Represents a learned user preference with confidence scoring."""

    preference_type: PreferenceType
    value: str
    confidence: float  # 0.0 to 1.0
    learned_from: str  # source of learning
    last_updated: datetime
    frequency: int = 0  # how often this preference was observed

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "preference_type": self.preference_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "learned_from": self.learned_from,
            "last_updated": self.last_updated.isoformat(),
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserPreference":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            preference_type=PreferenceType(data["preference_type"]),
            value=data["value"],
            confidence=data["confidence"],
            learned_from=data["learned_from"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
            frequency=data.get("frequency", 0),
        )


@dataclass
class InteractionContext:
    """Context information for a user interaction."""

    user_input: str
    response_generated: str
    timestamp: datetime
    interaction_mode: str  # "voice", "text"
    user_feedback: Optional[str] = None
    response_time: float = 0.0
    command_used: Optional[str] = None
    topic_detected: Optional[str] = None
    session_id: Optional[str] = None


class UserPreferenceLearner:
    """
    Main class for learning and adapting to user preferences.

    Uses online learning algorithms to continuously adapt to user behavior
    and preferences based on interaction patterns, feedback, and usage data.
    """

    def __init__(self, max_history: int = 1000, confidence_threshold: float = 0.7):
        """
        Initialize the preference learner.

        Args:
            max_history: Maximum number of interactions to keep in memory
            confidence_threshold: Minimum confidence to consider a preference learned
        """
        self.max_history = max_history
        self.confidence_threshold = confidence_threshold

        # User preferences storage
        self.preferences: Dict[PreferenceType, UserPreference] = {}

        # Interaction history
        self.interaction_history: deque = deque(maxlen=max_history)

        # Pattern tracking
        self.response_length_preferences = defaultdict(int)
        self.topic_interests = defaultdict(float)
        self.communication_style_scores = defaultdict(float)
        self.interaction_mode_counts = defaultdict(int)

        # Learning parameters
        self.learning_rate = 0.1
        self.decay_rate = 0.95
        self.min_observations = 3

        logger.info(f"UserPreferenceLearner initialized with max_history={max_history}")

    def record_interaction(self, context: InteractionContext) -> None:
        """
        Record a new user interaction for learning.

        Args:
            context: The interaction context containing user input, response, etc.
        """
        self.interaction_history.append(context)

        # Update various preference counters
        self._update_response_length_preference(context)
        self._update_topic_interests(context)
        self._update_communication_style(context)
        self._update_interaction_mode_preference(context)

        # Learn from explicit feedback if available
        if context.user_feedback:
            self._learn_from_feedback(context)

        logger.debug(f"Recorded interaction: {context.command_used or 'general'}")

    def _update_response_length_preference(self, context: InteractionContext) -> None:
        """Update preferences based on response length patterns."""
        response_length = len(context.response_generated.split())

        if response_length < 20:
            category = "brief"
        elif response_length < 100:
            category = "detailed"
        else:
            category = "comprehensive"

        self.response_length_preferences[category] += 1

        # Learn preference if we have enough data
        if sum(self.response_length_preferences.values()) >= self.min_observations:
            dominant_style = max(
                self.response_length_preferences.items(), key=lambda x: x[1]
            )
            confidence = dominant_style[1] / sum(
                self.response_length_preferences.values()
            )

            if confidence >= self.confidence_threshold:
                self._update_preference(
                    PreferenceType.RESPONSE_LENGTH,
                    dominant_style[0],
                    confidence,
                    "response_pattern_analysis",
                )

    def _update_topic_interests(self, context: InteractionContext) -> None:
        """Update topic interest scores based on user interactions."""
        if context.topic_detected:
            # Use exponential moving average for topic interest scoring
            current_score = self.topic_interests[context.topic_detected]
            self.topic_interests[context.topic_detected] = (
                current_score * self.decay_rate + (1 - self.decay_rate) * 1.0
            )

            # Decay other topics slightly
            for topic in self.topic_interests:
                if topic != context.topic_detected:
                    self.topic_interests[topic] *= self.decay_rate

    def _update_communication_style(self, context: InteractionContext) -> None:
        """Analyze communication style preferences."""
        user_input = context.user_input.lower()

        # Detect formal vs casual language
        formal_indicators = [
            "please",
            "could you",
            "would you",
            "thank you",
            "sir",
            "madam",
        ]
        casual_indicators = ["hey", "yo", "gonna", "wanna", "cool", "awesome", "dude"]
        technical_indicators = [
            "function",
            "class",
            "method",
            "algorithm",
            "architecture",
            "implementation",
        ]

        formal_score = sum(
            1 for indicator in formal_indicators if indicator in user_input
        )
        casual_score = sum(
            1 for indicator in casual_indicators if indicator in user_input
        )
        technical_score = sum(
            1 for indicator in technical_indicators if indicator in user_input
        )

        if formal_score > casual_score and formal_score > 0:
            self.communication_style_scores["formal"] += self.learning_rate
        elif casual_score > formal_score and casual_score > 0:
            self.communication_style_scores["casual"] += self.learning_rate

        if technical_score > 0:
            self.communication_style_scores["technical"] += (
                self.learning_rate * technical_score
            )

    def _update_interaction_mode_preference(self, context: InteractionContext) -> None:
        """Update interaction mode preferences."""
        self.interaction_mode_counts[context.interaction_mode] += 1

        # Learn mode preference if we have enough data
        total_interactions = sum(self.interaction_mode_counts.values())
        if total_interactions >= self.min_observations:
            dominant_mode = max(
                self.interaction_mode_counts.items(), key=lambda x: x[1]
            )
            confidence = dominant_mode[1] / total_interactions

            if confidence >= self.confidence_threshold:
                self._update_preference(
                    PreferenceType.INTERACTION_MODE,
                    dominant_mode[0],
                    confidence,
                    "interaction_mode_analysis",
                )

    def _learn_from_feedback(self, context: InteractionContext) -> None:
        """Learn from explicit user feedback."""
        feedback = context.user_feedback.lower()

        # Analyze feedback for preference indicators
        if any(word in feedback for word in ["too long", "verbose", "brief"]):
            self._update_preference(
                PreferenceType.RESPONSE_LENGTH, "brief", 0.8, "explicit_feedback"
            )
        elif any(word in feedback for word in ["more detail", "elaborate", "explain"]):
            self._update_preference(
                PreferenceType.RESPONSE_LENGTH, "detailed", 0.8, "explicit_feedback"
            )

        # Analyze for style feedback
        if any(word in feedback for word in ["formal", "professional"]):
            self._update_preference(
                PreferenceType.COMMUNICATION_STYLE, "formal", 0.8, "explicit_feedback"
            )
        elif any(word in feedback for word in ["casual", "friendly", "conversational"]):
            self._update_preference(
                PreferenceType.COMMUNICATION_STYLE,
                "conversational",
                0.8,
                "explicit_feedback",
            )

    def _update_preference(
        self, pref_type: PreferenceType, value: str, confidence: float, source: str
    ) -> None:
        """Update or create a user preference."""
        existing_pref = self.preferences.get(pref_type)

        if existing_pref:
            # Update existing preference with weighted average
            weight = confidence / (confidence + existing_pref.confidence)
            if existing_pref.value == value:
                # Reinforce existing preference
                existing_pref.confidence = min(1.0, existing_pref.confidence * 1.1)
                existing_pref.frequency += 1
            else:
                # Conflicting preference - use confidence-weighted approach
                if confidence > existing_pref.confidence:
                    existing_pref.value = value
                    existing_pref.confidence = (
                        confidence * weight + existing_pref.confidence * (1 - weight)
                    )

            existing_pref.last_updated = datetime.now()
            existing_pref.learned_from = f"{existing_pref.learned_from}, {source}"
        else:
            # Create new preference
            self.preferences[pref_type] = UserPreference(
                preference_type=pref_type,
                value=value,
                confidence=confidence,
                learned_from=source,
                last_updated=datetime.now(),
                frequency=1,
            )

        logger.debug(
            f"Updated preference: {pref_type.value} = {value} (confidence: {confidence:.2f})"
        )

    def analyze_user_patterns(self) -> Dict[str, Any]:
        """
        Analyze accumulated user interaction patterns.

        Returns:
            Dict containing analysis results and insights
        """
        if not self.interaction_history:
            return {"status": "no_data", "message": "No interaction history available"}

        analysis = {
            "total_interactions": len(self.interaction_history),
            "learned_preferences": len(self.preferences),
            "confidence_scores": {},
            "interaction_patterns": {},
            "recommendations": [],
        }

        # Analyze confidence scores
        for pref_type, preference in self.preferences.items():
            analysis["confidence_scores"][pref_type.value] = {
                "value": preference.value,
                "confidence": preference.confidence,
                "frequency": preference.frequency,
            }

        # Analyze interaction patterns
        recent_interactions = list(self.interaction_history)[
            -50:
        ]  # Last 50 interactions

        if recent_interactions:
            # Analyze response times
            response_times = [
                ctx.response_time
                for ctx in recent_interactions
                if ctx.response_time > 0
            ]
            if response_times:
                analysis["interaction_patterns"]["avg_response_time"] = statistics.mean(
                    response_times
                )
                analysis["interaction_patterns"]["response_time_trend"] = (
                    "improving"
                    if response_times[-5:] < response_times[:5]
                    else "stable"
                )

            # Analyze interaction modes
            mode_counts = defaultdict(int)
            for ctx in recent_interactions:
                mode_counts[ctx.interaction_mode] += 1
            analysis["interaction_patterns"]["mode_distribution"] = dict(mode_counts)

            # Analyze command usage
            command_counts = defaultdict(int)
            for ctx in recent_interactions:
                if ctx.command_used:
                    command_counts[ctx.command_used] += 1
            analysis["interaction_patterns"]["popular_commands"] = dict(
                sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            )

        # Generate recommendations
        if self.preferences.get(PreferenceType.RESPONSE_LENGTH):
            pref = self.preferences[PreferenceType.RESPONSE_LENGTH]
            if pref.confidence > 0.8:
                analysis["recommendations"].append(
                    f"User strongly prefers {pref.value} responses (confidence: {pref.confidence:.2f})"
                )

        if self.preferences.get(PreferenceType.COMMUNICATION_STYLE):
            pref = self.preferences[PreferenceType.COMMUNICATION_STYLE]
            if pref.confidence > 0.7:
                analysis["recommendations"].append(
                    f"Adapt communication style to be more {pref.value} (confidence: {pref.confidence:.2f})"
                )

        return analysis

    def adapt_response_style(
        self, base_response: str, context: Optional[InteractionContext] = None
    ) -> Tuple[str, Dict[str, str]]:
        """
        Adapt a response based on learned user preferences.

        Args:
            base_response: The original response to adapt
            context: Optional context for the current interaction

        Returns:
            Tuple of (adapted_response, adaptation_metadata)
        """
        adapted_response = base_response
        adaptations_applied = {}

        # Apply response length adaptation
        length_pref = self.preferences.get(PreferenceType.RESPONSE_LENGTH)
        if length_pref and length_pref.confidence >= self.confidence_threshold:
            if length_pref.value == "brief" and len(base_response.split()) > 30:
                adapted_response = self._make_response_brief(adapted_response)
                adaptations_applied["length"] = "shortened for brevity preference"
            elif (
                length_pref.value == "comprehensive" and len(base_response.split()) < 50
            ):
                adapted_response = self._make_response_comprehensive(adapted_response)
                adaptations_applied["length"] = "expanded for comprehensive preference"

        # Apply communication style adaptation
        style_pref = self.preferences.get(PreferenceType.COMMUNICATION_STYLE)
        if style_pref and style_pref.confidence >= self.confidence_threshold:
            if style_pref.value == "formal":
                adapted_response = self._make_response_formal(adapted_response)
                adaptations_applied["style"] = "formalized language"
            elif style_pref.value == "casual":
                adapted_response = self._make_response_casual(adapted_response)
                adaptations_applied["style"] = "casualized language"
            elif style_pref.value == "technical":
                adapted_response = self._make_response_technical(adapted_response)
                adaptations_applied["style"] = "enhanced technical detail"

        return adapted_response, adaptations_applied

    def _make_response_brief(self, response: str) -> str:
        """Make response more concise."""
        # Simple approach: keep first sentence and key points
        sentences = response.split(". ")
        if len(sentences) > 2:
            # Keep first sentence and any that contain key indicators
            key_sentences = [sentences[0]]  # Always keep first sentence
            for sentence in sentences[1:3]:  # Keep up to 2 more important sentences
                if any(
                    keyword in sentence.lower()
                    for keyword in ["important", "key", "main", "result", "solution"]
                ):
                    key_sentences.append(sentence)
            return ". ".join(key_sentences) + (
                "." if not key_sentences[-1].endswith(".") else ""
            )
        return response

    def _make_response_comprehensive(self, response: str) -> str:
        """Make response more comprehensive."""
        # Add helpful context and explanations
        if "Here's" in response or "This" in response:
            response += " Let me provide some additional context and considerations that might be helpful."
        return response

    def _make_response_formal(self, response: str) -> str:
        """Make response more formal."""
        # Replace casual contractions and informal language
        replacements = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "isn't": "is not",
            "aren't": "are not",
            "here's": "here is",
            "that's": "that is",
            "it's": "it is",
        }

        adapted = response
        for casual, formal in replacements.items():
            adapted = re.sub(
                r"\b" + casual + r"\b", formal, adapted, flags=re.IGNORECASE
            )

        return adapted

    def _make_response_casual(self, response: str) -> str:
        """Make response more casual and friendly."""
        # Add friendly interjections and casual language
        if not any(
            word in response.lower() for word in ["hey", "sure", "cool", "great"]
        ):
            if response.startswith("To"):
                response = "Sure! " + response
            elif response.startswith("The"):
                response = "So, " + response.lower()

        return response

    def _make_response_technical(self, response: str) -> str:
        """Enhance technical detail in response."""
        # This would typically involve adding more technical context
        # For now, we'll add a note about technical considerations
        if "implementation" in response.lower() or "code" in response.lower():
            response += " Consider performance implications and maintainability aspects as well."

        return response

    def get_preference_summary(self) -> Dict[str, Any]:
        """Get a summary of all learned preferences."""
        summary = {
            "preferences_count": len(self.preferences),
            "high_confidence_preferences": [],
            "learning_stats": {
                "total_interactions": len(self.interaction_history),
                "learning_sources": defaultdict(int),
            },
        }

        for pref_type, preference in self.preferences.items():
            if preference.confidence >= self.confidence_threshold:
                summary["high_confidence_preferences"].append(
                    {
                        "type": pref_type.value,
                        "value": preference.value,
                        "confidence": preference.confidence,
                        "frequency": preference.frequency,
                    }
                )

            summary["learning_stats"]["learning_sources"][preference.learned_from] += 1

        return summary

    def export_preferences(self) -> str:
        """Export preferences to JSON string for persistence."""
        export_data = {
            "preferences": {
                pref_type.value: pref.to_dict()
                for pref_type, pref in self.preferences.items()
            },
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_interactions": len(self.interaction_history),
                "learning_parameters": {
                    "learning_rate": self.learning_rate,
                    "confidence_threshold": self.confidence_threshold,
                    "max_history": self.max_history,
                },
            },
        }
        return json.dumps(export_data, indent=2)

    def import_preferences(self, json_data: str) -> bool:
        """Import preferences from JSON string."""
        try:
            data = json.loads(json_data)

            # Clear existing preferences
            self.preferences.clear()

            # Import preferences
            for pref_type_str, pref_data in data["preferences"].items():
                try:
                    pref_type = PreferenceType(pref_type_str)
                    self.preferences[pref_type] = UserPreference.from_dict(pref_data)
                except ValueError:
                    logger.warning(f"Unknown preference type: {pref_type_str}")
                    continue

            logger.info(f"Successfully imported {len(self.preferences)} preferences")
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to import preferences: {e}")
            return False

    def reset_learning(self) -> None:
        """Reset all learned preferences and history."""
        self.preferences.clear()
        self.interaction_history.clear()
        self.response_length_preferences.clear()
        self.topic_interests.clear()
        self.communication_style_scores.clear()
        self.interaction_mode_counts.clear()

        logger.info("Learning system reset - all preferences and history cleared")
