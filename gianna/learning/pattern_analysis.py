"""
Pattern Analysis Module for User Behavior Analytics.

This module provides comprehensive analysis of user interaction patterns,
including temporal patterns, command frequency, topic analysis, and behavioral insights.
"""

import logging
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from math import log
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TimeOfDay(Enum):
    """Time periods for temporal analysis."""

    EARLY_MORNING = "early_morning"  # 5:00-8:00
    MORNING = "morning"  # 8:00-12:00
    AFTERNOON = "afternoon"  # 12:00-17:00
    EVENING = "evening"  # 17:00-21:00
    NIGHT = "night"  # 21:00-1:00
    LATE_NIGHT = "late_night"  # 1:00-5:00


class InteractionType(Enum):
    """Types of user interactions."""

    COMMAND = "command"
    QUERY = "query"
    CONVERSATION = "conversation"
    FEEDBACK = "feedback"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class UsagePattern:
    """Represents a detected usage pattern."""

    pattern_type: str
    description: str
    frequency: float
    confidence: float
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "examples": self.examples,
            "metadata": self.metadata,
        }


@dataclass
class InteractionPattern:
    """Detailed interaction pattern with behavioral insights."""

    interaction_type: InteractionType
    common_phrases: List[str]
    typical_time_of_day: List[TimeOfDay]
    average_session_length: float
    success_rate: float
    common_topics: List[str]
    user_satisfaction_indicators: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "interaction_type": self.interaction_type.value,
            "common_phrases": self.common_phrases,
            "typical_time_of_day": [tod.value for tod in self.typical_time_of_day],
            "average_session_length": self.average_session_length,
            "success_rate": self.success_rate,
            "common_topics": self.common_topics,
            "user_satisfaction_indicators": self.user_satisfaction_indicators,
        }


class PatternAnalyzer:
    """
    Advanced pattern analyzer for user behavior and interaction analytics.

    Analyzes various aspects of user behavior including:
    - Temporal usage patterns
    - Command frequency and preferences
    - Topic interests and evolution
    - Interaction success patterns
    - Session behaviors
    """

    def __init__(
        self, min_pattern_frequency: int = 3, confidence_threshold: float = 0.6
    ):
        """
        Initialize the pattern analyzer.

        Args:
            min_pattern_frequency: Minimum frequency to consider a pattern significant
            confidence_threshold: Minimum confidence score for pattern recognition
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.confidence_threshold = confidence_threshold

        # Pattern storage
        self.detected_patterns: List[UsagePattern] = []
        self.interaction_patterns: Dict[InteractionType, InteractionPattern] = {}

        # Analysis caches
        self._temporal_cache: Dict[str, Any] = {}
        self._topic_cache: Dict[str, Any] = {}
        self._command_cache: Dict[str, Any] = {}

        logger.info("PatternAnalyzer initialized")

    def analyze_temporal_patterns(
        self, interaction_history: List[Any]
    ) -> Dict[str, Any]:
        """
        Analyze temporal usage patterns.

        Args:
            interaction_history: List of interaction contexts

        Returns:
            Dictionary with temporal analysis results
        """
        if not interaction_history:
            return {"error": "No interaction history provided"}

        # Group interactions by time of day
        time_distribution = defaultdict(list)
        day_of_week_distribution = defaultdict(list)
        hourly_distribution = defaultdict(int)

        for interaction in interaction_history:
            if hasattr(interaction, "timestamp") and interaction.timestamp:
                timestamp = interaction.timestamp

                # Time of day analysis
                hour = timestamp.hour
                time_period = self._get_time_period(hour)
                time_distribution[time_period].append(interaction)

                # Day of week analysis
                day_of_week = timestamp.strftime("%A")
                day_of_week_distribution[day_of_week].append(interaction)

                # Hourly analysis
                hourly_distribution[hour] += 1

        # Calculate peak usage times
        peak_hours = sorted(
            hourly_distribution.items(), key=lambda x: x[1], reverse=True
        )[:3]
        peak_time_periods = sorted(
            [
                (period, len(interactions))
                for period, interactions in time_distribution.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        # Calculate usage consistency
        daily_counts = [
            len(interactions) for interactions in day_of_week_distribution.values()
        ]
        usage_consistency = (
            1.0 - (statistics.stdev(daily_counts) / statistics.mean(daily_counts))
            if len(daily_counts) > 1
            else 1.0
        )

        analysis = {
            "total_interactions": len(interaction_history),
            "peak_hours": [
                {"hour": hour, "count": count} for hour, count in peak_hours
            ],
            "peak_time_periods": [
                {"period": period, "count": count}
                for period, count in peak_time_periods
            ],
            "usage_consistency": max(0.0, min(1.0, usage_consistency)),
            "daily_distribution": {
                day: len(interactions)
                for day, interactions in day_of_week_distribution.items()
            },
            "hourly_heatmap": dict(hourly_distribution),
        }

        # Detect temporal patterns
        self._detect_temporal_patterns(analysis)

        return analysis

    def _get_time_period(self, hour: int) -> str:
        """Classify hour into time period."""
        if 5 <= hour < 8:
            return TimeOfDay.EARLY_MORNING.value
        elif 8 <= hour < 12:
            return TimeOfDay.MORNING.value
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON.value
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING.value
        elif 21 <= hour < 24 or hour == 0:
            return TimeOfDay.NIGHT.value
        else:  # 1-4 AM
            return TimeOfDay.LATE_NIGHT.value

    def _detect_temporal_patterns(self, temporal_analysis: Dict[str, Any]) -> None:
        """Detect significant temporal patterns."""
        peak_periods = temporal_analysis.get("peak_time_periods", [])

        if peak_periods and len(peak_periods) > 0:
            dominant_period = peak_periods[0]
            total_interactions = temporal_analysis["total_interactions"]
            period_dominance = dominant_period["count"] / total_interactions

            if period_dominance > 0.4:  # More than 40% in one period
                pattern = UsagePattern(
                    pattern_type="temporal_preference",
                    description=f"User primarily active during {dominant_period['period']} ({period_dominance:.1%} of interactions)",
                    frequency=dominant_period["count"],
                    confidence=min(0.95, period_dominance * 2),
                    metadata={
                        "dominant_period": dominant_period["period"],
                        "dominance": period_dominance,
                    },
                )
                self.detected_patterns.append(pattern)

        # Detect consistency patterns
        consistency = temporal_analysis.get("usage_consistency", 0)
        if consistency > 0.8:
            pattern = UsagePattern(
                pattern_type="consistent_usage",
                description=f"User demonstrates consistent daily usage patterns (consistency: {consistency:.1%})",
                frequency=temporal_analysis["total_interactions"],
                confidence=consistency,
                metadata={"consistency_score": consistency},
            )
            self.detected_patterns.append(pattern)

    def analyze_command_frequency(
        self, interaction_history: List[Any]
    ) -> Dict[str, Any]:
        """
        Analyze command usage frequency and patterns.

        Args:
            interaction_history: List of interaction contexts

        Returns:
            Dictionary with command analysis results
        """
        if not interaction_history:
            return {"error": "No interaction history provided"}

        command_counts = Counter()
        command_success_rates = defaultdict(list)
        command_response_times = defaultdict(list)
        command_contexts = defaultdict(list)

        # Analyze each interaction
        for interaction in interaction_history:
            if hasattr(interaction, "command_used") and interaction.command_used:
                command = interaction.command_used
                command_counts[command] += 1

                # Track success indicators
                if hasattr(interaction, "user_feedback") and interaction.user_feedback:
                    success_indicators = [
                        "good",
                        "great",
                        "perfect",
                        "thanks",
                        "correct",
                    ]
                    failure_indicators = [
                        "wrong",
                        "error",
                        "bad",
                        "incorrect",
                        "failed",
                    ]

                    feedback = interaction.user_feedback.lower()
                    if any(indicator in feedback for indicator in success_indicators):
                        command_success_rates[command].append(1.0)
                    elif any(indicator in feedback for indicator in failure_indicators):
                        command_success_rates[command].append(0.0)
                    else:
                        command_success_rates[command].append(0.5)  # Neutral

                # Track response times
                if (
                    hasattr(interaction, "response_time")
                    and interaction.response_time > 0
                ):
                    command_response_times[command].append(interaction.response_time)

                # Store context for analysis
                if hasattr(interaction, "user_input"):
                    command_contexts[command].append(interaction.user_input)

        # Calculate statistics
        total_commands = sum(command_counts.values())
        command_stats = {}

        for command, count in command_counts.most_common():
            stats = {
                "count": count,
                "frequency": count / total_commands,
                "success_rate": (
                    statistics.mean(command_success_rates[command])
                    if command_success_rates[command]
                    else None
                ),
                "avg_response_time": (
                    statistics.mean(command_response_times[command])
                    if command_response_times[command]
                    else None
                ),
                "contexts": command_contexts[command][:5],  # Store sample contexts
            }
            command_stats[command] = stats

        analysis = {
            "total_commands": total_commands,
            "unique_commands": len(command_counts),
            "most_used_commands": dict(command_counts.most_common(10)),
            "command_statistics": command_stats,
            "command_diversity": len(command_counts)
            / max(1, total_commands),  # How diverse command usage is
        }

        # Detect command patterns
        self._detect_command_patterns(analysis)

        return analysis

    def _detect_command_patterns(self, command_analysis: Dict[str, Any]) -> None:
        """Detect significant command usage patterns."""
        command_stats = command_analysis.get("command_statistics", {})

        # Detect favorite commands
        for command, stats in command_stats.items():
            if stats["frequency"] > 0.2:  # Used in more than 20% of interactions
                pattern = UsagePattern(
                    pattern_type="favorite_command",
                    description=f"User frequently uses '{command}' command ({stats['frequency']:.1%} of interactions)",
                    frequency=stats["count"],
                    confidence=min(0.95, stats["frequency"] * 3),
                    examples=stats["contexts"][:3],
                    metadata={"command": command, "frequency": stats["frequency"]},
                )
                self.detected_patterns.append(pattern)

        # Detect command specialization
        diversity = command_analysis.get("command_diversity", 0)
        if diversity < 0.3:  # Low diversity indicates specialization
            pattern = UsagePattern(
                pattern_type="specialized_usage",
                description=f"User shows specialized command usage patterns (diversity: {diversity:.2f})",
                frequency=command_analysis["total_commands"],
                confidence=1.0 - diversity,
                metadata={"diversity_score": diversity},
            )
            self.detected_patterns.append(pattern)

    def analyze_topic_interests(self, interaction_history: List[Any]) -> Dict[str, Any]:
        """
        Analyze topic interests and evolution over time.

        Args:
            interaction_history: List of interaction contexts

        Returns:
            Dictionary with topic analysis results
        """
        if not interaction_history:
            return {"error": "No interaction history provided"}

        # Topic extraction keywords
        topic_keywords = {
            "programming": [
                "code",
                "function",
                "class",
                "programming",
                "debug",
                "algorithm",
                "software",
            ],
            "data_science": [
                "data",
                "analysis",
                "machine learning",
                "statistics",
                "model",
                "dataset",
            ],
            "web_development": [
                "html",
                "css",
                "javascript",
                "web",
                "frontend",
                "backend",
                "api",
            ],
            "system_admin": [
                "server",
                "deploy",
                "configuration",
                "system",
                "admin",
                "linux",
                "docker",
            ],
            "audio_processing": [
                "audio",
                "sound",
                "voice",
                "speech",
                "tts",
                "stt",
                "music",
            ],
            "file_management": [
                "file",
                "folder",
                "directory",
                "path",
                "save",
                "load",
                "import",
                "export",
            ],
            "automation": [
                "automate",
                "script",
                "batch",
                "schedule",
                "workflow",
                "pipeline",
            ],
        }

        topic_counts = defaultdict(float)
        topic_timeline = defaultdict(list)

        # Analyze each interaction for topics
        for interaction in interaction_history:
            text_content = ""
            if hasattr(interaction, "user_input") and interaction.user_input:
                text_content += interaction.user_input.lower() + " "
            if (
                hasattr(interaction, "response_generated")
                and interaction.response_generated
            ):
                text_content += interaction.response_generated.lower()

            if not text_content.strip():
                continue

            # Score topics based on keyword presence
            interaction_topics = {}
            for topic, keywords in topic_keywords.items():
                score = sum(text_content.count(keyword) for keyword in keywords)
                if score > 0:
                    # Apply TF-IDF-like scoring
                    topic_score = score * log(
                        len(topic_keywords)
                        / max(
                            1,
                            sum(
                                1
                                for t in topic_keywords
                                if any(kw in text_content for kw in topic_keywords[t])
                            ),
                        )
                    )
                    interaction_topics[topic] = topic_score
                    topic_counts[topic] += topic_score

            # Add to timeline
            if hasattr(interaction, "timestamp") and interaction_topics:
                for topic, score in interaction_topics.items():
                    topic_timeline[topic].append(
                        {"timestamp": interaction.timestamp, "score": score}
                    )

        # Calculate topic statistics
        total_topic_score = sum(topic_counts.values())
        topic_distribution = (
            {topic: score / total_topic_score for topic, score in topic_counts.items()}
            if total_topic_score > 0
            else {}
        )

        # Analyze topic evolution
        topic_trends = {}
        for topic, timeline in topic_timeline.items():
            if len(timeline) >= 3:  # Need at least 3 points for trend
                recent_scores = [
                    entry["score"] for entry in timeline[-5:]
                ]  # Last 5 interactions
                early_scores = [
                    entry["score"] for entry in timeline[:5]
                ]  # First 5 interactions

                recent_avg = statistics.mean(recent_scores) if recent_scores else 0
                early_avg = statistics.mean(early_scores) if early_scores else 0

                if early_avg > 0:
                    trend = (recent_avg - early_avg) / early_avg
                    if abs(trend) > 0.2:  # Significant change
                        topic_trends[topic] = (
                            "increasing" if trend > 0 else "decreasing"
                        )
                    else:
                        topic_trends[topic] = "stable"
                else:
                    topic_trends[topic] = "new"

        analysis = {
            "topic_distribution": dict(
                sorted(topic_distribution.items(), key=lambda x: x[1], reverse=True)
            ),
            "dominant_topics": [
                topic for topic, score in topic_distribution.items() if score > 0.15
            ],  # >15% of interactions
            "topic_trends": topic_trends,
            "topic_diversity": len(
                [score for score in topic_distribution.values() if score > 0.05]
            ),  # Topics with >5% presence
            "total_interactions_with_topics": len(
                [t for timeline in topic_timeline.values() for t in timeline]
            ),
        }

        # Detect topic patterns
        self._detect_topic_patterns(analysis)

        return analysis

    def _detect_topic_patterns(self, topic_analysis: Dict[str, Any]) -> None:
        """Detect significant topic interest patterns."""
        topic_distribution = topic_analysis.get("topic_distribution", {})
        topic_trends = topic_analysis.get("topic_trends", {})

        # Detect dominant interests
        for topic, score in topic_distribution.items():
            if score > 0.25:  # More than 25% of topic-related interactions
                pattern = UsagePattern(
                    pattern_type="dominant_interest",
                    description=f"Strong interest in {topic.replace('_', ' ')} ({score:.1%} of topic discussions)",
                    frequency=score
                    * topic_analysis.get("total_interactions_with_topics", 0),
                    confidence=min(0.95, score * 2),
                    metadata={"topic": topic, "score": score},
                )
                self.detected_patterns.append(pattern)

        # Detect evolving interests
        for topic, trend in topic_trends.items():
            if trend == "increasing":
                pattern = UsagePattern(
                    pattern_type="growing_interest",
                    description=f"Growing interest in {topic.replace('_', ' ')}",
                    frequency=topic_distribution.get(topic, 0),
                    confidence=0.7,
                    metadata={"topic": topic, "trend": trend},
                )
                self.detected_patterns.append(pattern)

    def analyze_session_behaviors(
        self, interaction_history: List[Any]
    ) -> Dict[str, Any]:
        """
        Analyze session-level behavioral patterns.

        Args:
            interaction_history: List of interaction contexts

        Returns:
            Dictionary with session analysis results
        """
        if not interaction_history:
            return {"error": "No interaction history provided"}

        # Group interactions by session
        sessions = self._group_by_sessions(interaction_history)

        session_lengths = []
        session_interaction_counts = []
        session_types = defaultdict(int)
        success_rates = []

        for session_id, interactions in sessions.items():
            # Calculate session duration
            if len(interactions) > 1:
                start_time = min(
                    i.timestamp
                    for i in interactions
                    if hasattr(i, "timestamp") and i.timestamp
                )
                end_time = max(
                    i.timestamp
                    for i in interactions
                    if hasattr(i, "timestamp") and i.timestamp
                )
                duration = (end_time - start_time).total_seconds() / 60.0  # minutes
                session_lengths.append(duration)

            session_interaction_counts.append(len(interactions))

            # Categorize session type
            commands_used = set()
            has_errors = False

            for interaction in interactions:
                if hasattr(interaction, "command_used") and interaction.command_used:
                    commands_used.add(interaction.command_used)

                if hasattr(interaction, "user_feedback") and interaction.user_feedback:
                    if any(
                        word in interaction.user_feedback.lower()
                        for word in ["error", "wrong", "failed"]
                    ):
                        has_errors = True

            # Classify session
            if len(commands_used) == 1:
                session_types["focused"] += 1
            elif len(commands_used) > 3:
                session_types["exploratory"] += 1
            elif has_errors:
                session_types["problem_solving"] += 1
            else:
                session_types["general"] += 1

            # Calculate session success rate (simplified)
            positive_feedback = sum(
                1
                for i in interactions
                if hasattr(i, "user_feedback")
                and i.user_feedback
                and any(
                    word in i.user_feedback.lower()
                    for word in ["good", "thanks", "great"]
                )
            )
            total_feedback = sum(
                1
                for i in interactions
                if hasattr(i, "user_feedback") and i.user_feedback
            )

            if total_feedback > 0:
                success_rates.append(positive_feedback / total_feedback)

        analysis = {
            "total_sessions": len(sessions),
            "avg_session_length": (
                statistics.mean(session_lengths) if session_lengths else 0
            ),
            "avg_interactions_per_session": statistics.mean(session_interaction_counts),
            "session_types": dict(session_types),
            "avg_success_rate": (
                statistics.mean(success_rates) if success_rates else None
            ),
            "session_length_distribution": {
                "short": len([s for s in session_lengths if s < 5]),  # < 5 min
                "medium": len([s for s in session_lengths if 5 <= s < 20]),  # 5-20 min
                "long": len([s for s in session_lengths if s >= 20]),  # > 20 min
            },
        }

        return analysis

    def _group_by_sessions(
        self, interaction_history: List[Any], session_timeout_minutes: int = 30
    ) -> Dict[str, List[Any]]:
        """Group interactions into sessions based on time gaps."""
        if not interaction_history:
            return {}

        # Sort by timestamp
        sorted_interactions = sorted(
            [i for i in interaction_history if hasattr(i, "timestamp") and i.timestamp],
            key=lambda x: x.timestamp,
        )

        sessions = {}
        current_session = []
        current_session_id = "session_1"
        last_timestamp = None
        session_counter = 1

        for interaction in sorted_interactions:
            timestamp = interaction.timestamp

            # Start new session if gap is too large
            if last_timestamp and (timestamp - last_timestamp).total_seconds() > (
                session_timeout_minutes * 60
            ):
                if current_session:
                    sessions[current_session_id] = current_session
                session_counter += 1
                current_session_id = f"session_{session_counter}"
                current_session = []

            current_session.append(interaction)
            last_timestamp = timestamp

        # Add final session
        if current_session:
            sessions[current_session_id] = current_session

        return sessions

    def get_comprehensive_analysis(
        self, interaction_history: List[Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive pattern analysis.

        Args:
            interaction_history: List of interaction contexts

        Returns:
            Complete analysis results
        """
        if not interaction_history:
            return {"error": "No interaction history provided"}

        logger.info(
            f"Starting comprehensive analysis of {len(interaction_history)} interactions"
        )

        analysis = {
            "overview": {
                "total_interactions": len(interaction_history),
                "analysis_timestamp": datetime.now().isoformat(),
                "date_range": self._get_date_range(interaction_history),
            },
            "temporal_patterns": self.analyze_temporal_patterns(interaction_history),
            "command_analysis": self.analyze_command_frequency(interaction_history),
            "topic_analysis": self.analyze_topic_interests(interaction_history),
            "session_analysis": self.analyze_session_behaviors(interaction_history),
            "detected_patterns": [
                pattern.to_dict() for pattern in self.detected_patterns
            ],
            "recommendations": self._generate_recommendations(),
        }

        logger.info(
            f"Analysis complete. Detected {len(self.detected_patterns)} patterns"
        )
        return analysis

    def _get_date_range(self, interaction_history: List[Any]) -> Dict[str, str]:
        """Get the date range of the interaction history."""
        timestamps = [
            i.timestamp
            for i in interaction_history
            if hasattr(i, "timestamp") and i.timestamp
        ]
        if timestamps:
            return {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "span_days": (max(timestamps) - min(timestamps)).days,
            }
        return {"start": None, "end": None, "span_days": 0}

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on detected patterns."""
        recommendations = []

        for pattern in self.detected_patterns:
            if pattern.confidence > 0.8:
                if pattern.pattern_type == "temporal_preference":
                    recommendations.append(
                        f"Schedule important interactions during user's preferred time: {pattern.metadata.get('dominant_period', 'unknown')}"
                    )
                elif pattern.pattern_type == "favorite_command":
                    recommendations.append(
                        f"Consider creating shortcuts or improvements for frequently used '{pattern.metadata.get('command')}' command"
                    )
                elif pattern.pattern_type == "dominant_interest":
                    recommendations.append(
                        f"Provide more advanced features or content related to {pattern.metadata.get('topic', '').replace('_', ' ')}"
                    )
                elif pattern.pattern_type == "specialized_usage":
                    recommendations.append(
                        "User shows specialized usage patterns - consider customized interface or advanced features"
                    )

        # General recommendations based on overall patterns
        if len(self.detected_patterns) < 3:
            recommendations.append(
                "Insufficient pattern data - continue monitoring user behavior for personalization"
            )

        return recommendations

    def export_patterns(self) -> str:
        """Export detected patterns to JSON string."""
        import json

        export_data = {
            "patterns": [pattern.to_dict() for pattern in self.detected_patterns],
            "analysis_metadata": {
                "pattern_count": len(self.detected_patterns),
                "export_timestamp": datetime.now().isoformat(),
                "analyzer_config": {
                    "min_pattern_frequency": self.min_pattern_frequency,
                    "confidence_threshold": self.confidence_threshold,
                },
            },
        }
        return json.dumps(export_data, indent=2)

    def clear_patterns(self) -> None:
        """Clear all detected patterns and caches."""
        self.detected_patterns.clear()
        self.interaction_patterns.clear()
        self._temporal_cache.clear()
        self._topic_cache.clear()
        self._command_cache.clear()
        logger.info("All patterns and caches cleared")
