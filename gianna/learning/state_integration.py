"""
State Integration Module for Learning System Persistence.

This module provides integration between the learning system and GiannaState
for persistent storage of user preferences, patterns, and adaptation models.
"""

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..core.state_manager import StateManager
from .adaptation_engine import AdaptationEngine, AdaptationStrategy, LearningMode
from .pattern_analysis import PatternAnalyzer, UsagePattern
from .user_adaptation import InteractionContext, PreferenceType, UserPreferenceLearner

logger = logging.getLogger(__name__)


class LearningStateManager:
    """
    Manages persistence and retrieval of learning system state.

    Integrates with GiannaState to provide seamless persistence of:
    - User preferences and confidence scores
    - Detected behavioral patterns
    - Adaptation model weights and parameters
    - Learning metrics and performance data
    """

    def __init__(self, state_manager: Optional[StateManager] = None):
        """
        Initialize the learning state manager.

        Args:
            state_manager: StateManager instance, creates new if None
        """
        self.state_manager = state_manager or StateManager()
        self.db_path = str(self.state_manager.db_path)

        # Initialize learning components
        self.preference_learner = UserPreferenceLearner()
        self.pattern_analyzer = PatternAnalyzer()
        self.adaptation_engine = AdaptationEngine()

        # Initialize database tables
        self._initialize_learning_tables()

        # Load existing learning state
        self._load_learning_state()

        logger.info("LearningStateManager initialized with database persistence")

    def _initialize_learning_tables(self) -> None:
        """Initialize database tables for learning system persistence."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # User preferences table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    preference_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    learned_from TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    frequency INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(preference_type) ON CONFLICT REPLACE
                )
            """
            )

            # Interaction history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS interaction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT,
                    response_generated TEXT,
                    timestamp TEXT NOT NULL,
                    interaction_mode TEXT NOT NULL,
                    user_feedback TEXT,
                    response_time REAL DEFAULT 0.0,
                    command_used TEXT,
                    topic_detected TEXT,
                    session_id TEXT,
                    satisfaction_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Detected patterns table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detected_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    frequency REAL NOT NULL,
                    confidence REAL NOT NULL,
                    examples TEXT, -- JSON array
                    metadata TEXT, -- JSON object
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Learning metrics table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_adaptations INTEGER DEFAULT 0,
                    successful_adaptations INTEGER DEFAULT 0,
                    user_satisfaction_score REAL DEFAULT 0.0,
                    adaptation_accuracy REAL DEFAULT 0.0,
                    learning_rate_effectiveness REAL DEFAULT 0.0,
                    last_updated TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Adaptation model weights table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptation_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL UNIQUE,
                    model_weights TEXT NOT NULL, -- JSON serialized weights
                    model_parameters TEXT, -- JSON serialized parameters
                    performance_metrics TEXT, -- JSON serialized metrics
                    last_updated TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name) ON CONFLICT REPLACE
                )
            """
            )

            conn.commit()
            logger.debug("Learning database tables initialized")

    def _load_learning_state(self) -> None:
        """Load existing learning state from database."""
        try:
            # Load user preferences
            self._load_user_preferences()

            # Load interaction history
            self._load_interaction_history()

            # Load detected patterns
            self._load_detected_patterns()

            # Load adaptation models
            self._load_adaptation_models()

            # Load learning metrics
            self._load_learning_metrics()

            logger.info("Learning state loaded from database")

        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")

    def _load_user_preferences(self) -> None:
        """Load user preferences from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_preferences ORDER BY last_updated DESC")

            for row in cursor.fetchall():
                try:
                    pref_type = PreferenceType(row[1])  # preference_type
                    preference_data = {
                        "preference_type": row[1],
                        "value": row[2],
                        "confidence": row[3],
                        "learned_from": row[4],
                        "last_updated": row[5],
                        "frequency": row[6],
                    }

                    # Import into preference learner
                    from .user_adaptation import UserPreference

                    preference = UserPreference.from_dict(preference_data)
                    self.preference_learner.preferences[pref_type] = preference

                except ValueError as e:
                    logger.warning(f"Skipping invalid preference type: {row[1]}")
                    continue

    def _load_interaction_history(self, limit: int = 500) -> None:
        """Load recent interaction history from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM interaction_history
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

            for row in cursor.fetchall():
                interaction = InteractionContext(
                    user_input=row[1] or "",
                    response_generated=row[2] or "",
                    timestamp=datetime.fromisoformat(row[3]),
                    interaction_mode=row[4],
                    user_feedback=row[5],
                    response_time=row[6] or 0.0,
                    command_used=row[7],
                    topic_detected=row[8],
                    session_id=row[9],
                )

                self.preference_learner.interaction_history.append(interaction)

    def _load_detected_patterns(self) -> None:
        """Load detected patterns from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM detected_patterns ORDER BY updated_at DESC")

            for row in cursor.fetchall():
                try:
                    examples = json.loads(row[5]) if row[5] else []
                    metadata = json.loads(row[6]) if row[6] else {}

                    pattern = UsagePattern(
                        pattern_type=row[1],
                        description=row[2],
                        frequency=row[3],
                        confidence=row[4],
                        examples=examples,
                        metadata=metadata,
                    )

                    self.pattern_analyzer.detected_patterns.append(pattern)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse pattern data: {e}")
                    continue

    def _load_adaptation_models(self) -> None:
        """Load adaptation model weights from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM adaptation_models ORDER BY last_updated DESC")

            for row in cursor.fetchall():
                try:
                    model_name = row[1]
                    weights_data = json.loads(row[2])

                    # Load weights into appropriate model
                    if model_name == "preference_learner":
                        self.adaptation_engine.preference_learner.weights.update(
                            weights_data
                        )
                    elif model_name == "satisfaction_predictor":
                        self.adaptation_engine.satisfaction_predictor.weights.update(
                            weights_data
                        )
                    elif model_name == "response_optimizer":
                        self.adaptation_engine.response_optimizer.weights.update(
                            weights_data
                        )

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to load model weights for {row[1]}: {e}")
                    continue

    def _load_learning_metrics(self) -> None:
        """Load learning metrics from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM learning_metrics ORDER BY last_updated DESC LIMIT 1"
            )

            row = cursor.fetchone()
            if row:
                self.adaptation_engine.metrics.total_adaptations = row[1]
                self.adaptation_engine.metrics.successful_adaptations = row[2]
                self.adaptation_engine.metrics.user_satisfaction_score = row[3]
                self.adaptation_engine.metrics.adaptation_accuracy = row[4]
                self.adaptation_engine.metrics.learning_rate_effectiveness = row[5]
                self.adaptation_engine.metrics.last_updated = datetime.fromisoformat(
                    row[6]
                )

    def record_interaction(
        self, context: InteractionContext, satisfaction_score: Optional[float] = None
    ) -> None:
        """
        Record a new interaction and update learning models.

        Args:
            context: The interaction context
            satisfaction_score: Optional satisfaction score (0-1)
        """
        # Record in preference learner
        self.preference_learner.record_interaction(context)

        # Learn from interaction in adaptation engine
        context_dict = {
            "user_input": context.user_input,
            "response": context.response_generated,
            "timestamp": context.timestamp,
            "interaction_mode": context.interaction_mode,
            "response_time": context.response_time,
            "command_used": context.command_used,
            "topic_detected": context.topic_detected,
        }

        self.adaptation_engine.learn_from_interaction(
            context_dict, satisfaction_score, context.user_feedback
        )

        # Persist to database
        self._save_interaction_to_db(context, satisfaction_score)

        # Save updated learning state periodically
        import random

        if random.random() < 0.1:  # 10% chance to save state
            self.save_learning_state()

    def _save_interaction_to_db(
        self, context: InteractionContext, satisfaction_score: Optional[float]
    ) -> None:
        """Save interaction to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO interaction_history
                (user_input, response_generated, timestamp, interaction_mode,
                 user_feedback, response_time, command_used, topic_detected,
                 session_id, satisfaction_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    context.user_input,
                    context.response_generated,
                    context.timestamp.isoformat(),
                    context.interaction_mode,
                    context.user_feedback,
                    context.response_time,
                    context.command_used,
                    context.topic_detected,
                    context.session_id,
                    satisfaction_score,
                ),
            )
            conn.commit()

    def save_learning_state(self) -> None:
        """Save complete learning state to database."""
        try:
            self._save_user_preferences()
            self._save_detected_patterns()
            self._save_adaptation_models()
            self._save_learning_metrics()

            logger.debug("Learning state saved to database")

        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")

    def _save_user_preferences(self) -> None:
        """Save user preferences to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for pref_type, preference in self.preference_learner.preferences.items():
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO user_preferences
                    (preference_type, value, confidence, learned_from,
                     last_updated, frequency)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        pref_type.value,
                        preference.value,
                        preference.confidence,
                        preference.learned_from,
                        preference.last_updated.isoformat(),
                        preference.frequency,
                    ),
                )

            conn.commit()

    def _save_detected_patterns(self) -> None:
        """Save detected patterns to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Clear existing patterns
            cursor.execute("DELETE FROM detected_patterns")

            # Save new patterns
            for pattern in self.pattern_analyzer.detected_patterns:
                cursor.execute(
                    """
                    INSERT INTO detected_patterns
                    (pattern_type, description, frequency, confidence, examples, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern.pattern_type,
                        pattern.description,
                        pattern.frequency,
                        pattern.confidence,
                        json.dumps(pattern.examples),
                        json.dumps(pattern.metadata),
                    ),
                )

            conn.commit()

    def _save_adaptation_models(self) -> None:
        """Save adaptation model weights to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            models = {
                "preference_learner": dict(
                    self.adaptation_engine.preference_learner.weights
                ),
                "satisfaction_predictor": dict(
                    self.adaptation_engine.satisfaction_predictor.weights
                ),
                "response_optimizer": dict(
                    self.adaptation_engine.response_optimizer.weights
                ),
            }

            for model_name, weights in models.items():
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO adaptation_models
                    (model_name, model_weights, last_updated)
                    VALUES (?, ?, ?)
                """,
                    (model_name, json.dumps(weights), datetime.now().isoformat()),
                )

            conn.commit()

    def _save_learning_metrics(self) -> None:
        """Save learning metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            metrics = self.adaptation_engine.metrics
            cursor.execute(
                """
                INSERT OR REPLACE INTO learning_metrics
                (id, total_adaptations, successful_adaptations, user_satisfaction_score,
                 adaptation_accuracy, learning_rate_effectiveness, last_updated)
                VALUES (1, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.total_adaptations,
                    metrics.successful_adaptations,
                    metrics.user_satisfaction_score,
                    metrics.adaptation_accuracy,
                    metrics.learning_rate_effectiveness,
                    metrics.last_updated.isoformat(),
                ),
            )

            conn.commit()

    def get_user_profile(self) -> Dict[str, Any]:
        """Get comprehensive user profile based on learned data."""
        # Get preference summary
        preference_summary = self.preference_learner.get_preference_summary()

        # Get recent interaction analysis
        recent_interactions = list(self.preference_learner.interaction_history)[-50:]
        if recent_interactions:
            pattern_analysis = self.pattern_analyzer.get_comprehensive_analysis(
                recent_interactions
            )
        else:
            pattern_analysis = {"error": "No recent interactions"}

        # Get adaptation insights
        adaptation_insights = self.adaptation_engine.get_learning_insights()

        profile = {
            "user_id": "default",  # Could be extended for multi-user support
            "profile_created": datetime.now().isoformat(),
            "preferences": preference_summary,
            "behavioral_patterns": pattern_analysis,
            "adaptation_performance": adaptation_insights,
            "interaction_stats": {
                "total_interactions": len(self.preference_learner.interaction_history),
                "learning_confidence": self._calculate_overall_confidence(),
            },
        }

        return profile

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence in learned user model."""
        if not self.preference_learner.preferences:
            return 0.0

        confidences = [
            pref.confidence for pref in self.preference_learner.preferences.values()
        ]
        interaction_count = len(self.preference_learner.interaction_history)

        # Weight confidence by number of interactions
        base_confidence = sum(confidences) / len(confidences)
        interaction_weight = min(
            1.0, interaction_count / 100.0
        )  # Full confidence at 100+ interactions

        return base_confidence * interaction_weight

    def adapt_response(
        self, response: str, context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Adapt response using all learned models.

        Args:
            response: Original response
            context: Interaction context

        Returns:
            Tuple of (adapted_response, adaptation_metadata)
        """
        # First apply preference-based adaptation
        pref_adapted, pref_metadata = self.preference_learner.adapt_response_style(
            response, None
        )

        # Then apply ML-based adaptation
        adaptation_result = self.adaptation_engine.adapt_response(pref_adapted, context)

        final_response = context.get("adapted_response", pref_adapted)

        # Combine metadata
        combined_metadata = {
            "preference_adaptations": pref_metadata,
            "ml_adaptations": adaptation_result.adaptations_applied,
            "confidence_score": adaptation_result.confidence_score,
            "explanation": adaptation_result.explanation,
        }

        return final_response, combined_metadata

    def export_learning_data(self) -> Dict[str, Any]:
        """Export all learning data for backup or analysis."""
        return {
            "preferences": self.preference_learner.export_preferences(),
            "patterns": self.pattern_analyzer.export_patterns(),
            "adaptation_models": self.adaptation_engine.export_learning_state(),
            "export_timestamp": datetime.now().isoformat(),
        }

    def import_learning_data(self, data: Dict[str, Any]) -> bool:
        """Import learning data from backup."""
        try:
            success = True

            if "preferences" in data:
                success &= self.preference_learner.import_preferences(
                    data["preferences"]
                )

            if "adaptation_models" in data:
                success &= self.adaptation_engine.import_learning_state(
                    data["adaptation_models"]
                )

            if success:
                self.save_learning_state()
                logger.info("Successfully imported learning data")
            else:
                logger.error("Failed to import some learning data")

            return success

        except Exception as e:
            logger.error(f"Failed to import learning data: {e}")
            return False

    def reset_all_learning(self) -> None:
        """Reset all learning data and models."""
        # Reset learning components
        self.preference_learner.reset_learning()
        self.pattern_analyzer.clear_patterns()
        self.adaptation_engine.reset_learning()

        # Clear database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_preferences")
            cursor.execute("DELETE FROM detected_patterns")
            cursor.execute("DELETE FROM learning_metrics")
            cursor.execute("DELETE FROM adaptation_models")
            # Keep interaction history for future learning
            conn.commit()

        logger.info("All learning data reset")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning system statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Database statistics
            cursor.execute("SELECT COUNT(*) FROM interaction_history")
            interaction_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM user_preferences")
            preference_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM detected_patterns")
            pattern_count = cursor.fetchone()[0]

            # Recent activity
            cursor.execute(
                """
                SELECT COUNT(*) FROM interaction_history
                WHERE timestamp > datetime('now', '-7 days')
            """
            )
            recent_interactions = cursor.fetchone()[0]

        stats = {
            "database_stats": {
                "total_interactions": interaction_count,
                "stored_preferences": preference_count,
                "detected_patterns": pattern_count,
                "recent_activity": recent_interactions,
            },
            "learning_performance": self.adaptation_engine.metrics.to_dict(),
            "overall_confidence": self._calculate_overall_confidence(),
            "system_health": (
                "good" if self._calculate_overall_confidence() > 0.5 else "learning"
            ),
            "last_updated": datetime.now().isoformat(),
        }

        return stats
