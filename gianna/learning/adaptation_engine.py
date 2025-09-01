"""
Adaptation Engine with Advanced Learning Algorithms.

This module implements the core adaptation engine that uses machine learning
techniques to continuously improve user experience through behavioral adaptation.
"""

import json
import logging
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Different adaptation strategies."""

    CONSERVATIVE = "conservative"  # Slow, careful adaptation
    BALANCED = "balanced"  # Moderate adaptation speed
    AGGRESSIVE = "aggressive"  # Fast adaptation to changes
    CONTEXT_AWARE = "context_aware"  # Adapts based on context


class LearningMode(Enum):
    """Learning modes for the adaptation engine."""

    ONLINE = "online"  # Real-time learning
    BATCH = "batch"  # Periodic batch learning
    HYBRID = "hybrid"  # Combination of both


@dataclass
class AdaptationResult:
    """Result of an adaptation operation."""

    success: bool
    adaptations_applied: Dict[str, str]
    confidence_score: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class LearningMetrics:
    """Metrics for tracking learning performance."""

    total_adaptations: int = 0
    successful_adaptations: int = 0
    user_satisfaction_score: float = 0.0
    adaptation_accuracy: float = 0.0
    learning_rate_effectiveness: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def success_rate(self) -> float:
        """Calculate adaptation success rate."""
        return self.successful_adaptations / max(1, self.total_adaptations)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data["last_updated"] = self.last_updated.isoformat()
        data["success_rate"] = self.success_rate()
        return data


class BaseLearningAlgorithm(ABC):
    """Abstract base class for learning algorithms."""

    @abstractmethod
    def update(
        self, features: Dict[str, float], target: float, weight: float = 1.0
    ) -> None:
        """Update the algorithm with new data."""
        pass

    @abstractmethod
    def predict(self, features: Dict[str, float]) -> float:
        """Make a prediction based on current model."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass


class SimpleLinearLearning(BaseLearningAlgorithm):
    """Simple linear learning algorithm for preference adaptation."""

    def __init__(self, learning_rate: float = 0.1, decay_rate: float = 0.99):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.weights: Dict[str, float] = defaultdict(float)
        self.feature_counts: Dict[str, int] = defaultdict(int)
        self.bias = 0.0

    def update(
        self, features: Dict[str, float], target: float, weight: float = 1.0
    ) -> None:
        """Update weights using simple gradient descent."""
        # Calculate prediction
        prediction = self.predict(features)
        error = target - prediction

        # Update weights
        for feature, value in features.items():
            self.weights[feature] += self.learning_rate * error * value * weight
            self.feature_counts[feature] += 1

        # Update bias
        self.bias += self.learning_rate * error * weight

        # Apply decay
        for feature in self.weights:
            self.weights[feature] *= self.decay_rate

    def predict(self, features: Dict[str, float]) -> float:
        """Make prediction using linear combination."""
        prediction = self.bias
        for feature, value in features.items():
            prediction += self.weights[feature] * value

        # Apply sigmoid to bound between 0 and 1
        return 1.0 / (1.0 + np.exp(-prediction))

    def get_feature_importance(self) -> Dict[str, float]:
        """Get normalized feature importance."""
        if not self.weights:
            return {}

        total_weight = sum(abs(w) for w in self.weights.values())
        if total_weight == 0:
            return {feature: 0.0 for feature in self.weights}

        return {
            feature: abs(weight) / total_weight
            for feature, weight in self.weights.items()
        }


class ExponentialMovingAverage:
    """Exponential moving average for adaptive learning rates."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def update(self, new_value: float) -> float:
        """Update the moving average."""
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value

        return self.value

    def get_value(self) -> float:
        """Get current moving average value."""
        return self.value


class AdaptationEngine:
    """
    Advanced adaptation engine with multiple learning algorithms.

    Uses online learning, pattern recognition, and user feedback to
    continuously improve response quality and user satisfaction.
    """

    def __init__(
        self,
        strategy: AdaptationStrategy = AdaptationStrategy.BALANCED,
        learning_mode: LearningMode = LearningMode.HYBRID,
        max_history_size: int = 1000,
    ):
        """
        Initialize the adaptation engine.

        Args:
            strategy: Adaptation strategy to use
            learning_mode: Online, batch, or hybrid learning
            max_history_size: Maximum number of interactions to keep
        """
        self.strategy = strategy
        self.learning_mode = learning_mode
        self.max_history_size = max_history_size

        # Learning components
        self.preference_learner = SimpleLinearLearning()
        self.satisfaction_predictor = SimpleLinearLearning()
        self.response_optimizer = SimpleLinearLearning()

        # Adaptation state
        self.adaptation_history: deque = deque(maxlen=max_history_size)
        self.user_feedback_buffer: deque = deque(maxlen=100)

        # Performance tracking
        self.metrics = LearningMetrics()
        self.satisfaction_ema = ExponentialMovingAverage(alpha=0.2)
        self.adaptation_success_ema = ExponentialMovingAverage(alpha=0.15)

        # Feature extractors
        self.feature_extractors = self._initialize_feature_extractors()

        # Configuration based on strategy
        self._configure_strategy()

        logger.info(
            f"AdaptationEngine initialized with {strategy.value} strategy and {learning_mode.value} learning"
        )

    def _configure_strategy(self) -> None:
        """Configure engine parameters based on adaptation strategy."""
        if self.strategy == AdaptationStrategy.CONSERVATIVE:
            self.preference_learner.learning_rate = 0.05
            self.confidence_threshold = 0.8
            self.adaptation_frequency = 10  # Adapt every 10 interactions
        elif self.strategy == AdaptationStrategy.AGGRESSIVE:
            self.preference_learner.learning_rate = 0.2
            self.confidence_threshold = 0.6
            self.adaptation_frequency = 2
        elif self.strategy == AdaptationStrategy.CONTEXT_AWARE:
            self.preference_learner.learning_rate = 0.1
            self.confidence_threshold = 0.7
            self.adaptation_frequency = 5
            self.context_weight_multiplier = 1.5
        else:  # BALANCED
            self.preference_learner.learning_rate = 0.1
            self.confidence_threshold = 0.7
            self.adaptation_frequency = 5

    def _initialize_feature_extractors(self) -> Dict[str, callable]:
        """Initialize feature extraction functions."""
        return {
            "response_length": lambda ctx: len(ctx.get("response", "").split()) / 100.0,
            "user_input_length": lambda ctx: len(ctx.get("user_input", "").split())
            / 50.0,
            "response_time": lambda ctx: min(ctx.get("response_time", 0) / 10.0, 1.0),
            "time_of_day": lambda ctx: self._extract_time_features(ctx),
            "interaction_mode": lambda ctx: (
                1.0 if ctx.get("interaction_mode") == "voice" else 0.0
            ),
            "has_command": lambda ctx: 1.0 if ctx.get("command_used") else 0.0,
            "topic_complexity": lambda ctx: self._estimate_topic_complexity(ctx),
            "user_expertise": lambda ctx: self._estimate_user_expertise(ctx),
        }

    def _extract_time_features(self, context: Dict[str, Any]) -> float:
        """Extract time-based features."""
        if "timestamp" not in context:
            return 0.5

        timestamp = context["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Convert hour to cyclical feature (0-1)
        hour = timestamp.hour
        return hour / 24.0

    def _estimate_topic_complexity(self, context: Dict[str, Any]) -> float:
        """Estimate topic complexity based on content."""
        text = context.get("user_input", "") + " " + context.get("response", "")

        # Simple heuristics for complexity
        complexity_indicators = [
            "algorithm",
            "architecture",
            "implementation",
            "optimization",
            "performance",
            "scalability",
            "integration",
            "configuration",
        ]

        matches = sum(
            1 for indicator in complexity_indicators if indicator in text.lower()
        )
        return min(matches / 5.0, 1.0)  # Normalize to 0-1

    def _estimate_user_expertise(self, context: Dict[str, Any]) -> float:
        """Estimate user expertise based on language used."""
        text = context.get("user_input", "").lower()

        technical_terms = [
            "class",
            "function",
            "method",
            "api",
            "database",
            "server",
            "client",
            "protocol",
            "framework",
            "library",
            "module",
        ]

        technical_count = sum(1 for term in technical_terms if term in text)
        return min(technical_count / 10.0, 1.0)

    def learn_from_interaction(
        self,
        context: Dict[str, Any],
        user_satisfaction: Optional[float] = None,
        explicit_feedback: Optional[str] = None,
    ) -> None:
        """
        Learn from a user interaction.

        Args:
            context: Interaction context dictionary
            user_satisfaction: Satisfaction score (0-1) if available
            explicit_feedback: Text feedback from user
        """
        # Extract features
        features = {}
        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = extractor(context)
            except Exception as e:
                logger.warning(f"Feature extraction failed for {feature_name}: {e}")
                features[feature_name] = 0.0

        # Infer satisfaction if not provided
        if user_satisfaction is None:
            user_satisfaction = self._infer_satisfaction(context, explicit_feedback)

        # Update learning algorithms
        self._update_learners(features, user_satisfaction, context)

        # Store interaction for batch learning
        interaction_data = {
            "timestamp": datetime.now(),
            "features": features,
            "satisfaction": user_satisfaction,
            "context": context,
            "feedback": explicit_feedback,
        }
        self.adaptation_history.append(interaction_data)

        # Update metrics
        self.metrics.total_adaptations += 1
        self.satisfaction_ema.update(user_satisfaction)

        # Trigger batch learning if needed
        if (
            self.learning_mode in [LearningMode.BATCH, LearningMode.HYBRID]
            and len(self.adaptation_history) % 50 == 0
        ):
            self._batch_learning_update()

        logger.debug(
            f"Learned from interaction with satisfaction: {user_satisfaction:.2f}"
        )

    def _infer_satisfaction(
        self, context: Dict[str, Any], feedback: Optional[str]
    ) -> float:
        """Infer user satisfaction from context and feedback."""
        satisfaction = 0.5  # Default neutral satisfaction

        if feedback:
            feedback_lower = feedback.lower()

            # Positive indicators
            positive_words = [
                "good",
                "great",
                "perfect",
                "excellent",
                "thanks",
                "correct",
                "helpful",
            ]
            positive_count = sum(1 for word in positive_words if word in feedback_lower)

            # Negative indicators
            negative_words = [
                "bad",
                "wrong",
                "error",
                "incorrect",
                "useless",
                "terrible",
                "failed",
            ]
            negative_count = sum(1 for word in negative_words if word in feedback_lower)

            if positive_count > negative_count:
                satisfaction = min(1.0, 0.7 + (positive_count - negative_count) * 0.1)
            elif negative_count > positive_count:
                satisfaction = max(0.0, 0.3 - (negative_count - positive_count) * 0.1)

        # Adjust based on context
        if context.get("response_time", 0) > 10:  # Slow response
            satisfaction *= 0.9

        if (
            context.get("command_used")
            and "successful" in context.get("command_result", "").lower()
        ):
            satisfaction = min(1.0, satisfaction + 0.1)

        return satisfaction

    def _update_learners(
        self, features: Dict[str, float], satisfaction: float, context: Dict[str, Any]
    ) -> None:
        """Update all learning algorithms."""
        # Update satisfaction predictor
        self.satisfaction_predictor.update(features, satisfaction)

        # Update preference learner based on user choices
        if "user_preference_signal" in context:
            preference_signal = context["user_preference_signal"]
            self.preference_learner.update(features, preference_signal)

        # Update response optimizer
        response_quality = self._calculate_response_quality(context, satisfaction)
        self.response_optimizer.update(features, response_quality)

    def _calculate_response_quality(
        self, context: Dict[str, Any], satisfaction: float
    ) -> float:
        """Calculate overall response quality score."""
        base_quality = satisfaction

        # Adjust for response time
        response_time = context.get("response_time", 5)
        if response_time < 2:
            time_bonus = 0.1
        elif response_time > 10:
            time_bonus = -0.1
        else:
            time_bonus = 0.0

        # Adjust for response completeness
        response = context.get("response", "")
        if len(response.split()) < 10:
            completeness_penalty = -0.05
        elif len(response.split()) > 200:
            completeness_penalty = -0.02  # Might be too verbose
        else:
            completeness_penalty = 0.0

        return max(0.0, min(1.0, base_quality + time_bonus + completeness_penalty))

    def _batch_learning_update(self) -> None:
        """Perform batch learning update on accumulated data."""
        if len(self.adaptation_history) < 10:
            return

        logger.info("Performing batch learning update")

        # Get recent interactions
        recent_interactions = list(self.adaptation_history)[-50:]

        # Batch update for pattern recognition
        feature_importance = self._analyze_feature_importance(recent_interactions)
        self._adjust_learning_rates(feature_importance)

        # Update metrics
        satisfactions = [
            interaction["satisfaction"] for interaction in recent_interactions
        ]
        self.metrics.user_satisfaction_score = statistics.mean(satisfactions)
        self.metrics.last_updated = datetime.now()

    def _analyze_feature_importance(
        self, interactions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze which features are most important for satisfaction."""
        feature_correlations = defaultdict(list)

        for interaction in interactions:
            satisfaction = interaction["satisfaction"]
            features = interaction["features"]

            for feature_name, feature_value in features.items():
                feature_correlations[feature_name].append((feature_value, satisfaction))

        # Calculate correlation coefficients
        importance_scores = {}
        for feature_name, values in feature_correlations.items():
            if len(values) < 5:
                importance_scores[feature_name] = 0.0
                continue

            x_values = [v[0] for v in values]
            y_values = [v[1] for v in values]

            try:
                correlation = np.corrcoef(x_values, y_values)[0, 1]
                importance_scores[feature_name] = (
                    abs(correlation) if not np.isnan(correlation) else 0.0
                )
            except:
                importance_scores[feature_name] = 0.0

        return importance_scores

    def _adjust_learning_rates(self, feature_importance: Dict[str, float]) -> None:
        """Adjust learning rates based on feature importance."""
        if not feature_importance:
            return

        # Find most important features
        max_importance = max(feature_importance.values()) if feature_importance else 0

        if max_importance > 0.3:  # Strong correlation found
            self.preference_learner.learning_rate *= 1.1  # Increase learning rate
        elif max_importance < 0.1:  # Weak correlations
            self.preference_learner.learning_rate *= 0.9  # Decrease learning rate

        # Bound learning rate
        self.preference_learner.learning_rate = max(
            0.01, min(0.5, self.preference_learner.learning_rate)
        )

    def adapt_response(
        self, base_response: str, context: Dict[str, Any]
    ) -> AdaptationResult:
        """
        Adapt a response based on learned preferences.

        Args:
            base_response: Original response to adapt
            context: Current interaction context

        Returns:
            AdaptationResult with adapted response and metadata
        """
        # Extract features for current context
        features = {}
        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = extractor(context)
            except:
                features[feature_name] = 0.0

        # Predict user satisfaction for current response
        predicted_satisfaction = self.satisfaction_predictor.predict(features)

        # Determine adaptations needed
        adaptations_applied = {}
        adapted_response = base_response
        confidence_score = predicted_satisfaction

        # Length adaptation
        if features.get("response_length", 0.5) > 0.8 and predicted_satisfaction < 0.6:
            adapted_response = self._shorten_response(adapted_response)
            adaptations_applied["length"] = "shortened for better satisfaction"
            confidence_score += 0.1
        elif (
            features.get("response_length", 0.5) < 0.3 and predicted_satisfaction < 0.6
        ):
            adapted_response = self._expand_response(adapted_response)
            adaptations_applied["length"] = "expanded for completeness"
            confidence_score += 0.05

        # Technical level adaptation
        user_expertise = features.get("user_expertise", 0.5)
        topic_complexity = features.get("topic_complexity", 0.5)

        if user_expertise < 0.3 and topic_complexity > 0.7:
            adapted_response = self._simplify_response(adapted_response)
            adaptations_applied["complexity"] = "simplified for user level"
            confidence_score += 0.1
        elif user_expertise > 0.7 and topic_complexity < 0.3:
            adapted_response = self._add_technical_detail(adapted_response)
            adaptations_applied["complexity"] = "enhanced technical detail"
            confidence_score += 0.05

        # Update adaptation success tracking
        self.metrics.total_adaptations += 1
        if len(adaptations_applied) > 0:
            self.metrics.successful_adaptations += 1

        # Create result
        result = AdaptationResult(
            success=len(adaptations_applied) > 0,
            adaptations_applied=adaptations_applied,
            confidence_score=min(1.0, confidence_score),
            explanation=self._generate_explanation(
                adaptations_applied, predicted_satisfaction
            ),
            metadata={
                "original_length": len(base_response.split()),
                "adapted_length": len(adapted_response.split()),
                "predicted_satisfaction": predicted_satisfaction,
                "feature_scores": features,
            },
        )

        # Store adapted response in context for learning
        context["adapted_response"] = adapted_response
        context["adaptation_result"] = result.to_dict()

        return result

    def _shorten_response(self, response: str) -> str:
        """Shorten a response while preserving key information."""
        sentences = response.split(". ")
        if len(sentences) <= 2:
            return response

        # Keep first sentence and most important ones
        important_sentences = [sentences[0]]
        for sentence in sentences[1:]:
            if any(
                keyword in sentence.lower()
                for keyword in [
                    "important",
                    "key",
                    "main",
                    "result",
                    "solution",
                    "answer",
                ]
            ):
                important_sentences.append(sentence)
                if len(important_sentences) >= 3:
                    break

        return ". ".join(important_sentences) + (
            "." if not important_sentences[-1].endswith(".") else ""
        )

    def _expand_response(self, response: str) -> str:
        """Expand a response with additional helpful context."""
        if len(response) < 50:
            return response + " I can provide more details if you need them."
        return response

    def _simplify_response(self, response: str) -> str:
        """Simplify technical language in response."""
        simplifications = {
            "implementation": "setup",
            "configuration": "settings",
            "optimization": "improvement",
            "algorithm": "method",
            "architecture": "structure",
        }

        simplified = response
        for technical, simple in simplifications.items():
            simplified = simplified.replace(technical, simple)

        return simplified

    def _add_technical_detail(self, response: str) -> str:
        """Add technical detail for expert users."""
        if "configure" in response.lower() and "parameter" not in response.lower():
            response += (
                " Consider the relevant parameters and their impact on performance."
            )
        elif "implement" in response.lower() and "pattern" not in response.lower():
            response += " You might want to consider appropriate design patterns for this implementation."

        return response

    def _generate_explanation(
        self, adaptations: Dict[str, str], predicted_satisfaction: float
    ) -> str:
        """Generate explanation for adaptations applied."""
        if not adaptations:
            return f"No adaptations needed (predicted satisfaction: {predicted_satisfaction:.2f})"

        explanations = []
        for adaptation_type, description in adaptations.items():
            explanations.append(f"{adaptation_type}: {description}")

        return f"Applied {len(adaptations)} adaptations: {'; '.join(explanations)} (predicted satisfaction improvement: {predicted_satisfaction:.2f})"

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning process."""
        feature_importance = self.preference_learner.get_feature_importance()

        insights = {
            "learning_metrics": self.metrics.to_dict(),
            "feature_importance": feature_importance,
            "most_important_features": sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "current_satisfaction_trend": self.satisfaction_ema.get_value(),
            "adaptation_success_rate": self.adaptation_success_ema.get_value(),
            "learning_algorithm_stats": {
                "preference_learner_weights": dict(self.preference_learner.weights),
                "satisfaction_predictor_weights": dict(
                    self.satisfaction_predictor.weights
                ),
            },
            "recommendations": self._generate_learning_recommendations(),
        }

        return insights

    def _generate_learning_recommendations(self) -> List[str]:
        """Generate recommendations for improving learning."""
        recommendations = []

        if self.metrics.success_rate() < 0.6:
            recommendations.append(
                "Consider adjusting adaptation thresholds - current success rate is low"
            )

        if self.satisfaction_ema.get_value() < 0.5:
            recommendations.append(
                "User satisfaction trend is concerning - review adaptation strategies"
            )

        feature_importance = self.preference_learner.get_feature_importance()
        if feature_importance:
            top_feature = max(feature_importance.items(), key=lambda x: x[1])
            if top_feature[1] > 0.3:
                recommendations.append(
                    f"Focus on optimizing '{top_feature[0]}' - it's the most impactful feature"
                )

        if len(self.adaptation_history) < 50:
            recommendations.append(
                "Collect more interaction data to improve adaptation accuracy"
            )

        return recommendations

    def export_learning_state(self) -> str:
        """Export complete learning state to JSON."""
        state = {
            "strategy": self.strategy.value,
            "learning_mode": self.learning_mode.value,
            "metrics": self.metrics.to_dict(),
            "learner_weights": {
                "preference_learner": dict(self.preference_learner.weights),
                "satisfaction_predictor": dict(self.satisfaction_predictor.weights),
                "response_optimizer": dict(self.response_optimizer.weights),
            },
            "satisfaction_ema": self.satisfaction_ema.get_value(),
            "export_timestamp": datetime.now().isoformat(),
        }

        return json.dumps(state, indent=2)

    def import_learning_state(self, json_data: str) -> bool:
        """Import learning state from JSON."""
        try:
            state = json.loads(json_data)

            # Restore weights
            if "learner_weights" in state:
                weights = state["learner_weights"]

                if "preference_learner" in weights:
                    self.preference_learner.weights.update(
                        weights["preference_learner"]
                    )

                if "satisfaction_predictor" in weights:
                    self.satisfaction_predictor.weights.update(
                        weights["satisfaction_predictor"]
                    )

                if "response_optimizer" in weights:
                    self.response_optimizer.weights.update(
                        weights["response_optimizer"]
                    )

            # Restore EMA
            if "satisfaction_ema" in state:
                self.satisfaction_ema.value = state["satisfaction_ema"]
                self.satisfaction_ema.initialized = True

            logger.info("Successfully imported learning state")
            return True

        except Exception as e:
            logger.error(f"Failed to import learning state: {e}")
            return False

    def reset_learning(self) -> None:
        """Reset all learning state."""
        self.preference_learner = SimpleLinearLearning()
        self.satisfaction_predictor = SimpleLinearLearning()
        self.response_optimizer = SimpleLinearLearning()

        self.adaptation_history.clear()
        self.user_feedback_buffer.clear()

        self.metrics = LearningMetrics()
        self.satisfaction_ema = ExponentialMovingAverage(alpha=0.2)
        self.adaptation_success_ema = ExponentialMovingAverage(alpha=0.15)

        logger.info("Learning state reset - all models cleared")
