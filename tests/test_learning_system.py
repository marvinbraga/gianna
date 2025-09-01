#!/usr/bin/env python3
"""
Basic tests for the Gianna Learning System.

This module provides basic unit tests and integration tests
for the learning system components.
"""

import json
import os

# Import the learning system components
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gianna.learning.adaptation_engine import (
    AdaptationEngine,
    AdaptationResult,
    AdaptationStrategy,
    LearningMode,
)
from gianna.learning.pattern_analysis import (
    InteractionType,
    PatternAnalyzer,
    UsagePattern,
)
from gianna.learning.state_integration import LearningStateManager
from gianna.learning.user_adaptation import (
    InteractionContext,
    PreferenceType,
    UserPreference,
    UserPreferenceLearner,
)


class TestUserPreferenceLearner(unittest.TestCase):
    """Test cases for UserPreferenceLearner."""

    def setUp(self):
        """Set up test fixtures."""
        self.learner = UserPreferenceLearner(max_history=50)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(len(self.learner.preferences), 0)
        self.assertEqual(len(self.learner.interaction_history), 0)
        self.assertEqual(self.learner.max_history, 50)

    def test_record_interaction(self):
        """Test recording interactions."""
        context = InteractionContext(
            user_input="How do I create a function in Python?",
            response_generated="To create a function, use the 'def' keyword.",
            timestamp=datetime.now(),
            interaction_mode="text",
            user_feedback="Too brief, can you elaborate?",
        )

        self.learner.record_interaction(context)

        self.assertEqual(len(self.learner.interaction_history), 1)
        self.assertEqual(self.learner.interaction_history[0], context)

    def test_preference_learning_from_feedback(self):
        """Test learning preferences from explicit feedback."""
        # Create interactions with clear preference signals
        contexts = [
            InteractionContext(
                user_input="Explain Python classes",
                response_generated="A very long detailed explanation about Python classes with many examples...",
                timestamp=datetime.now(),
                interaction_mode="text",
                user_feedback="Too long, please be brief",
            ),
            InteractionContext(
                user_input="How to use loops?",
                response_generated="Use for loops for iteration.",
                timestamp=datetime.now(),
                interaction_mode="text",
                user_feedback="Perfect length!",
            ),
            InteractionContext(
                user_input="What are decorators?",
                response_generated="Short answer here.",
                timestamp=datetime.now(),
                interaction_mode="text",
                user_feedback="Great, concise response",
            ),
        ]

        for context in contexts:
            self.learner.record_interaction(context)

        # Check if brief preference was learned
        if PreferenceType.RESPONSE_LENGTH in self.learner.preferences:
            pref = self.learner.preferences[PreferenceType.RESPONSE_LENGTH]
            self.assertEqual(pref.value, "brief")
            self.assertGreater(pref.confidence, 0.5)

    def test_response_adaptation(self):
        """Test response adaptation based on preferences."""
        # Set up a preference
        self.learner._update_preference(
            PreferenceType.RESPONSE_LENGTH, "brief", 0.8, "test"
        )

        long_response = "This is a very long response with lots of details about the topic that could be shortened."
        adapted, metadata = self.learner.adapt_response_style(long_response)

        # Response should be shortened
        self.assertLess(len(adapted.split()), len(long_response.split()))
        self.assertIn("length", metadata)

    def test_preference_export_import(self):
        """Test exporting and importing preferences."""
        # Create some preferences
        self.learner._update_preference(
            PreferenceType.RESPONSE_LENGTH, "brief", 0.8, "test"
        )
        self.learner._update_preference(
            PreferenceType.COMMUNICATION_STYLE, "casual", 0.7, "test"
        )

        # Export
        export_data = self.learner.export_preferences()
        self.assertIsInstance(export_data, str)

        # Clear preferences
        self.learner.preferences.clear()
        self.assertEqual(len(self.learner.preferences), 0)

        # Import
        success = self.learner.import_preferences(export_data)
        self.assertTrue(success)
        self.assertEqual(len(self.learner.preferences), 2)

    def test_pattern_analysis(self):
        """Test user pattern analysis."""
        # Add multiple interactions
        for i in range(5):
            context = InteractionContext(
                user_input=f"Test input {i}",
                response_generated=f"Test response {i}",
                timestamp=datetime.now() - timedelta(days=i),
                interaction_mode="text",
                response_time=1.0 + i * 0.5,
            )
            self.learner.record_interaction(context)

        analysis = self.learner.analyze_user_patterns()

        self.assertIn("total_interactions", analysis)
        self.assertEqual(analysis["total_interactions"], 5)
        self.assertIn("interaction_patterns", analysis)


class TestPatternAnalyzer(unittest.TestCase):
    """Test cases for PatternAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = PatternAnalyzer()

        # Create sample interactions
        self.sample_interactions = [
            Mock(
                timestamp=datetime(2024, 1, 1, 9, 0),  # Morning
                command_used="create",
                user_input="create a function",
                response_generated="Here's how to create a function...",
                user_feedback="good",
            ),
            Mock(
                timestamp=datetime(2024, 1, 1, 14, 0),  # Afternoon
                command_used="analyze",
                user_input="analyze this code",
                response_generated="The code analysis shows...",
                user_feedback="helpful",
            ),
            Mock(
                timestamp=datetime(2024, 1, 2, 9, 30),  # Next day morning
                command_used="create",
                user_input="create another function",
                response_generated="Another function example...",
                user_feedback="perfect",
            ),
        ]

    def test_temporal_pattern_analysis(self):
        """Test temporal pattern detection."""
        analysis = self.analyzer.analyze_temporal_patterns(self.sample_interactions)

        self.assertIn("total_interactions", analysis)
        self.assertIn("peak_time_periods", analysis)
        self.assertEqual(analysis["total_interactions"], 3)

    def test_command_frequency_analysis(self):
        """Test command frequency analysis."""
        analysis = self.analyzer.analyze_command_frequency(self.sample_interactions)

        self.assertIn("total_commands", analysis)
        self.assertIn("most_used_commands", analysis)
        self.assertEqual(analysis["most_used_commands"]["create"], 2)
        self.assertEqual(analysis["most_used_commands"]["analyze"], 1)

    def test_topic_interest_analysis(self):
        """Test topic interest analysis."""
        # Add programming-related interactions
        interactions = []
        for i in range(3):
            mock_interaction = Mock()
            mock_interaction.user_input = "python programming function class"
            mock_interaction.response_generated = "programming concepts and algorithms"
            interactions.append(mock_interaction)

        analysis = self.analyzer.analyze_topic_interests(interactions)

        self.assertIn("topic_distribution", analysis)
        # Should detect programming as a topic
        if analysis["topic_distribution"]:
            self.assertIn("programming", analysis["topic_distribution"])

    def test_comprehensive_analysis(self):
        """Test comprehensive pattern analysis."""
        analysis = self.analyzer.get_comprehensive_analysis(self.sample_interactions)

        self.assertIn("overview", analysis)
        self.assertIn("temporal_patterns", analysis)
        self.assertIn("command_analysis", analysis)
        self.assertIn("detected_patterns", analysis)
        self.assertEqual(analysis["overview"]["total_interactions"], 3)


class TestAdaptationEngine(unittest.TestCase):
    """Test cases for AdaptationEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = AdaptationEngine(strategy=AdaptationStrategy.BALANCED)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.engine.strategy, AdaptationStrategy.BALANCED)
        self.assertEqual(self.engine.learning_mode, LearningMode.HYBRID)
        self.assertEqual(len(self.engine.adaptation_history), 0)

    def test_learning_from_interaction(self):
        """Test learning from interactions."""
        context = {
            "user_input": "Test input",
            "response": "Test response",
            "timestamp": datetime.now(),
            "interaction_mode": "text",
            "response_time": 2.0,
        }

        self.engine.learn_from_interaction(context, user_satisfaction=0.8)

        self.assertEqual(len(self.engine.adaptation_history), 1)
        self.assertEqual(self.engine.metrics.total_adaptations, 1)

    def test_response_adaptation(self):
        """Test response adaptation."""
        base_response = "This is a test response that might need adaptation."
        context = {
            "user_input": "Test question",
            "interaction_mode": "text",
            "response_time": 2.0,
            "user_expertise": 0.5,
            "topic_complexity": 0.6,
        }

        result = self.engine.adapt_response(base_response, context)

        self.assertIsInstance(result, AdaptationResult)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.adaptations_applied, dict)
        self.assertIsInstance(result.confidence_score, float)

    def test_feature_extraction(self):
        """Test feature extraction."""
        context = {
            "response": "Test response with multiple words here",
            "user_input": "Short input",
            "response_time": 3.0,
            "interaction_mode": "voice",
            "command_used": "test",
            "timestamp": datetime.now(),
        }

        features = {}
        for feature_name, extractor in self.engine.feature_extractors.items():
            features[feature_name] = extractor(context)

        self.assertIn("response_length", features)
        self.assertIn("interaction_mode", features)
        self.assertEqual(features["interaction_mode"], 1.0)  # Voice mode
        self.assertEqual(features["has_command"], 1.0)  # Has command

    def test_learning_state_export_import(self):
        """Test learning state export/import."""
        # Train the model a bit
        context = {
            "user_input": "Test",
            "response": "Test response",
            "timestamp": datetime.now(),
            "interaction_mode": "text",
        }

        self.engine.learn_from_interaction(context, 0.8)

        # Export state
        export_data = self.engine.export_learning_state()
        self.assertIsInstance(export_data, str)

        # Reset and import
        self.engine.reset_learning()
        success = self.engine.import_learning_state(export_data)
        self.assertTrue(success)


class TestLearningStateManager(unittest.TestCase):
    """Test cases for LearningStateManager integration."""

    def setUp(self):
        """Set up test fixtures with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_gianna.db")

        # Mock StateManager to use temp database
        with patch("gianna.learning.state_integration.StateManager") as mock_state:
            mock_state.return_value.db_path = self.db_path
            self.state_manager = LearningStateManager()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertIsNotNone(self.state_manager.preference_learner)
        self.assertIsNotNone(self.state_manager.pattern_analyzer)
        self.assertIsNotNone(self.state_manager.adaptation_engine)

    def test_record_and_persist_interaction(self):
        """Test recording and persisting interactions."""
        context = InteractionContext(
            user_input="Test input",
            response_generated="Test response",
            timestamp=datetime.now(),
            interaction_mode="text",
            user_feedback="Good response",
        )

        initial_count = len(self.state_manager.preference_learner.interaction_history)
        self.state_manager.record_interaction(context, satisfaction_score=0.8)

        # Should have recorded the interaction
        final_count = len(self.state_manager.preference_learner.interaction_history)
        self.assertEqual(final_count, initial_count + 1)

    def test_user_profile_generation(self):
        """Test user profile generation."""
        # Add some interactions
        context = InteractionContext(
            user_input="Test query",
            response_generated="Test response",
            timestamp=datetime.now(),
            interaction_mode="text",
            command_used="test",
        )

        self.state_manager.record_interaction(context, satisfaction_score=0.7)

        profile = self.state_manager.get_user_profile()

        self.assertIn("user_id", profile)
        self.assertIn("preferences", profile)
        self.assertIn("behavioral_patterns", profile)
        self.assertIn("interaction_stats", profile)

    def test_learning_statistics(self):
        """Test learning statistics generation."""
        stats = self.state_manager.get_learning_statistics()

        self.assertIn("database_stats", stats)
        self.assertIn("learning_performance", stats)
        self.assertIn("overall_confidence", stats)
        self.assertIn("system_health", stats)


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for complete learning workflow."""

    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "integration_test.db")

    def tearDown(self):
        """Clean up."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_learning_workflow(self):
        """Test complete learning workflow integration."""
        # Initialize system
        with patch("gianna.learning.state_integration.StateManager") as mock_state:
            mock_state.return_value.db_path = self.db_path
            learning_manager = LearningStateManager()

        # Simulate user interactions
        interactions = [
            (
                "How to code in Python?",
                "Here's how to code in Python...",
                "text",
                "Great!",
            ),
            (
                "Explain functions",
                "Functions are reusable blocks of code...",
                "text",
                "Too detailed",
            ),
            ("Quick example please", "def example(): pass", "text", "Perfect!"),
        ]

        # Process interactions
        for i, (user_input, response, mode, feedback) in enumerate(interactions):
            context = InteractionContext(
                user_input=user_input,
                response_generated=response,
                timestamp=datetime.now() - timedelta(minutes=30 - i * 10),
                interaction_mode=mode,
                user_feedback=feedback,
            )

            learning_manager.record_interaction(
                context, satisfaction_score=0.7 + i * 0.1
            )

        # Test adaptation
        test_response = "This is a very long and detailed explanation that should be adapted based on learned preferences."
        adapted_response, metadata = learning_manager.adapt_response(
            test_response, {"user_input": "Test question", "interaction_mode": "text"}
        )

        # Should have some adaptation
        self.assertIsInstance(adapted_response, str)
        self.assertIsInstance(metadata, dict)

        # Test profile generation
        profile = learning_manager.get_user_profile()
        self.assertGreater(profile["interaction_stats"]["total_interactions"], 0)

        # Test persistence
        learning_manager.save_learning_state()
        stats = learning_manager.get_learning_statistics()
        self.assertGreater(stats["database_stats"]["total_interactions"], 0)


def run_basic_tests():
    """Run basic test suite."""
    # Create test suite
    test_classes = [
        TestUserPreferenceLearner,
        TestPatternAnalyzer,
        TestAdaptationEngine,
        TestLearningStateManager,
        TestIntegrationWorkflow,
    ]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Gianna Learning System Tests")
    print("=" * 50)

    success = run_basic_tests()

    if success:
        print("\n✅ All tests passed!")
        exit_code = 0
    else:
        print("\n❌ Some tests failed!")
        exit_code = 1

    sys.exit(exit_code)
