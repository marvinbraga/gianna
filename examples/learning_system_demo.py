#!/usr/bin/env python3
"""
Learning System Demo - Complete demonstration of Gianna's learning capabilities.

This demo shows how to use the learning system for user adaptation,
pattern analysis, and response optimization.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gianna.learning.adaptation_engine import AdaptationStrategy, LearningMode
from gianna.learning.state_integration import LearningStateManager
from gianna.learning.user_adaptation import InteractionContext


def demonstrate_basic_learning():
    """Demonstrate basic learning functionality."""
    print("=" * 60)
    print("BASIC LEARNING DEMONSTRATION")
    print("=" * 60)

    # Initialize learning system
    learning_manager = LearningStateManager()
    print("✓ Learning system initialized")

    # Simulate user interactions
    interactions = [
        {
            "user_input": "How do I implement a REST API?",
            "response": "To implement a REST API, you need to create endpoints that handle HTTP requests. Here's a detailed explanation of the process...",
            "mode": "text",
            "feedback": "Too detailed, can you be more brief?",
            "command": None,
        },
        {
            "user_input": "Show me a quick example of Python classes",
            "response": "Here's a simple Python class example:\n\nclass Person:\n    def __init__(self, name):\n        self.name = name",
            "mode": "text",
            "feedback": "Perfect, thanks!",
            "command": None,
        },
        {
            "user_input": "How to optimize database queries?",
            "response": "Use indexes, limit results, optimize JOINs.",
            "mode": "text",
            "feedback": "Good but could use more examples",
            "command": "analyze",
        },
        {
            "user_input": "Create a simple web form",
            "response": "Here's a basic HTML form with validation and styling...",
            "mode": "text",
            "feedback": "Great explanation!",
            "command": "create",
        },
    ]

    # Process interactions
    print("\nProcessing user interactions...")
    for i, interaction in enumerate(interactions):
        context = InteractionContext(
            user_input=interaction["user_input"],
            response_generated=interaction["response"],
            timestamp=datetime.now()
            - timedelta(days=7 - i),  # Simulate interactions over past week
            interaction_mode=interaction["mode"],
            user_feedback=interaction["feedback"],
            response_time=2.0 + i * 0.5,  # Simulate varying response times
            command_used=interaction["command"],
        )

        # Infer satisfaction from feedback
        feedback = interaction["feedback"].lower()
        if any(word in feedback for word in ["perfect", "great", "good"]):
            satisfaction = 0.9
        elif any(word in feedback for word in ["too", "could", "more"]):
            satisfaction = 0.6
        else:
            satisfaction = 0.7

        learning_manager.record_interaction(context, satisfaction)
        print(f"  ✓ Processed interaction {i+1}: {interaction['user_input'][:30]}...")

    print(f"\n📊 Learning Summary:")
    print(f"  - Total interactions recorded: {len(interactions)}")

    # Show learned preferences
    preferences = learning_manager.preference_learner.get_preference_summary()
    print(f"  - Preferences learned: {preferences['preferences_count']}")

    if preferences["high_confidence_preferences"]:
        print("  - High confidence preferences:")
        for pref in preferences["high_confidence_preferences"]:
            print(
                f"    • {pref['type']}: {pref['value']} (confidence: {pref['confidence']:.2f})"
            )

    return learning_manager


def demonstrate_pattern_analysis(learning_manager):
    """Demonstrate pattern analysis capabilities."""
    print("\n" + "=" * 60)
    print("PATTERN ANALYSIS DEMONSTRATION")
    print("=" * 60)

    # Add more diverse interactions for pattern analysis
    additional_interactions = [
        # Morning interactions
        (
            "Good morning! How to set up a development environment?",
            "Here's how to set up your dev environment...",
            "09:00",
            "text",
            "setup",
        ),
        (
            "Can you help me debug this code?",
            "Let's analyze your code step by step...",
            "09:30",
            "text",
            "debug",
        ),
        # Afternoon interactions
        (
            "Implement user authentication",
            "I'll show you how to implement secure authentication...",
            "14:00",
            "text",
            "implement",
        ),
        (
            "Optimize this SQL query",
            "Here are several optimization techniques...",
            "15:30",
            "text",
            "optimize",
        ),
        # Evening interactions
        (
            "Voice command: create documentation",
            "I'll help you create comprehensive documentation...",
            "19:00",
            "voice",
            "create",
        ),
        (
            "Voice: explain machine learning concepts",
            "Machine learning is a subset of AI that...",
            "20:00",
            "voice",
            "explain",
        ),
    ]

    print("Adding diverse interactions for pattern analysis...")
    base_time = datetime.now() - timedelta(days=1)

    for i, (user_input, response, time_str, mode, command) in enumerate(
        additional_interactions
    ):
        # Parse time and create realistic timestamp
        hour, minute = map(int, time_str.split(":"))
        interaction_time = base_time.replace(hour=hour, minute=minute)

        context = InteractionContext(
            user_input=user_input,
            response_generated=response,
            timestamp=interaction_time,
            interaction_mode=mode,
            user_feedback="Helpful, thanks!" if i % 2 == 0 else None,
            response_time=1.5 + (i * 0.2),
            command_used=command,
            topic_detected=(
                "programming"
                if "code" in user_input or "debug" in user_input
                else "general"
            ),
        )

        learning_manager.record_interaction(context, 0.8)
        print(f"  ✓ Added {mode} interaction at {time_str}")

    # Analyze patterns
    print("\nAnalyzing behavioral patterns...")
    interactions_list = list(learning_manager.preference_learner.interaction_history)
    analysis = learning_manager.pattern_analyzer.get_comprehensive_analysis(
        interactions_list
    )

    print(f"\n📈 Pattern Analysis Results:")

    # Temporal patterns
    if (
        "temporal_patterns" in analysis
        and "peak_time_periods" in analysis["temporal_patterns"]
    ):
        peak_periods = analysis["temporal_patterns"]["peak_time_periods"][:3]
        print(f"  🕒 Peak usage times:")
        for period in peak_periods:
            print(f"    • {period['period']}: {period['count']} interactions")

    # Command analysis
    if (
        "command_analysis" in analysis
        and "most_used_commands" in analysis["command_analysis"]
    ):
        commands = analysis["command_analysis"]["most_used_commands"]
        print(f"  🔧 Most used commands:")
        for cmd, count in list(commands.items())[:3]:
            print(f"    • {cmd}: {count} times")

    # Topic analysis
    if (
        "topic_analysis" in analysis
        and "topic_distribution" in analysis["topic_analysis"]
    ):
        topics = analysis["topic_analysis"]["topic_distribution"]
        print(f"  📚 Topic interests:")
        for topic, score in list(topics.items())[:3]:
            print(f"    • {topic.replace('_', ' ').title()}: {score:.1%}")

    # Detected patterns
    patterns = learning_manager.pattern_analyzer.detected_patterns
    if patterns:
        print(f"  🔍 Detected behavioral patterns:")
        for pattern in patterns[:3]:
            print(f"    • {pattern.pattern_type}: {pattern.description}")
            print(f"      Confidence: {pattern.confidence:.2f}")

    return analysis


def demonstrate_response_adaptation(learning_manager):
    """Demonstrate intelligent response adaptation."""
    print("\n" + "=" * 60)
    print("RESPONSE ADAPTATION DEMONSTRATION")
    print("=" * 60)

    # Test responses that should be adapted based on learned preferences
    test_cases = [
        {
            "response": "To implement a REST API, you need to understand HTTP methods (GET, POST, PUT, DELETE), status codes (200, 404, 500), request/response formats (JSON, XML), authentication mechanisms (JWT, OAuth), error handling strategies, database integration patterns, caching strategies, rate limiting, documentation generation, testing methodologies, deployment considerations, monitoring and logging, security best practices, and performance optimization techniques.",
            "context": {
                "user_input": "How to build a REST API?",
                "interaction_mode": "text",
                "user_expertise": 0.3,  # Beginner
                "topic_complexity": 0.7,  # Complex topic
                "response_time": 3.0,
            },
            "description": "Long technical response for beginner user",
        },
        {
            "response": "Use indexes.",
            "context": {
                "user_input": "How to optimize database performance?",
                "interaction_mode": "text",
                "user_expertise": 0.8,  # Expert
                "topic_complexity": 0.6,  # Moderate complexity
                "response_time": 1.0,
            },
            "description": "Too brief response for expert user",
        },
        {
            "response": "Here's a basic implementation of user authentication with proper security measures.",
            "context": {
                "user_input": "Implement user login",
                "interaction_mode": "voice",
                "user_expertise": 0.5,  # Intermediate
                "topic_complexity": 0.5,  # Moderate
                "response_time": 2.0,
            },
            "description": "Balanced response",
        },
    ]

    print("Testing response adaptation...")

    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 Test Case {i+1}: {test_case['description']}")
        print(f"Original response length: {len(test_case['response'].split())} words")

        # Apply preference-based adaptation
        pref_adapted, pref_metadata = (
            learning_manager.preference_learner.adapt_response_style(
                test_case["response"]
            )
        )

        # Apply ML-based adaptation
        adapted_response, adaptation_metadata = learning_manager.adapt_response(
            test_case["response"], test_case["context"]
        )

        print(f"Adapted response length: {len(adapted_response.split())} words")

        if pref_metadata:
            print(f"Preference adaptations: {pref_metadata}")

        if adaptation_metadata["ml_adaptations"]:
            print(f"ML adaptations: {adaptation_metadata['ml_adaptations']}")
            print(f"Confidence score: {adaptation_metadata['confidence_score']:.2f}")

        if adaptation_metadata.get("explanation"):
            print(f"Explanation: {adaptation_metadata['explanation']}")

        # Show sample of adapted response
        if len(adapted_response) > 100:
            sample = adapted_response[:100] + "..."
        else:
            sample = adapted_response

        print(f'Sample adapted response: "{sample}"')


def demonstrate_learning_persistence():
    """Demonstrate learning state persistence."""
    print("\n" + "=" * 60)
    print("LEARNING PERSISTENCE DEMONSTRATION")
    print("=" * 60)

    # Create a new learning manager
    learning_manager1 = LearningStateManager()

    # Add some learning data
    print("Adding learning data to first instance...")
    context = InteractionContext(
        user_input="Show me Python best practices",
        response_generated="Here are key Python best practices...",
        timestamp=datetime.now(),
        interaction_mode="text",
        user_feedback="Very helpful, thanks!",
        command_used="explain",
    )

    learning_manager1.record_interaction(context, satisfaction_score=0.9)
    learning_manager1.save_learning_state()
    print("✓ Learning data saved")

    # Get initial statistics
    stats1 = learning_manager1.get_learning_statistics()
    print(
        f"✓ Initial stats - Interactions: {stats1['database_stats']['total_interactions']}"
    )

    # Create second learning manager (should load existing data)
    print("\nCreating second instance (should load existing data)...")
    learning_manager2 = LearningStateManager()

    stats2 = learning_manager2.get_learning_statistics()
    print(
        f"✓ Loaded stats - Interactions: {stats2['database_stats']['total_interactions']}"
    )

    if stats2["database_stats"]["total_interactions"] > 0:
        print("✅ Learning persistence working correctly!")
    else:
        print("⚠️  No data loaded - check database persistence")

    return learning_manager2


def demonstrate_user_profile_generation(learning_manager):
    """Demonstrate comprehensive user profile generation."""
    print("\n" + "=" * 60)
    print("USER PROFILE GENERATION DEMONSTRATION")
    print("=" * 60)

    # Generate comprehensive user profile
    print("Generating user profile based on all learned data...")
    profile = learning_manager.get_user_profile()

    print(f"\n👤 User Profile Summary:")
    print(f"  Profile created: {profile['profile_created']}")
    print(
        f"  Overall confidence: {profile['interaction_stats']['learning_confidence']:.2f}"
    )

    # Preferences
    if profile["preferences"]["high_confidence_preferences"]:
        print(f"\n  🎯 High Confidence Preferences:")
        for pref in profile["preferences"]["high_confidence_preferences"]:
            print(
                f"    • {pref['type']}: {pref['value']} (confidence: {pref['confidence']:.2f})"
            )

    # Behavioral patterns
    if "detected_patterns" in profile["behavioral_patterns"]:
        patterns = profile["behavioral_patterns"]["detected_patterns"][:3]
        print(f"\n  📊 Key Behavioral Patterns:")
        for pattern in patterns:
            print(f"    • {pattern['pattern_type']}: {pattern['description']}")

    # Adaptation performance
    adaptation_perf = profile["adaptation_performance"]["learning_metrics"]
    print(f"\n  ⚡ Adaptation Performance:")
    print(f"    • Success rate: {adaptation_perf.get('success_rate', 0):.1%}")
    print(f"    • Total adaptations: {adaptation_perf.get('total_adaptations', 0)}")

    # Recommendations
    if "recommendations" in profile["adaptation_performance"]:
        recommendations = profile["adaptation_performance"]["recommendations"][:3]
        if recommendations:
            print(f"\n  💡 Recommendations:")
            for rec in recommendations:
                print(f"    • {rec}")

    return profile


def demonstrate_learning_export_import():
    """Demonstrate learning data export/import functionality."""
    print("\n" + "=" * 60)
    print("LEARNING EXPORT/IMPORT DEMONSTRATION")
    print("=" * 60)

    # Create learning manager with data
    learning_manager = LearningStateManager()

    # Add some test data
    context = InteractionContext(
        user_input="Test export functionality",
        response_generated="This is a test response for export",
        timestamp=datetime.now(),
        interaction_mode="text",
        user_feedback="Good test!",
        command_used="test",
    )

    learning_manager.record_interaction(context, satisfaction_score=0.8)

    # Export learning data
    print("Exporting learning data...")
    export_data = learning_manager.export_learning_data()
    print(f"✓ Exported data size: {len(json.dumps(export_data))} characters")

    # Reset learning system
    print("Resetting learning system...")
    learning_manager.reset_all_learning()

    stats_after_reset = learning_manager.get_learning_statistics()
    print(
        f"✓ After reset - Preferences: {stats_after_reset['database_stats']['stored_preferences']}"
    )

    # Import learning data
    print("Importing learning data...")
    success = learning_manager.import_learning_data(export_data)

    if success:
        print("✅ Import successful!")
        stats_after_import = learning_manager.get_learning_statistics()
        print(
            f"✓ After import - Overall confidence: {stats_after_import['overall_confidence']:.2f}"
        )
    else:
        print("❌ Import failed!")


def main():
    """Run the complete learning system demonstration."""
    print("🧠 GIANNA LEARNING SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This demo shows all features of Gianna's learning and adaptation system.")
    print()

    try:
        # Basic learning
        learning_manager = demonstrate_basic_learning()

        # Pattern analysis
        demonstrate_pattern_analysis(learning_manager)

        # Response adaptation
        demonstrate_response_adaptation(learning_manager)

        # Persistence
        learning_manager = demonstrate_learning_persistence()

        # User profile
        demonstrate_user_profile_generation(learning_manager)

        # Export/Import
        demonstrate_learning_export_import()

        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The learning system has successfully demonstrated:")
        print("• User preference learning and adaptation")
        print("• Behavioral pattern analysis and detection")
        print("• Intelligent response adaptation")
        print("• Persistent learning state management")
        print("• Comprehensive user profiling")
        print("• Data export and import capabilities")
        print("\nThe learning system is ready for production use!")

    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
