#!/usr/bin/env python3
"""
Simple Learning System Example

A straightforward example of how to use the Gianna learning system.
"""

import os
import sys
import tempfile
from datetime import datetime
from unittest.mock import Mock

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gianna.learning import (
    AdaptationEngine,
    InteractionContext,
    LearningStateManager,
    UserPreferenceLearner,
)


def simple_example():
    """Run a simple learning example."""
    print("🧠 Simple Gianna Learning System Example")
    print("=" * 50)

    # Setup with mock state manager (for demo purposes)
    temp_dir = tempfile.mkdtemp()
    mock_state = Mock()
    mock_state.db_path = os.path.join(temp_dir, "learning_demo.db")

    # Initialize learning manager
    learning_manager = LearningStateManager(mock_state)
    print("✓ Learning system initialized")

    # Simulate some user interactions
    interactions = [
        {
            "input": "How do I create a Python function?",
            "response": 'To create a Python function, you use the "def" keyword followed by the function name and parameters. Here\'s the basic syntax:\n\ndef function_name(parameters):\n    # function body\n    return value\n\nFor example:\n\ndef greet(name):\n    return f"Hello, {name}!"\n\nThis creates a simple greeting function that takes a name parameter and returns a greeting message.',
            "feedback": "Too detailed, can you be more brief?",
            "mode": "text",
        },
        {
            "input": "Show me a loop example",
            "response": "for i in range(5):\n    print(i)",
            "feedback": "Perfect, short and clear!",
            "mode": "text",
        },
        {
            "input": "What is a dictionary in Python?",
            "response": 'A dictionary is a collection of key-value pairs. Example: my_dict = {"key": "value"}',
            "feedback": "Great explanation!",
            "mode": "text",
        },
    ]

    print("\n📝 Recording user interactions...")

    for i, interaction in enumerate(interactions, 1):
        # Create interaction context
        context = InteractionContext(
            user_input=interaction["input"],
            response_generated=interaction["response"],
            timestamp=datetime.now(),
            interaction_mode=interaction["mode"],
            user_feedback=interaction["feedback"],
            response_time=1.5 + i * 0.3,
        )

        # Determine satisfaction from feedback
        feedback_lower = interaction["feedback"].lower()
        if "perfect" in feedback_lower or "great" in feedback_lower:
            satisfaction = 0.9
        elif "too" in feedback_lower or "detailed" in feedback_lower:
            satisfaction = 0.5
        else:
            satisfaction = 0.7

        # Record the interaction
        learning_manager.record_interaction(context, satisfaction_score=satisfaction)
        print(f"  ✓ Recorded interaction {i}: {interaction['input'][:40]}...")

    print(f"\n📊 Learning Summary:")

    # Check learned preferences
    preferences = learning_manager.preference_learner.get_preference_summary()
    print(
        f"  - Total interactions: {preferences['learning_stats']['total_interactions']}"
    )
    print(f"  - Preferences learned: {preferences['preferences_count']}")

    if preferences["high_confidence_preferences"]:
        print("  - High confidence preferences:")
        for pref in preferences["high_confidence_preferences"]:
            print(
                f"    • {pref['type']}: {pref['value']} (confidence: {pref['confidence']:.2f})"
            )

    print("\n🔧 Testing Response Adaptation:")

    # Test response adaptation with a long response
    original_response = """
    Python classes are object-oriented programming constructs that allow you to create
    custom data types. A class defines a blueprint for creating objects (instances) that
    share common attributes and methods. Here's how to define a class:

    class MyClass:
        def __init__(self, attribute):
            self.attribute = attribute

        def my_method(self):
            return f"Attribute value: {self.attribute}"

    You can then create instances: obj = MyClass("value")
    And call methods: result = obj.my_method()

    Classes support inheritance, encapsulation, and polymorphism, which are the core
    principles of object-oriented programming.
    """.strip()

    print(f"  Original response length: {len(original_response.split())} words")

    # Adapt the response based on learned preferences
    adapted_response, metadata = learning_manager.adapt_response(
        original_response,
        {
            "user_input": "What are Python classes?",
            "interaction_mode": "text",
            "user_expertise": 0.3,  # Beginner level
            "topic_complexity": 0.7,
        },
    )

    print(f"  Adapted response length: {len(adapted_response.split())} words")

    if metadata["preference_adaptations"]:
        print(f"  Preference adaptations: {metadata['preference_adaptations']}")

    if metadata["ml_adaptations"]:
        print(f"  ML adaptations: {metadata['ml_adaptations']}")

    print(f"  Confidence score: {metadata['confidence_score']:.2f}")

    # Show a sample of the adapted response
    sample = (
        adapted_response[:100] + "..."
        if len(adapted_response) > 100
        else adapted_response
    )
    print(f'  Adapted response sample: "{sample}"')

    print("\n📈 System Statistics:")
    stats = learning_manager.get_learning_statistics()
    print(f"  - System health: {stats['system_health']}")
    print(f"  - Overall confidence: {stats['overall_confidence']:.2f}")
    print(f"  - Database interactions: {stats['database_stats']['total_interactions']}")

    print("\n✅ Learning example completed successfully!")
    print("\nThe system learned that the user prefers brief responses and")
    print("will automatically adapt future responses to be more concise.")

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


def demonstrate_individual_components():
    """Demonstrate individual learning components."""
    print("\n" + "=" * 50)
    print("INDIVIDUAL COMPONENTS DEMONSTRATION")
    print("=" * 50)

    # 1. User Preference Learner
    print("\n1. 🎯 User Preference Learner")
    learner = UserPreferenceLearner()

    context = InteractionContext(
        user_input="Explain Python decorators",
        response_generated="Very long explanation...",
        timestamp=datetime.now(),
        interaction_mode="text",
        user_feedback="Too verbose, please be brief",
    )

    learner.record_interaction(context)
    print("   ✓ Recorded interaction with brevity preference")

    # Test adaptation
    long_text = (
        "This is a very long explanation with lots of details that could be shortened."
    )
    adapted, metadata = learner.adapt_response_style(long_text)
    print(
        f"   ✓ Adapted response: {len(adapted.split())} words (was {len(long_text.split())} words)"
    )

    # 2. Adaptation Engine
    print("\n2. ⚡ Adaptation Engine")
    engine = AdaptationEngine()

    context_dict = {
        "user_input": "Test question",
        "response": "Test response",
        "timestamp": datetime.now(),
        "interaction_mode": "text",
    }

    engine.learn_from_interaction(context_dict, user_satisfaction=0.8)
    print("   ✓ Learned from interaction")

    result = engine.adapt_response("Test response to adapt", context_dict)
    print(
        f"   ✓ Adaptation result: success={result.success}, confidence={result.confidence_score:.2f}"
    )

    print("\n🎉 Individual components working correctly!")


if __name__ == "__main__":
    try:
        simple_example()
        demonstrate_individual_components()
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
