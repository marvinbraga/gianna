#!/usr/bin/env python3
"""
LangGraph Migration Layer Demo for Gianna AI Assistant

This example demonstrates the LangGraph Migration Layer functionality,
showing backward compatibility and new features.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def demo_backward_compatibility():
    """Demonstrate that existing code continues to work unchanged."""
    print("=== Backward Compatibility Demo ===")

    try:
        from gianna.assistants.models.factory_method import get_chain_instance

        # This is the EXACT same code from the notebook
        chain_processor = get_chain_instance(
            model_registered_name="grok-3-mini",
            prompt="Atue como um assistente virtual que fala especificamente sobre café e nada mais.",
        )

        # This should work exactly the same as before
        result = chain_processor.process({"input": "Fale sobre futebol."})
        print(f"Traditional chain result: {result.output[:100]}...")

        # Test invoke method as well
        invoke_result = chain_processor.invoke(
            {"input": "Como fazer um café expresso?"}
        )
        if hasattr(invoke_result, "output"):
            print(f"Invoke method result: {invoke_result.output[:100]}...")
        else:
            print(
                f"Invoke method result: {invoke_result.get('output', str(invoke_result))[:100]}..."
            )

        print("✅ Backward compatibility verified!")
        return True

    except Exception as e:
        print(f"❌ Error in backward compatibility test: {e}")
        return False


def demo_langgraph_features():
    """Demonstrate new LangGraph features if available."""
    print("\n=== LangGraph Features Demo ===")

    try:
        from gianna.core import LANGGRAPH_AVAILABLE, get_enhanced_chain_instance

        if not LANGGRAPH_AVAILABLE:
            print("⚠️  LangGraph not available - install with: pip install langgraph")
            return False

        print("✅ LangGraph is available!")

        # Create a LangGraph-enabled chain
        langgraph_chain = get_enhanced_chain_instance(
            model_registered_name="grok-3-mini",
            prompt="You are a helpful AI assistant. Respond helpfully and concisely.",
            prefer_langgraph=True,
            session_id="demo_session_001",
        )

        print(f"Chain type: {type(langgraph_chain).__name__}")

        # Test with state management
        result1 = langgraph_chain.invoke(
            "Hello, my name is Alice. What can you help me with?",
            session_id="demo_session_001",
        )
        print(f"First interaction: {result1.get('output', 'No output')[:100]}...")

        # Test conversation continuity
        result2 = langgraph_chain.invoke(
            "What's my name?", session_id="demo_session_001"
        )
        print(f"Follow-up question: {result2.get('output', 'No output')[:100]}...")

        # Test conversation history
        if hasattr(langgraph_chain, "get_conversation_history"):
            history = langgraph_chain.get_conversation_history("demo_session_001")
            print(f"Conversation history: {len(history)} messages")
            for i, msg in enumerate(history[-2:]):  # Show last 2 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:50]
                print(f"  {i+1}. {role}: {content}...")

        print("✅ LangGraph features working!")
        return True

    except Exception as e:
        print(f"❌ Error in LangGraph features test: {e}")
        import traceback

        traceback.print_exc()
        return False


def demo_migration_utilities():
    """Demonstrate migration utilities."""
    print("\n=== Migration Utilities Demo ===")

    try:
        from gianna.core import (
            LANGGRAPH_INTEGRATION,
            detect_chain_type,
            get_migration_recommendations,
        )

        if not LANGGRAPH_INTEGRATION:
            print("⚠️  Migration utilities not available")
            return False

        # Test chain type detection
        from gianna.assistants.models.factory_method import get_chain_instance

        traditional_chain = get_chain_instance(
            "grok-3-mini", "Test prompt", use_langgraph=False
        )
        chain_type = detect_chain_type(traditional_chain)
        print(f"Traditional chain type: {chain_type}")

        # Test migration recommendations
        usage_patterns = {
            "maintains_conversation_history": True,
            "multiple_sessions": True,
            "complex_workflows": False,
            "audio_integration": True,
        }

        recommendations = get_migration_recommendations(usage_patterns)
        print(f"Should migrate: {recommendations['should_migrate']}")
        print(f"Migration complexity: {recommendations['migration_complexity']}")
        print(f"Benefits: {len(recommendations['benefits'])} identified")
        print(f"Reasons: {', '.join(recommendations['reasons'][:2])}")

        print("✅ Migration utilities working!")
        return True

    except Exception as e:
        print(f"❌ Error in migration utilities test: {e}")
        import traceback

        traceback.print_exc()
        return False


def demo_explicit_langgraph_usage():
    """Demonstrate explicit LangGraph usage."""
    print("\n=== Explicit LangGraph Usage Demo ===")

    try:
        from gianna.assistants.models.factory_method import create_langgraph_chain
        from gianna.core import LANGGRAPH_AVAILABLE

        if not LANGGRAPH_AVAILABLE:
            print("⚠️  LangGraph not available for explicit usage")
            return False

        # Explicitly create a LangGraph chain
        langgraph_chain = create_langgraph_chain(
            model_registered_name="grok-3-mini",
            prompt="You are a knowledgeable AI assistant specializing in technology.",
            session_id="tech_session_001",
        )

        print(f"Explicit LangGraph chain type: {type(langgraph_chain).__name__}")
        print(
            f"Has state management: {getattr(langgraph_chain, 'has_state_management', False)}"
        )

        # Test with technical questions
        result = langgraph_chain.invoke(
            "Explain what LangGraph is and how it differs from traditional chains.",
            session_id="tech_session_001",
        )

        print(f"Technical explanation: {result.get('output', 'No output')[:150]}...")

        print("✅ Explicit LangGraph usage working!")
        return True

    except Exception as e:
        print(f"❌ Error in explicit LangGraph usage: {e}")
        import traceback

        traceback.print_exc()
        return False


def demo_compatibility_wrapper():
    """Demonstrate backward compatibility wrapper."""
    print("\n=== Compatibility Wrapper Demo ===")

    try:
        from gianna.core import LANGGRAPH_INTEGRATION, create_compatible_chain

        if not LANGGRAPH_INTEGRATION:
            print("⚠️  Compatibility wrapper not available")
            return False

        # Create a compatible chain (wrapped for backward compatibility)
        compatible_chain = create_compatible_chain(
            model_name="grok-3-mini",
            prompt="You are a helpful assistant.",
            backward_compatible=True,
        )

        print(f"Compatible chain type: {type(compatible_chain).__name__}")
        print(f"Is LangGraph: {getattr(compatible_chain, 'is_langgraph', False)}")

        # Test both invoke and process methods
        invoke_result = compatible_chain.invoke("What is artificial intelligence?")
        print(f"Invoke result type: {type(invoke_result)}")

        process_result = compatible_chain.process("Explain machine learning briefly.")
        print(f"Process result type: {type(process_result)}")
        print(f"Process output: {process_result.output[:100]}...")

        print("✅ Compatibility wrapper working!")
        return True

    except Exception as e:
        print(f"❌ Error in compatibility wrapper test: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_feature_availability():
    """Check which features are available."""
    print("=== Feature Availability Check ===")

    try:
        from gianna.assistants.models.factory_method import get_available_features
        from gianna.core import LANGGRAPH_AVAILABLE, LANGGRAPH_INTEGRATION

        print(f"LangGraph Available: {LANGGRAPH_AVAILABLE}")
        print(f"LangGraph Integration: {LANGGRAPH_INTEGRATION}")

        features = get_available_features()
        print("Available features:")
        for feature, available in features.items():
            status = "✅" if available else "❌"
            print(f"  {status} {feature}")

        if LANGGRAPH_INTEGRATION:
            from gianna.core import get_langgraph_capabilities

            capabilities = get_langgraph_capabilities()
            print(
                f"\nLangGraph models available: {len(capabilities.get('models', []))}"
            )

        return True

    except Exception as e:
        print(f"❌ Error checking features: {e}")
        return False


def main():
    """Run all demos."""
    print("🚀 Gianna LangGraph Migration Layer Demo")
    print("=" * 50)

    results = []

    # Check feature availability first
    results.append(check_feature_availability())

    # Run backward compatibility test (most important)
    results.append(demo_backward_compatibility())

    # Run LangGraph-specific tests if available
    results.append(demo_langgraph_features())
    results.append(demo_migration_utilities())
    results.append(demo_explicit_langgraph_usage())
    results.append(demo_compatibility_wrapper())

    # Summary
    print("\n" + "=" * 50)
    print("📊 Demo Results Summary")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! LangGraph Migration Layer is working correctly.")
    elif results[1]:  # Backward compatibility is working
        print(
            "✅ Backward compatibility confirmed - existing code will continue to work."
        )
        print(
            "⚠️  Some LangGraph features may not be available (likely missing dependencies)."
        )
    else:
        print("❌ Critical issues found - backward compatibility may be broken.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
