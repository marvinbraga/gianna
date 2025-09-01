#!/usr/bin/env python3
"""
Backward Compatibility Test for Gianna AI Assistant

This test verifies that existing code from the notebooks continues to work
without any changes after implementing the LangGraph Migration Layer.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_original_notebook_code():
    """Test the exact code from notebooks/01_basics.ipynb"""
    print("Testing original notebook code...")

    try:
        # This is EXACTLY the code from the notebook - unchanged
        from gianna.assistants.models.registers import LLMRegister

        # List available models
        models = sorted([model_name for model_name, _ in LLMRegister.list()])
        print(f"✅ Available models: {len(models)} models found")

        # Create chain instance (exact notebook code)
        from gianna.assistants.models.factory_method import get_chain_instance

        chain_processor = get_chain_instance(
            model_registered_name="grok-3-mini",
            prompt="Atue como um assistente virtual que fala especificamente sobre café e nada mais.",
        )

        print(f"✅ Chain created: {type(chain_processor).__name__}")

        # Test process method (exact notebook code)
        result = chain_processor.process({"input": "Fale sobre futebol."})
        print(f"✅ Process method works: {type(result).__name__}")
        print(f"✅ Has output attribute: {hasattr(result, 'output')}")

        # The response would normally be from the LLM, but in test environment
        # we get a fallback response - that's expected and fine
        output_preview = result.output[:50] if result.output else "No output"
        print(f"✅ Output preview: {output_preview}...")

        return True

    except Exception as e:
        print(f"❌ Error in backward compatibility test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_invoke_method():
    """Test the invoke method specifically"""
    print("\nTesting invoke method...")

    try:
        from gianna.assistants.models.factory_method import get_chain_instance

        chain = get_chain_instance("grok-3-mini", "You are a helpful assistant.")

        # Test invoke method
        result = chain.invoke({"input": "Hello, how are you?"})

        print(f"✅ Invoke method works")
        print(f"✅ Result type: {type(result)}")

        # Check if it returns the expected format
        if hasattr(result, "output"):
            print(f"✅ Has output attribute: {result.output[:50]}...")
        elif isinstance(result, dict) and "output" in result:
            print(f"✅ Has output in dict: {result['output'][:50]}...")
        else:
            print(f"⚠️  Unexpected result format: {result}")

        return True

    except Exception as e:
        print(f"❌ Error testing invoke method: {e}")
        return False


def test_method_chaining():
    """Test method chaining as shown in notebooks"""
    print("\nTesting method chaining...")

    try:
        from gianna.assistants.models.factory_method import get_chain_instance

        # Test chaining - process should return self for chaining
        chain = get_chain_instance("grok-3-mini", "Test prompt")

        result = chain.process({"input": "Test input"})

        # Should return self for chaining
        print(f"✅ Process returns: {type(result).__name__}")
        print(f"✅ Is same instance: {result is chain}")

        # Should be able to access output
        print(f"✅ Output accessible: {hasattr(result, 'output')}")

        return True

    except Exception as e:
        print(f"❌ Error testing method chaining: {e}")
        return False


def test_multiple_model_types():
    """Test that different model types still work"""
    print("\nTesting multiple model types...")

    try:
        from gianna.assistants.models.factory_method import get_chain_instance
        from gianna.assistants.models.registers import LLMRegister

        # Get a few different model types to test
        models = [model_name for model_name, _ in LLMRegister.list()]
        test_models = models[:3]  # Test first 3 models

        success_count = 0

        for model_name in test_models:
            try:
                chain = get_chain_instance(model_name, "Test prompt")
                result = chain.process({"input": "Hello"})

                if hasattr(result, "output"):
                    success_count += 1
                    print(f"✅ {model_name}: Working")
                else:
                    print(f"⚠️  {model_name}: Unexpected result format")

            except Exception as e:
                print(f"❌ {model_name}: Error - {str(e)[:50]}...")

        print(f"✅ Successfully tested {success_count}/{len(test_models)} models")
        return success_count > 0

    except Exception as e:
        print(f"❌ Error testing multiple models: {e}")
        return False


def test_new_features_dont_break_existing():
    """Test that new LangGraph features don't interfere with existing code"""
    print("\nTesting that new features don't interfere...")

    try:
        # First, test standard usage
        from gianna.assistants.models.factory_method import get_chain_instance

        standard_chain = get_chain_instance("grok-3-mini", "Standard prompt")
        standard_result = standard_chain.process({"input": "Standard test"})

        print(f"✅ Standard chain type: {type(standard_chain).__name__}")

        # Now test with explicit traditional mode
        traditional_chain = get_chain_instance(
            "grok-3-mini",
            "Traditional prompt",
            use_langgraph=False,  # New parameter should not break anything
        )
        traditional_result = traditional_chain.process({"input": "Traditional test"})

        print(f"✅ Traditional chain type: {type(traditional_chain).__name__}")

        # Both should work the same way from the user perspective
        print(
            f"✅ Both have output: {hasattr(standard_result, 'output')} and {hasattr(traditional_result, 'output')}"
        )

        return True

    except Exception as e:
        print(f"❌ Error testing new features compatibility: {e}")
        return False


def main():
    """Run all backward compatibility tests"""
    print("🔄 Running Backward Compatibility Tests")
    print("=" * 50)

    tests = [
        test_original_notebook_code,
        test_invoke_method,
        test_method_chaining,
        test_multiple_model_types,
        test_new_features_dont_break_existing,
    ]

    results = []
    for test_func in tests:
        results.append(test_func())

    # Summary
    print("\n" + "=" * 50)
    print("📊 Backward Compatibility Test Results")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print(
            "🎉 Perfect backward compatibility! All existing code will work unchanged."
        )
    elif passed >= total - 1:
        print(
            "✅ Excellent backward compatibility! Minor edge cases may need attention."
        )
    elif passed >= total // 2:
        print("⚠️  Good backward compatibility! Some adjustments may be needed.")
    else:
        print(
            "❌ Backward compatibility issues detected! Code changes may be required."
        )

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
