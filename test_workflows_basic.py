#!/usr/bin/env python3
"""
Basic test script for Gianna workflow templates

This script performs basic functionality tests to verify that the workflow
templates can be imported, created, and executed without errors.
"""

import sys
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all workflow components can be imported."""
    print("Testing imports...")

    try:
        from gianna.workflows import (
            WorkflowBuilder,
            WorkflowConfig,
            WorkflowError,
            WorkflowStateError,
            create_command_workflow,
            create_conversation_workflow,
            create_voice_workflow,
        )

        print("✅ All workflow templates imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        traceback.print_exc()
        return False


def test_state_imports():
    """Test that state management can be imported."""
    print("Testing state imports...")

    try:
        from gianna.core.state import GiannaState, create_initial_state
        from gianna.core.state_manager import StateManager

        print("✅ State management imported successfully")
        return True
    except ImportError as e:
        print(f"❌ State import failed: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error during state import: {e}")
        traceback.print_exc()
        return False


def test_workflow_creation():
    """Test that workflows can be created."""
    print("Testing workflow creation...")

    try:
        from gianna.workflows import create_conversation_workflow

        # Test basic creation without LangGraph (should handle gracefully)
        workflow = create_conversation_workflow(
            name="test_workflow",
            enable_state_management=False,  # Disable to avoid DB dependencies
        )

        print("✅ Workflow creation successful")
        return True
    except Exception as e:
        print(f"❌ Workflow creation failed: {e}")
        traceback.print_exc()
        return False


def test_state_creation():
    """Test that state objects can be created."""
    print("Testing state creation...")

    try:
        from uuid import uuid4

        from gianna.core.state import create_initial_state

        # Create initial state
        session_id = str(uuid4())
        state = create_initial_state(session_id)

        # Verify state structure
        assert "conversation" in state
        assert "audio" in state
        assert "commands" in state
        assert "metadata" in state

        # Test adding a message
        state["conversation"].messages.append(
            {"role": "user", "content": "Test message", "timestamp": str(uuid4())}
        )

        print(f"✅ State creation successful (session: {session_id})")
        return True
    except Exception as e:
        print(f"❌ State creation failed: {e}")
        traceback.print_exc()
        return False


def test_workflow_config():
    """Test workflow configuration."""
    print("Testing workflow configuration...")

    try:
        from gianna.workflows.templates import WorkflowConfig, WorkflowType

        # Create workflow config
        config = WorkflowConfig(
            workflow_type=WorkflowType.CONVERSATION,
            name="test_config",
            description="Test configuration",
            enable_state_management=False,
            enable_error_recovery=True,
        )

        # Verify config
        assert config.workflow_type == WorkflowType.CONVERSATION
        assert config.name == "test_config"
        assert config.enable_error_recovery is True

        print("✅ Workflow configuration successful")
        return True
    except Exception as e:
        print(f"❌ Workflow configuration failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all basic tests."""
    print("=" * 50)
    print("GIANNA WORKFLOW TEMPLATES - BASIC TESTS")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("State Import Test", test_state_imports),
        ("State Creation Test", test_state_creation),
        ("Workflow Config Test", test_workflow_config),
        ("Workflow Creation Test", test_workflow_creation),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
        print()

    # Summary
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    if passed == total:
        print("\n🎉 All basic tests passed! Workflow templates are ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
