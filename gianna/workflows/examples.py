"""
Usage Examples for Gianna Workflow Templates

This module provides practical examples of how to use the workflow templates
for different use cases. Each example demonstrates proper setup, configuration,
and execution patterns.
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from ..core.state import GiannaState, create_initial_state
from ..core.state_manager import StateManager
from .templates import (
    WorkflowBuilder,
    WorkflowConfig,
    WorkflowType,
    create_command_workflow,
    create_conversation_workflow,
    create_voice_workflow,
)


def example_basic_conversation():
    """
    Example: Basic conversation workflow

    Demonstrates how to create and use a simple conversation workflow
    for text-based interactions.
    """
    print("=== Basic Conversation Workflow Example ===")

    # Create the workflow
    workflow = create_conversation_workflow(
        name="basic_chat", enable_state_management=True, enable_error_recovery=True
    )

    # Create initial state
    session_id = str(uuid4())
    state = create_initial_state(session_id)

    # Add user message
    state["conversation"].messages.append(
        {
            "role": "user",
            "content": "Hello! How are you today?",
            "timestamp": str(uuid4()),
        }
    )

    # Configuration for LangGraph
    config = {"configurable": {"thread_id": session_id}}

    try:
        # Execute workflow
        result = workflow.invoke(state, config)

        # Print results
        print(f"Session ID: {session_id}")
        print(f"Messages count: {len(result['conversation'].messages)}")
        print(f"Last message: {result['conversation'].messages[-1]['content']}")
        print(
            f"Processing stage: {result['metadata'].get('processing_stage', 'unknown')}"
        )

        return result

    except Exception as e:
        print(f"Error in conversation workflow: {e}")
        return None


def example_command_execution():
    """
    Example: Command execution workflow

    Demonstrates how to create and use a command execution workflow
    with safety validation.
    """
    print("=== Command Execution Workflow Example ===")

    # Create the workflow with safety validation
    workflow = create_command_workflow(
        name="safe_command_executor",
        enable_state_management=True,
        enable_error_recovery=True,
        enable_safety_validation=True,
    )

    # Create initial state
    session_id = str(uuid4())
    state = create_initial_state(session_id)

    # Add command message
    state["conversation"].messages.append(
        {
            "role": "user",
            "content": "ls -la /home",  # Safe command
            "timestamp": str(uuid4()),
        }
    )

    # Configuration
    config = {"configurable": {"thread_id": session_id}}

    try:
        # Execute workflow
        result = workflow.invoke(state, config)

        # Print results
        print(f"Session ID: {session_id}")
        print(
            f"Command executed: {result['metadata'].get('parsed_command', {}).get('raw_command', 'N/A')}"
        )
        print(f"Safety check: {result['metadata'].get('safety_check', 'N/A')}")
        print(
            f"Execution result: {result['metadata'].get('execution_result', {}).get('stdout', 'N/A')}"
        )
        print(
            f"Processing stage: {result['metadata'].get('processing_stage', 'unknown')}"
        )

        return result

    except Exception as e:
        print(f"Error in command workflow: {e}")
        return None


def example_unsafe_command():
    """
    Example: Unsafe command handling

    Demonstrates how the command workflow handles unsafe commands.
    """
    print("=== Unsafe Command Handling Example ===")

    workflow = create_command_workflow(
        name="safety_demo", enable_safety_validation=True
    )

    session_id = str(uuid4())
    state = create_initial_state(session_id)

    # Add unsafe command
    state["conversation"].messages.append(
        {
            "role": "user",
            "content": "rm -rf /",  # Dangerous command
            "timestamp": str(uuid4()),
        }
    )

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = workflow.invoke(state, config)

        print(f"Unsafe command: rm -rf /")
        print(f"Safety check: {result['metadata'].get('safety_check', 'N/A')}")
        print(
            f"Command executed: {'No' if result['metadata'].get('safety_check') == 'failed' else 'Yes'}"
        )

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


def example_voice_interaction():
    """
    Example: Complete voice interaction workflow

    Demonstrates the full voice workflow including STT, processing, and TTS.
    """
    print("=== Voice Interaction Workflow Example ===")

    # Create voice workflow
    workflow = create_voice_workflow(
        name="voice_assistant",
        enable_state_management=True,
        enable_error_recovery=True,
        enable_async_processing=False,  # Sync for this example
    )

    # Create initial state with voice configuration
    session_id = str(uuid4())
    state = create_initial_state(session_id)

    # Configure audio settings
    state["audio"].speech_type = "google"
    state["audio"].language = "en-US"
    state["audio"].current_mode = "listening"

    # Voice settings
    state["audio"].voice_settings = {
        "voice_speed": 1.0,
        "voice_pitch": 0.0,
        "voice_volume": 0.8,
    }

    # Configuration
    config = {"configurable": {"thread_id": session_id}}

    try:
        # Execute workflow
        result = workflow.invoke(state, config)

        # Print results
        print(f"Session ID: {session_id}")
        print(f"Audio mode: {result['audio'].current_mode}")
        print(f"STT result: {result['metadata'].get('stt_result', 'N/A')}")
        print(f"Voice response: {result['metadata'].get('voice_response', 'N/A')}")
        print(
            f"TTS metadata: {result['metadata'].get('tts_result', {}).get('text', 'N/A')}"
        )
        print(
            f"Processing stage: {result['metadata'].get('processing_stage', 'unknown')}"
        )

        return result

    except Exception as e:
        print(f"Error in voice workflow: {e}")
        return None


def example_custom_workflow():
    """
    Example: Custom workflow with additional nodes

    Demonstrates how to create a custom workflow by adding custom nodes
    to an existing template.
    """
    print("=== Custom Workflow Example ===")

    def custom_preprocessing_node(state: GiannaState) -> GiannaState:
        """Custom node that preprocesses input"""
        try:
            # Add custom preprocessing
            state["metadata"]["custom_preprocessing"] = "completed"
            state["metadata"]["preprocessing_timestamp"] = str(uuid4())

            # Modify the last message
            if state["conversation"].messages:
                last_message = state["conversation"].messages[-1]
                content = last_message.get("content", "")
                last_message["content"] = f"[Preprocessed] {content}"

            print("Custom preprocessing completed")
            return state
        except Exception as e:
            state["metadata"]["error"] = f"Custom preprocessing failed: {e}"
            return state

    def custom_postprocessing_node(state: GiannaState) -> GiannaState:
        """Custom node that postprocesses output"""
        try:
            # Add custom postprocessing
            state["metadata"]["custom_postprocessing"] = "completed"

            # Modify the assistant response
            assistant_messages = [
                msg
                for msg in state["conversation"].messages
                if msg.get("role") == "assistant"
            ]
            if assistant_messages:
                last_response = assistant_messages[-1]
                content = last_response.get("content", "")
                last_response["content"] = f"{content} [Custom postprocessed]"

            print("Custom postprocessing completed")
            return state
        except Exception as e:
            state["metadata"]["error"] = f"Custom postprocessing failed: {e}"
            return state

    # Create workflow with custom nodes
    custom_nodes = {
        "custom_preprocessing": custom_preprocessing_node,
        "custom_postprocessing": custom_postprocessing_node,
    }

    # Custom edges to integrate the new nodes
    custom_edges = [
        ("custom_preprocessing", "process_conversation"),
        ("format_output", "custom_postprocessing"),
    ]

    # Create workflow configuration
    config = WorkflowConfig(
        workflow_type=WorkflowType.CONVERSATION,
        name="custom_conversation_workflow",
        description="Conversation workflow with custom pre/post processing",
        enable_state_management=True,
        enable_error_recovery=True,
        custom_nodes=custom_nodes,
        custom_edges=custom_edges,
    )

    # Build and compile workflow
    builder = WorkflowBuilder(config)
    workflow = builder.compile()

    # Test the custom workflow
    session_id = str(uuid4())
    state = create_initial_state(session_id)

    state["conversation"].messages.append(
        {"role": "user", "content": "Test custom workflow", "timestamp": str(uuid4())}
    )

    workflow_config = {"configurable": {"thread_id": session_id}}

    try:
        result = workflow.invoke(state, workflow_config)

        print(
            f"Custom preprocessing: {result['metadata'].get('custom_preprocessing', 'N/A')}"
        )
        print(
            f"Custom postprocessing: {result['metadata'].get('custom_postprocessing', 'N/A')}"
        )
        print(f"Final message: {result['conversation'].messages[-1]['content']}")

        return result

    except Exception as e:
        print(f"Error in custom workflow: {e}")
        return None


def example_state_persistence():
    """
    Example: State persistence across workflow executions

    Demonstrates how to use state management for conversation continuity.
    """
    print("=== State Persistence Example ===")

    # Create workflow with state management
    workflow = create_conversation_workflow(
        name="persistent_chat", enable_state_management=True
    )

    # Use consistent session ID
    session_id = "persistent_session_123"

    # First interaction
    print("First interaction:")
    state1 = create_initial_state(session_id)
    state1["conversation"].messages.append(
        {"role": "user", "content": "My name is Alice", "timestamp": str(uuid4())}
    )

    config = {"configurable": {"thread_id": session_id}}
    result1 = workflow.invoke(state1, config)
    print(f"Messages after first: {len(result1['conversation'].messages)}")

    # Second interaction (should load previous state)
    print("\nSecond interaction:")
    state2 = create_initial_state(session_id)
    state2["conversation"].messages.append(
        {"role": "user", "content": "What is my name?", "timestamp": str(uuid4())}
    )

    result2 = workflow.invoke(state2, config)
    print(f"Messages after second: {len(result2['conversation'].messages)}")
    print("Conversation history:")
    for i, msg in enumerate(result2["conversation"].messages):
        print(f"  {i+1}. {msg['role']}: {msg['content']}")

    return result2


def example_error_recovery():
    """
    Example: Error recovery in workflows

    Demonstrates how workflows handle and recover from errors.
    """
    print("=== Error Recovery Example ===")

    def failing_custom_node(state: GiannaState) -> GiannaState:
        """Custom node that intentionally fails"""
        raise Exception("Simulated node failure for testing")

    # Create workflow with error recovery and a failing node
    custom_nodes = {"failing_node": failing_custom_node}
    custom_edges = [
        ("process_conversation", "failing_node"),
        ("failing_node", "generate_response"),
    ]

    workflow = create_conversation_workflow(
        name="error_recovery_test",
        enable_error_recovery=True,
        custom_nodes=custom_nodes,
    )

    session_id = str(uuid4())
    state = create_initial_state(session_id)
    state["conversation"].messages.append(
        {"role": "user", "content": "Test error recovery", "timestamp": str(uuid4())}
    )

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = workflow.invoke(state, config)

        print(f"Workflow completed despite error")
        print(f"Error recovered: {result['metadata'].get('error_handled', False)}")
        print(f"Final stage: {result['metadata'].get('processing_stage', 'unknown')}")

        # Check if error message was added
        error_messages = [
            msg
            for msg in result["conversation"].messages
            if msg.get("metadata", {}).get("error_recovery")
        ]
        print(f"Error recovery messages: {len(error_messages)}")

        return result

    except Exception as e:
        print(f"Workflow failed: {e}")
        return None


def run_all_examples():
    """Run all examples to demonstrate workflow capabilities."""
    print("Running all workflow examples...\n")

    examples = [
        ("Basic Conversation", example_basic_conversation),
        ("Command Execution", example_command_execution),
        ("Unsafe Command Handling", example_unsafe_command),
        ("Voice Interaction", example_voice_interaction),
        ("Custom Workflow", example_custom_workflow),
        ("State Persistence", example_state_persistence),
        ("Error Recovery", example_error_recovery),
    ]

    results = {}

    for name, example_func in examples:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")

        try:
            result = example_func()
            results[name] = result
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = None

        print()  # Add spacing between examples

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")

    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)

    print(f"Successful examples: {successful}/{total}")
    for name, result in results.items():
        status = "✅ PASS" if result is not None else "❌ FAIL"
        print(f"  {name}: {status}")

    return results


if __name__ == "__main__":
    # Run all examples when script is executed directly
    run_all_examples()
