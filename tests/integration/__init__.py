"""
Integration Tests for Gianna Components

This package contains integration tests that verify component interactions and
multi-system workflows. Integration tests focus on how components work together
rather than individual component functionality.

Modules:
- test_voice_workflows: Complete voice processing pipelines
- test_agent_coordination: Multi-agent orchestration
- test_memory_integration: Memory system integration
- test_llm_integration: LLM provider integration
- test_audio_workflows: Audio processing workflows
- test_tool_integration: Tool usage and coordination
- test_state_workflows: State management integration
- test_streaming_integration: Real-time streaming workflows

Testing Philosophy:
- Test component interactions
- Verify end-to-end workflows
- Test realistic usage scenarios
- Moderate execution time (< 10 seconds per test)
- Focus on integration points and data flow
- Test error handling across components
"""

# Integration test categories
INTEGRATION_TEST_CATEGORIES = {
    "voice_workflows": "Voice processing pipeline integration",
    "agent_coordination": "Multi-agent orchestration testing",
    "memory_integration": "Memory system integration workflows",
    "llm_integration": "LLM provider integration testing",
    "audio_workflows": "Audio processing workflow testing",
    "tool_integration": "Tool usage and coordination",
    "state_workflows": "State management workflow integration",
    "streaming_integration": "Real-time streaming integration",
}

# Integration test performance requirements
INTEGRATION_TEST_PERFORMANCE = {
    "max_execution_time": 10.0,  # seconds per test
    "memory_limit": 200,  # MB per test
    "coverage_threshold": 0.70,  # 70% integration coverage
    "component_interaction_coverage": 0.80,  # 80% interaction coverage
}

# Common integration test scenarios
COMMON_SCENARIOS = {
    "simple_conversation": "Basic text-based conversation workflow",
    "voice_interaction": "Complete voice input/output workflow",
    "command_execution": "Command generation and execution workflow",
    "multi_modal": "Mixed text/voice interaction workflow",
    "memory_enabled": "Conversation with memory/learning enabled",
    "error_recovery": "Workflow error handling and recovery",
    "concurrent_sessions": "Multiple concurrent session handling",
    "streaming_audio": "Real-time audio streaming workflow",
}
