"""
Unit Tests for Gianna Components

This package contains unit tests for individual Gianna components, organized by
module and functionality. Unit tests focus on isolated component behavior with
minimal dependencies.

Modules:
- test_core: Core state management, LangGraph integration
- test_models: LLM providers and chain implementations
- test_audio: Audio processing, TTS, STT, VAD
- test_tools: Individual tool functionality
- test_agents: ReAct agent behavior
- test_memory: Semantic memory and learning systems
- test_workflows: Workflow template functionality
- test_coordination: Orchestrator and routing
- test_optimization: Performance and caching

Testing Philosophy:
- Test one thing at a time
- Minimal external dependencies
- Fast execution (< 1 second per test)
- High code coverage (>90% for individual components)
- Comprehensive edge case coverage
"""

# Test categories for unit tests
UNIT_TEST_CATEGORIES = {
    "core": "State management, LangGraph chains",
    "models": "LLM providers and integrations",
    "audio": "Audio processing pipeline",
    "tools": "Individual tool functionality",
    "agents": "ReAct agent implementations",
    "memory": "Memory and learning systems",
    "workflows": "Workflow templates",
    "coordination": "Agent orchestration",
    "optimization": "Performance optimizations",
}

# Unit test performance requirements
UNIT_TEST_PERFORMANCE = {
    "max_execution_time": 1.0,  # seconds per test
    "memory_limit": 50,  # MB per test
    "coverage_threshold": 0.90,  # 90% code coverage
}
