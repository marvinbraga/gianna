"""
Gianna Testing Framework

This package contains comprehensive tests for all Gianna components, organized by
test type and complexity level.

Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions
- performance/: Performance and load tests
- fixtures/: Shared test fixtures and utilities
- data/: Test data files

Test Categories:
- FASE 1: Core state management and LangGraph chains
- FASE 2: ReAct agents, tools, and orchestration
- FASE 3: VAD, streaming pipeline, voice workflows
- FASE 4: Semantic memory, learning, and optimization
- FASE 5: End-to-end integration and production readiness

Usage:
    # Run all tests
    pytest

    # Run specific test category
    pytest -m unit
    pytest -m integration
    pytest -m performance

    # Run tests for specific phase
    pytest tests/unit/test_core.py
    pytest tests/integration/test_voice_workflows.py

    # Run with coverage
    pytest --cov=gianna --cov-report=html
"""

__version__ = "1.0.0"
__author__ = "Gianna Testing Team"

# Test categories and markers
TEST_CATEGORIES = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interaction",
    "performance": "Performance and load testing",
    "voice": "Voice processing and audio tests",
    "llm": "Large Language Model integration tests",
    "agents": "ReAct agents and coordination tests",
    "memory": "Semantic memory and learning tests",
    "optimization": "Performance optimization tests",
    "end_to_end": "Complete workflow tests",
    "slow": "Tests that take more than 5 seconds",
    "external_api": "Tests requiring external API access",
    "database": "Tests requiring database operations",
    "audio_hardware": "Tests requiring audio hardware",
    "mock_only": "Tests using mocks for all external dependencies",
}

# Test phases based on project phases
TEST_PHASES = {
    "fase1": "Core state management, LangGraph chains",
    "fase2": "ReAct agents, tools, orchestrator",
    "fase3": "VAD, streaming pipeline, voice workflows",
    "fase4": "Semantic memory, learning, optimization",
    "fase5": "End-to-end integration, production readiness",
}

# Performance benchmarks and thresholds
PERFORMANCE_THRESHOLDS = {
    "response_time": {
        "simple_command": 2.0,  # seconds
        "complex_workflow": 10.0,  # seconds
        "llm_response": 5.0,  # seconds
    },
    "memory_usage": {
        "normal_operation": 500,  # MB
        "peak_usage": 1000,  # MB
    },
    "throughput": {
        "interactions_per_minute": 10,
        "concurrent_sessions": 5,
    },
    "uptime": 0.99,  # 99% uptime during load tests
    "cache_hit_rate": 0.7,  # 70% cache hit rate
}

# Test data locations
TEST_DATA_DIR = "tests/data"
AUDIO_TEST_FILES = f"{TEST_DATA_DIR}/audio"
CONFIG_TEST_FILES = f"{TEST_DATA_DIR}/config"
MOCK_RESPONSES_DIR = f"{TEST_DATA_DIR}/mock_responses"
