"""
Test Fixtures and Utilities for Gianna Testing Framework

This package contains shared test fixtures, mock utilities, and testing
data that can be reused across different test modules and categories.

Modules:
- mock_data: Mock data generators and sample data
- audio_fixtures: Audio-related test fixtures and sample files
- state_fixtures: State management fixtures and helpers
- llm_fixtures: LLM and chain testing fixtures
- performance_fixtures: Performance testing utilities
- integration_fixtures: Integration test helpers

Utilities:
- Test data generation
- Mock object creation
- Performance measurement
- Error simulation
- Resource management
"""

# Test fixture categories
FIXTURE_CATEGORIES = {
    "mock_data": "Mock data generators and sample data",
    "audio_fixtures": "Audio-related test fixtures",
    "state_fixtures": "State management test fixtures",
    "llm_fixtures": "LLM and chain testing fixtures",
    "performance_fixtures": "Performance testing utilities",
    "integration_fixtures": "Integration test helpers",
}

# Common test data constants
TEST_DATA_CONSTANTS = {
    "sample_messages": [
        {"role": "user", "content": "Hello", "timestamp": "2024-01-01T10:00:00"},
        {
            "role": "assistant",
            "content": "Hi there! How can I help?",
            "timestamp": "2024-01-01T10:00:01",
        },
        {
            "role": "user",
            "content": "What's the weather?",
            "timestamp": "2024-01-01T10:00:02",
        },
        {
            "role": "assistant",
            "content": "I'd need your location to check weather.",
            "timestamp": "2024-01-01T10:00:03",
        },
    ],
    "sample_commands": [
        {"command": "ls -la", "result": "file listing", "success": True},
        {"command": "pwd", "result": "/home/user", "success": True},
        {"command": "echo test", "result": "test", "success": True},
        {"command": "invalid_command", "result": "command not found", "success": False},
    ],
    "sample_audio_texts": [
        "Hello, this is a test audio message",
        "Please process this voice command",
        "Convert this text to speech",
        "Transcribe this audio file",
    ],
    "sample_llm_responses": [
        {"output": "I understand your request and will help you."},
        {"output": "Here's the information you asked for."},
        {"output": "I've completed that task successfully."},
        {"output": "Let me help you with that question."},
    ],
}

# Performance test data
PERFORMANCE_TEST_DATA = {
    "response_times": {
        "fast": [0.1, 0.15, 0.12, 0.08, 0.11],  # Fast responses
        "normal": [0.5, 0.8, 0.6, 0.7, 0.9],  # Normal responses
        "slow": [2.0, 2.5, 1.8, 2.2, 2.1],  # Slow responses
        "timeout": [5.0, 6.0, 5.5, 5.8, 5.2],  # Near-timeout responses
    },
    "memory_usage": {
        "baseline": 100.0,  # MB
        "moderate": 250.0,  # MB
        "high": 450.0,  # MB
        "peak": 800.0,  # MB
    },
    "error_rates": {
        "excellent": 0.001,  # 0.1% error rate
        "good": 0.01,  # 1% error rate
        "acceptable": 0.05,  # 5% error rate
        "poor": 0.10,  # 10% error rate
    },
}

# Mock configuration templates
MOCK_CONFIG_TEMPLATES = {
    "llm_config": {
        "provider": "test_provider",
        "model": "test_model",
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout": 30.0,
    },
    "audio_config": {
        "tts_engine": "test_tts",
        "stt_engine": "test_stt",
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
    },
    "state_config": {
        "db_path": ":memory:",
        "session_timeout": 3600,
        "max_messages": 1000,
        "enable_persistence": False,
    },
}
