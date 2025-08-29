"""
Global test configuration and fixtures for Gianna testing framework.

This module provides shared fixtures, mock utilities, and test configuration
for all test modules in the Gianna project.
"""

import asyncio
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Core imports
from gianna.core.state import (
    AudioState,
    CommandState,
    ConversationState,
    GiannaState,
    create_initial_state,
)
from gianna.core.state_manager import StateManager

# ============================================================================
# TEST ENVIRONMENT SETUP
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """Setup test environment variables and configuration."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"

    # Mock API keys for testing
    test_api_keys = {
        "OPENAI_API_KEY": "test-openai-key",
        "GOOGLE_API_KEY": "test-google-key",
        "ELEVEN_LABS_API_KEY": "test-elevenlabs-key",
        "GROQ_API_KEY": "test-groq-key",
        "NVIDIA_API_KEY": "test-nvidia-key",
        "COHERE_API_KEY": "test-cohere-key",
        "XAI_API_KEY": "test-xai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
    }

    for key, value in test_api_keys.items():
        if key not in os.environ:
            os.environ[key] = value

    yield

    # Cleanup
    for key in test_api_keys:
        if os.environ.get(key) == test_api_keys[key]:
            del os.environ[key]


# ============================================================================
# DATABASE AND STATE FIXTURES
# ============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def test_session_id():
    """Generate a unique test session ID."""
    return f"test-session-{uuid4()}"


@pytest.fixture
def gianna_state(test_session_id):
    """Create a fresh GiannaState for testing."""
    return create_initial_state(test_session_id)


@pytest.fixture
def state_manager(temp_db):
    """Create a StateManager with temporary database."""
    return StateManager(temp_db)


@pytest.fixture
def conversation_state():
    """Create a ConversationState for testing."""
    return ConversationState(
        messages=[
            {
                "role": "system",
                "content": "You are a test assistant",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "role": "user",
                "content": "Hello",
                "timestamp": datetime.now().isoformat(),
            },
        ],
        session_id="test-session",
        user_preferences={"language": "pt-br", "voice_speed": 1.0},
        context_summary="Test conversation context",
    )


@pytest.fixture
def audio_state():
    """Create an AudioState for testing."""
    return AudioState(
        current_mode="idle",
        voice_settings={"speed": 1.0, "pitch": 1.0},
        speech_type="google",
        language="pt-br",
    )


@pytest.fixture
def command_state():
    """Create a CommandState for testing."""
    return CommandState(
        execution_history=[
            {
                "command": "echo test",
                "result": "test",
                "timestamp": datetime.now().isoformat(),
            }
        ],
        pending_operations=[],
    )


# ============================================================================
# LLM AND CHAIN FIXTURES
# ============================================================================


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = MagicMock()
    llm.invoke.return_value = "Test response from mock LLM"
    llm.ainvoke = AsyncMock(return_value="Test async response from mock LLM")
    llm.model_name = "test-model"
    return llm


@pytest.fixture
def mock_chain():
    """Mock LangChain for testing."""
    chain = MagicMock()
    chain.invoke.return_value = {"output": "Test response from mock chain"}
    chain.ainvoke = AsyncMock(
        return_value={"output": "Test async response from mock chain"}
    )
    return chain


@pytest.fixture
def mock_langgraph_chain():
    """Mock LangGraphChain for testing."""
    with patch("gianna.core.langgraph_chain.LangGraphChain") as mock_class:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = {"output": "Test LangGraph response"}
        mock_instance.ainvoke = AsyncMock(
            return_value={"output": "Test async LangGraph response"}
        )
        mock_class.return_value = mock_instance
        yield mock_instance


# ============================================================================
# AUDIO FIXTURES
# ============================================================================


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    import numpy as np

    # Generate 1 second of sine wave at 16kHz
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio_data


@pytest.fixture
def temp_audio_file(sample_audio_data):
    """Create a temporary audio file for testing."""
    import wave

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_file = f.name

    # Write audio data to file
    with wave.open(audio_file, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        # Convert float32 to int16
        audio_int16 = (sample_audio_data * 32767).astype("int16")
        wav_file.writeframes(audio_int16.tobytes())

    yield audio_file

    # Cleanup
    try:
        os.unlink(audio_file)
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_tts():
    """Mock text-to-speech service."""
    with patch("gianna.assistants.audio.tts.factory_method.text_to_speech") as mock_tts:
        mock_tts.return_value = MagicMock()
        yield mock_tts


@pytest.fixture
def mock_stt():
    """Mock speech-to-text service."""
    with patch("gianna.assistants.audio.stt.factory_method.speech_to_text") as mock_stt:
        mock_stt.return_value = "Test transcribed text"
        yield mock_stt


@pytest.fixture
def mock_vad():
    """Mock Voice Activity Detection."""
    with patch("gianna.audio.streaming.VoiceActivityDetector") as mock_vad_class:
        mock_vad = MagicMock()
        mock_vad.detect_activity.return_value = True
        mock_vad.is_speech_active = False
        mock_vad_class.return_value = mock_vad
        yield mock_vad


# ============================================================================
# AGENT AND TOOL FIXTURES
# ============================================================================


@pytest.fixture
def mock_react_agent():
    """Mock ReAct agent for testing."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.invoke.return_value = {"output": "Agent response"}
    agent.ainvoke = AsyncMock(return_value={"output": "Async agent response"})
    return agent


@pytest.fixture
def mock_shell_tool():
    """Mock shell tool for testing."""
    with patch("gianna.tools.shell_tools.ShellExecutorTool") as mock_tool:
        tool = MagicMock()
        tool.name = "shell_executor"
        tool._run.return_value = (
            '{"exit_code": 0, "stdout": "test output", "stderr": "", "success": true}'
        )
        mock_tool.return_value = tool
        yield tool


@pytest.fixture
def mock_audio_tools():
    """Mock audio tools for testing."""
    with (
        patch("gianna.tools.audio_tools.AudioProcessorTool") as mock_audio,
        patch("gianna.tools.audio_tools.TTSTool") as mock_tts,
        patch("gianna.tools.audio_tools.STTTool") as mock_stt,
    ):

        tools = {
            "audio_processor": MagicMock(),
            "tts_tool": MagicMock(),
            "stt_tool": MagicMock(),
        }

        mock_audio.return_value = tools["audio_processor"]
        mock_tts.return_value = tools["tts_tool"]
        mock_stt.return_value = tools["stt_tool"]

        yield tools


# ============================================================================
# MEMORY AND LEARNING FIXTURES
# ============================================================================


@pytest.fixture
def mock_semantic_memory():
    """Mock semantic memory system."""
    with patch("gianna.memory.semantic_memory.SemanticMemory") as mock_memory:
        memory = MagicMock()
        memory.store_interaction.return_value = None
        memory.search_similar_interactions.return_value = [
            {"content": "Similar interaction", "metadata": {"score": 0.8}}
        ]
        memory.get_context_summary.return_value = "Test context summary"
        mock_memory.return_value = memory
        yield memory


@pytest.fixture
def mock_learning_system():
    """Mock learning and adaptation system."""
    with patch("gianna.learning.user_adaptation.UserPreferenceLearner") as mock_learner:
        learner = MagicMock()
        learner.analyze_user_patterns.return_value = None
        learner.adapt_response_style.return_value = "Adapted response"
        mock_learner.return_value = learner
        yield learner


# ============================================================================
# PERFORMANCE AND MONITORING FIXTURES
# ============================================================================


@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitoring system."""
    with patch("gianna.optimization.monitoring.PerformanceMonitor") as mock_monitor:
        monitor = MagicMock()
        monitor.get_metrics.return_value = {
            "response_time": 0.5,
            "memory_usage": 100.0,
            "cpu_usage": 25.0,
            "cache_hit_rate": 0.8,
        }
        mock_monitor.return_value = monitor
        yield monitor


@pytest.fixture
def benchmark_timer():
    """Timer for performance benchmarking."""

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


# ============================================================================
# WORKFLOW FIXTURES
# ============================================================================


@pytest.fixture
def mock_workflow_config():
    """Mock workflow configuration."""
    from gianna.workflows.templates import WorkflowConfig, WorkflowType

    return WorkflowConfig(
        workflow_type=WorkflowType.CONVERSATION,
        name="test_workflow",
        description="Test workflow configuration",
        enable_state_management=True,
        enable_error_recovery=True,
        timeout=30.0,
    )


@pytest.fixture
def mock_orchestrator():
    """Mock agent orchestrator."""
    with patch("gianna.coordination.orchestrator.AgentOrchestrator") as mock_orch:
        orchestrator = MagicMock()
        orchestrator.register_agent.return_value = None
        orchestrator.route_request.return_value = "conversation_agent"
        orchestrator.coordinate_parallel_execution.return_value = None
        mock_orch.return_value = orchestrator
        yield orchestrator


# ============================================================================
# UTILITY FIXTURES
# ============================================================================


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        temp_path = f.name
        f.write("Test file content")

    yield temp_path

    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def async_test_runner():
    """Utility for running async tests."""

    def run_async(coro):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    return run_async


# ============================================================================
# PARAMETRIZED FIXTURES
# ============================================================================


@pytest.fixture(params=["gpt35", "gpt4", "claude", "gemini"])
def llm_model_name(request):
    """Parametrized fixture for different LLM models."""
    return request.param


@pytest.fixture(params=["google", "elevenlabs", "whisper"])
def tts_engine(request):
    """Parametrized fixture for different TTS engines."""
    return request.param


@pytest.fixture(params=["whisper", "whisper_local"])
def stt_engine(request):
    """Parametrized fixture for different STT engines."""
    return request.param


# ============================================================================
# CLEANUP UTILITIES
# ============================================================================


def pytest_runtest_teardown(item, nextitem):
    """Clean up after each test."""
    # Close any open database connections
    import sqlite3

    # Clear any cached modules
    import sys

    modules_to_remove = [
        name for name in sys.modules.keys() if name.startswith("test_")
    ]
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]


# ============================================================================
# CUSTOM ASSERTIONS
# ============================================================================


def assert_gianna_state_valid(state: GiannaState):
    """Assert that a GiannaState is properly formatted."""
    assert isinstance(state, dict)
    assert "conversation" in state
    assert "audio" in state
    assert "commands" in state
    assert "metadata" in state

    # Validate conversation state
    conv = state["conversation"]
    assert hasattr(conv, "messages")
    assert hasattr(conv, "session_id")

    # Validate audio state
    audio = state["audio"]
    assert hasattr(audio, "current_mode")
    assert audio.current_mode in ["idle", "listening", "speaking", "processing"]

    # Validate command state
    commands = state["commands"]
    assert hasattr(commands, "execution_history")
    assert hasattr(commands, "pending_operations")


def assert_response_format_valid(response: Dict[str, Any]):
    """Assert that a response follows expected format."""
    assert isinstance(response, dict)
    assert "output" in response
    assert isinstance(response["output"], str)
    assert len(response["output"]) > 0


# Make assertions available to all test modules
pytest.assert_gianna_state_valid = assert_gianna_state_valid
pytest.assert_response_format_valid = assert_response_format_valid
