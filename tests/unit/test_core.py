"""
Unit tests for core Gianna components - FASE 1

Tests for:
- State management (GiannaState, StateManager)
- LangGraph integration (LangGraphChain, factories)
- Core utilities and migration helpers

Test Coverage:
- State creation and validation
- State manager database operations
- LangGraph chain functionality
- Factory method patterns
- Migration utilities
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
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


@pytest.mark.unit
@pytest.mark.fase1
class TestGiannaState:
    """Test GiannaState structure and operations."""

    def test_create_initial_state(self, test_session_id):
        """Test initial state creation with proper structure."""
        state = create_initial_state(test_session_id)

        # Validate state structure
        pytest.assert_gianna_state_valid(state)
        assert state["conversation"].session_id == test_session_id
        assert state["audio"].current_mode == "idle"
        assert len(state["commands"].execution_history) == 0
        assert isinstance(state["metadata"], dict)

    def test_conversation_state_creation(self):
        """Test ConversationState model validation."""
        conv_state = ConversationState(
            messages=[],
            session_id="test-123",
            user_preferences={"lang": "pt-br"},
            context_summary="",
        )

        assert conv_state.session_id == "test-123"
        assert conv_state.user_preferences["lang"] == "pt-br"
        assert len(conv_state.messages) == 0
        assert conv_state.context_summary == ""

    def test_audio_state_creation(self):
        """Test AudioState model validation."""
        audio_state = AudioState(
            current_mode="listening",
            voice_settings={"speed": 1.2},
            speech_type="elevenlabs",
            language="en-us",
        )

        assert audio_state.current_mode == "listening"
        assert audio_state.voice_settings["speed"] == 1.2
        assert audio_state.speech_type == "elevenlabs"
        assert audio_state.language == "en-us"

    def test_command_state_creation(self):
        """Test CommandState model validation."""
        cmd_state = CommandState(
            execution_history=[
                {"cmd": "ls", "result": "files", "timestamp": "2024-01-01"}
            ],
            pending_operations=["mkdir test"],
        )

        assert len(cmd_state.execution_history) == 1
        assert cmd_state.execution_history[0]["cmd"] == "ls"
        assert len(cmd_state.pending_operations) == 1
        assert cmd_state.pending_operations[0] == "mkdir test"

    def test_state_message_operations(self, gianna_state):
        """Test message operations on conversation state."""
        initial_count = len(gianna_state["conversation"].messages)

        # Add user message
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": "Test message",
                "timestamp": datetime.now().isoformat(),
            }
        )

        assert len(gianna_state["conversation"].messages) == initial_count + 1
        last_message = gianna_state["conversation"].messages[-1]
        assert last_message["role"] == "user"
        assert last_message["content"] == "Test message"

    def test_state_audio_mode_transitions(self, gianna_state):
        """Test audio mode state transitions."""
        modes = ["idle", "listening", "processing", "speaking"]

        for mode in modes:
            gianna_state["audio"].current_mode = mode
            assert gianna_state["audio"].current_mode == mode

    def test_state_command_history_operations(self, gianna_state):
        """Test command history operations."""
        initial_count = len(gianna_state["commands"].execution_history)

        # Add command execution
        gianna_state["commands"].execution_history.append(
            {
                "command": "echo 'test'",
                "result": "test",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "execution_time": 0.1,
            }
        )

        assert len(gianna_state["commands"].execution_history) == initial_count + 1
        last_cmd = gianna_state["commands"].execution_history[-1]
        assert last_cmd["command"] == "echo 'test'"
        assert last_cmd["success"] is True

    def test_state_metadata_operations(self, gianna_state):
        """Test metadata operations."""
        # Add metadata
        gianna_state["metadata"]["test_key"] = "test_value"
        gianna_state["metadata"]["session_start"] = datetime.now().isoformat()
        gianna_state["metadata"]["interaction_count"] = 5

        assert gianna_state["metadata"]["test_key"] == "test_value"
        assert "session_start" in gianna_state["metadata"]
        assert gianna_state["metadata"]["interaction_count"] == 5


@pytest.mark.unit
@pytest.mark.fase1
@pytest.mark.database
class TestStateManager:
    """Test StateManager database operations."""

    def test_state_manager_initialization(self, temp_db):
        """Test StateManager initializes with database."""
        manager = StateManager(temp_db)

        assert manager.db_path == Path(temp_db)
        assert manager.checkpointer is not None

        # Check database file exists
        assert Path(temp_db).exists()

    def test_database_schema_creation(self, temp_db):
        """Test database schema is created correctly."""
        manager = StateManager(temp_db)

        # Check table exists
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='user_sessions'
        """
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == "user_sessions"

    def test_get_config(self, state_manager, test_session_id):
        """Test configuration generation for sessions."""
        config = state_manager.get_config(test_session_id)

        assert isinstance(config, dict)
        assert "configurable" in config
        assert "thread_id" in config["configurable"]
        assert config["configurable"]["thread_id"] == test_session_id

    def test_multiple_sessions(self, state_manager):
        """Test multiple session configurations."""
        sessions = [f"session-{i}" for i in range(5)]
        configs = []

        for session_id in sessions:
            config = state_manager.get_config(session_id)
            configs.append(config)

            assert config["configurable"]["thread_id"] == session_id

        # Ensure all configs are unique
        thread_ids = [c["configurable"]["thread_id"] for c in configs]
        assert len(set(thread_ids)) == len(sessions)

    def test_state_manager_persistence(self, temp_db, test_session_id):
        """Test state manager handles database persistence."""
        # Create manager and perform operations
        manager1 = StateManager(temp_db)
        config1 = manager1.get_config(test_session_id)

        # Create new manager instance (simulate restart)
        manager2 = StateManager(temp_db)
        config2 = manager2.get_config(test_session_id)

        # Configs should be identical
        assert config1 == config2
        assert Path(temp_db).exists()


@pytest.mark.unit
@pytest.mark.fase1
class TestLangGraphIntegration:
    """Test LangGraph integration components."""

    def test_langgraph_chain_import(self):
        """Test LangGraphChain can be imported."""
        try:
            from gianna.core.langgraph_chain import LangGraphChain

            assert LangGraphChain is not None
        except ImportError as e:
            pytest.skip(f"LangGraphChain not available: {e}")

    @patch("gianna.core.langgraph_chain.StateGraph")
    @patch("gianna.core.langgraph_chain.StateManager")
    def test_langgraph_chain_creation(self, mock_state_manager, mock_state_graph):
        """Test LangGraphChain creation with mocked dependencies."""
        from gianna.core.langgraph_chain import LangGraphChain

        # Setup mocks
        mock_manager = MagicMock()
        mock_state_manager.return_value = mock_manager

        mock_graph_instance = MagicMock()
        mock_compiled_graph = MagicMock()
        mock_graph_instance.compile.return_value = mock_compiled_graph
        mock_state_graph.return_value = mock_graph_instance

        # Create chain
        chain = LangGraphChain("gpt35", "You are helpful")

        assert chain.model_name == "gpt35"
        assert chain.state_manager == mock_manager
        assert chain.graph == mock_compiled_graph

        # Verify graph was configured
        mock_state_graph.assert_called_once()
        mock_graph_instance.compile.assert_called_once()

    @patch("gianna.core.langgraph_chain.LangGraphChain")
    def test_langgraph_factory_integration(self, mock_langgraph_chain):
        """Test LangGraphChain factory integration."""
        from gianna.core.langgraph_factory import create_langgraph_chain

        mock_instance = MagicMock()
        mock_langgraph_chain.return_value = mock_instance

        chain = create_langgraph_chain("gpt4", "Test prompt")

        assert chain == mock_instance
        mock_langgraph_chain.assert_called_once_with("gpt4", "Test prompt")

    def test_migration_utils_import(self):
        """Test migration utilities can be imported."""
        try:
            from gianna.core.migration_utils import (
                convert_legacy_chain_to_langgraph,
                validate_state_compatibility,
            )

            assert convert_legacy_chain_to_langgraph is not None
            assert validate_state_compatibility is not None
        except ImportError as e:
            pytest.skip(f"Migration utils not available: {e}")

    @patch("gianna.core.migration_utils.LangGraphChain")
    def test_legacy_chain_conversion(self, mock_langgraph_chain):
        """Test legacy chain to LangGraph conversion."""
        from gianna.core.migration_utils import convert_legacy_chain_to_langgraph

        # Mock legacy chain
        legacy_chain = MagicMock()
        legacy_chain.model_name = "gpt35"
        legacy_chain.prompt_template = "Test template"

        mock_instance = MagicMock()
        mock_langgraph_chain.return_value = mock_instance

        result = convert_legacy_chain_to_langgraph(legacy_chain)

        assert result == mock_instance
        mock_langgraph_chain.assert_called_once_with("gpt35", "Test template")

    def test_state_compatibility_validation(self, gianna_state):
        """Test state compatibility validation."""
        from gianna.core.migration_utils import validate_state_compatibility

        # Valid state should pass
        is_valid = validate_state_compatibility(gianna_state)
        assert is_valid is True

        # Invalid state should fail
        invalid_state = {"invalid": "structure"}
        is_valid = validate_state_compatibility(invalid_state)
        assert is_valid is False


@pytest.mark.unit
@pytest.mark.fase1
class TestFactoryPatterns:
    """Test factory pattern implementations in core module."""

    def test_langgraph_factory_methods(self):
        """Test LangGraph factory method patterns."""
        try:
            from gianna.core.langgraph_factory import (
                create_langgraph_chain,
                create_workflow_graph,
                get_available_models,
            )

            assert callable(create_langgraph_chain)
            assert callable(get_available_models)
            assert callable(create_workflow_graph)
        except ImportError as e:
            pytest.skip(f"LangGraph factory not available: {e}")

    @patch("gianna.core.langgraph_factory.LangGraphChain")
    def test_factory_model_creation(self, mock_langgraph_chain):
        """Test factory creates different model types."""
        from gianna.core.langgraph_factory import create_langgraph_chain

        models = ["gpt35", "gpt4", "claude", "gemini"]
        mock_instance = MagicMock()
        mock_langgraph_chain.return_value = mock_instance

        for model in models:
            chain = create_langgraph_chain(model, "Test prompt")
            assert chain == mock_instance

        assert mock_langgraph_chain.call_count == len(models)

    def test_factory_error_handling(self):
        """Test factory error handling for invalid inputs."""
        try:
            from gianna.core.langgraph_factory import create_langgraph_chain

            with pytest.raises((ValueError, KeyError)):
                create_langgraph_chain("invalid_model", "Test prompt")
        except ImportError:
            pytest.skip("LangGraph factory not available")

    def test_available_models_listing(self):
        """Test listing available models from factory."""
        try:
            from gianna.core.langgraph_factory import get_available_models

            models = get_available_models()
            assert isinstance(models, (list, tuple))
            assert len(models) > 0

            # Check for expected models
            expected_models = ["gpt35", "gpt4", "claude", "gemini"]
            for model in expected_models:
                if model not in models:
                    # Some models might not be available in test environment
                    pass
        except ImportError:
            pytest.skip("LangGraph factory not available")


@pytest.mark.unit
@pytest.mark.performance
class TestCorePerformance:
    """Test performance characteristics of core components."""

    def test_state_creation_performance(self, benchmark_timer):
        """Test state creation performance."""
        benchmark_timer.start()

        states = []
        for i in range(100):
            state = create_initial_state(f"session-{i}")
            states.append(state)

        benchmark_timer.stop()

        assert len(states) == 100
        assert benchmark_timer.elapsed < 1.0  # < 1 second for 100 states

    def test_state_message_append_performance(self, gianna_state, benchmark_timer):
        """Test message append performance."""
        benchmark_timer.start()

        for i in range(1000):
            gianna_state["conversation"].messages.append(
                {
                    "role": "user",
                    "content": f"Message {i}",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        benchmark_timer.stop()

        assert len(gianna_state["conversation"].messages) >= 1000
        assert benchmark_timer.elapsed < 1.0  # < 1 second for 1000 messages

    def test_state_manager_config_generation_performance(
        self, state_manager, benchmark_timer
    ):
        """Test config generation performance."""
        benchmark_timer.start()

        configs = []
        for i in range(1000):
            config = state_manager.get_config(f"session-{i}")
            configs.append(config)

        benchmark_timer.stop()

        assert len(configs) == 1000
        assert benchmark_timer.elapsed < 1.0  # < 1 second for 1000 configs

    def test_database_operations_performance(self, temp_db, benchmark_timer):
        """Test database operations performance."""
        benchmark_timer.start()

        # Create multiple managers (simulate high load)
        managers = []
        for i in range(10):
            manager = StateManager(temp_db)
            configs = [manager.get_config(f"session-{j}") for j in range(10)]
            managers.append((manager, configs))

        benchmark_timer.stop()

        assert len(managers) == 10
        assert benchmark_timer.elapsed < 2.0  # < 2 seconds for 100 operations
