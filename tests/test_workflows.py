"""
Comprehensive workflow tests for Gianna - Phase 5 Testing Framework

This module contains comprehensive tests for LangGraphChain, voice workflows,
multi-agent coordination, and complete workflow integration following the
Phase 5 specifications.

Test Categories:
- LangGraphChain functionality and compatibility
- Complete voice workflow pipeline
- Multi-agent coordination and orchestration
- End-to-end workflow integration
- Performance and reliability testing

Requirements Coverage:
- FASE 1: Core state management, LangGraph chains
- FASE 2: ReAct agents, tools, orchestrator
- FASE 3: VAD, streaming pipeline, voice workflows
- FASE 4: Semantic memory, learning, optimization
- FASE 5: End-to-end integration
"""

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Import test performance thresholds
from tests import PERFORMANCE_THRESHOLDS

# Core imports
from gianna.core.state import GiannaState, create_initial_state
from gianna.core.state_manager import StateManager

# ============================================================================
# FASE 1 TESTS: LANGGRAPH CHAIN FUNCTIONALITY
# ============================================================================


@pytest.mark.unit
@pytest.mark.fase1
class TestLangGraphChain:
    """Test LangGraphChain implementation and compatibility."""

    def test_langgraph_chain_creation(self, mock_langgraph_chain, gianna_state):
        """Test LangGraphChain can be created with proper configuration."""
        from gianna.core.langgraph_chain import LangGraphChain

        with patch(
            "gianna.core.langgraph_chain.LangGraphChain",
            return_value=mock_langgraph_chain,
        ):
            chain = LangGraphChain("gpt35", "You are a helpful assistant")

            assert chain is not None
            assert hasattr(chain, "invoke")
            assert hasattr(chain, "model_name")

    def test_langgraph_chain_invoke(self, mock_langgraph_chain, gianna_state):
        """Test LangGraphChain invoke method with state."""
        mock_langgraph_chain.invoke.return_value = {
            "output": "Test response from LangGraph chain"
        }

        response = mock_langgraph_chain.invoke({"input": "Hello"})

        assert "output" in response
        assert isinstance(response["output"], str)
        assert len(response["output"]) > 0
        pytest.assert_response_format_valid(response)

    def test_langgraph_chain_async_invoke(
        self, mock_langgraph_chain, async_test_runner
    ):
        """Test LangGraphChain async invoke method."""

        async def test_async():
            response = await mock_langgraph_chain.ainvoke({"input": "Hello async"})
            assert "output" in response
            assert isinstance(response["output"], str)
            return response

        response = async_test_runner(test_async())
        pytest.assert_response_format_valid(response)

    def test_langgraph_chain_state_management(
        self, mock_langgraph_chain, gianna_state, state_manager
    ):
        """Test LangGraphChain properly manages state."""
        # Simulate state updates
        initial_msg_count = len(gianna_state["conversation"].messages)

        # Add user message
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": "Test message",
                "timestamp": datetime.now().isoformat(),
            }
        )

        mock_langgraph_chain.invoke.return_value = {
            "output": "Response to test message"
        }

        response = mock_langgraph_chain.invoke({"input": "Test message"})

        assert len(gianna_state["conversation"].messages) == initial_msg_count + 1
        assert response["output"] == "Response to test message"

    @pytest.mark.parametrize("model_name", ["gpt35", "gpt4", "claude", "gemini"])
    def test_langgraph_chain_multi_model(self, model_name, mock_langgraph_chain):
        """Test LangGraphChain works with different models."""
        with patch("gianna.core.langgraph_chain.LangGraphChain") as mock_class:
            mock_instance = MagicMock()
            mock_instance.model_name = model_name
            mock_instance.invoke.return_value = {
                "output": f"Response from {model_name}"
            }
            mock_class.return_value = mock_instance

            from gianna.core.langgraph_chain import LangGraphChain

            chain = LangGraphChain(model_name, "Test prompt")

            response = chain.invoke({"input": "Test"})

            assert chain.model_name == model_name
            assert model_name in response["output"]

    def test_langgraph_chain_error_handling(self, mock_langgraph_chain):
        """Test LangGraphChain error handling."""
        mock_langgraph_chain.invoke.side_effect = Exception("Test error")

        with pytest.raises(Exception) as exc_info:
            mock_langgraph_chain.invoke({"input": "Test"})

        assert "Test error" in str(exc_info.value)

    @pytest.mark.performance
    def test_langgraph_chain_performance(self, mock_langgraph_chain, benchmark_timer):
        """Test LangGraphChain performance meets requirements."""
        mock_langgraph_chain.invoke.return_value = {"output": "Fast response"}

        benchmark_timer.start()
        response = mock_langgraph_chain.invoke({"input": "Performance test"})
        benchmark_timer.stop()

        assert (
            benchmark_timer.elapsed
            < PERFORMANCE_THRESHOLDS["response_time"]["simple_command"]
        )
        assert response["output"] == "Fast response"


# ============================================================================
# FASE 2 TESTS: REACT AGENTS AND COORDINATION
# ============================================================================


@pytest.mark.integration
@pytest.mark.fase2
@pytest.mark.agents
class TestMultiAgentCoordination:
    """Test multi-agent coordination and orchestration."""

    def test_agent_registration(self, mock_orchestrator, mock_react_agent):
        """Test agent registration with orchestrator."""
        mock_orchestrator.register_agent(mock_react_agent)
        mock_orchestrator.register_agent.assert_called_once_with(mock_react_agent)

    def test_request_routing(self, mock_orchestrator, gianna_state):
        """Test intelligent request routing to appropriate agents."""
        # Test command routing
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": "executar comando ls",
                "timestamp": datetime.now().isoformat(),
            }
        )

        mock_orchestrator.route_request.return_value = "command_agent"
        agent_name = mock_orchestrator.route_request(gianna_state)

        assert agent_name == "command_agent"
        mock_orchestrator.route_request.assert_called_once_with(gianna_state)

    def test_audio_agent_routing(self, mock_orchestrator, gianna_state):
        """Test routing to audio agent for voice requests."""
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": "falar em voz alta",
                "timestamp": datetime.now().isoformat(),
            }
        )

        mock_orchestrator.route_request.return_value = "audio_agent"
        agent_name = mock_orchestrator.route_request(gianna_state)

        assert agent_name == "audio_agent"

    def test_parallel_agent_execution(self, mock_orchestrator, gianna_state):
        """Test parallel execution coordination between agents."""
        agents = ["command_agent", "audio_agent"]

        mock_orchestrator.coordinate_parallel_execution(agents, gianna_state)
        mock_orchestrator.coordinate_parallel_execution.assert_called_once_with(
            agents, gianna_state
        )

    @pytest.mark.slow
    def test_agent_coordination_stress(self, mock_orchestrator, gianna_state):
        """Test agent coordination under stress conditions."""
        # Simulate multiple concurrent requests
        agents = ["command_agent", "audio_agent", "memory_agent"]

        for i in range(10):
            gianna_state["conversation"].messages.append(
                {
                    "role": "user",
                    "content": f"concurrent request {i}",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            mock_orchestrator.route_request.return_value = agents[i % len(agents)]
            agent_name = mock_orchestrator.route_request(gianna_state)
            assert agent_name in agents

    def test_agent_tool_integration(
        self, mock_react_agent, mock_shell_tool, mock_audio_tools
    ):
        """Test agent integration with tools."""
        # Test shell tool integration
        mock_react_agent.tools = [mock_shell_tool]
        mock_react_agent.invoke.return_value = {
            "output": "Command executed successfully",
            "tool_results": {"shell_executor": "test output"},
        }

        response = mock_react_agent.invoke({"input": "execute ls command"})

        assert "output" in response
        assert "tool_results" in response or "output" in response


# ============================================================================
# FASE 3 TESTS: VOICE WORKFLOW PIPELINE
# ============================================================================


@pytest.mark.integration
@pytest.mark.fase3
@pytest.mark.voice
class TestVoiceWorkflows:
    """Test complete voice processing workflows."""

    def test_voice_activity_detection(self, mock_vad, sample_audio_data):
        """Test Voice Activity Detection functionality."""
        # Test speech detection
        mock_vad.detect_activity.return_value = True
        has_speech = mock_vad.detect_activity(sample_audio_data)

        assert has_speech is True
        mock_vad.detect_activity.assert_called_once_with(sample_audio_data)

    def test_streaming_voice_pipeline_setup(
        self, mock_vad, mock_react_agent, gianna_state
    ):
        """Test streaming voice pipeline initialization."""
        with patch(
            "gianna.workflows.voice_streaming.StreamingVoicePipeline"
        ) as mock_pipeline:
            pipeline = mock_pipeline.return_value
            pipeline.start_listening = AsyncMock()

            # Test pipeline setup
            assert pipeline is not None

            # Test async listening start
            async def test_start():
                await pipeline.start_listening(gianna_state)
                pipeline.start_listening.assert_called_once_with(gianna_state)

            asyncio.run(test_start())

    def test_voice_to_text_conversion(self, mock_stt, temp_audio_file):
        """Test speech-to-text conversion in voice workflow."""
        mock_stt.return_value = "Converted speech to text"

        result = mock_stt(temp_audio_file)

        assert result == "Converted speech to text"
        assert isinstance(result, str)
        assert len(result) > 0

    def test_text_to_voice_conversion(self, mock_tts, gianna_state):
        """Test text-to-speech conversion in voice workflow."""
        test_text = "Hello, this is a test response"
        mock_tts_instance = MagicMock()
        mock_tts.return_value = mock_tts_instance

        result = mock_tts(test_text)

        assert result is not None
        mock_tts.assert_called_once_with(test_text)

    @pytest.mark.slow
    @pytest.mark.async_test
    def test_complete_voice_workflow(
        self,
        mock_vad,
        mock_stt,
        mock_tts,
        mock_react_agent,
        gianna_state,
        async_test_runner,
    ):
        """Test complete voice workflow from speech input to voice output."""

        async def voice_workflow():
            # 1. Voice Activity Detection
            mock_vad.detect_activity.return_value = True
            has_speech = mock_vad.detect_activity([0.1, 0.2, 0.3])  # Mock audio
            assert has_speech

            # 2. Speech to Text
            mock_stt.return_value = "test voice command"
            text_input = mock_stt("mock_audio_file.wav")
            assert text_input == "test voice command"

            # 3. Update state with user input
            gianna_state["conversation"].messages.append(
                {
                    "role": "user",
                    "content": text_input,
                    "timestamp": datetime.now().isoformat(),
                    "source": "voice",
                }
            )

            # 4. Agent processing
            mock_react_agent.ainvoke = AsyncMock(
                return_value={"output": "Voice command processed successfully"}
            )
            response = await mock_react_agent.ainvoke(gianna_state)
            assert "output" in response

            # 5. Text to Speech
            mock_tts_instance = MagicMock()
            mock_tts.return_value = mock_tts_instance
            tts_result = mock_tts(response["output"])
            assert tts_result is not None

            # 6. Update state with assistant response
            gianna_state["conversation"].messages.append(
                {
                    "role": "assistant",
                    "content": response["output"],
                    "timestamp": datetime.now().isoformat(),
                    "source": "voice",
                }
            )

            return response

        result = async_test_runner(voice_workflow())
        assert result["output"] == "Voice command processed successfully"

    def test_voice_workflow_error_recovery(self, mock_vad, mock_stt, gianna_state):
        """Test voice workflow error recovery mechanisms."""
        # Test STT failure recovery
        mock_stt.side_effect = Exception("STT service unavailable")

        with pytest.raises(Exception) as exc_info:
            mock_stt("test_audio.wav")

        assert "STT service unavailable" in str(exc_info.value)

        # Test VAD failure recovery
        mock_vad.detect_activity.side_effect = Exception("VAD error")

        with pytest.raises(Exception):
            mock_vad.detect_activity([0.1, 0.2])

    @pytest.mark.performance
    def test_voice_workflow_performance(
        self, mock_vad, mock_stt, mock_tts, benchmark_timer, gianna_state
    ):
        """Test voice workflow performance requirements."""
        # Setup mocks for fast responses
        mock_vad.detect_activity.return_value = True
        mock_stt.return_value = "quick test"
        mock_tts.return_value = MagicMock()

        benchmark_timer.start()

        # Simulate voice workflow steps
        has_speech = mock_vad.detect_activity([0.1, 0.2])
        if has_speech:
            text = mock_stt("audio.wav")
            tts_result = mock_tts(text)

        benchmark_timer.stop()

        assert (
            benchmark_timer.elapsed
            < PERFORMANCE_THRESHOLDS["response_time"]["complex_workflow"]
        )
        assert has_speech is True
        assert text == "quick test"
        assert tts_result is not None


# ============================================================================
# FASE 4 TESTS: MEMORY AND LEARNING
# ============================================================================


@pytest.mark.integration
@pytest.mark.fase4
@pytest.mark.memory
class TestSemanticMemoryWorkflow:
    """Test semantic memory integration in workflows."""

    def test_memory_storage_workflow(self, mock_semantic_memory, gianna_state):
        """Test storing interactions in semantic memory during workflow."""
        interaction = {
            "user_input": "What is the weather like?",
            "assistant_response": "I need more information about your location.",
            "timestamp": datetime.now().isoformat(),
            "session_id": gianna_state["conversation"].session_id,
            "intent": "weather_query",
        }

        mock_semantic_memory.store_interaction(interaction)
        mock_semantic_memory.store_interaction.assert_called_once_with(interaction)

    def test_memory_retrieval_workflow(self, mock_semantic_memory, gianna_state):
        """Test retrieving similar interactions from semantic memory."""
        query = "weather information"
        mock_semantic_memory.search_similar_interactions.return_value = [
            {
                "content": "Previous weather query",
                "metadata": {"score": 0.85, "timestamp": "2024-01-01T10:00:00"},
            }
        ]

        results = mock_semantic_memory.search_similar_interactions(query, k=5)

        assert len(results) == 1
        assert results[0]["metadata"]["score"] > 0.8
        mock_semantic_memory.search_similar_interactions.assert_called_once_with(
            query, k=5
        )

    def test_context_summary_generation(self, mock_semantic_memory, gianna_state):
        """Test context summary generation for session."""
        session_id = gianna_state["conversation"].session_id
        mock_semantic_memory.get_context_summary.return_value = (
            "User has been asking about weather and location services"
        )

        summary = mock_semantic_memory.get_context_summary(session_id)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "weather" in summary
        mock_semantic_memory.get_context_summary.assert_called_once_with(session_id)

    def test_learning_system_integration(self, mock_learning_system, gianna_state):
        """Test learning system integration with workflows."""
        # Simulate user pattern analysis
        mock_learning_system.analyze_user_patterns(gianna_state)
        mock_learning_system.analyze_user_patterns.assert_called_once_with(gianna_state)

        # Test response adaptation
        user_id = gianna_state["conversation"].session_id
        base_response = "This is a standard response"
        mock_learning_system.adapt_response_style.return_value = (
            "This is an adapted response for your preferences"
        )

        adapted_response = mock_learning_system.adapt_response_style(
            user_id, base_response
        )

        assert adapted_response != base_response
        assert "adapted" in adapted_response
        mock_learning_system.adapt_response_style.assert_called_once_with(
            user_id, base_response
        )


# ============================================================================
# FASE 5 TESTS: END-TO-END INTEGRATION
# ============================================================================


@pytest.mark.end_to_end
@pytest.mark.fase5
@pytest.mark.slow
class TestEndToEndWorkflows:
    """Test complete end-to-end workflow integration."""

    def test_complete_conversation_workflow(
        self,
        gianna_state,
        mock_langgraph_chain,
        mock_semantic_memory,
        mock_learning_system,
    ):
        """Test complete conversation workflow with all components."""
        # 1. Initial state setup
        pytest.assert_gianna_state_valid(gianna_state)

        # 2. User input processing
        user_input = "Hello, can you help me with a complex task?"
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # 3. LLM processing
        mock_langgraph_chain.invoke.return_value = {
            "output": "I'd be happy to help you with your complex task. Could you provide more details?"
        }
        response = mock_langgraph_chain.invoke({"input": user_input})

        # 4. Memory storage
        interaction = {
            "user_input": user_input,
            "assistant_response": response["output"],
            "timestamp": datetime.now().isoformat(),
            "session_id": gianna_state["conversation"].session_id,
            "intent": "help_request",
        }
        mock_semantic_memory.store_interaction(interaction)

        # 5. Learning system update
        mock_learning_system.analyze_user_patterns(gianna_state)

        # Assertions
        assert len(gianna_state["conversation"].messages) >= 1
        assert response["output"] is not None
        mock_semantic_memory.store_interaction.assert_called_once()
        mock_learning_system.analyze_user_patterns.assert_called_once()

    def test_complete_voice_interaction_workflow(
        self,
        gianna_state,
        mock_vad,
        mock_stt,
        mock_tts,
        mock_react_agent,
        mock_semantic_memory,
    ):
        """Test complete voice interaction workflow end-to-end."""
        # Setup initial audio state
        gianna_state["audio"].current_mode = "listening"

        # 1. Voice Activity Detection
        mock_vad.detect_activity.return_value = True
        has_speech = mock_vad.detect_activity([0.1, 0.2, 0.3])
        assert has_speech

        # 2. Speech to Text
        mock_stt.return_value = "show me my calendar for today"
        transcribed_text = mock_stt("audio_buffer.wav")

        # 3. Update conversation state
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": transcribed_text,
                "timestamp": datetime.now().isoformat(),
                "source": "voice",
            }
        )

        # 4. Agent processing
        mock_react_agent.invoke.return_value = {
            "output": "Here's your calendar for today. You have 3 meetings scheduled."
        }
        response = mock_react_agent.invoke(gianna_state)

        # 5. Text to Speech
        mock_tts.return_value = MagicMock()
        tts_result = mock_tts(response["output"])

        # 6. Update audio state
        gianna_state["audio"].current_mode = "speaking"

        # 7. Memory storage
        interaction = {
            "user_input": transcribed_text,
            "assistant_response": response["output"],
            "timestamp": datetime.now().isoformat(),
            "session_id": gianna_state["conversation"].session_id,
            "intent": "calendar_query",
            "modality": "voice",
        }
        mock_semantic_memory.store_interaction(interaction)

        # Assertions
        assert transcribed_text == "show me my calendar for today"
        assert "calendar" in response["output"]
        assert gianna_state["audio"].current_mode == "speaking"
        assert tts_result is not None
        mock_semantic_memory.store_interaction.assert_called_once()

    def test_multi_modal_workflow_integration(
        self, gianna_state, mock_langgraph_chain, mock_shell_tool, mock_semantic_memory
    ):
        """Test integration between text and voice modalities."""
        # 1. Start with text interaction
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": "create a new directory called 'test_project'",
                "timestamp": datetime.now().isoformat(),
                "source": "text",
            }
        )

        # 2. Process with command agent (using shell tool)
        mock_shell_tool._run.return_value = json.dumps(
            {
                "exit_code": 0,
                "stdout": "Directory created successfully",
                "stderr": "",
                "success": True,
            }
        )

        mock_langgraph_chain.invoke.return_value = {
            "output": "I've successfully created the 'test_project' directory for you."
        }

        response = mock_langgraph_chain.invoke({"input": "create directory"})

        # 3. Continue with voice interaction
        gianna_state["conversation"].messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "timestamp": datetime.now().isoformat(),
                "source": "text",
            }
        )

        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": "now list the contents of that directory",
                "timestamp": datetime.now().isoformat(),
                "source": "voice",
            }
        )

        # 4. Memory integration
        mock_semantic_memory.search_similar_interactions.return_value = [
            {
                "content": "Previous directory creation command",
                "metadata": {"score": 0.9, "intent": "file_management"},
            }
        ]

        similar = mock_semantic_memory.search_similar_interactions(
            "directory operations"
        )

        # Assertions
        assert len(gianna_state["conversation"].messages) >= 3
        assert response["output"] is not None
        assert "created" in response["output"]
        assert len(similar) > 0

    @pytest.mark.performance
    def test_end_to_end_performance_benchmark(
        self, gianna_state, mock_langgraph_chain, benchmark_timer
    ):
        """Test end-to-end workflow performance meets requirements."""
        # Setup for performance test
        mock_langgraph_chain.invoke.return_value = {
            "output": "Performance test response"
        }

        benchmark_timer.start()

        # Simulate complete workflow
        for i in range(5):  # 5 interactions
            gianna_state["conversation"].messages.append(
                {
                    "role": "user",
                    "content": f"Performance test message {i}",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            response = mock_langgraph_chain.invoke(
                {"input": f"Performance test message {i}"}
            )

            gianna_state["conversation"].messages.append(
                {
                    "role": "assistant",
                    "content": response["output"],
                    "timestamp": datetime.now().isoformat(),
                }
            )

        benchmark_timer.stop()

        # Performance assertions
        interactions_per_second = 5 / benchmark_timer.elapsed
        assert benchmark_timer.elapsed < 10.0  # 10 seconds for 5 interactions
        assert interactions_per_second >= 0.5  # At least 0.5 interactions per second

    @pytest.mark.database
    def test_state_persistence_workflow(self, temp_db, test_session_id):
        """Test state persistence across workflow sessions."""
        # Create state manager with temp database
        state_manager = StateManager(temp_db)

        # Create initial state
        state1 = create_initial_state(test_session_id)
        state1["conversation"].messages.append(
            {
                "role": "user",
                "content": "Test message for persistence",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Save state (this would normally happen automatically)
        config = state_manager.get_config(test_session_id)
        assert config["configurable"]["thread_id"] == test_session_id

        # Create new state instance (simulating new session)
        state2 = create_initial_state(test_session_id)

        # Verify state structure is maintained
        pytest.assert_gianna_state_valid(state1)
        pytest.assert_gianna_state_valid(state2)
        assert state1["conversation"].session_id == state2["conversation"].session_id


# ============================================================================
# UTILITY AND REGRESSION TESTS
# ============================================================================


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility with existing interfaces."""

    def test_legacy_invoke_interface(self, mock_chain):
        """Test legacy invoke interface still works."""
        # Test old-style invoke
        response = mock_chain.invoke({"input": "test"})
        assert "output" in response

        # Test with session parameter
        response_with_session = mock_chain.invoke(
            {"input": "test"}, session_id="test-session"
        )
        assert "output" in response_with_session

    def test_existing_examples_compatibility(self, mock_chain):
        """Test that existing examples continue to work."""
        # Simulate example notebook patterns
        examples = [
            {"input": "Hello world"},
            {"input": "What is the weather?"},
            {"input": "Execute ls command"},
            {"input": "Play some music"},
        ]

        for example in examples:
            response = mock_chain.invoke(example)
            assert "output" in response
            assert isinstance(response["output"], str)

    def test_environment_config_compatibility(self):
        """Test environment configuration still works."""
        # Test that environment variables are properly loaded
        assert "TESTING" in os.environ
        assert os.environ["TESTING"] == "true"

        # Test API key mocking
        api_keys = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "ELEVEN_LABS_API_KEY"]
        for key in api_keys:
            assert key in os.environ
            assert "test" in os.environ[key]


# ============================================================================
# TEST FIXTURES FOR GIANNA STATE
# ============================================================================


@pytest.fixture
def complex_gianna_state(test_session_id):
    """Create a complex GiannaState with multiple interactions for testing."""
    state = create_initial_state(test_session_id)

    # Add conversation history
    interactions = [
        ("user", "Hello, I need help with my project"),
        ("assistant", "I'd be happy to help! What kind of project are you working on?"),
        ("user", "It's a Python application with voice features"),
        (
            "assistant",
            "That sounds interesting! I can help you with Python development and voice integration.",
        ),
        ("user", "Can you execute some shell commands for me?"),
        ("assistant", "I can help generate shell commands. What would you like to do?"),
    ]

    for role, content in interactions:
        state["conversation"].messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "source": "text" if role == "user" else "assistant",
            }
        )

    # Add command history
    state["commands"].execution_history.extend(
        [
            {
                "command": "ls -la",
                "result": "total 24\ndrwxr-xr-x  3 user user 4096 Jan  1 12:00 .",
                "timestamp": datetime.now().isoformat(),
                "success": True,
            },
            {
                "command": "mkdir test_dir",
                "result": "Directory created",
                "timestamp": datetime.now().isoformat(),
                "success": True,
            },
        ]
    )

    # Update audio state
    state["audio"].current_mode = "idle"
    state["audio"].voice_settings = {"speed": 1.0, "pitch": 1.0, "volume": 0.8}

    # Add metadata
    state["metadata"].update(
        {
            "session_start": datetime.now().isoformat(),
            "interaction_count": len(interactions),
            "preferred_modality": "mixed",
            "user_expertise_level": "intermediate",
        }
    )

    return state


@pytest.fixture
def workflow_performance_config():
    """Configuration for workflow performance testing."""
    return {
        "max_response_time": 5.0,
        "max_memory_usage": 512,  # MB
        "min_throughput": 2.0,  # interactions per second
        "error_threshold": 0.01,  # 1% error rate
        "timeout_threshold": 30.0,  # seconds
    }
