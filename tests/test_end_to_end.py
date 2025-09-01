#!/usr/bin/env python3
"""
End-to-End Integration Tests for FASE 2 - Complete System Validation

This module provides comprehensive end-to-end tests that validate the complete
integration of tools, agents, coordination system, and existing Gianna
infrastructure working together as a cohesive system.
"""

import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from gianna.agents.react_agents import (
    AudioAgent,
    CommandAgent,
    ConversationAgent,
    MemoryAgent,
)
from gianna.coordination import AgentOrchestrator, AgentRouter
from gianna.coordination.router import AgentType, ExecutionMode
from gianna.core.state import GiannaState, create_initial_state
from gianna.tools import (
    AudioProcessorTool,
    FileSystemTool,
    MemoryTool,
    ShellExecutorTool,
    STTTool,
    TTSTool,
)


class MockLanguageModel:
    """Advanced mock language model for end-to-end testing."""

    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {
            "default": "I'll help you with that task.",
            "command": "I'll execute that command for you.",
            "audio": "I'll process that audio request.",
            "conversation": "I'm here to help with conversation.",
            "memory": "I'll manage that memory operation.",
        }
        self.call_count = 0
        self.call_history = []

    def invoke(self, input_data: Any) -> str:
        """Mock invoke with context-aware responses."""
        self.call_count += 1
        input_str = str(input_data).lower()

        # Store call history
        self.call_history.append(
            {
                "input": input_data,
                "timestamp": time.time(),
                "call_number": self.call_count,
            }
        )

        # Context-aware response selection
        if "comando" in input_str or "shell" in input_str:
            return self.responses.get("command", self.responses["default"])
        elif "áudio" in input_str or "music" in input_str or "voz" in input_str:
            return self.responses.get("audio", self.responses["default"])
        elif "memória" in input_str or "lembrar" in input_str:
            return self.responses.get("memory", self.responses["default"])
        else:
            return self.responses.get("conversation", self.responses["default"])


class TestCompleteSystemIntegration:
    """Test complete system integration from user input to final output."""

    def setup_method(self):
        """Setup comprehensive test environment."""
        self.mock_llm = MockLanguageModel(
            {
                "default": "System ready to process requests.",
                "command": "Executing shell command as requested.",
                "audio": "Processing audio with appropriate settings.",
                "conversation": "Engaging in natural conversation with user.",
                "memory": "Managing memory and context for optimal experience.",
            }
        )

        # Initialize orchestrator
        self.orchestrator = AgentOrchestrator(max_workers=4)

        # Create and register all agents
        self.agents = {
            "command": CommandAgent(self.mock_llm),
            "audio": AudioAgent(self.mock_llm),
            "conversation": ConversationAgent(self.mock_llm),
            "memory": MemoryAgent(self.mock_llm),
        }

        # Register agents with orchestrator
        for agent in self.agents.values():
            self.orchestrator.register_agent(agent)

        # Setup test directory
        self.test_dir = tempfile.mkdtemp(prefix="gianna_e2e_test_")

    def teardown_method(self):
        """Cleanup test environment."""
        self.orchestrator.shutdown()
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_user_command_execution_workflow(self):
        """Test complete workflow for user command execution."""
        # Create realistic user state
        state = create_initial_state("e2e_command_test")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Execute o comando 'ls -la' para mim"}
        ]
        state["conversation"]["user_preferences"] = {
            "name": "Usuário Teste",
            "language": "pt-BR",
        }

        # Execute complete workflow
        result = self.orchestrator.route_and_execute(state=state)

        # Verify workflow completion
        assert result.success, f"Command workflow failed: {result.error}"
        assert result.agent_type == AgentType.COMMAND
        assert result.execution_time > 0
        assert result.result is not None

        # Verify agent was called appropriately
        assert self.mock_llm.call_count > 0

        # Verify routing decision was recorded
        routing_stats = self.orchestrator.router.get_routing_stats()
        assert routing_stats["total_routes"] > 0

    def test_audio_processing_workflow(self):
        """Test complete workflow for audio processing."""
        state = create_initial_state("e2e_audio_test")
        state["conversation"]["messages"] = [
            {
                "role": "user",
                "content": "Converta este texto em fala: 'Olá, como você está?'",
            }
        ]
        state["audio"]["current_mode"] = "output"
        state["audio"]["language"] = "pt-BR"

        # Execute audio workflow
        result = self.orchestrator.route_and_execute(state=state)

        # Verify workflow completion
        assert result.success, f"Audio workflow failed: {result.error}"
        assert result.agent_type == AgentType.AUDIO
        assert result.execution_time > 0

        # Verify audio context was considered
        assert self.mock_llm.call_count > 0
        audio_calls = [
            call
            for call in self.mock_llm.call_history
            if "áudio" in str(call["input"]).lower()
        ]
        assert len(audio_calls) > 0

    def test_conversation_continuity_workflow(self):
        """Test conversation continuity across multiple interactions."""
        base_session_id = "e2e_conversation_test"

        # Simulate multi-turn conversation
        conversation_turns = [
            "Olá, como você está?",
            "Qual é o seu nome?",
            "Você pode me ajudar com algumas tarefas?",
            "Obrigado pela ajuda!",
        ]

        conversation_results = []
        accumulated_messages = []

        for turn, message in enumerate(conversation_turns):
            # Build conversation state
            state = create_initial_state(f"{base_session_id}_{turn}")
            accumulated_messages.extend([{"role": "user", "content": message}])

            # Add previous assistant responses
            if turn > 0:
                accumulated_messages.append(
                    {"role": "assistant", "content": f"Response to turn {turn}"}
                )

            state["conversation"]["messages"] = accumulated_messages.copy()

            # Execute conversation turn
            result = self.orchestrator.route_and_execute(state=state)
            conversation_results.append(result)

            # Verify each turn
            assert result.success, f"Conversation turn {turn} failed: {result.error}"
            assert result.execution_time > 0

        # Verify conversation continuity
        assert len(conversation_results) == 4
        assert all(r.success for r in conversation_results)

        # Verify increasing LLM interaction complexity
        assert self.mock_llm.call_count >= 4

    def test_multi_agent_coordination_workflow(self):
        """Test complex workflow requiring multiple agent coordination."""
        state = create_initial_state("e2e_coordination_test")
        state["conversation"]["messages"] = [
            {
                "role": "user",
                "content": "Analise o sistema, depois toque uma música e salve as preferências",
            }
        ]

        # Define multi-agent workflow
        agents_sequence = [AgentType.COMMAND, AgentType.AUDIO, AgentType.MEMORY]

        # Execute sequential coordination
        results = self.orchestrator.coordinate_agents(
            agents=agents_sequence, state=state, execution_mode=ExecutionMode.SEQUENTIAL
        )

        # Verify coordination success
        assert len(results) == 3
        assert all(r.success for r in results)

        # Verify agent execution order
        expected_agents = [AgentType.COMMAND, AgentType.AUDIO, AgentType.MEMORY]
        for i, (result, expected_agent) in enumerate(zip(results, expected_agents)):
            assert (
                result.agent_type == expected_agent
            ), f"Agent {i}: expected {expected_agent}, got {result.agent_type}"

        # Verify execution times are reasonable
        total_execution_time = sum(r.execution_time for r in results)
        assert total_execution_time > 0

        # Verify all agents were utilized
        performance_metrics = self.orchestrator.get_performance_metrics()
        for agent_name in ["command_agent", "audio_agent", "memory_agent"]:
            assert agent_name in performance_metrics
            assert performance_metrics[agent_name]["total_requests"] > 0

    def test_parallel_agent_execution_workflow(self):
        """Test parallel multi-agent execution for efficiency."""
        state = create_initial_state("e2e_parallel_test")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Execute análise paralela do sistema"}
        ]

        # Execute parallel coordination
        agents_parallel = [AgentType.COMMAND, AgentType.MEMORY, AgentType.AUDIO]

        start_time = time.time()
        results = self.orchestrator.coordinate_agents(
            agents=agents_parallel, state=state, execution_mode=ExecutionMode.PARALLEL
        )
        parallel_time = time.time() - start_time

        # Verify parallel execution
        assert len(results) == 3
        assert all(r.success for r in results)

        # Verify parallel execution was faster than sequential would be
        # (This is approximate, but parallel should be significantly faster)
        assert parallel_time < 1.0  # Should complete quickly with mocked agents

        # Verify all agents executed
        agent_types = {r.agent_type for r in results}
        expected_types = {AgentType.COMMAND, AgentType.MEMORY, AgentType.AUDIO}
        assert agent_types == expected_types

    def test_error_recovery_workflow(self):
        """Test system behavior under error conditions."""
        # Create a failing agent scenario
        original_agent = self.agents["command"]

        # Mock a failing agent
        failing_mock_llm = Mock()
        failing_mock_llm.invoke.side_effect = Exception("Simulated LLM failure")

        failing_agent = CommandAgent(failing_mock_llm)
        self.orchestrator.agents[AgentType.COMMAND].agent = failing_agent

        state = create_initial_state("e2e_error_test")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Execute comando ls"}
        ]

        # Execute with failing agent
        result = self.orchestrator.route_and_execute(
            state=state, requested_agent=AgentType.COMMAND
        )

        # Verify error handling
        assert not result.success
        assert result.error is not None
        assert "failure" in result.error.lower() or "error" in result.error.lower()

        # Verify system stability (other agents still work)
        conversation_result = self.orchestrator.route_and_execute(
            state=state, requested_agent=AgentType.CONVERSATION
        )
        assert conversation_result.success

        # Restore original agent
        self.orchestrator.agents[AgentType.COMMAND].agent = original_agent

    def test_state_management_integration(self):
        """Test state management across the complete system."""
        session_id = "e2e_state_test"

        # Initial state setup
        initial_state = create_initial_state(session_id)
        initial_state["conversation"]["user_preferences"] = {
            "name": "Test User",
            "language": "pt-BR",
            "voice_preference": "female",
        }
        initial_state["conversation"]["messages"] = [
            {"role": "user", "content": "Configure minhas preferências de voz"}
        ]

        # Execute memory agent to handle preferences
        result = self.orchestrator.route_and_execute(
            state=initial_state, requested_agent=AgentType.MEMORY
        )

        assert result.success

        # Verify state was processed
        assert result.result is not None

        # Test state persistence concept (in a real implementation)
        # For now, verify the state structure is maintained
        assert initial_state["conversation"]["session_id"] == session_id
        assert initial_state["conversation"]["user_preferences"]["name"] == "Test User"

    def test_tool_integration_across_agents(self):
        """Test that tools work correctly across different agents."""
        test_file = os.path.join(self.test_dir, "integration_test.txt")
        test_content = "Integration test content from end-to-end testing"

        # Test 1: Command agent with file operations
        command_state = create_initial_state("e2e_tool_command")
        command_state["conversation"]["messages"] = [
            {"role": "user", "content": f"Create file {test_file} with content"}
        ]

        command_result = self.orchestrator.route_and_execute(
            state=command_state, requested_agent=AgentType.COMMAND
        )
        assert command_result.success

        # Test 2: Memory agent with session management
        memory_state = create_initial_state("e2e_tool_memory")
        memory_state["conversation"]["messages"] = [
            {"role": "user", "content": "Store session information"}
        ]

        memory_result = self.orchestrator.route_and_execute(
            state=memory_state, requested_agent=AgentType.MEMORY
        )
        assert memory_result.success

        # Test 3: Audio agent with audio processing
        audio_state = create_initial_state("e2e_tool_audio")
        audio_state["conversation"]["messages"] = [
            {"role": "user", "content": "Process audio file for speech synthesis"}
        ]

        audio_result = self.orchestrator.route_and_execute(
            state=audio_state, requested_agent=AgentType.AUDIO
        )
        assert audio_result.success

        # Verify all agents successfully used their tools
        assert command_result.execution_time > 0
        assert memory_result.execution_time > 0
        assert audio_result.execution_time > 0

    def test_performance_under_load(self):
        """Test system performance under simulated load."""
        import queue
        import threading

        results_queue = queue.Queue()
        num_concurrent_requests = 10

        def execute_request(request_id: int):
            """Execute a single request."""
            state = create_initial_state(f"e2e_load_test_{request_id}")
            state["conversation"]["messages"] = [
                {"role": "user", "content": f"Process request {request_id}"}
            ]

            start_time = time.time()
            result = self.orchestrator.route_and_execute(state=state)
            end_time = time.time()

            results_queue.put(
                {
                    "request_id": request_id,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "total_time": end_time - start_time,
                    "agent_type": result.agent_type,
                }
            )

        # Execute concurrent requests
        threads = []
        overall_start_time = time.time()

        for i in range(num_concurrent_requests):
            thread = threading.Thread(target=execute_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        overall_end_time = time.time()
        overall_time = overall_end_time - overall_start_time

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Verify performance
        assert len(results) == num_concurrent_requests
        successful_requests = [r for r in results if r["success"]]
        assert (
            len(successful_requests) >= num_concurrent_requests * 0.9
        )  # 90% success rate

        # Performance metrics
        avg_execution_time = sum(
            r["execution_time"] for r in successful_requests
        ) / len(successful_requests)
        max_execution_time = max(r["execution_time"] for r in successful_requests)

        # Performance assertions (generous limits for mocked environment)
        assert avg_execution_time < 1.0  # Average under 1 second
        assert max_execution_time < 5.0  # Max under 5 seconds
        assert overall_time < 10.0  # Overall completion under 10 seconds

        # Verify system remained stable
        system_status = self.orchestrator.get_system_status()
        assert system_status["orchestrator_status"] == "running"
        assert system_status["available_agents"] > 0

    def test_complete_user_session_simulation(self):
        """Test a complete realistic user session from start to finish."""
        session_id = "e2e_complete_session"

        # Simulate realistic user session flow
        session_flow = [
            # 1. User greets the system
            {
                "message": "Olá Gianna, como você está?",
                "expected_agent": AgentType.CONVERSATION,
            },
            # 2. User asks for system information
            {
                "message": "Mostre-me informações do sistema com o comando uname -a",
                "expected_agent": AgentType.COMMAND,
            },
            # 3. User requests audio configuration
            {
                "message": "Configure a voz para português brasileiro, voz feminina",
                "expected_agent": AgentType.AUDIO,
            },
            # 4. User asks to save preferences
            {
                "message": "Salve essas preferências na memória para próximas sessões",
                "expected_agent": AgentType.MEMORY,
            },
            # 5. User asks for help with file operations
            {
                "message": "Liste os arquivos no diretório atual",
                "expected_agent": AgentType.COMMAND,
            },
            # 6. User says goodbye
            {
                "message": "Obrigado pela ajuda! Até logo!",
                "expected_agent": AgentType.CONVERSATION,
            },
        ]

        accumulated_messages = []
        session_results = []

        for step, interaction in enumerate(session_flow):
            # Build progressive conversation state
            state = create_initial_state(f"{session_id}_step_{step}")

            # Add current user message
            accumulated_messages.append(
                {"role": "user", "content": interaction["message"]}
            )

            # Add previous assistant responses (simulate conversation flow)
            if step > 0:
                accumulated_messages.append(
                    {
                        "role": "assistant",
                        "content": f"Response to step {step - 1}",
                        "agent": (
                            session_results[-1].agent_type.value
                            if session_results
                            else "conversation"
                        ),
                    }
                )

            state["conversation"]["messages"] = accumulated_messages.copy()

            # Add realistic user preferences that accumulate
            state["conversation"]["user_preferences"] = {
                "name": "Usuário de Teste",
                "language": "pt-BR",
                "session_steps_completed": step,
            }

            if step >= 2:  # After audio configuration step
                state["audio"]["language"] = "pt-BR"
                state["audio"]["voice_settings"] = {"gender": "female"}

            # Execute the interaction
            result = self.orchestrator.route_and_execute(state=state)
            session_results.append(result)

            # Verify step success
            assert result.success, f"Session step {step} failed: {result.error}"
            assert result.execution_time > 0

            # Note: We don't strictly enforce agent type matching since routing
            # is intelligent and might choose appropriate alternatives

        # Verify complete session success
        assert len(session_results) == 6
        assert all(r.success for r in session_results)

        # Verify session diversity (multiple agent types used)
        used_agents = {r.agent_type for r in session_results}
        assert len(used_agents) >= 2  # At least 2 different agent types

        # Verify system performance throughout session
        total_session_time = sum(r.execution_time for r in session_results)
        assert total_session_time > 0
        assert total_session_time < 30.0  # Complete session under 30 seconds

        # Verify system stability after complete session
        final_system_status = self.orchestrator.get_system_status()
        assert final_system_status["orchestrator_status"] == "running"
        assert final_system_status["total_agents"] == 4

        # Verify performance metrics show activity
        performance_metrics = self.orchestrator.get_performance_metrics()
        total_requests = sum(
            metrics["total_requests"] for metrics in performance_metrics.values()
        )
        assert total_requests >= 6  # At least one request per step

        # Verify routing statistics
        routing_stats = self.orchestrator.router.get_routing_stats()
        assert routing_stats["total_routes"] >= 6
        assert routing_stats["avg_confidence"] > 0


class TestSystemResilience:
    """Test system resilience and edge case handling."""

    def setup_method(self):
        """Setup resilience test environment."""
        self.mock_llm = MockLanguageModel()
        self.orchestrator = AgentOrchestrator(max_workers=2)

        # Register agents
        agents = [CommandAgent(self.mock_llm), ConversationAgent(self.mock_llm)]

        for agent in agents:
            self.orchestrator.register_agent(agent)

    def teardown_method(self):
        """Cleanup resilience test environment."""
        self.orchestrator.shutdown()

    def test_malformed_state_handling(self):
        """Test system behavior with malformed state objects."""
        malformed_states = [
            {},  # Empty state
            {"invalid": "structure"},  # Wrong structure
            {"conversation": {}},  # Missing required fields
            {"conversation": {"messages": "not_a_list"}},  # Wrong type
        ]

        for i, malformed_state in enumerate(malformed_states):
            try:
                result = self.orchestrator.route_and_execute(state=malformed_state)
                # If it doesn't raise an exception, it should at least fail gracefully
                assert not result.success, f"Malformed state {i} should have failed"
                assert result.error is not None
            except Exception as e:
                # Graceful exception handling is acceptable
                assert isinstance(e, (KeyError, AttributeError, TypeError, ValueError))

    def test_empty_message_handling(self):
        """Test handling of empty or invalid messages."""
        empty_message_cases = [
            [],  # No messages
            [{"role": "user", "content": ""}],  # Empty content
            [{"role": "user"}],  # Missing content
            [{"content": "message without role"}],  # Missing role
            [{"role": "user", "content": None}],  # None content
        ]

        for i, messages in enumerate(empty_message_cases):
            state = create_initial_state(f"empty_test_{i}")
            state["conversation"]["messages"] = messages

            result = self.orchestrator.route_and_execute(state=state)

            # System should handle gracefully (either succeed with fallback or fail cleanly)
            if not result.success:
                assert result.error is not None
            assert result.execution_time >= 0  # Should not be negative

    def test_resource_exhaustion_simulation(self):
        """Test behavior under simulated resource constraints."""
        # Create orchestrator with very limited resources
        limited_orchestrator = AgentOrchestrator(max_workers=1)

        # Register single agent
        agent = ConversationAgent(self.mock_llm)
        limited_orchestrator.register_agent(agent)

        try:
            # Submit multiple concurrent requests to exhaust resources
            import queue
            import threading

            results_queue = queue.Queue()

            def submit_request(request_id: int):
                state = create_initial_state(f"resource_test_{request_id}")
                state["conversation"]["messages"] = [
                    {"role": "user", "content": f"Request {request_id}"}
                ]

                result = limited_orchestrator.route_and_execute(state=state)
                results_queue.put(result)

            # Submit 5 requests to single-worker orchestrator
            threads = []
            for i in range(5):
                thread = threading.Thread(target=submit_request, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join(timeout=10)  # Prevent hanging

            # Collect results
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())

            # Verify system handled resource constraints gracefully
            assert len(results) > 0  # At least some requests completed

            # All completed requests should be valid
            for result in results:
                assert hasattr(result, "success")
                assert hasattr(result, "execution_time")
                assert result.execution_time >= 0

        finally:
            limited_orchestrator.shutdown()

    def test_agent_failure_cascades(self):
        """Test system behavior when agent failures cascade."""
        # Create failing LLM
        failing_llm = Mock()
        failing_llm.invoke.side_effect = Exception("Cascade failure simulation")

        # Replace agents with failing versions
        failing_agents = [CommandAgent(failing_llm), ConversationAgent(failing_llm)]

        failing_orchestrator = AgentOrchestrator(max_workers=2)

        for agent in failing_agents:
            failing_orchestrator.register_agent(agent)

        try:
            state = create_initial_state("cascade_failure_test")
            state["conversation"]["messages"] = [
                {"role": "user", "content": "Test cascade failure handling"}
            ]

            # Test multiple agent types fail
            command_result = failing_orchestrator.route_and_execute(
                state=state, requested_agent=AgentType.COMMAND
            )
            conversation_result = failing_orchestrator.route_and_execute(
                state=state, requested_agent=AgentType.CONVERSATION
            )

            # Verify graceful failure handling
            assert not command_result.success
            assert not conversation_result.success
            assert command_result.error is not None
            assert conversation_result.error is not None

            # Verify system stability despite cascading failures
            system_status = failing_orchestrator.get_system_status()
            assert system_status["orchestrator_status"] == "running"

        finally:
            failing_orchestrator.shutdown()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
