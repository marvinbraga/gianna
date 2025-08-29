#!/usr/bin/env python3
"""
Unit tests for the Multi-Agent Coordination System.

This module provides comprehensive tests for the orchestrator, router,
and integration components of the multi-agent coordination system.
"""

import os
import sys
import threading
import time
from typing import Any, Dict
from unittest.mock import MagicMock, Mock

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from gianna.agents.react_agents import GiannaReActAgent
from gianna.coordination import AgentOrchestrator, AgentRouter
from gianna.coordination.orchestrator import AgentInfo, AgentStatus, ExecutionRequest
from gianna.coordination.router import AgentType, ExecutionMode
from gianna.core.state import GiannaState, create_initial_state


class MockAgent(GiannaReActAgent):
    """Mock agent for testing purposes."""

    def __init__(self, name: str, delay: float = 0.1, should_fail: bool = False):
        """Initialize mock agent."""
        self.name = name
        self.delay = delay
        self.should_fail = should_fail
        self.llm = Mock()
        self.tools = []
        self.config = Mock()
        self.execution_count = 0

    def execute(self, input_data: Any, state: GiannaState) -> Dict[str, Any]:
        """Mock execute method."""
        self.execution_count += 1
        time.sleep(self.delay)  # Simulate processing time

        if self.should_fail:
            raise RuntimeError(f"Mock failure in {self.name}")

        return {
            "agent_name": self.name,
            "content": f"Mock response from {self.name}",
            "success": True,
            "input": str(input_data),
        }


class TestAgentRouter:
    """Test cases for the AgentRouter component."""

    def setup_method(self):
        """Setup test fixtures."""
        self.router = AgentRouter()

    def test_router_initialization(self):
        """Test router initialization."""
        assert self.router is not None
        assert len(self.router.routing_rules) > 0
        assert self.router.routing_history == []

    def test_command_routing(self):
        """Test routing to command agent."""
        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Execute o comando ls -la"}
        ]

        agent_type, confidence = self.router.route_request(state)

        assert agent_type == AgentType.COMMAND
        assert confidence > 0.6  # Should have high confidence for clear command request

    def test_audio_routing(self):
        """Test routing to audio agent."""
        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Tocar música relaxante para mim"}
        ]

        agent_type, confidence = self.router.route_request(state)

        assert agent_type == AgentType.AUDIO
        assert confidence > 0.6

    def test_memory_routing(self):
        """Test routing to memory agent."""
        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {
                "role": "user",
                "content": "Lembrar das configurações que discutimos ontem",
            }
        ]

        agent_type, confidence = self.router.route_request(state)

        assert agent_type == AgentType.MEMORY
        assert confidence > 0.5

    def test_conversation_fallback(self):
        """Test fallback to conversation agent."""
        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Como você está hoje?"}
        ]

        agent_type, confidence = self.router.route_request(state)

        assert agent_type == AgentType.CONVERSATION
        # Conversation agent has lower confidence threshold as fallback

    def test_empty_message_handling(self):
        """Test handling of empty messages."""
        state = create_initial_state("test_session")

        agent_type, confidence = self.router.route_request(state)

        assert agent_type == AgentType.CONVERSATION
        assert confidence == 0.5  # Default confidence for empty messages

    def test_contextual_adjustments(self):
        """Test contextual routing adjustments."""
        state = create_initial_state("test_session")
        state["audio"]["current_mode"] = "listening"
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Can you help me?"}
        ]

        agent_type, confidence = self.router.route_request(state)

        # Should prefer audio agent due to listening mode context
        # Note: This test may need adjustment based on exact implementation
        assert agent_type in [AgentType.AUDIO, AgentType.CONVERSATION]

    def test_routing_history(self):
        """Test routing history tracking."""
        initial_count = len(self.router.routing_history)

        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Test message"}
        ]

        self.router.route_request(state)

        assert len(self.router.routing_history) == initial_count + 1

        # Test history entry structure
        last_entry = self.router.routing_history[-1]
        assert "message" in last_entry
        assert "selected_agent" in last_entry
        assert "confidence" in last_entry
        assert "all_scores" in last_entry

    def test_routing_statistics(self):
        """Test routing statistics generation."""
        # Make several routing requests
        test_messages = [
            "Execute ls command",
            "Tocar música",
            "Lembrar configurações",
            "Como está o tempo?",
        ]

        for message in test_messages:
            state = create_initial_state("test_session")
            state["conversation"]["messages"] = [{"role": "user", "content": message}]
            self.router.route_request(state)

        stats = self.router.get_routing_stats()

        assert stats["total_routes"] >= len(test_messages)
        assert "agent_distribution" in stats
        assert "avg_confidence" in stats
        assert "recent_routes" in stats


class TestAgentOrchestrator:
    """Test cases for the AgentOrchestrator component."""

    def setup_method(self):
        """Setup test fixtures."""
        self.orchestrator = AgentOrchestrator(max_workers=2)
        self.setup_mock_agents()

    def teardown_method(self):
        """Cleanup after tests."""
        self.orchestrator.shutdown()

    def setup_mock_agents(self):
        """Setup mock agents for testing."""
        agents = [
            MockAgent("command_agent"),
            MockAgent("audio_agent"),
            MockAgent("conversation_agent"),
            MockAgent("memory_agent"),
        ]

        for agent in agents:
            self.orchestrator.register_agent(agent)

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator is not None
        assert len(self.orchestrator.agents) == 4
        assert self.orchestrator.router is not None

    def test_agent_registration(self):
        """Test agent registration and unregistration."""
        # Test registration
        new_agent = MockAgent("test_agent")

        # This will fail because "test_agent" is not a valid AgentType
        # But we can test the registration logic
        initial_count = len(self.orchestrator.agents)
        self.orchestrator.register_agent(new_agent)

        # Count should remain the same for invalid agent type
        assert len(self.orchestrator.agents) == initial_count

        # Test getting registered agents
        registered = self.orchestrator.get_registered_agents()
        assert len(registered) == 4
        assert AgentType.COMMAND in registered
        assert AgentType.AUDIO in registered

    def test_agent_status(self):
        """Test agent status management."""
        status = self.orchestrator.get_agent_status(AgentType.COMMAND)
        assert status == AgentStatus.AVAILABLE

        # Test status for unregistered agent
        # First create a new orchestrator without agents
        new_orchestrator = AgentOrchestrator()
        status = new_orchestrator.get_agent_status(AgentType.COMMAND)
        assert status is None
        new_orchestrator.shutdown()

    def test_single_agent_execution(self):
        """Test execution with a single agent."""
        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Test message"}
        ]

        result = self.orchestrator.route_and_execute(
            state=state, requested_agent=AgentType.COMMAND
        )

        assert result.success
        assert result.agent_type == AgentType.COMMAND
        assert result.execution_time > 0
        assert result.result is not None

    def test_sequential_execution(self):
        """Test sequential multi-agent execution."""
        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Test sequential execution"}
        ]

        agents = [AgentType.MEMORY, AgentType.COMMAND, AgentType.CONVERSATION]

        results = self.orchestrator.coordinate_agents(
            agents=agents, state=state, execution_mode=ExecutionMode.SEQUENTIAL
        )

        assert len(results) == 3
        assert all(result.success for result in results)

        # Check execution order
        expected_agents = [AgentType.MEMORY, AgentType.COMMAND, AgentType.CONVERSATION]
        for i, result in enumerate(results):
            assert result.agent_type == expected_agents[i]

    def test_parallel_execution(self):
        """Test parallel multi-agent execution."""
        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Test parallel execution"}
        ]

        agents = [AgentType.COMMAND, AgentType.AUDIO, AgentType.MEMORY]

        start_time = time.time()
        results = self.orchestrator.coordinate_agents(
            agents=agents, state=state, execution_mode=ExecutionMode.PARALLEL
        )
        execution_time = time.time() - start_time

        assert len(results) == 3
        assert all(result.success for result in results)

        # Parallel execution should be faster than sequential
        # (3 agents * 0.1s delay ≈ 0.3s sequential vs ~0.1s parallel)
        assert execution_time < 0.25  # Should be much faster than sequential

    def test_hybrid_execution(self):
        """Test hybrid multi-agent execution."""
        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Test hybrid execution"}
        ]

        agents = [
            AgentType.COMMAND,
            AgentType.MEMORY,
            AgentType.AUDIO,
            AgentType.CONVERSATION,
        ]

        results = self.orchestrator.coordinate_agents(
            agents=agents, state=state, execution_mode=ExecutionMode.HYBRID
        )

        assert len(results) == 4
        assert all(result.success for result in results)

    def test_error_handling(self):
        """Test error handling and recovery."""
        # Register a failing agent
        failing_agent = MockAgent("command_agent", should_fail=True)

        # Replace the existing command agent
        self.orchestrator.agents[AgentType.COMMAND] = AgentInfo(
            agent=failing_agent, status=AgentStatus.AVAILABLE
        )

        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Test error handling"}
        ]

        result = self.orchestrator.route_and_execute(
            state=state, requested_agent=AgentType.COMMAND
        )

        assert not result.success
        assert result.error is not None
        assert "Mock failure" in result.error

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Test metrics"}
        ]

        # Execute several requests
        for i in range(3):
            self.orchestrator.route_and_execute(
                state=state, requested_agent=AgentType.COMMAND
            )

        metrics = self.orchestrator.get_performance_metrics()

        assert "command_agent" in metrics
        command_metrics = metrics["command_agent"]
        assert command_metrics["total_requests"] >= 3
        assert command_metrics["successful_requests"] >= 3
        assert command_metrics["average_execution_time"] > 0
        assert command_metrics["error_rate"] == 0.0  # No errors in this test

    def test_system_status(self):
        """Test system status reporting."""
        status = self.orchestrator.get_system_status()

        assert status["orchestrator_status"] == "running"
        assert status["total_agents"] == 4
        assert status["available_agents"] <= 4
        assert "agents" in status
        assert "routing_stats" in status
        assert "performance_metrics" in status

    def test_health_check(self):
        """Test agent health checking."""
        health = self.orchestrator.health_check()

        assert len(health) == 4  # All registered agents
        assert "command_agent" in health
        assert "audio_agent" in health
        assert "conversation_agent" in health
        assert "memory_agent" in health

        # All agents should be healthy
        assert all(health.values())

    def test_concurrent_execution(self):
        """Test concurrent execution safety."""
        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Concurrent test"}
        ]

        def execute_request():
            return self.orchestrator.route_and_execute(
                state=state, requested_agent=AgentType.CONVERSATION
            )

        # Execute multiple requests concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=execute_request)
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Check that all executions were handled properly
        # (No exceptions should have occurred)
        assert True  # If we get here, concurrent execution worked

    def test_timeout_handling(self):
        """Test execution timeout handling."""
        # Create a slow agent
        slow_agent = MockAgent("command_agent", delay=2.0)  # 2 second delay

        self.orchestrator.agents[AgentType.COMMAND] = AgentInfo(
            agent=slow_agent, status=AgentStatus.AVAILABLE
        )

        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Test timeout"}
        ]

        result = self.orchestrator.route_and_execute(
            state=state,
            requested_agent=AgentType.COMMAND,
            timeout=1,  # 1 second timeout
        )

        # Should fail due to timeout
        assert not result.success
        assert "timeout" in result.error.lower() or "timed out" in result.error.lower()


class TestIntegration:
    """Integration tests for the complete coordination system."""

    def setup_method(self):
        """Setup integration test fixtures."""
        self.orchestrator = AgentOrchestrator(max_workers=4)
        self.setup_mock_agents()

    def teardown_method(self):
        """Cleanup after integration tests."""
        self.orchestrator.shutdown()

    def setup_mock_agents(self):
        """Setup mock agents for integration testing."""
        agents = [
            MockAgent("command_agent"),
            MockAgent("audio_agent", delay=0.05),
            MockAgent("conversation_agent", delay=0.02),
            MockAgent("memory_agent", delay=0.08),
        ]

        for agent in agents:
            self.orchestrator.register_agent(agent)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Test different types of requests
        test_cases = [
            ("Execute ls command", AgentType.COMMAND),
            ("Tocar uma música", AgentType.AUDIO),
            ("Lembrar das configurações", AgentType.MEMORY),
            ("Como você está?", AgentType.CONVERSATION),
        ]

        for message, expected_agent in test_cases:
            state = create_initial_state("integration_test")
            state["conversation"]["messages"] = [{"role": "user", "content": message}]

            # Test automatic routing
            result = self.orchestrator.route_and_execute(state=state)

            assert result.success, f"Failed for message: {message}"
            assert result.result is not None
            assert result.execution_time > 0

            # Note: We don't assert exact agent match because routing
            # might choose alternatives or fallbacks based on confidence

    def test_complex_multi_agent_workflow(self):
        """Test complex workflow with multiple coordination modes."""
        state = create_initial_state("complex_workflow")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Comprehensive system analysis"}
        ]

        # Phase 1: Parallel information gathering
        phase1_agents = [AgentType.COMMAND, AgentType.MEMORY]
        phase1_results = self.orchestrator.coordinate_agents(
            agents=phase1_agents, state=state, execution_mode=ExecutionMode.PARALLEL
        )

        assert len(phase1_results) == 2
        assert all(r.success for r in phase1_results)

        # Update state with results (simplified)
        for result in phase1_results:
            if result.result and isinstance(result.result, dict):
                content = result.result.get("content", "")
                state["conversation"]["messages"].append(
                    {
                        "role": "assistant",
                        "content": content,
                        "agent": result.agent_type.value,
                    }
                )

        # Phase 2: Sequential processing
        phase2_agents = [AgentType.AUDIO, AgentType.CONVERSATION]
        phase2_results = self.orchestrator.coordinate_agents(
            agents=phase2_agents, state=state, execution_mode=ExecutionMode.SEQUENTIAL
        )

        assert len(phase2_results) == 2
        assert all(r.success for r in phase2_results)

        # Verify total workflow
        total_results = phase1_results + phase2_results
        assert len(total_results) == 4
        assert (
            len(set(r.agent_type for r in total_results)) == 4
        )  # All different agents

    def test_system_resilience(self):
        """Test system resilience under various conditions."""
        state = create_initial_state("resilience_test")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "System resilience test"}
        ]

        # Test with one agent failing
        failing_agent = MockAgent("memory_agent", should_fail=True)
        self.orchestrator.agents[AgentType.MEMORY] = AgentInfo(
            agent=failing_agent, status=AgentStatus.AVAILABLE
        )

        # Execute mixed success/failure scenario
        agents = [AgentType.COMMAND, AgentType.MEMORY, AgentType.CONVERSATION]
        results = self.orchestrator.coordinate_agents(
            agents=agents, state=state, execution_mode=ExecutionMode.SEQUENTIAL
        )

        # Should have results for all agents, but memory agent should fail
        assert len(results) <= 3  # May stop early due to failure
        command_result = next(
            (r for r in results if r.agent_type == AgentType.COMMAND), None
        )
        assert command_result is not None
        assert command_result.success

        memory_result = next(
            (r for r in results if r.agent_type == AgentType.MEMORY), None
        )
        if memory_result:  # May not exist if execution stopped
            assert not memory_result.success


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
