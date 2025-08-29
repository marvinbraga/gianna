#!/usr/bin/env python3
"""
Multi-Agent Coordination System Example

This example demonstrates how to use the Gianna Multi-Agent Coordination System
with intelligent routing, agent orchestration, and workflow management.
"""

import asyncio
import os
import sys
from typing import Any, Dict

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from langchain_openai import ChatOpenAI
from loguru import logger

from gianna.agents.react_agents import (
    AudioAgent,
    CommandAgent,
    ConversationAgent,
    MemoryAgent,
)
from gianna.coordination import AgentOrchestrator, AgentRouter
from gianna.coordination.orchestrator import ExecutionMode
from gianna.coordination.router import AgentType
from gianna.core.state import GiannaState, create_initial_state


class MultiAgentCoordinationDemo:
    """Demonstration of multi-agent coordination capabilities."""

    def __init__(self):
        """Initialize the demonstration with mock LLM and agents."""
        self.orchestrator = AgentOrchestrator(max_workers=4)
        self.setup_agents()

    def setup_agents(self):
        """Setup and register all specialized agents."""
        # For demo purposes, we'll use a mock LLM
        # In production, this would be a real LLM instance
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        except Exception:
            # Fallback to mock if no API key available
            llm = MockLLM()

        # Create and register agents
        agents = [
            CommandAgent(llm),
            AudioAgent(llm),
            ConversationAgent(llm),
            MemoryAgent(llm),
        ]

        for agent in agents:
            self.orchestrator.register_agent(agent)

        logger.info(f"Registered {len(agents)} agents with orchestrator")

    def demo_intelligent_routing(self):
        """Demonstrate intelligent routing based on Portuguese keywords."""
        print("\n🧠 DEMONSTRATING INTELLIGENT ROUTING")
        print("=" * 50)

        test_messages = [
            "Execute o comando ls -la no terminal",
            "Fale sobre os benefícios da inteligência artificial",
            "Toque uma música relaxante para mim",
            "Lembrar da nossa conversa sobre Python",
            "Como posso configurar o Git no meu sistema?",
            "Preciso gravar um áudio com minha voz",
            "Qual foi o último comando que executamos?",
        ]

        for message in test_messages:
            state = self.create_test_state(message)

            # Get routing decision
            router = AgentRouter()
            agent_type, confidence = router.route_request(state)

            print(f"📝 Message: '{message}'")
            print(f"🎯 Routed to: {agent_type.value}")
            print(f"📊 Confidence: {confidence:.2f}")
            print("-" * 30)

    def demo_sequential_execution(self):
        """Demonstrate sequential multi-agent execution."""
        print("\n🔄 DEMONSTRATING SEQUENTIAL EXECUTION")
        print("=" * 50)

        # Simulate a complex workflow requiring multiple agents
        state = self.create_test_state(
            "Lembrar das configurações de áudio e depois tocar música"
        )

        # Define agent sequence
        agent_sequence = [
            AgentType.MEMORY,  # First retrieve audio preferences
            AgentType.AUDIO,  # Then handle audio playback
            AgentType.CONVERSATION,  # Finally provide user feedback
        ]

        print(f"🎼 Executing sequence: {' → '.join([a.value for a in agent_sequence])}")

        try:
            results = self.orchestrator.coordinate_agents(
                agents=agent_sequence,
                state=state,
                execution_mode=ExecutionMode.SEQUENTIAL,
            )

            print(f"✅ Sequential execution completed with {len(results)} results")
            for i, result in enumerate(results):
                print(
                    f"  {i+1}. {result.agent_type.value}: {'✅' if result.success else '❌'} "
                    f"({result.execution_time:.2f}s)"
                )

        except Exception as e:
            print(f"❌ Sequential execution failed: {e}")

    def demo_parallel_execution(self):
        """Demonstrate parallel multi-agent execution."""
        print("\n⚡ DEMONSTRATING PARALLEL EXECUTION")
        print("=" * 50)

        # Simulate independent operations that can run in parallel
        state = self.create_test_state(
            "Análise geral do sistema e preparação de relatório"
        )

        # Define agents for parallel execution
        parallel_agents = [
            AgentType.COMMAND,  # System status check
            AgentType.MEMORY,  # Context analysis
            AgentType.CONVERSATION,  # Report preparation
        ]

        print(
            f"⚡ Executing in parallel: {' | '.join([a.value for a in parallel_agents])}"
        )

        try:
            results = self.orchestrator.coordinate_agents(
                agents=parallel_agents,
                state=state,
                execution_mode=ExecutionMode.PARALLEL,
            )

            print(f"✅ Parallel execution completed with {len(results)} results")
            for result in results:
                print(
                    f"  • {result.agent_type.value}: {'✅' if result.success else '❌'} "
                    f"({result.execution_time:.2f}s)"
                )

        except Exception as e:
            print(f"❌ Parallel execution failed: {e}")

    def demo_hybrid_execution(self):
        """Demonstrate hybrid execution mode."""
        print("\n🔀 DEMONSTRATING HYBRID EXECUTION")
        print("=" * 50)

        state = self.create_test_state(
            "Configurar sistema, preparar áudio e iniciar conversa"
        )

        # Mix of agents requiring different execution strategies
        hybrid_agents = [
            AgentType.COMMAND,  # Must run first (high priority)
            AgentType.MEMORY,  # Must run first (high priority)
            AgentType.AUDIO,  # Can run parallel (medium priority)
            AgentType.CONVERSATION,  # Should run last (low priority)
        ]

        print(f"🔀 Executing in hybrid mode with {len(hybrid_agents)} agents")

        try:
            results = self.orchestrator.coordinate_agents(
                agents=hybrid_agents, state=state, execution_mode=ExecutionMode.HYBRID
            )

            print(f"✅ Hybrid execution completed with {len(results)} results")

            # Group results by priority for display
            priority_groups = {
                "High Priority": [AgentType.COMMAND, AgentType.MEMORY],
                "Medium Priority": [AgentType.AUDIO],
                "Low Priority": [AgentType.CONVERSATION],
            }

            for group_name, group_agents in priority_groups.items():
                group_results = [r for r in results if r.agent_type in group_agents]
                if group_results:
                    print(f"  {group_name}:")
                    for result in group_results:
                        print(
                            f"    • {result.agent_type.value}: {'✅' if result.success else '❌'}"
                        )

        except Exception as e:
            print(f"❌ Hybrid execution failed: {e}")

    def demo_error_recovery(self):
        """Demonstrate error recovery and alternative routing."""
        print("\n🔧 DEMONSTRATING ERROR RECOVERY")
        print("=" * 50)

        # Simulate agent failure scenario
        print("📋 System Status Before Error Simulation:")
        self.print_system_status()

        # Manually set an agent to error state for demonstration
        with self.orchestrator._lock:
            if AgentType.COMMAND in self.orchestrator.agents:
                self.orchestrator.agents[AgentType.COMMAND].status = (
                    self.orchestrator.agents[AgentType.COMMAND].__class__.__dict__.get(
                        "ERROR", "error"
                    )
                )
                print("❌ Simulated CommandAgent failure")

        # Try to execute a command that would normally go to CommandAgent
        state = self.create_test_state("Execute ls command")

        try:
            result = self.orchestrator.route_and_execute(
                state=state, requested_agent=AgentType.COMMAND
            )

            print(
                f"🔄 Fallback result: Agent={result.agent_type.value}, Success={result.success}"
            )

        except Exception as e:
            print(f"❌ Error recovery failed: {e}")

        # Reset agent errors
        self.orchestrator.reset_agent_errors()
        print("✅ Agent errors reset")

    def demo_performance_monitoring(self):
        """Demonstrate performance monitoring and metrics."""
        print("\n📊 DEMONSTRATING PERFORMANCE MONITORING")
        print("=" * 50)

        # Execute several requests to generate metrics
        test_requests = [
            ("Olá, como você está?", AgentType.CONVERSATION),
            ("Execute pwd", AgentType.COMMAND),
            ("Tocar música", AgentType.AUDIO),
            ("Lembrar preferências", AgentType.MEMORY),
        ]

        print("🏃 Executing test requests to generate metrics...")
        for message, agent_type in test_requests:
            state = self.create_test_state(message)
            try:
                self.orchestrator.route_and_execute(
                    state=state, requested_agent=agent_type
                )
            except Exception as e:
                print(f"  ⚠️  Request failed: {e}")

        # Display performance metrics
        print("\n📈 Performance Metrics:")
        metrics = self.orchestrator.get_performance_metrics()
        for agent_name, agent_metrics in metrics.items():
            print(f"  {agent_name}:")
            print(f"    • Total Requests: {agent_metrics.get('total_requests', 0)}")
            print(f"    • Success Rate: {agent_metrics.get('error_rate', 0):.1%}")
            print(
                f"    • Avg Execution Time: {agent_metrics.get('average_execution_time', 0):.2f}s"
            )

        # Display routing statistics
        print("\n🎯 Routing Statistics:")
        routing_stats = self.orchestrator.router.get_routing_stats()
        print(f"  • Total Routes: {routing_stats.get('total_routes', 0)}")
        print(f"  • Average Confidence: {routing_stats.get('avg_confidence', 0):.2f}")

        agent_dist = routing_stats.get("agent_distribution", {})
        if agent_dist:
            print("  • Agent Distribution:")
            for agent, percentage in agent_dist.items():
                print(f"    - {agent}: {percentage:.1%}")

    def print_system_status(self):
        """Print comprehensive system status."""
        status = self.orchestrator.get_system_status()

        print(f"🔍 Orchestrator Status: {status['orchestrator_status']}")
        print(
            f"📊 Agents - Total: {status['total_agents']}, "
            f"Available: {status['available_agents']}, "
            f"Busy: {status['busy_agents']}, "
            f"Error: {status['error_agents']}"
        )

        print("🤖 Individual Agent Status:")
        for agent_name, agent_info in status["agents"].items():
            print(
                f"  • {agent_name}: {agent_info['status']} "
                f"(executions: {agent_info['total_executions']}, "
                f"success: {agent_info['success_rate']:.1%})"
            )

    def create_test_state(self, message: str) -> GiannaState:
        """
        Create a test state with a user message.

        Args:
            message: User message content

        Returns:
            GiannaState: Test state with the message
        """
        state = create_initial_state("demo_session")
        state["conversation"].messages = [{"role": "user", "content": message}]
        return state

    def run_full_demo(self):
        """Run the complete demonstration."""
        print("🚀 GIANNA MULTI-AGENT COORDINATION SYSTEM DEMO")
        print("=" * 60)

        try:
            # Run all demonstrations
            self.demo_intelligent_routing()
            self.demo_sequential_execution()
            self.demo_parallel_execution()
            self.demo_hybrid_execution()
            self.demo_error_recovery()
            self.demo_performance_monitoring()

            print(f"\n✅ DEMONSTRATION COMPLETED")
            print("🔍 Final System Status:")
            self.print_system_status()

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")

        finally:
            # Cleanup
            print("\n🧹 Cleaning up resources...")
            self.orchestrator.shutdown()
            print("✅ Cleanup completed")


class MockLLM:
    """Mock LLM for demonstration when real LLM is not available."""

    def invoke(self, prompt: str) -> str:
        """Mock invoke method."""
        return f"Mock response to: {prompt[:50]}..."

    def __call__(self, prompt: str) -> str:
        """Make the mock callable."""
        return self.invoke(prompt)


def main():
    """Run the multi-agent coordination demonstration."""
    demo = MultiAgentCoordinationDemo()
    demo.run_full_demo()


if __name__ == "__main__":
    main()
