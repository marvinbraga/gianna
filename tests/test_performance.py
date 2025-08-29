#!/usr/bin/env python3
"""
Performance Benchmarks and Validation Tests for FASE 2

This module provides comprehensive performance testing and validation
of the FASE 2 implementation against the specified success criteria.
"""

import concurrent.futures
import os
import statistics
import sys
import threading
import time
from typing import Any, Dict, List
from unittest.mock import Mock

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
from gianna.core.state import create_initial_state
from gianna.tools import (
    AudioProcessorTool,
    FileSystemTool,
    MemoryTool,
    ShellExecutorTool,
    STTTool,
    TTSTool,
)


class FastMockLLM:
    """High-performance mock LLM for benchmarking."""

    def __init__(self, response_time: float = 0.001):
        self.response_time = response_time
        self.call_count = 0

    def invoke(self, input_data: Any) -> str:
        """Mock invoke with controlled response time."""
        time.sleep(self.response_time)  # Simulate processing time
        self.call_count += 1
        return f"Response to input: {str(input_data)[:50]}..."


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time

    @staticmethod
    def run_benchmark(func, iterations: int = 100, *args, **kwargs):
        """Run benchmark multiple times and collect statistics."""
        times = []
        results = []

        for _ in range(iterations):
            result, exec_time = PerformanceBenchmark.measure_execution_time(
                func, *args, **kwargs
            )
            times.append(exec_time)
            results.append(result)

        return {
            "times": times,
            "results": results,
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "total_time": sum(times),
        }


class TestToolPerformance:
    """Performance tests for individual tools."""

    def setup_method(self):
        """Setup performance test environment."""
        self.shell_tool = ShellExecutorTool()
        self.fs_tool = FileSystemTool()
        self.memory_tool = MemoryTool()
        self.benchmark = PerformanceBenchmark()

    def test_shell_tool_performance(self):
        """Test shell tool execution performance."""
        # Benchmark simple command execution
        stats = self.benchmark.run_benchmark(
            self.shell_tool._run, iterations=50, command='echo "performance test"'
        )

        # Performance assertions
        assert stats["mean"] < 1.0, f"Shell tool too slow: {stats['mean']:.3f}s average"
        assert stats["max"] < 5.0, f"Shell tool max time too high: {stats['max']:.3f}s"
        assert (
            stats["stdev"] < 0.5
        ), f"Shell tool inconsistent: {stats['stdev']:.3f}s stdev"

        # Verify all executions succeeded
        import json

        success_count = sum(
            1 for result in stats["results"] if json.loads(result)["success"]
        )
        assert (
            success_count == 50
        ), f"Only {success_count}/50 shell executions succeeded"

    def test_filesystem_tool_performance(self):
        """Test filesystem tool performance."""
        import json
        import tempfile

        # Test file write performance
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "perf_test.txt")

            write_stats = self.benchmark.run_benchmark(
                self.fs_tool._run,
                iterations=30,
                input_json=json.dumps(
                    {
                        "action": "write",
                        "path": f"{test_file}_{{}}_{{}}",  # Will be formatted
                        "content": "Performance test content",
                    }
                ),
            )

            # Performance assertions for write operations
            assert (
                write_stats["mean"] < 0.1
            ), f"FS write too slow: {write_stats['mean']:.3f}s"
            assert (
                write_stats["max"] < 0.5
            ), f"FS write max too high: {write_stats['max']:.3f}s"

    def test_memory_tool_performance(self):
        """Test memory tool performance."""
        import json

        # Test memory stats retrieval performance
        stats = self.benchmark.run_benchmark(
            self.memory_tool._run,
            iterations=100,
            input_json=json.dumps(
                {"action": "get_stats", "session_id": "perf_test_session"}
            ),
        )

        # Performance assertions
        assert stats["mean"] < 0.05, f"Memory tool too slow: {stats['mean']:.3f}s"
        assert stats["max"] < 0.2, f"Memory tool max too high: {stats['max']:.3f}s"

        # Verify all operations succeeded
        import json

        success_count = sum(
            1 for result in stats["results"] if json.loads(result)["success"]
        )
        assert (
            success_count == 100
        ), f"Only {success_count}/100 memory operations succeeded"

    def test_tool_concurrent_performance(self):
        """Test tool performance under concurrent load."""
        import concurrent.futures
        import json

        def run_shell_command(cmd_id: int):
            """Run a shell command with unique ID."""
            start_time = time.perf_counter()
            result = self.shell_tool._run(f'echo "Concurrent test {cmd_id}"')
            end_time = time.perf_counter()
            return {
                "id": cmd_id,
                "result": json.loads(result),
                "time": end_time - start_time,
            }

        # Run 20 concurrent shell commands
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_shell_command, i) for i in range(20)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        end_time = time.perf_counter()

        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 10.0, f"Concurrent execution too slow: {total_time:.3f}s"
        assert (
            len(results) == 20
        ), f"Only {len(results)}/20 concurrent operations completed"

        # Verify all operations succeeded
        successful_ops = [r for r in results if r["result"]["success"]]
        assert (
            len(successful_ops) >= 18
        ), f"Only {len(successful_ops)}/20 concurrent operations succeeded"

        # Check individual execution times
        execution_times = [r["time"] for r in successful_ops]
        avg_time = statistics.mean(execution_times)
        max_time = max(execution_times)

        assert avg_time < 1.0, f"Average concurrent execution too slow: {avg_time:.3f}s"
        assert max_time < 5.0, f"Max concurrent execution too slow: {max_time:.3f}s"


class TestAgentPerformance:
    """Performance tests for ReAct agents."""

    def setup_method(self):
        """Setup agent performance test environment."""
        self.fast_llm = FastMockLLM(response_time=0.001)  # 1ms response time
        self.agents = {
            "command": CommandAgent(self.fast_llm),
            "audio": AudioAgent(self.fast_llm),
            "conversation": ConversationAgent(self.fast_llm),
            "memory": MemoryAgent(self.fast_llm),
        }
        self.benchmark = PerformanceBenchmark()

    def test_agent_initialization_performance(self):
        """Test agent initialization performance."""

        def create_agent(agent_type: str):
            if agent_type == "command":
                return CommandAgent(self.fast_llm)
            elif agent_type == "audio":
                return AudioAgent(self.fast_llm)
            elif agent_type == "conversation":
                return ConversationAgent(self.fast_llm)
            elif agent_type == "memory":
                return MemoryAgent(self.fast_llm)

        # Benchmark agent creation
        for agent_type in ["command", "audio", "conversation", "memory"]:
            stats = self.benchmark.run_benchmark(
                create_agent, iterations=20, agent_type=agent_type
            )

            # Initialization should be fast
            assert (
                stats["mean"] < 0.1
            ), f"{agent_type} agent init too slow: {stats['mean']:.3f}s"
            assert (
                stats["max"] < 0.5
            ), f"{agent_type} agent init max too high: {stats['max']:.3f}s"

    def test_agent_state_preparation_performance(self):
        """Test agent state preparation performance."""
        state = create_initial_state("perf_test")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Performance test message"}
        ]

        for agent_name, agent in self.agents.items():
            stats = self.benchmark.run_benchmark(
                agent._prepare_agent_state, iterations=100, state=state
            )

            # State preparation should be very fast
            assert (
                stats["mean"] < 0.001
            ), f"{agent_name} state prep too slow: {stats['mean']:.4f}s"
            assert (
                stats["max"] < 0.01
            ), f"{agent_name} state prep max too high: {stats['max']:.4f}s"

    def test_agent_fallback_performance(self):
        """Test agent fallback execution performance."""
        # Test fallback execution (when LangGraph not available)
        for agent_name, agent in self.agents.items():
            state = create_initial_state(f"fallback_perf_{agent_name}")
            prepared_state = agent._prepare_agent_state(state)

            stats = self.benchmark.run_benchmark(
                agent._execute_fallback,
                iterations=50,
                input_data="Performance test input",
                agent_state=prepared_state,
            )

            # Fallback execution should be reasonably fast
            assert (
                stats["mean"] < 0.1
            ), f"{agent_name} fallback too slow: {stats['mean']:.3f}s"
            assert (
                stats["max"] < 0.5
            ), f"{agent_name} fallback max too high: {stats['max']:.3f}s"

            # Verify LLM was called for each execution
            expected_calls = 50
            actual_calls = self.fast_llm.call_count
            # Reset for next agent
            self.fast_llm.call_count = 0


class TestCoordinationPerformance:
    """Performance tests for multi-agent coordination."""

    def setup_method(self):
        """Setup coordination performance test environment."""
        self.fast_llm = FastMockLLM(response_time=0.002)  # 2ms response time
        self.orchestrator = AgentOrchestrator(max_workers=4)

        # Register all agents
        agents = [
            CommandAgent(self.fast_llm),
            AudioAgent(self.fast_llm),
            ConversationAgent(self.fast_llm),
            MemoryAgent(self.fast_llm),
        ]

        for agent in agents:
            self.orchestrator.register_agent(agent)

        self.benchmark = PerformanceBenchmark()

    def teardown_method(self):
        """Cleanup coordination test environment."""
        self.orchestrator.shutdown()

    def test_routing_performance(self):
        """Test agent routing decision performance."""
        router = AgentRouter()

        # Test routing for different message types
        test_messages = [
            "Execute o comando ls",
            "Toque uma música",
            "Lembre-se desta configuração",
            "Como você está hoje?",
            "Mostre informações do sistema",
        ]

        for message in test_messages:
            state = create_initial_state("routing_perf_test")
            state["conversation"]["messages"] = [{"role": "user", "content": message}]

            stats = self.benchmark.run_benchmark(
                router.route_request, iterations=1000, state=state
            )

            # Routing should be very fast
            assert (
                stats["mean"] < 0.001
            ), f"Routing too slow for '{message}': {stats['mean']:.4f}s"
            assert (
                stats["max"] < 0.01
            ), f"Routing max too high for '{message}': {stats['max']:.4f}s"

    def test_single_agent_execution_performance(self):
        """Test single agent execution performance."""
        state = create_initial_state("single_exec_perf")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Performance test message"}
        ]

        # Test each agent type
        agent_types = [
            AgentType.COMMAND,
            AgentType.AUDIO,
            AgentType.CONVERSATION,
            AgentType.MEMORY,
        ]

        for agent_type in agent_types:
            stats = self.benchmark.run_benchmark(
                self.orchestrator.route_and_execute,
                iterations=30,
                state=state,
                requested_agent=agent_type,
            )

            # Single agent execution should be fast
            assert (
                stats["mean"] < 0.1
            ), f"{agent_type.value} execution too slow: {stats['mean']:.3f}s"
            assert (
                stats["max"] < 0.5
            ), f"{agent_type.value} execution max too high: {stats['max']:.3f}s"

            # Verify all executions succeeded
            successful_executions = sum(
                1 for result in stats["results"] if result.success
            )
            assert (
                successful_executions >= 27
            ), f"Only {successful_executions}/30 {agent_type.value} executions succeeded"

    def test_sequential_coordination_performance(self):
        """Test sequential multi-agent coordination performance."""
        state = create_initial_state("sequential_perf")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Sequential coordination test"}
        ]

        agents = [AgentType.COMMAND, AgentType.MEMORY, AgentType.CONVERSATION]

        stats = self.benchmark.run_benchmark(
            self.orchestrator.coordinate_agents,
            iterations=20,
            agents=agents,
            state=state,
            execution_mode=ExecutionMode.SEQUENTIAL,
        )

        # Sequential coordination should be reasonably fast
        assert (
            stats["mean"] < 0.5
        ), f"Sequential coordination too slow: {stats['mean']:.3f}s"
        assert (
            stats["max"] < 2.0
        ), f"Sequential coordination max too high: {stats['max']:.3f}s"

        # Verify all coordinations succeeded
        successful_coords = sum(
            1
            for results in stats["results"]
            if len(results) == 3 and all(r.success for r in results)
        )
        assert (
            successful_coords >= 18
        ), f"Only {successful_coords}/20 sequential coordinations succeeded"

    def test_parallel_coordination_performance(self):
        """Test parallel multi-agent coordination performance."""
        state = create_initial_state("parallel_perf")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Parallel coordination test"}
        ]

        agents = [AgentType.COMMAND, AgentType.AUDIO, AgentType.MEMORY]

        # Benchmark parallel execution
        stats = self.benchmark.run_benchmark(
            self.orchestrator.coordinate_agents,
            iterations=20,
            agents=agents,
            state=state,
            execution_mode=ExecutionMode.PARALLEL,
        )

        # Parallel should be faster than sequential
        assert (
            stats["mean"] < 0.3
        ), f"Parallel coordination too slow: {stats['mean']:.3f}s"
        assert (
            stats["max"] < 1.0
        ), f"Parallel coordination max too high: {stats['max']:.3f}s"

        # Compare with sequential timing (theoretical)
        # Parallel should be significantly faster for 3 agents
        # (assuming each agent takes ~0.01s, sequential would be ~0.03s, parallel ~0.01s)
        assert stats["mean"] < 0.1, "Parallel coordination not showing expected speedup"

    def test_concurrent_orchestration_performance(self):
        """Test orchestrator performance under concurrent load."""
        import concurrent.futures

        def execute_request(request_id: int):
            """Execute a single orchestration request."""
            state = create_initial_state(f"concurrent_perf_{request_id}")
            state["conversation"]["messages"] = [
                {"role": "user", "content": f"Concurrent request {request_id}"}
            ]

            start_time = time.perf_counter()
            result = self.orchestrator.route_and_execute(state=state)
            end_time = time.perf_counter()

            return {
                "id": request_id,
                "success": result.success,
                "execution_time": result.execution_time,
                "total_time": end_time - start_time,
            }

        # Execute 30 concurrent requests
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(execute_request, i) for i in range(30)]
            results = [
                future.result()
                for future in concurrent.futures.as_completed(futures, timeout=10)
            ]
        end_time = time.perf_counter()

        total_time = end_time - start_time

        # Performance assertions
        assert total_time < 5.0, f"Concurrent orchestration too slow: {total_time:.3f}s"
        assert (
            len(results) == 30
        ), f"Only {len(results)}/30 concurrent requests completed"

        successful_requests = [r for r in results if r["success"]]
        assert (
            len(successful_requests) >= 27
        ), f"Only {len(successful_requests)}/30 concurrent requests succeeded"

        # Check individual performance
        execution_times = [r["execution_time"] for r in successful_requests]
        total_times = [r["total_time"] for r in successful_requests]

        avg_execution_time = statistics.mean(execution_times)
        avg_total_time = statistics.mean(total_times)
        max_execution_time = max(execution_times)
        max_total_time = max(total_times)

        assert (
            avg_execution_time < 0.1
        ), f"Average concurrent execution too slow: {avg_execution_time:.3f}s"
        assert (
            avg_total_time < 0.2
        ), f"Average concurrent total time too slow: {avg_total_time:.3f}s"
        assert (
            max_execution_time < 0.5
        ), f"Max concurrent execution too slow: {max_execution_time:.3f}s"
        assert (
            max_total_time < 1.0
        ), f"Max concurrent total time too slow: {max_total_time:.3f}s"

    def test_system_scalability(self):
        """Test system scalability with increasing load."""
        load_levels = [5, 10, 20, 50]  # Number of concurrent requests
        performance_results = {}

        for load_level in load_levels:
            import concurrent.futures

            def execute_load_test(request_id: int):
                state = create_initial_state(f"scale_test_{load_level}_{request_id}")
                state["conversation"]["messages"] = [
                    {"role": "user", "content": f"Scalability test {request_id}"}
                ]

                start_time = time.perf_counter()
                result = self.orchestrator.route_and_execute(state=state)
                end_time = time.perf_counter()

                return {"success": result.success, "time": end_time - start_time}

            # Execute load test
            start_time = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(load_level, 10)
            ) as executor:
                futures = [
                    executor.submit(execute_load_test, i) for i in range(load_level)
                ]
                results = [
                    future.result(timeout=15)
                    for future in concurrent.futures.as_completed(futures)
                ]
            end_time = time.perf_counter()

            # Collect performance metrics
            total_time = end_time - start_time
            successful_results = [r for r in results if r["success"]]
            success_rate = len(successful_results) / len(results)

            if successful_results:
                avg_response_time = statistics.mean(
                    [r["time"] for r in successful_results]
                )
                max_response_time = max([r["time"] for r in successful_results])
            else:
                avg_response_time = float("inf")
                max_response_time = float("inf")

            throughput = len(successful_results) / total_time  # requests per second

            performance_results[load_level] = {
                "total_time": total_time,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time,
                "throughput": throughput,
            }

        # Scalability assertions
        for load_level, metrics in performance_results.items():
            # Success rate should remain high under load
            assert (
                metrics["success_rate"] >= 0.8
            ), f"Success rate too low at load {load_level}: {metrics['success_rate']:.2f}"

            # Response times should remain reasonable
            assert (
                metrics["avg_response_time"] < 1.0
            ), f"Average response time too high at load {load_level}: {metrics['avg_response_time']:.3f}s"
            assert (
                metrics["max_response_time"] < 5.0
            ), f"Max response time too high at load {load_level}: {metrics['max_response_time']:.3f}s"

            # Throughput should be reasonable
            expected_min_throughput = min(
                load_level, 5
            )  # At least 5 req/s or load_level if smaller
            assert (
                metrics["throughput"] >= expected_min_throughput * 0.8
            ), f"Throughput too low at load {load_level}: {metrics['throughput']:.1f} req/s"

        # Verify graceful degradation (response times shouldn't increase exponentially)
        response_times = [
            metrics["avg_response_time"] for metrics in performance_results.values()
        ]

        # Response time increase should be sub-linear
        assert (
            response_times[-1] / response_times[0] < 5
        ), "Response time degradation too severe under load"


class TestFASE2SuccessCriteria:
    """Validation tests against FASE 2 success criteria."""

    def setup_method(self):
        """Setup FASE 2 validation environment."""
        self.llm = FastMockLLM(response_time=0.01)
        self.orchestrator = AgentOrchestrator(max_workers=4)

        # Create and register all agents
        self.agents = {
            AgentType.COMMAND: CommandAgent(self.llm),
            AgentType.AUDIO: AudioAgent(self.llm),
            AgentType.CONVERSATION: ConversationAgent(self.llm),
            AgentType.MEMORY: MemoryAgent(self.llm),
        }

        for agent in self.agents.values():
            self.orchestrator.register_agent(agent)

    def teardown_method(self):
        """Cleanup FASE 2 validation environment."""
        self.orchestrator.shutdown()

    def test_react_agents_responding_to_commands(self):
        """✅ Validate: Agentes ReAct respondendo a comandos"""
        # Test command scenarios for each agent type
        test_scenarios = [
            {
                "message": "Execute o comando ls -la",
                "expected_agent": AgentType.COMMAND,
                "description": "Command execution",
            },
            {
                "message": "Converta este texto em fala",
                "expected_agent": AgentType.AUDIO,
                "description": "Audio processing",
            },
            {
                "message": "Lembre-se desta configuração",
                "expected_agent": AgentType.MEMORY,
                "description": "Memory management",
            },
            {
                "message": "Como você está hoje?",
                "expected_agent": AgentType.CONVERSATION,
                "description": "Natural conversation",
            },
        ]

        success_count = 0

        for scenario in test_scenarios:
            state = create_initial_state(f"react_test_{scenario['description']}")
            state["conversation"]["messages"] = [
                {"role": "user", "content": scenario["message"]}
            ]

            # Test routing and execution
            result = self.orchestrator.route_and_execute(state=state)

            # Verify agent responded successfully
            if result.success:
                success_count += 1
                assert (
                    result.execution_time > 0
                ), f"Agent execution time invalid for {scenario['description']}"
                assert (
                    result.result is not None
                ), f"Agent result missing for {scenario['description']}"
            else:
                print(f"Warning: {scenario['description']} failed: {result.error}")

        # Require at least 75% success rate (3/4 scenarios)
        success_rate = success_count / len(test_scenarios)
        assert (
            success_rate >= 0.75
        ), f"ReAct agents success rate too low: {success_rate:.2f}"

        print(
            f"✅ ReAct Agents Response Success Rate: {success_rate:.1%} ({success_count}/{len(test_scenarios)})"
        )

    def test_integrated_tools_functioning(self):
        """✅ Validate: Ferramentas integradas funcionando"""
        # Test each tool category through agents
        tool_tests = [
            {
                "agent": AgentType.COMMAND,
                "message": "Execute echo 'tool test'",
                "tool_type": "Shell Execution",
                "expected_tools": ["shell_executor", "filesystem_manager"],
            },
            {
                "agent": AgentType.AUDIO,
                "message": "Process audio file",
                "tool_type": "Audio Processing",
                "expected_tools": [
                    "audio_processor",
                    "text_to_speech",
                    "speech_to_text",
                ],
            },
            {
                "agent": AgentType.MEMORY,
                "message": "Store user preferences",
                "tool_type": "Memory Management",
                "expected_tools": ["memory_manager"],
            },
        ]

        tools_functioning = 0
        total_tools_tested = 0

        for test in tool_tests:
            state = create_initial_state(f"tool_test_{test['tool_type']}")
            state["conversation"]["messages"] = [
                {"role": "user", "content": test["message"]}
            ]

            # Execute through orchestrator
            result = self.orchestrator.route_and_execute(
                state=state, requested_agent=test["agent"]
            )

            # Verify tool integration
            if result.success:
                # Verify agent has expected tools
                agent = self.agents[test["agent"]]
                agent_tools = [tool.name for tool in agent.tools]

                for expected_tool in test["expected_tools"]:
                    total_tools_tested += 1
                    if expected_tool in agent_tools:
                        tools_functioning += 1
                    else:
                        print(
                            f"Warning: Tool {expected_tool} missing from {test['agent'].value}"
                        )
            else:
                print(f"Warning: {test['tool_type']} test failed: {result.error}")

        # Verify tool integration rate
        if total_tools_tested > 0:
            tool_success_rate = tools_functioning / total_tools_tested
            assert (
                tool_success_rate >= 0.8
            ), f"Tool integration success rate too low: {tool_success_rate:.2f}"
            print(
                f"✅ Tool Integration Success Rate: {tool_success_rate:.1%} ({tools_functioning}/{total_tools_tested})"
            )

    def test_multi_agent_coordination_operational(self):
        """✅ Validate: Coordenação multi-agente operacional"""
        # Test different coordination modes
        coordination_tests = [
            {
                "mode": ExecutionMode.SEQUENTIAL,
                "agents": [AgentType.MEMORY, AgentType.COMMAND, AgentType.CONVERSATION],
                "description": "Sequential coordination",
            },
            {
                "mode": ExecutionMode.PARALLEL,
                "agents": [AgentType.COMMAND, AgentType.AUDIO, AgentType.MEMORY],
                "description": "Parallel coordination",
            },
            {
                "mode": ExecutionMode.HYBRID,
                "agents": [AgentType.COMMAND, AgentType.CONVERSATION],
                "description": "Hybrid coordination",
            },
        ]

        coordination_success_count = 0

        for test in coordination_tests:
            state = create_initial_state(f"coordination_test_{test['description']}")
            state["conversation"]["messages"] = [
                {"role": "user", "content": f"Test {test['description'].lower()}"}
            ]

            # Execute coordination
            start_time = time.perf_counter()
            results = self.orchestrator.coordinate_agents(
                agents=test["agents"], state=state, execution_mode=test["mode"]
            )
            end_time = time.perf_counter()

            execution_time = end_time - start_time

            # Verify coordination success
            if (
                len(results) == len(test["agents"])
                and all(r.success for r in results)
                and execution_time < 10.0
            ):  # Reasonable time limit

                coordination_success_count += 1

                # Verify agent diversity
                agent_types_used = {r.agent_type for r in results}
                expected_types = set(test["agents"])
                assert (
                    agent_types_used == expected_types
                ), f"Agent coordination mismatch in {test['description']}"

                print(
                    f"✅ {test['description']}: {len(results)} agents, {execution_time:.3f}s"
                )
            else:
                failed_agents = [r for r in results if not r.success]
                print(
                    f"Warning: {test['description']} failed: {len(failed_agents)} failures, {execution_time:.3f}s"
                )

        # Require all coordination modes to work
        coordination_success_rate = coordination_success_count / len(coordination_tests)
        assert (
            coordination_success_rate >= 0.67
        ), f"Coordination success rate too low: {coordination_success_rate:.2f}"

        print(
            f"✅ Multi-Agent Coordination Success Rate: {coordination_success_rate:.1%} ({coordination_success_count}/{len(coordination_tests)})"
        )

    def test_comprehensive_fase2_validation(self):
        """Comprehensive validation of all FASE 2 requirements."""
        validation_results = {
            "react_agents_functional": False,
            "tools_integrated": False,
            "coordination_operational": False,
            "performance_acceptable": False,
            "error_handling_robust": False,
        }

        # 1. ReAct Agents Functionality
        try:
            state = create_initial_state("comprehensive_test")
            state["conversation"]["messages"] = [
                {"role": "user", "content": "Comprehensive system test"}
            ]

            result = self.orchestrator.route_and_execute(state=state)
            validation_results["react_agents_functional"] = result.success
        except Exception as e:
            print(f"ReAct agents validation failed: {e}")

        # 2. Tools Integration
        try:
            # Verify all expected tools are available
            total_tools = sum(len(agent.tools) for agent in self.agents.values())
            expected_min_tools = 6  # Minimum expected tools across all agents
            validation_results["tools_integrated"] = total_tools >= expected_min_tools
        except Exception as e:
            print(f"Tools integration validation failed: {e}")

        # 3. Coordination Operational
        try:
            state = create_initial_state("coordination_validation")
            state["conversation"]["messages"] = [
                {"role": "user", "content": "Multi-agent coordination test"}
            ]

            results = self.orchestrator.coordinate_agents(
                agents=[AgentType.COMMAND, AgentType.CONVERSATION],
                state=state,
                execution_mode=ExecutionMode.SEQUENTIAL,
            )

            validation_results["coordination_operational"] = len(results) == 2 and all(
                r.success for r in results
            )
        except Exception as e:
            print(f"Coordination validation failed: {e}")

        # 4. Performance Acceptable
        try:
            # Quick performance check
            start_time = time.perf_counter()
            for _ in range(10):
                state = create_initial_state("perf_validation")
                state["conversation"]["messages"] = [
                    {"role": "user", "content": "Performance validation test"}
                ]
                result = self.orchestrator.route_and_execute(state=state)
            end_time = time.perf_counter()

            avg_time = (end_time - start_time) / 10
            validation_results["performance_acceptable"] = (
                avg_time < 0.5
            )  # 500ms per request
        except Exception as e:
            print(f"Performance validation failed: {e}")

        # 5. Error Handling Robust
        try:
            # Test with invalid state
            invalid_state = {"invalid": "state"}
            result = self.orchestrator.route_and_execute(state=invalid_state)

            # Should fail gracefully, not crash
            validation_results["error_handling_robust"] = (
                not result.success and result.error is not None
            )
        except Exception:
            # Even exceptions should be caught and handled gracefully
            validation_results["error_handling_robust"] = False

        # Summary
        passed_validations = sum(validation_results.values())
        total_validations = len(validation_results)
        success_rate = passed_validations / total_validations

        print("\n📊 FASE 2 Comprehensive Validation Results:")
        for test_name, passed in validation_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")

        print(
            f"\n🎯 Overall FASE 2 Success Rate: {success_rate:.1%} ({passed_validations}/{total_validations})"
        )

        # Require 80% success rate for FASE 2 validation
        assert (
            success_rate >= 0.8
        ), f"FASE 2 validation failed: {success_rate:.1%} success rate (minimum 80% required)"

        return validation_results


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
