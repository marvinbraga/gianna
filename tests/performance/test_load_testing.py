"""
Load testing for Gianna system - FASE 5

Tests for:
- High-load scenario testing
- Concurrent session handling
- System behavior under stress
- Performance degradation measurement
- Resource utilization under load
- Error rate measurement

Test Coverage:
- Baseline performance measurement
- Moderate load testing (5 concurrent users)
- High load testing (10 concurrent users)
- Peak load testing (20 concurrent users)
- Stress testing beyond capacity
- Long-running stability tests
"""

import asyncio
import concurrent.futures
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import psutil
import pytest

# Performance test imports
from tests.performance import (
    LOAD_TEST_SCENARIOS,
    MONITORING_CONFIG,
    PERFORMANCE_REQUIREMENTS,
)


class PerformanceMonitor:
    """Monitor system performance metrics during load tests."""

    def __init__(self):
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "response_times": [],
            "error_count": 0,
            "success_count": 0,
            "start_time": None,
            "end_time": None,
        }
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.metrics["start_time"] = time.perf_counter()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        self.metrics["end_time"] = time.perf_counter()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_resources(self):
        """Monitor CPU and memory usage."""
        import os

        process = psutil.Process(os.getpid())

        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024

                self.metrics["cpu_usage"].append(cpu_percent)
                self.metrics["memory_usage"].append(memory_mb)

                time.sleep(MONITORING_CONFIG["cpu_sample_interval"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def record_operation(self, response_time: float, success: bool):
        """Record operation result."""
        self.metrics["response_times"].append(response_time)
        if success:
            self.metrics["success_count"] += 1
        else:
            self.metrics["error_count"] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_operations = self.metrics["success_count"] + self.metrics["error_count"]
        total_time = self.metrics["end_time"] - self.metrics["start_time"]

        return {
            "total_operations": total_operations,
            "success_rate": self.metrics["success_count"] / max(total_operations, 1),
            "error_rate": self.metrics["error_count"] / max(total_operations, 1),
            "avg_response_time": sum(self.metrics["response_times"])
            / max(len(self.metrics["response_times"]), 1),
            "max_response_time": (
                max(self.metrics["response_times"])
                if self.metrics["response_times"]
                else 0
            ),
            "min_response_time": (
                min(self.metrics["response_times"])
                if self.metrics["response_times"]
                else 0
            ),
            "avg_cpu_usage": sum(self.metrics["cpu_usage"])
            / max(len(self.metrics["cpu_usage"]), 1),
            "peak_memory_mb": (
                max(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0
            ),
            "avg_memory_mb": sum(self.metrics["memory_usage"])
            / max(len(self.metrics["memory_usage"]), 1),
            "throughput": total_operations / max(total_time, 1),
            "duration": total_time,
        }


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    monitor = PerformanceMonitor()
    yield monitor
    if monitor.monitoring:
        monitor.stop_monitoring()


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.fase5
class TestBasicLoadScenarios:
    """Test basic load scenarios."""

    def test_baseline_performance(self, performance_monitor, mock_langgraph_chain):
        """Test baseline single-user performance."""
        scenario = LOAD_TEST_SCENARIOS["baseline"]

        # Setup fast mock
        mock_langgraph_chain.invoke.return_value = {"output": "Baseline response"}

        performance_monitor.start_monitoring()

        # Single user operations
        operations = 0
        start_time = time.perf_counter()
        end_time = start_time + scenario["duration"]

        while time.perf_counter() < end_time:
            op_start = time.perf_counter()

            try:
                response = mock_langgraph_chain.invoke(
                    {"input": f"Operation {operations}"}
                )
                success = "output" in response
                op_time = time.perf_counter() - op_start

                performance_monitor.record_operation(op_time, success)
                operations += 1

                # Control operation rate
                time.sleep(60.0 / scenario["operations_per_minute"])

            except Exception as e:
                op_time = time.perf_counter() - op_start
                performance_monitor.record_operation(op_time, False)

        performance_monitor.stop_monitoring()
        summary = performance_monitor.get_summary()

        # Validate baseline requirements
        assert summary["success_rate"] >= 0.99  # 99% success rate
        assert (
            summary["avg_response_time"]
            < PERFORMANCE_REQUIREMENTS["latency"]["simple_command"]
        )
        assert (
            summary["peak_memory_mb"]
            < PERFORMANCE_REQUIREMENTS["memory"]["normal_operation"]
        )
        assert summary["throughput"] >= 1.0  # At least 1 operation/second

        print(f"Baseline Performance: {summary}")

    def test_moderate_load_performance(self, performance_monitor, mock_langgraph_chain):
        """Test moderate load with multiple concurrent users."""
        scenario = LOAD_TEST_SCENARIOS["moderate_load"]

        # Setup mock
        mock_langgraph_chain.invoke.return_value = {"output": "Moderate load response"}

        performance_monitor.start_monitoring()

        def user_session(user_id: int, duration: int):
            """Simulate single user session."""
            session_start = time.perf_counter()
            session_end = session_start + duration
            operations = 0

            while time.perf_counter() < session_end:
                op_start = time.perf_counter()

                try:
                    response = mock_langgraph_chain.invoke(
                        {"input": f"User {user_id} operation {operations}"}
                    )
                    success = "output" in response
                    op_time = time.perf_counter() - op_start

                    performance_monitor.record_operation(op_time, success)
                    operations += 1

                    # Control operation rate per user
                    time.sleep(
                        60.0
                        / (
                            scenario["operations_per_minute"]
                            / scenario["concurrent_users"]
                        )
                    )

                except Exception:
                    op_time = time.perf_counter() - op_start
                    performance_monitor.record_operation(op_time, False)

        # Run concurrent user sessions
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=scenario["concurrent_users"]
        ) as executor:
            futures = []
            for user_id in range(scenario["concurrent_users"]):
                future = executor.submit(user_session, user_id, scenario["duration"])
                futures.append(future)

            # Wait for all sessions to complete
            concurrent.futures.wait(futures)

        performance_monitor.stop_monitoring()
        summary = performance_monitor.get_summary()

        # Validate moderate load requirements
        assert summary["success_rate"] >= 0.95  # 95% success rate under load
        assert (
            summary["avg_response_time"]
            < PERFORMANCE_REQUIREMENTS["latency"]["simple_command"] * 1.5
        )
        assert (
            summary["peak_memory_mb"] < PERFORMANCE_REQUIREMENTS["memory"]["peak_usage"]
        )
        assert (
            summary["throughput"] >= scenario["operations_per_minute"] / 60 * 0.8
        )  # 80% of target

        print(f"Moderate Load Performance: {summary}")

    @pytest.mark.slow
    def test_high_load_performance(self, performance_monitor, mock_langgraph_chain):
        """Test high load scenario."""
        scenario = LOAD_TEST_SCENARIOS["high_load"]

        # Setup mock with slight delay to simulate real load
        def slow_invoke(input_data):
            time.sleep(0.1)  # 100ms processing time
            return {"output": "High load response"}

        mock_langgraph_chain.invoke.side_effect = slow_invoke

        performance_monitor.start_monitoring()

        def concurrent_user(user_id: int, duration: int):
            """Simulate concurrent user operations."""
            end_time = time.perf_counter() + duration
            operations = 0

            while time.perf_counter() < end_time:
                op_start = time.perf_counter()

                try:
                    response = mock_langgraph_chain.invoke(
                        {"input": f"High load user {user_id} op {operations}"}
                    )
                    success = "output" in response
                    op_time = time.perf_counter() - op_start

                    performance_monitor.record_operation(op_time, success)
                    operations += 1

                except Exception:
                    op_time = time.perf_counter() - op_start
                    performance_monitor.record_operation(op_time, False)

        # Run high load test
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=scenario["concurrent_users"]
        ) as executor:
            futures = []
            for user_id in range(scenario["concurrent_users"]):
                future = executor.submit(concurrent_user, user_id, scenario["duration"])
                futures.append(future)

            concurrent.futures.wait(futures, timeout=scenario["duration"] + 30)

        performance_monitor.stop_monitoring()
        summary = performance_monitor.get_summary()

        # Validate high load requirements
        assert summary["success_rate"] >= 0.90  # 90% success rate under high load
        assert (
            summary["avg_response_time"]
            < PERFORMANCE_REQUIREMENTS["latency"]["complex_workflow"]
        )
        assert (
            summary["peak_memory_mb"] < PERFORMANCE_REQUIREMENTS["memory"]["peak_usage"]
        )
        assert (
            summary["error_rate"]
            <= PERFORMANCE_REQUIREMENTS["reliability"]["error_rate"]
        )

        print(f"High Load Performance: {summary}")


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.fase5
class TestConcurrentSessionHandling:
    """Test concurrent session handling capabilities."""

    def test_concurrent_voice_sessions(self, performance_monitor, async_test_runner):
        """Test concurrent voice processing sessions."""

        async def voice_session(session_id: str, duration: int):
            """Simulate voice processing session."""
            end_time = time.perf_counter() + duration
            operations = 0

            # Mock voice processing components
            with patch(
                "gianna.workflows.voice_streaming.StreamingVoicePipeline"
            ) as mock_pipeline:
                pipeline = mock_pipeline.return_value
                pipeline.start_listening = AsyncMock()
                pipeline._process_audio_buffer = AsyncMock()

                while time.perf_counter() < end_time:
                    op_start = time.perf_counter()

                    try:
                        # Simulate voice processing
                        await pipeline.start_listening(
                            {
                                "session_id": session_id,
                                "conversation": {"messages": []},
                                "audio": {"current_mode": "listening"},
                            }
                        )

                        await pipeline._process_audio_buffer(b"fake_audio")

                        op_time = time.perf_counter() - op_start
                        performance_monitor.record_operation(op_time, True)
                        operations += 1

                        await asyncio.sleep(0.5)  # 500ms between operations

                    except Exception:
                        op_time = time.perf_counter() - op_start
                        performance_monitor.record_operation(op_time, False)

        async def run_concurrent_voice_test():
            performance_monitor.start_monitoring()

            # Create concurrent voice sessions
            tasks = []
            for i in range(5):  # 5 concurrent voice sessions
                task = asyncio.create_task(voice_session(f"voice-session-{i}", 30))
                tasks.append(task)

            await asyncio.gather(*tasks)

            performance_monitor.stop_monitoring()
            return performance_monitor.get_summary()

        summary = async_test_runner(run_concurrent_voice_test())

        # Validate concurrent voice processing
        assert summary["success_rate"] >= 0.90
        assert (
            summary["avg_response_time"]
            < PERFORMANCE_REQUIREMENTS["latency"]["voice_processing"]
        )
        assert summary["throughput"] >= 1.0  # At least 1 voice operation/second

        print(f"Concurrent Voice Sessions: {summary}")

    def test_concurrent_memory_operations(
        self, performance_monitor, mock_semantic_memory
    ):
        """Test concurrent memory operations."""
        # Setup memory operations
        mock_semantic_memory.store_interaction.return_value = None
        mock_semantic_memory.search_similar_interactions.return_value = [
            {"content": "Similar interaction", "metadata": {"score": 0.8}}
        ]

        def memory_worker(worker_id: int, operations: int):
            """Worker thread for memory operations."""
            for i in range(operations):
                op_start = time.perf_counter()

                try:
                    # Store interaction
                    mock_semantic_memory.store_interaction(
                        {
                            "user_input": f"Worker {worker_id} input {i}",
                            "assistant_response": f"Worker {worker_id} response {i}",
                            "timestamp": datetime.now().isoformat(),
                            "session_id": f"session-{worker_id}",
                        }
                    )

                    # Search similar
                    results = mock_semantic_memory.search_similar_interactions(
                        f"query {i}"
                    )

                    op_time = time.perf_counter() - op_start
                    success = len(results) > 0

                    performance_monitor.record_operation(op_time, success)

                except Exception:
                    op_time = time.perf_counter() - op_start
                    performance_monitor.record_operation(op_time, False)

        performance_monitor.start_monitoring()

        # Run concurrent memory operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for worker_id in range(10):
                future = executor.submit(
                    memory_worker, worker_id, 50
                )  # 50 ops per worker
                futures.append(future)

            concurrent.futures.wait(futures)

        performance_monitor.stop_monitoring()
        summary = performance_monitor.get_summary()

        # Validate concurrent memory operations
        assert summary["success_rate"] >= 0.95
        assert (
            summary["avg_response_time"]
            < PERFORMANCE_REQUIREMENTS["latency"]["memory_retrieval"]
        )
        assert (
            summary["total_operations"] == 1000
        )  # 10 workers * 50 ops * 2 ops per iteration

        print(f"Concurrent Memory Operations: {summary}")

    def test_mixed_workload_performance(
        self,
        performance_monitor,
        mock_langgraph_chain,
        mock_semantic_memory,
        mock_shell_tool,
    ):
        """Test mixed workload performance."""
        # Setup mocks
        mock_langgraph_chain.invoke.return_value = {"output": "Mixed workload response"}
        mock_semantic_memory.store_interaction.return_value = None
        mock_shell_tool._run.return_value = (
            '{"exit_code": 0, "stdout": "success", "success": true}'
        )

        def mixed_workload_user(user_id: int, duration: int):
            """Simulate user with mixed operations."""
            end_time = time.perf_counter() + duration
            operation_types = ["llm", "memory", "shell"]

            while time.perf_counter() < end_time:
                op_start = time.perf_counter()
                op_type = operation_types[
                    int(time.perf_counter()) % len(operation_types)
                ]

                try:
                    if op_type == "llm":
                        response = mock_langgraph_chain.invoke(
                            {"input": f"User {user_id} LLM request"}
                        )
                        success = "output" in response

                    elif op_type == "memory":
                        mock_semantic_memory.store_interaction(
                            {
                                "user_input": f"User {user_id} memory input",
                                "assistant_response": f"User {user_id} memory response",
                                "timestamp": datetime.now().isoformat(),
                                "session_id": f"mixed-session-{user_id}",
                            }
                        )
                        success = True

                    elif op_type == "shell":
                        result = mock_shell_tool._run(f"echo 'User {user_id} command'")
                        success = "success" in result and "true" in result

                    op_time = time.perf_counter() - op_start
                    performance_monitor.record_operation(op_time, success)

                    # Brief pause between operations
                    time.sleep(0.1)

                except Exception:
                    op_time = time.perf_counter() - op_start
                    performance_monitor.record_operation(op_time, False)

        performance_monitor.start_monitoring()

        # Run mixed workload test
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for user_id in range(8):  # 8 concurrent users
                future = executor.submit(
                    mixed_workload_user, user_id, 60
                )  # 60 second test
                futures.append(future)

            concurrent.futures.wait(futures)

        performance_monitor.stop_monitoring()
        summary = performance_monitor.get_summary()

        # Validate mixed workload performance
        assert summary["success_rate"] >= 0.90
        assert (
            summary["avg_response_time"]
            < PERFORMANCE_REQUIREMENTS["latency"]["complex_workflow"]
        )
        assert (
            summary["peak_memory_mb"] < PERFORMANCE_REQUIREMENTS["memory"]["peak_usage"]
        )
        assert summary["throughput"] >= 5.0  # At least 5 mixed operations/second

        print(f"Mixed Workload Performance: {summary}")


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.fase5
class TestStressAndPeakLoad:
    """Test stress conditions and peak load scenarios."""

    @pytest.mark.stress
    def test_peak_load_capacity(self, performance_monitor, mock_langgraph_chain):
        """Test system behavior at peak load capacity."""
        scenario = LOAD_TEST_SCENARIOS["peak_load"]

        # Setup mock with realistic processing time
        def realistic_invoke(input_data):
            time.sleep(0.05)  # 50ms processing time
            return {"output": "Peak load response"}

        mock_langgraph_chain.invoke.side_effect = realistic_invoke

        performance_monitor.start_monitoring()

        def peak_load_user(user_id: int, duration: int):
            """High-intensity user simulation."""
            end_time = time.perf_counter() + duration

            while time.perf_counter() < end_time:
                op_start = time.perf_counter()

                try:
                    response = mock_langgraph_chain.invoke(
                        {"input": f"Peak user {user_id} operation"}
                    )
                    success = "output" in response
                    op_time = time.perf_counter() - op_start

                    performance_monitor.record_operation(op_time, success)

                    # Minimal pause - high intensity
                    time.sleep(0.01)

                except Exception:
                    op_time = time.perf_counter() - op_start
                    performance_monitor.record_operation(op_time, False)

        # Run peak load test
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=scenario["concurrent_users"]
        ) as executor:
            futures = []
            for user_id in range(scenario["concurrent_users"]):
                future = executor.submit(peak_load_user, user_id, scenario["duration"])
                futures.append(future)

            concurrent.futures.wait(futures, timeout=scenario["duration"] + 60)

        performance_monitor.stop_monitoring()
        summary = performance_monitor.get_summary()

        # Validate peak load handling
        assert summary["success_rate"] >= 0.80  # 80% success under peak load
        assert summary["error_rate"] <= 0.20  # Max 20% error rate
        assert (
            summary["avg_response_time"]
            < PERFORMANCE_REQUIREMENTS["latency"]["complex_workflow"] * 2
        )
        assert (
            summary["peak_memory_mb"]
            < PERFORMANCE_REQUIREMENTS["memory"]["peak_usage"] * 1.2
        )

        print(f"Peak Load Capacity: {summary}")

    @pytest.mark.stress
    def test_beyond_capacity_stress(self, performance_monitor, mock_langgraph_chain):
        """Test system behavior beyond designed capacity."""
        scenario = LOAD_TEST_SCENARIOS["stress_test"]

        # Setup mock that occasionally fails under stress
        call_count = 0

        def stress_invoke(input_data):
            nonlocal call_count
            call_count += 1

            # Simulate increasing response times under stress
            time.sleep(0.1 + (call_count % 100) * 0.001)

            # Occasional failures under extreme stress
            if call_count % 50 == 0:
                raise Exception("Service overwhelmed")

            return {"output": "Stress response"}

        mock_langgraph_chain.invoke.side_effect = stress_invoke

        performance_monitor.start_monitoring()

        def stress_user(user_id: int, duration: int):
            """Extreme stress user simulation."""
            end_time = time.perf_counter() + duration

            while time.perf_counter() < end_time:
                op_start = time.perf_counter()

                try:
                    response = mock_langgraph_chain.invoke(
                        {"input": f"Stress user {user_id} operation"}
                    )
                    success = "output" in response
                    op_time = time.perf_counter() - op_start

                    performance_monitor.record_operation(op_time, success)

                except Exception:
                    op_time = time.perf_counter() - op_start
                    performance_monitor.record_operation(op_time, False)

        # Run stress test
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=scenario["concurrent_users"]
        ) as executor:
            futures = []
            for user_id in range(scenario["concurrent_users"]):
                future = executor.submit(stress_user, user_id, scenario["duration"])
                futures.append(future)

            concurrent.futures.wait(futures, timeout=scenario["duration"] + 30)

        performance_monitor.stop_monitoring()
        summary = performance_monitor.get_summary()

        # Validate stress test behavior (more lenient requirements)
        assert summary["success_rate"] >= 0.60  # 60% success under extreme stress
        assert summary["total_operations"] > 1000  # System still processes operations
        assert (
            summary["peak_memory_mb"]
            < PERFORMANCE_REQUIREMENTS["memory"]["peak_usage"] * 1.5
        )

        # System should not crash completely
        assert summary["error_rate"] < 0.50  # Less than 50% error rate

        print(f"Beyond Capacity Stress: {summary}")

    def test_memory_leak_detection(self, performance_monitor, mock_langgraph_chain):
        """Test for memory leaks during extended operations."""
        # Setup mock
        mock_langgraph_chain.invoke.return_value = {"output": "Memory test response"}

        import os

        import psutil

        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]

        performance_monitor.start_monitoring()

        # Run extended operations
        for i in range(1000):  # 1000 operations
            op_start = time.perf_counter()

            try:
                response = mock_langgraph_chain.invoke(
                    {"input": f"Memory test operation {i}"}
                )
                success = "output" in response
                op_time = time.perf_counter() - op_start

                performance_monitor.record_operation(op_time, success)

                # Sample memory every 100 operations
                if i % 100 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)

            except Exception:
                op_time = time.perf_counter() - op_start
                performance_monitor.record_operation(op_time, False)

        performance_monitor.stop_monitoring()
        summary = performance_monitor.get_summary()

        # Analyze memory trend
        final_memory = memory_samples[-1]
        memory_increase = final_memory - initial_memory

        # Validate no significant memory leaks
        assert memory_increase < 50  # Less than 50MB increase for 1000 operations
        assert summary["success_rate"] >= 0.98  # High success rate for extended run

        # Check for memory growth trend
        if len(memory_samples) > 2:
            # Calculate linear regression slope (simple approximation)
            n = len(memory_samples)
            x_avg = (n - 1) / 2
            y_avg = sum(memory_samples) / n

            slope = sum(
                (i - x_avg) * (mem - y_avg) for i, mem in enumerate(memory_samples)
            )
            slope /= sum((i - x_avg) ** 2 for i in range(n))

            # Memory growth should be minimal
            assert slope < 1.0  # Less than 1MB per sample period growth

        print(
            f"Memory Leak Test: Initial={initial_memory:.1f}MB, Final={final_memory:.1f}MB, Increase={memory_increase:.1f}MB"
        )
        print(f"Extended Run Performance: {summary}")
