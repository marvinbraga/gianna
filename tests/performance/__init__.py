"""
Performance Tests for Gianna Components

This package contains performance and load tests that verify system behavior
under stress, measure throughput, latency, and resource usage, and ensure
scalability requirements are met.

Modules:
- test_load_testing: High-load scenario testing
- test_throughput: Throughput measurement and validation
- test_latency: Response time and latency testing
- test_memory_usage: Memory usage and leak testing
- test_concurrency: Concurrent session handling
- test_scalability: Scalability testing
- test_stress: Stress testing under extreme conditions
- test_benchmarks: Comparative benchmarking

Testing Philosophy:
- Measure actual performance metrics
- Test realistic load scenarios
- Identify performance bottlenecks
- Validate against requirements
- Test resource utilization
- Long-running tests (> 30 seconds)
- Focus on performance regressions
"""

# Performance test categories
PERFORMANCE_TEST_CATEGORIES = {
    "load_testing": "High-load scenario testing",
    "throughput": "Throughput measurement and validation",
    "latency": "Response time and latency testing",
    "memory_usage": "Memory usage and leak testing",
    "concurrency": "Concurrent session handling",
    "scalability": "Scalability testing",
    "stress": "Stress testing under extreme conditions",
    "benchmarks": "Comparative benchmarking",
}

# Performance test thresholds (from project requirements)
PERFORMANCE_REQUIREMENTS = {
    # Response Time Requirements
    "latency": {
        "simple_command": 2.0,  # < 2s for simple commands
        "complex_workflow": 10.0,  # < 10s for complex workflows
        "llm_response": 5.0,  # < 5s for LLM responses
        "voice_processing": 3.0,  # < 3s for voice processing
        "audio_conversion": 2.0,  # < 2s for audio conversion
        "memory_retrieval": 1.0,  # < 1s for memory operations
    },
    # Memory Usage Requirements
    "memory": {
        "normal_operation": 500,  # < 500MB normal operation
        "peak_usage": 1000,  # < 1GB peak usage
        "memory_leak_rate": 1.0,  # < 1MB/hour leak rate
        "session_memory": 50,  # < 50MB per session
    },
    # Throughput Requirements
    "throughput": {
        "interactions_per_minute": 10,  # > 10 interactions/minute
        "concurrent_sessions": 5,  # > 5 concurrent sessions
        "llm_requests_per_second": 2,  # > 2 LLM requests/second
        "voice_commands_per_minute": 8,  # > 8 voice commands/minute
        "audio_processing_rate": 1.5,  # > 1.5x real-time for audio
    },
    # Reliability Requirements
    "reliability": {
        "uptime": 0.99,  # > 99% uptime during load tests
        "error_rate": 0.01,  # < 1% error rate
        "recovery_time": 5.0,  # < 5s recovery from errors
        "cache_hit_rate": 0.70,  # > 70% cache hit rate
    },
    # Scalability Requirements
    "scalability": {
        "max_concurrent_users": 20,  # > 20 concurrent users
        "response_time_degradation": 2.0,  # < 2x response time at max load
        "memory_scaling_factor": 1.5,  # < 1.5x memory per additional user
        "cpu_scaling_factor": 1.2,  # < 1.2x CPU per additional user
    },
}

# Load test scenarios
LOAD_TEST_SCENARIOS = {
    "baseline": {
        "description": "Single user, normal operations",
        "concurrent_users": 1,
        "duration": 60,  # seconds
        "operations_per_minute": 10,
    },
    "moderate_load": {
        "description": "Moderate load with multiple users",
        "concurrent_users": 5,
        "duration": 300,  # 5 minutes
        "operations_per_minute": 30,
    },
    "high_load": {
        "description": "High load stress test",
        "concurrent_users": 10,
        "duration": 600,  # 10 minutes
        "operations_per_minute": 60,
    },
    "peak_load": {
        "description": "Peak load capacity test",
        "concurrent_users": 20,
        "duration": 300,  # 5 minutes
        "operations_per_minute": 100,
    },
    "stress_test": {
        "description": "Beyond capacity stress test",
        "concurrent_users": 50,
        "duration": 120,  # 2 minutes
        "operations_per_minute": 200,
    },
}

# Benchmark test operations
BENCHMARK_OPERATIONS = {
    "llm_invoke": "Single LLM chain invocation",
    "voice_processing": "Complete voice input/output cycle",
    "memory_storage": "Store interaction in semantic memory",
    "memory_retrieval": "Retrieve similar interactions",
    "state_update": "Update conversation state",
    "audio_conversion": "Convert audio format",
    "tool_execution": "Execute shell tool command",
    "agent_coordination": "Route request through orchestrator",
}

# Performance monitoring intervals
MONITORING_CONFIG = {
    "cpu_sample_interval": 0.1,  # seconds
    "memory_sample_interval": 1.0,  # seconds
    "disk_sample_interval": 5.0,  # seconds
    "network_sample_interval": 1.0,  # seconds
    "response_time_percentiles": [50, 90, 95, 99],  # percentiles to track
}
