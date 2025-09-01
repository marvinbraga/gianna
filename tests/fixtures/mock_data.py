"""
Mock data generators and sample data for testing.

Provides utilities for generating realistic test data including:
- Conversation messages and history
- Audio processing data
- LLM responses and prompts
- Performance metrics
- Error scenarios
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List
from uuid import uuid4

import numpy as np
from tests.fixtures import TEST_DATA_CONSTANTS


class MockDataGenerator:
    """Generate realistic mock data for testing."""

    def __init__(self, seed: int = 42):
        """Initialize with optional random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    def generate_conversation_history(
        self, message_count: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate realistic conversation history."""
        messages = []
        base_time = datetime.now() - timedelta(hours=1)

        conversation_templates = [
            ("user", "Hello, how are you today?"),
            ("assistant", "I'm doing well, thank you! How can I help you?"),
            ("user", "Can you help me with a Python problem?"),
            (
                "assistant",
                "Of course! I'd be happy to help with Python. What's the issue?",
            ),
            ("user", "I need to process some audio files"),
            (
                "assistant",
                "I can help with audio processing. What format are your files?",
            ),
            ("user", "They are WAV files that need conversion"),
            ("assistant", "I can help convert WAV files. What format do you need?"),
            ("user", "Execute the command: ls -la"),
            ("assistant", "I'll help you execute that command safely."),
        ]

        for i in range(message_count):
            template_idx = i % len(conversation_templates)
            role, content_template = conversation_templates[template_idx]

            # Add some variation to the content
            if "Python" in content_template and random.random() > 0.5:
                content_template = content_template.replace(
                    "Python", random.choice(["JavaScript", "Java", "C++", "Go"])
                )

            message = {
                "role": role,
                "content": content_template,
                "timestamp": (base_time + timedelta(minutes=i * 2)).isoformat(),
                "source": random.choice(["text", "voice"]),
                "metadata": {
                    "message_id": str(uuid4()),
                    "processing_time": round(random.uniform(0.1, 2.0), 3),
                    "confidence": (
                        round(random.uniform(0.8, 1.0), 3)
                        if role == "assistant"
                        else None
                    ),
                },
            }
            messages.append(message)

        return messages

    def generate_audio_data(
        self, duration: float = 1.0, sample_rate: int = 16000
    ) -> np.ndarray:
        """Generate synthetic audio data for testing."""
        num_samples = int(duration * sample_rate)

        # Generate a mix of sine waves to simulate speech
        t = np.linspace(0, duration, num_samples)
        audio = np.zeros(num_samples)

        # Add multiple frequency components
        frequencies = [440, 880, 220, 1320]  # A4, A5, A3, E6
        for freq in frequencies:
            amplitude = random.uniform(0.1, 0.3)
            phase = random.uniform(0, 2 * np.pi)
            audio += amplitude * np.sin(2 * np.pi * freq * t + phase)

        # Add some noise
        noise_amplitude = 0.05
        audio += noise_amplitude * np.random.normal(0, 1, num_samples)

        # Normalize
        audio = audio / np.max(np.abs(audio))

        return audio.astype(np.float32)

    def generate_llm_responses(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic LLM responses."""
        response_templates = [
            "I understand your request. Let me help you with that.",
            "Here's the information you're looking for: {}",
            "I can assist you with {}. Here's how to proceed:",
            "That's a great question about {}. The answer is:",
            "I've processed your request for {}. Here are the results:",
            "To solve this {} problem, you can follow these steps:",
            "Based on your input about {}, I recommend:",
            "I've completed the {} task successfully. Here's what happened:",
        ]

        topics = [
            "audio processing",
            "file management",
            "Python programming",
            "data analysis",
            "system administration",
            "web development",
            "machine learning",
            "database queries",
            "API integration",
            "troubleshooting",
            "optimization",
            "security",
        ]

        responses = []
        for i in range(count):
            template = random.choice(response_templates)
            topic = random.choice(topics)

            if "{}" in template:
                content = template.format(topic)
            else:
                content = template

            response = {
                "output": content,
                "metadata": {
                    "model": random.choice(["gpt35", "gpt4", "claude", "gemini"]),
                    "tokens_used": random.randint(50, 500),
                    "processing_time": round(random.uniform(0.5, 3.0), 3),
                    "confidence": round(random.uniform(0.7, 0.95), 3),
                    "timestamp": datetime.now().isoformat(),
                },
            }
            responses.append(response)

        return responses

    def generate_command_history(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate realistic command execution history."""
        command_templates = [
            (
                "ls -la",
                "total 24\\ndrwxr-xr-x  3 user user 4096 Jan  1 10:00 .\\n-rw-r--r--  1 user user 1024 Jan  1 10:00 file.txt",
                True,
            ),
            ("pwd", "/home/user/project", True),
            ("echo 'Hello World'", "Hello World", True),
            ("mkdir test_dir", "", True),
            ("rm test_file.txt", "", True),
            (
                "cat nonexistent.txt",
                "cat: nonexistent.txt: No such file or directory",
                False,
            ),
            ("python script.py", "Script executed successfully", True),
            ("invalid_command", "invalid_command: command not found", False),
            ("chmod 755 file.sh", "", True),
            ("grep 'pattern' file.txt", "matching line with pattern", True),
        ]

        history = []
        base_time = datetime.now() - timedelta(hours=2)

        for i in range(count):
            template_idx = i % len(command_templates)
            command, output, success = command_templates[template_idx]

            # Add some variation
            if "test" in command:
                command = command.replace("test", f"test_{i}")
                output = output.replace("test", f"test_{i}")

            entry = {
                "command": command,
                "result": output,
                "success": success,
                "exit_code": 0 if success else random.randint(1, 127),
                "execution_time": round(random.uniform(0.01, 2.0), 3),
                "timestamp": (base_time + timedelta(minutes=i * 3)).isoformat(),
                "user": "test_user",
                "working_directory": f"/home/test_user/dir_{i % 5}",
            }
            history.append(entry)

        return history

    def generate_performance_metrics(
        self, duration: int = 60
    ) -> Dict[str, List[float]]:
        """Generate realistic performance metrics over time."""
        # Sample every second
        timestamps = list(range(duration))

        # CPU usage (0-100%)
        cpu_base = random.uniform(10, 30)
        cpu_usage = []
        for i in timestamps:
            # Add some periodicity and noise
            periodic = 10 * np.sin(2 * np.pi * i / 20)  # 20-second period
            noise = random.uniform(-5, 5)
            spike = 30 if random.random() < 0.05 else 0  # 5% chance of spike

            cpu = max(0, min(100, cpu_base + periodic + noise + spike))
            cpu_usage.append(round(cpu, 2))

        # Memory usage (MB)
        memory_base = random.uniform(100, 300)
        memory_usage = []
        memory_current = memory_base

        for i in timestamps:
            # Gradual increase with occasional drops (garbage collection)
            if random.random() < 0.1:  # 10% chance of GC
                memory_current *= 0.9
            else:
                memory_current += random.uniform(-2, 5)  # Slight increase trend

            memory_current = max(50, memory_current)  # Minimum 50MB
            memory_usage.append(round(memory_current, 2))

        # Response times (seconds)
        response_base = random.uniform(0.1, 0.5)
        response_times = []

        for i in timestamps:
            # Response times increase with load
            load_factor = 1 + (cpu_usage[i] / 100) * 2  # Up to 3x slower at 100% CPU
            noise = random.uniform(0.8, 1.2)
            timeout = 5.0 if random.random() < 0.001 else 0  # 0.1% chance of timeout

            response_time = response_base * load_factor * noise + timeout
            response_times.append(round(response_time, 3))

        return {
            "timestamps": timestamps,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "response_times": response_times,
            "throughput": [round(1.0 / max(rt, 0.001), 2) for rt in response_times],
            "error_rate": [0.01 + random.uniform(-0.005, 0.005) for _ in timestamps],
        }

    def generate_error_scenarios(self) -> List[Dict[str, Any]]:
        """Generate various error scenarios for testing."""
        error_scenarios = [
            {
                "type": "network_error",
                "message": "Connection timeout after 30 seconds",
                "code": "TIMEOUT",
                "recoverable": True,
                "retry_after": 5.0,
            },
            {
                "type": "api_error",
                "message": "API rate limit exceeded",
                "code": "RATE_LIMIT",
                "recoverable": True,
                "retry_after": 60.0,
            },
            {
                "type": "validation_error",
                "message": "Invalid input format",
                "code": "INVALID_INPUT",
                "recoverable": False,
                "retry_after": None,
            },
            {
                "type": "service_error",
                "message": "External service unavailable",
                "code": "SERVICE_DOWN",
                "recoverable": True,
                "retry_after": 30.0,
            },
            {
                "type": "memory_error",
                "message": "Insufficient memory to complete operation",
                "code": "OUT_OF_MEMORY",
                "recoverable": False,
                "retry_after": None,
            },
            {
                "type": "file_error",
                "message": "File not found or access denied",
                "code": "FILE_ERROR",
                "recoverable": False,
                "retry_after": None,
            },
            {
                "type": "audio_error",
                "message": "Audio device not available",
                "code": "AUDIO_ERROR",
                "recoverable": True,
                "retry_after": 10.0,
            },
            {
                "type": "llm_error",
                "message": "Language model service temporarily unavailable",
                "code": "LLM_ERROR",
                "recoverable": True,
                "retry_after": 15.0,
            },
        ]

        # Add context and timestamps
        for error in error_scenarios:
            error.update(
                {
                    "timestamp": datetime.now().isoformat(),
                    "session_id": str(uuid4()),
                    "component": random.choice(
                        ["core", "audio", "llm", "memory", "tools"]
                    ),
                    "severity": random.choice(["low", "medium", "high", "critical"]),
                    "user_impact": random.choice(
                        ["none", "minor", "major", "blocking"]
                    ),
                }
            )

        return error_scenarios

    def generate_memory_interactions(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate interactions for semantic memory testing."""
        interaction_types = [
            (
                "weather",
                "What's the weather like?",
                "I need your location to check the weather.",
            ),
            ("calculation", "Calculate 15 * 24", "15 * 24 equals 360."),
            (
                "programming",
                "How do I sort a list in Python?",
                "You can use the sorted() function or .sort() method.",
            ),
            (
                "file_ops",
                "How do I copy files?",
                "You can use the cp command or shutil.copy() in Python.",
            ),
            (
                "system",
                "Check system status",
                "System is running normally with 75% CPU usage.",
            ),
            (
                "audio",
                "Convert audio to MP3",
                "I can help convert audio files to MP3 format.",
            ),
            (
                "help",
                "I need help with my project",
                "I'd be happy to help! What kind of project are you working on?",
            ),
            (
                "explanation",
                "Explain machine learning",
                "Machine learning is a method of data analysis that automates analytical model building.",
            ),
        ]

        interactions = []
        base_time = datetime.now() - timedelta(days=30)

        for i in range(count):
            category, user_input, assistant_response = random.choice(interaction_types)

            # Add variation to inputs
            variations = {
                "weather": [
                    f"weather in {city}"
                    for city in ["New York", "London", "Tokyo", "Paris"]
                ],
                "calculation": [
                    f"calculate {a} * {b}" for a, b in [(12, 34), (7, 89), (15, 23)]
                ],
                "programming": [
                    "Python list sorting",
                    "sort array JavaScript",
                    "C++ vector sort",
                ],
                "file_ops": [
                    "copy files Linux",
                    "move files Windows",
                    "file operations",
                ],
                "system": ["system check", "server status", "health monitoring"],
                "audio": ["audio conversion", "sound file processing", "MP3 encoding"],
                "help": ["project assistance", "coding help", "technical support"],
                "explanation": ["explain AI", "what is ML", "define neural networks"],
            }

            if category in variations:
                user_input = random.choice(variations[category])

            interaction = {
                "user_input": user_input,
                "assistant_response": assistant_response,
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "session_id": f"session-{(i // 10) + 1}",  # Group into sessions
                "intent": category,
                "confidence": round(random.uniform(0.7, 0.95), 3),
                "tokens": random.randint(20, 200),
                "processing_time": round(random.uniform(0.3, 2.0), 3),
                "feedback": random.choice(
                    [None, None, None, "helpful", "not_helpful"]
                ),  # Sparse feedback
            }
            interactions.append(interaction)

        return interactions


class MockResponseGenerator:
    """Generate mock responses for various components."""

    @staticmethod
    def llm_response(content: str, model: str = "gpt35") -> Dict[str, Any]:
        """Generate mock LLM response."""
        return {
            "output": content,
            "metadata": {
                "model": model,
                "tokens": len(content.split()) * 1.3,  # Approximate token count
                "processing_time": round(random.uniform(0.5, 2.0), 3),
                "timestamp": datetime.now().isoformat(),
            },
        }

    @staticmethod
    def tool_response(tool_name: str, success: bool = True, output: str = None) -> str:
        """Generate mock tool response."""
        if output is None:
            output = (
                f"Tool {tool_name} executed successfully"
                if success
                else f"Tool {tool_name} failed"
            )

        response = {
            "tool": tool_name,
            "success": success,
            "output": output,
            "execution_time": round(random.uniform(0.1, 1.0), 3),
            "timestamp": datetime.now().isoformat(),
        }

        return json.dumps(response)

    @staticmethod
    def voice_response(text: str, engine: str = "google") -> Any:
        """Generate mock voice response."""

        # Return a mock audio object
        class MockAudio:
            def __init__(self, text: str, engine: str):
                self.text = text
                self.engine = engine
                self.duration = len(text) * 0.1  # Approximate duration

            def play(self):
                return f"Playing: {self.text} with {self.engine}"

            def save(self, filename: str):
                return f"Saved {filename} with {self.engine}"

        return MockAudio(text, engine)


def create_test_data_file(filename: str, data: Any, format: str = "json"):
    """Create test data file for use in tests."""
    import os

    # Create tests/data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    filepath = os.path.join(data_dir, filename)

    if format == "json":
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif format == "txt":
        with open(filepath, "w") as f:
            if isinstance(data, list):
                for item in data:
                    f.write(f"{item}\n")
            else:
                f.write(str(data))

    return filepath


# Convenience functions for quick mock data generation
def quick_conversation(message_count: int = 5) -> List[Dict[str, Any]]:
    """Quick conversation history generation."""
    return MockDataGenerator().generate_conversation_history(message_count)


def quick_audio(duration: float = 1.0) -> np.ndarray:
    """Quick audio data generation."""
    return MockDataGenerator().generate_audio_data(duration)


def quick_llm_responses(count: int = 3) -> List[Dict[str, Any]]:
    """Quick LLM responses generation."""
    return MockDataGenerator().generate_llm_responses(count)


def quick_performance_metrics(duration: int = 30) -> Dict[str, List[float]]:
    """Quick performance metrics generation."""
    return MockDataGenerator().generate_performance_metrics(duration)
