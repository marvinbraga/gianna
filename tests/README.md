# Gianna Testing Framework

Comprehensive testing suite for the Gianna voice assistant framework, providing thorough validation across all project phases and components.

## 📋 Overview

The Gianna Testing Framework is designed to ensure reliability, performance, and quality across all system components through comprehensive testing strategies:

- **Unit Tests**: Individual component validation with >80% coverage target
- **Integration Tests**: Multi-component workflow testing
- **Performance Tests**: Load, stress, and benchmark testing
- **End-to-End Tests**: Complete workflow validation
- **Voice Processing Tests**: Audio pipeline and streaming tests

## 🗂️ Structure

```
tests/
├── conftest.py              # Global fixtures and configuration
├── pytest.ini              # Pytest configuration
├── README.md               # This file
├── run_tests.py            # Main test runner script
├── run_coverage.py         # Coverage analysis tool
├── test_workflows.py       # Main workflow tests (FASE 5 spec)
├── unit/                   # Unit tests by component
│   ├── test_core.py        # Core state management tests
│   ├── test_models.py      # LLM provider tests
│   ├── test_audio.py       # Audio processing tests
│   └── ...
├── integration/            # Integration tests
│   ├── test_voice_workflows.py # Voice pipeline integration
│   └── ...
├── performance/            # Performance and load tests
│   ├── test_load_testing.py    # Load and stress tests
│   └── ...
├── fixtures/               # Shared test fixtures
│   ├── mock_data.py        # Mock data generators
│   └── ...
└── data/                   # Test data files
    ├── audio/              # Sample audio files
    ├── config/             # Test configurations
    └── mock_responses/     # Mock API responses
```

## 🚀 Quick Start

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-performance    # Performance tests only

# Run tests by phase
make phase1             # Core state management
make phase2             # ReAct agents & tools
make phase3             # Voice workflows
make phase4             # Memory & learning
make phase5             # End-to-end integration

# Run with coverage
make coverage
make coverage-html      # Generate HTML report

# Full CI pipeline
make ci
```

### Using Test Runner Scripts

```bash
# Basic usage
python tests/run_tests.py

# Advanced usage
python tests/run_tests.py --unit --coverage --verbose
python tests/run_tests.py --phase fase3 --slow
python tests/run_tests.py --performance --parallel 4

# Coverage analysis
python tests/run_coverage.py --html --gaps --trend
```

## 📊 Test Categories

### Unit Tests (>80% Coverage Target)

Individual component testing with minimal dependencies:

- **Core Tests** (`test_core.py`): State management, LangGraph integration
- **Model Tests** (`test_models.py`): LLM providers, chains, factories
- **Audio Tests** (`test_audio.py`): TTS, STT, VAD, audio processing
- **Agent Tests**: ReAct agents, tool integration
- **Memory Tests**: Semantic memory, learning systems

```bash
# Run unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=gianna --cov-report=html
```

### Integration Tests

Multi-component workflow validation:

- **Voice Workflows**: Complete voice input/output pipelines
- **Agent Coordination**: Multi-agent orchestration
- **Memory Integration**: Semantic memory with workflows
- **Tool Integration**: Tool usage across components

```bash
# Run integration tests
pytest tests/integration/ -v

# Specific workflows
pytest tests/integration/test_voice_workflows.py -v
```

### Performance Tests

Load, stress, and benchmark testing:

- **Load Testing**: Concurrent user scenarios
- **Stress Testing**: Beyond-capacity scenarios
- **Memory Testing**: Memory leak detection
- **Throughput Testing**: Operations per second measurement

```bash
# Run performance tests
pytest tests/performance/ -v -m performance

# Load testing specifically
python -m pytest tests/performance/test_load_testing.py -v
```

## 🧪 Test Phase Mapping

Tests are organized by project implementation phases:

### FASE 1: Foundation (Core State Management)
- `test_core.py`: GiannaState, StateManager, LangGraph integration
- `test_models.py`: LLM providers and chain factories

### FASE 2: Agent System (ReAct Agents)
- `test_react_agents.py`: Agent implementations
- `test_tools_integration.py`: Tool usage and coordination
- `test_coordination.py`: Agent orchestration

### FASE 3: Voice Pipeline (Audio Processing)
- `test_audio.py`: TTS, STT, VAD components
- `test_voice_workflows.py`: Complete voice workflows
- `test_streaming_integration.py`: Real-time audio streaming

### FASE 4: Advanced Features (Memory & Learning)
- `test_memory_integration.py`: Semantic memory integration
- `test_learning_system.py`: User adaptation and learning
- `test_optimization_performance.py`: Performance optimizations

### FASE 5: Production Readiness (End-to-End)
- `test_workflows.py`: Complete workflow integration
- `test_end_to_end.py`: Full system validation
- `test_performance.py`: Production load testing

## 🎯 Coverage Analysis

### Coverage Targets

- **Overall Coverage**: >80%
- **Unit Test Coverage**: >90% per component
- **Integration Coverage**: >70% workflow coverage
- **Branch Coverage**: >75%

### Coverage Tools

```bash
# Basic coverage
make coverage

# Detailed coverage analysis
python tests/run_coverage.py --html --gaps --trend

# Component-specific coverage
python tests/run_coverage.py --component core
```

### Coverage Reports

- **HTML Report**: `htmlcov/index.html` - Interactive coverage browser
- **XML Report**: `coverage.xml` - CI/CD integration format
- **Terminal Report**: Immediate console feedback
- **Gap Analysis**: Identifies specific coverage improvements needed

## ⚙️ Configuration

### Pytest Configuration (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
addopts =
    --strict-markers
    --cov=gianna
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-fail-under=80
    --durations=10

markers =
    unit: Unit tests for individual components
    integration: Integration tests for component interaction
    performance: Performance and load testing
    voice: Voice processing and audio tests
    slow: Tests that take more than 5 seconds
    external_api: Tests requiring external API access
```

### Environment Variables

Testing framework uses these environment variables:

```bash
export TESTING=true               # Enable test mode
export LOG_LEVEL=DEBUG           # Detailed logging
export OPENAI_API_KEY=test-key   # Mock API keys
export GOOGLE_API_KEY=test-key
# ... other API keys for testing
```

## 🔧 Fixtures and Utilities

### Global Fixtures (`conftest.py`)

- **gianna_state**: Fresh GiannaState instance
- **mock_llm**: Mocked LLM for testing
- **mock_audio**: Audio processing mocks
- **temp_files**: Temporary file management
- **performance_monitor**: Performance metrics collection

### Mock Data Generators (`fixtures/mock_data.py`)

```python
from tests.fixtures.mock_data import (
    quick_conversation,      # Generate conversation history
    quick_audio,            # Generate audio data
    quick_llm_responses,    # Generate LLM responses
    MockDataGenerator       # Full mock data suite
)

# Example usage
conversation = quick_conversation(message_count=10)
audio_data = quick_audio(duration=2.0)
llm_responses = quick_llm_responses(count=5)
```

## 📈 Performance Requirements

### Response Time Thresholds
- Simple commands: <2 seconds
- Complex workflows: <10 seconds
- LLM responses: <5 seconds
- Voice processing: <3 seconds

### Memory Usage Limits
- Normal operation: <500MB
- Peak usage: <1GB
- Per session: <50MB
- Memory leak rate: <1MB/hour

### Throughput Targets
- Interactions per minute: >10
- Concurrent sessions: >5
- LLM requests per second: >2
- Voice commands per minute: >8

## 🚨 Error Handling Testing

### Error Scenarios
- Network timeouts and connection failures
- API rate limiting and service unavailability
- Invalid input formats and data corruption
- Memory exhaustion and resource limits
- Hardware unavailability (audio devices)

### Recovery Testing
- Graceful degradation mechanisms
- Automatic retry strategies
- Fallback system activation
- Error state recovery procedures

## 🔄 Continuous Integration

### CI Pipeline (`make ci`)

1. **Environment Setup**: Install dependencies, configure environment
2. **Code Quality**: Lint, format check, type checking
3. **Unit Tests**: Run all unit tests with coverage
4. **Integration Tests**: Run workflow integration tests
5. **Performance Tests**: Run basic performance validation
6. **Coverage Analysis**: Generate coverage reports
7. **Artifact Generation**: Create test reports and documentation

### CI Configuration

```bash
# Full CI pipeline
make ci

# Fast CI (no slow tests)
make ci-fast

# Component-specific CI
make phase1 phase2 phase3 phase4 phase5
```

## 📝 Writing New Tests

### Test Structure Guidelines

```python
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.unit
@pytest.mark.fase1
class TestNewComponent:
    """Test new component functionality."""

    def test_basic_functionality(self, gianna_state):
        """Test basic component functionality."""
        # Arrange
        component = NewComponent()

        # Act
        result = component.process(gianna_state)

        # Assert
        assert result is not None
        pytest.assert_gianna_state_valid(gianna_state)
```

### Best Practices

1. **Test Organization**: Group related tests in classes
2. **Descriptive Names**: Use clear, descriptive test names
3. **AAA Pattern**: Arrange, Act, Assert structure
4. **Mock External Dependencies**: Use mocks for external services
5. **Test Edge Cases**: Include boundary and error conditions
6. **Performance Awareness**: Mark slow tests appropriately
7. **Documentation**: Include docstrings for complex tests

### Markers Usage

```python
@pytest.mark.unit           # Unit test
@pytest.mark.integration    # Integration test
@pytest.mark.performance    # Performance test
@pytest.mark.slow          # Takes >5 seconds
@pytest.mark.voice         # Voice processing test
@pytest.mark.fase1         # Phase 1 test
@pytest.mark.external_api  # Requires external API
```

## 🐛 Debugging Tests

### Debug Commands

```bash
# Run with maximum verbosity
make test-verbose

# Run only failed tests
make test-failed

# Run single test with debugging
pytest tests/unit/test_core.py::TestGiannaState::test_create_initial_state -vv -s

# Run with debugger
pytest --pdb tests/unit/test_core.py
```

### Common Issues

1. **Import Errors**: Check PYTHONPATH and module imports
2. **Mock Issues**: Verify mock patches and return values
3. **Async Tests**: Use `pytest-asyncio` and proper async fixtures
4. **Resource Conflicts**: Check for port conflicts, file locks
5. **Environment Issues**: Verify test environment variables

## 📚 Additional Resources

### Performance Benchmarking

```bash
# Run benchmarks
pytest --benchmark-only

# Save benchmark results
pytest --benchmark-save=baseline

# Compare benchmarks
pytest --benchmark-compare=baseline
```

### Test Data Management

- **Audio Files**: Sample audio files in `tests/data/audio/`
- **Mock Responses**: Stored API responses in `tests/data/mock_responses/`
- **Configuration Files**: Test configurations in `tests/data/config/`

### Integration with Development Tools

- **Pre-commit Hooks**: Automatically run tests before commits
- **IDE Integration**: Configure IDE to run tests and show coverage
- **Continuous Deployment**: Integrate with CD pipelines

## 🆘 Troubleshooting

### Common Solutions

```bash
# Install missing dependencies
poetry install --with test

# Clear pytest cache
make clean

# Check test dependencies
make check-deps

# Regenerate coverage data
make coverage-report
```

### Getting Help

1. Check this README for common patterns
2. Examine existing tests for examples
3. Review test fixtures in `conftest.py`
4. Use `make list-tests` to see available tests
5. Run `python tests/run_tests.py --help` for options

---

**Happy Testing!** 🧪✨

For questions or contributions to the testing framework, please ensure all new tests follow the established patterns and include appropriate documentation.
