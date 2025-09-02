# Gianna Audio VAD Module

A comprehensive Voice Activity Detection (VAD) system for the Gianna assistant framework, providing robust voice activity detection with multiple algorithms, extensive configuration options, and thread-safe implementations.

## Features

- **Multiple VAD Algorithms**: Energy-based (implemented), with extensible architecture for spectral, ML-based, and other algorithms
- **Thread-Safe**: All implementations use proper locking mechanisms for concurrent access
- **Configurable**: Extensive configuration options with validation
- **Real-Time Processing**: Optimized for streaming audio with low latency
- **Legacy Compatible**: Maintains compatibility with existing `VoiceActivityDetector` interface
- **Comprehensive Results**: Detailed analysis results with confidence scores, energy levels, and state information
- **Audio Utilities**: Rich set of audio processing and signal analysis utilities

## Quick Start

### Basic Usage

```python
from gianna.audio.vad import create_vad
import numpy as np

# Create a VAD detector
vad = create_vad("energy", threshold=0.03, min_silence_duration=1.5)

# Process audio chunk (numpy array or bytes)
audio_chunk = np.random.randint(-1000, 1000, 1024, dtype=np.int16)
result = vad.process_stream(audio_chunk)

# Check results
if result.is_voice_active:
    print(f"Voice detected! Energy: {result.energy_level:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
```

### Advanced Configuration

```python
from gianna.audio.vad import VADConfig, EnergyVAD

# Create detailed configuration
config = VADConfig(
    threshold=0.025,
    min_silence_duration=1.0,
    min_speech_duration=0.2,
    sample_rate=16000,
    chunk_size=1024,
    enable_callbacks=True
)

# Create VAD with configuration
vad = EnergyVAD(config)

# Set up callbacks
def on_speech_start(result):
    print(f"Speech started at {result.timestamp}")

def on_speech_end(result):
    print(f"Speech ended after {result.silence_duration:.2f}s")

vad.set_speech_start_callback(on_speech_start)
vad.set_speech_end_callback(on_speech_end)
```

### Real-Time Optimized

```python
from gianna.audio.vad import create_realtime_vad

# Create real-time optimized VAD
vad = create_realtime_vad("energy", latency_mode="low")

# Context manager usage
with vad:
    result = vad.process_stream(audio_chunk)
    print(f"Voice active: {result.is_voice_active}")
```

## Architecture

### Core Components

- **BaseVAD**: Abstract base class defining the VAD interface
- **EnergyVAD**: Energy-based VAD implementation using RMS analysis
- **VADConfig**: Configuration dataclass with validation
- **VADResult**: Comprehensive result structure with detailed information
- **VADStatistics**: Performance monitoring and metrics

### Type System

- **VADAlgorithm**: Enumeration of supported algorithms
- **VADState**: Processing state enumeration (IDLE, LISTENING, SPEAKING, SILENT, ERROR)
- **VADEventType**: Event type enumeration for callbacks
- **AudioChunk**: Type alias for audio data (numpy array or bytes)

### Utilities

- **Audio Processing**: Format conversion, normalization, resampling, validation
- **Signal Processing**: Pre-emphasis, bandpass filtering, spectral analysis, MFCC computation

## Algorithms

### Energy-Based VAD (Default)

Uses RMS (Root Mean Square) energy analysis with configurable thresholds and timing parameters:

- Computes RMS energy of audio chunks
- Compares against configurable threshold
- Manages state transitions with minimum duration requirements
- Supports adaptive threshold based on noise floor estimation
- Provides confidence scores and signal-to-noise ratio

## Configuration Options

### Core Parameters

- `threshold`: Energy threshold for voice detection (0.0-1.0, default: 0.02)
- `min_silence_duration`: Minimum silence before speech end (seconds, default: 1.0)
- `min_speech_duration`: Minimum speech duration before speech start (seconds, default: 0.1)
- `sample_rate`: Audio sample rate in Hz (default: 16000)
- `chunk_size`: Audio chunk size for processing (default: 1024)

### Advanced Parameters

- `algorithm_params`: Algorithm-specific parameters dictionary
- `enable_callbacks`: Enable/disable callback execution (default: True)
- `callback_timeout`: Maximum callback execution time (default: 5.0s)
- `buffer_size`: Internal buffer size (default: 4096)
- `max_history_length`: Maximum energy history length (default: 100)

## Legacy Compatibility

The module maintains full backward compatibility with the original `VoiceActivityDetector`:

```python
from gianna.audio.vad import VoiceActivityDetector, create_vad_detector

# Legacy interface (deprecated but supported)
vad = create_vad_detector(threshold=0.02, min_silence_duration=1.0)
energy = vad.calculate_rms_energy(audio_chunk)
stats = vad.get_statistics()
```

## Thread Safety

All VAD implementations are thread-safe with proper locking mechanisms:

- State management protected by RLock
- Callback execution serialized
- Statistics updates atomic
- Configuration changes synchronized

## Performance

- **Low Latency**: Optimized for real-time processing with <1ms overhead
- **Memory Efficient**: Bounded memory usage with configurable history limits
- **CPU Optimized**: Vectorized operations using NumPy
- **Scalable**: Suitable for both single-stream and multi-stream processing

## Extension Points

The module is designed for extensibility:

- Register custom VAD algorithms with `register_vad_algorithm()`
- Implement custom algorithms by inheriting from `BaseVAD`
- Add custom audio processing utilities
- Extend configuration with algorithm-specific parameters

## Examples

See the `examples/` directory for comprehensive usage examples including:
- Real-time audio processing
- Callback system usage
- Custom algorithm implementation
- Performance optimization techniques

## API Reference

For detailed API documentation, see the docstrings in each module file:
- `base.py`: Abstract base class and interface
- `energy_vad.py`: Energy-based VAD implementation
- `types.py`: Type definitions and data structures
- `utils/`: Audio processing and signal analysis utilities
