# Gianna Voice Activity Detection (VAD) System

A comprehensive Voice Activity Detection system with multiple advanced algorithms, extensive configuration options, and a clean factory-based architecture.

## Overview

The Gianna VAD module provides state-of-the-art voice activity detection capabilities through multiple specialized algorithms, each optimized for different use cases and audio conditions. The system follows a clean factory pattern with optional dependency handling and graceful fallbacks.

## Supported Algorithms

### 1. Energy-based VAD (`energy`)
**Always Available** - No external dependencies

**Best For**: Quiet environments, real-time applications, resource-constrained systems
- **Accuracy**: Good (80-90%)
- **CPU Usage**: Very Low
- **Memory**: Very Low (~1MB)
- **Latency**: Excellent (<1ms)

```python
from gianna.audio.vad import create_vad

# Basic energy VAD
vad = create_vad("energy", threshold=0.02, min_silence_duration=1.0)

# With adaptive threshold
from gianna.audio.vad import create_energy_vad
vad = create_energy_vad(threshold=0.02, adaptive_threshold=True)
```

### 2. Spectral VAD (`spectral`)
**Requires**: `scipy`, `numpy`

**Best For**: Noisy environments, music rejection, multi-speaker scenarios
- **Accuracy**: Very Good (85-95%)
- **CPU Usage**: Medium
- **Memory**: Medium (~5-20MB)
- **Latency**: Medium (5-20ms)

```python
# Spectral VAD optimized for noisy environments
vad = create_vad("spectral",
                threshold=0.3,
                fft_size=1024,
                use_spectral_flux=True,
                noise_floor_adaptation=True)

# Using convenience function
from gianna.audio.vad import create_spectral_vad
vad = create_spectral_vad(
    threshold=0.3,
    use_spectral_flux=True,
    noise_floor_adaptation=True
)
```

### 3. WebRTC VAD (`webrtc`)
**Requires**: `webrtcvad`

**Best For**: Real-time communication, VoIP, streaming applications
- **Accuracy**: Good-Very Good (85-95%)
- **CPU Usage**: Low
- **Memory**: Very Low (~1-5MB)
- **Latency**: Excellent (<5ms)

```python
# WebRTC VAD with balanced aggressiveness
vad = create_vad("webrtc",
                aggressiveness=2,
                frame_duration_ms=20,
                confidence_smoothing=True)

# Using convenience function
from gianna.audio.vad import create_webrtc_vad
vad = create_webrtc_vad(aggressiveness=2, confidence_smoothing=True)
```

### 4. Silero VAD (`silero`)
**Requires**: `torch`, `torchaudio`

**Best For**: Highest accuracy requirements, challenging audio conditions
- **Accuracy**: Excellent (95-98%)
- **CPU Usage**: Medium-High
- **Memory**: Medium (~20-50MB)
- **Latency**: Medium (10-50ms)
- **GPU Support**: Yes

```python
# Silero VAD with GPU acceleration
vad = create_vad("silero",
                speech_threshold=0.5,
                device="cuda",  # or "cpu", "auto"
                model_name="silero_vad")

# Using convenience function
from gianna.audio.vad import create_silero_vad
vad = create_silero_vad(speech_threshold=0.6, device="auto")
```

### 5. Adaptive VAD (`adaptive`)
**Combines**: Multiple algorithms with voting strategies

**Best For**: Production systems, diverse audio conditions, maximum robustness
- **Accuracy**: Excellent (90-98%)
- **CPU Usage**: Variable (sum of constituent algorithms)
- **Memory**: Medium-High
- **Latency**: Variable (limited by slowest algorithm)

```python
# Adaptive VAD combining multiple algorithms
vad = create_vad("adaptive",
                algorithms=["webrtc", "spectral", "silero"],
                voting_strategy="weighted",
                adaptation_mode="hybrid")

# Using convenience function
from gianna.audio.vad import create_adaptive_vad
vad = create_adaptive_vad(
    algorithms=["webrtc", "spectral"],
    voting_strategy="consensus",
    consensus_threshold=0.7
)
```

## Quick Start Guide

### Basic Usage

```python
from gianna.audio.vad import create_vad
import numpy as np

# Create a VAD detector
vad = create_vad("webrtc", aggressiveness=2)

# Initialize the detector
if not vad.initialize():
    raise RuntimeError("Failed to initialize VAD")

# Process audio data
audio_chunk = np.random.randint(-32768, 32767, 1024, dtype=np.int16)
result = vad.process_stream(audio_chunk)

# Check results
if result.is_voice_active:
    print(f"Voice detected! Confidence: {result.confidence:.3f}")
else:
    print(f"Silence detected. Energy: {result.energy_level:.3f}")

# Cleanup when done
vad.cleanup()
```

### Context Manager Usage

```python
from gianna.audio.vad import create_vad

with create_vad("silero") as vad:
    result = vad.process_stream(audio_chunk)
    if result.is_voice_active:
        print("Voice activity detected!")
```

### Real-time Processing

```python
from gianna.audio.vad import create_realtime_vad

# Optimized for low-latency real-time processing
vad = create_realtime_vad("webrtc", latency_mode="low")

# Process audio stream
for audio_chunk in audio_stream:
    result = vad.process_stream(audio_chunk)

    # Handle voice activity events
    if result.state_changed:
        if result.event_type.value == "speech_start":
            print("Speech started!")
        elif result.event_type.value == "speech_end":
            print("Speech ended!")
```

### Advanced Configuration

```python
from gianna.audio.vad import SpectralVADConfig, SpectralVAD

# Create detailed configuration
config = SpectralVADConfig(
    threshold=0.3,
    min_silence_duration=0.5,
    min_speech_duration=0.1,
    sample_rate=16000,
    fft_size=2048,
    hop_length=512,
    use_spectral_centroid=True,
    use_spectral_rolloff=True,
    use_spectral_flux=True,
    noise_floor_adaptation=True,
    spectral_subtraction=True
)

# Create VAD with custom configuration
vad = SpectralVAD(config)
```

## Callback System

```python
from gianna.audio.vad import create_vad, VADEventType

def on_speech_start(result):
    print(f"Speech started at {result.timestamp}")

def on_speech_end(result):
    print(f"Speech ended at {result.timestamp}")

def on_threshold_changed(event_type, event_data):
    print(f"Threshold changed: {event_data}")

# Set up callbacks
vad = create_vad("energy")
vad.set_speech_start_callback(on_speech_start)
vad.set_speech_end_callback(on_speech_end)
vad.add_event_callback(VADEventType.THRESHOLD_CHANGED, on_threshold_changed)
```

## Performance Monitoring

### Algorithm Benchmarking

```python
# Benchmark different algorithms
algorithms = ["energy", "webrtc", "spectral", "silero"]
results = {}

for alg_name in algorithms:
    try:
        vad = create_vad(alg_name)
        if vad.initialize():
            # Run benchmark
            if hasattr(vad, 'get_model_benchmark'):
                benchmark = vad.get_model_benchmark(test_duration=10.0)
                results[alg_name] = benchmark
                print(f"{alg_name}: {benchmark['real_time_factor']:.2f}x real-time")
    except Exception as e:
        print(f"{alg_name} unavailable: {e}")
```

### Performance Statistics

```python
# Get detailed performance statistics
vad = create_vad("spectral")
vad.initialize()

# Process some audio...
for chunk in audio_chunks:
    vad.process_stream(chunk)

# Get statistics
stats = vad.statistics.to_dict()
print(f"Processed {stats['total_chunks_processed']} chunks")
print(f"Speech detection ratio: {stats['speech_detection_ratio']:.3f}")
print(f"Average processing time: {stats['average_processing_time']:.3f}ms")

# Algorithm-specific stats
if hasattr(vad, 'performance_stats'):
    perf_stats = vad.performance_stats
    print(f"Algorithm performance: {perf_stats}")
```

## Advanced Features

### Adaptive Algorithm Weighting

```python
from gianna.audio.vad import create_adaptive_vad, VotingStrategy, AdaptationMode

# Create adaptive VAD with custom configuration
vad = create_adaptive_vad(
    algorithms=["webrtc", "spectral", "silero"],
    voting_strategy=VotingStrategy.WEIGHTED,
    adaptation_mode=AdaptationMode.HYBRID,
    adaptation_rate=0.05
)

# Monitor adaptation
vad.initialize()
for chunk in audio_chunks:
    result = vad.process_stream(chunk)

    # Check current algorithm weights
    status = vad.get_algorithm_status()
    for alg_name, info in status.items():
        print(f"{alg_name}: weight={info['current_weight']:.3f}")
```

### Custom Voting Strategies

```python
# Different voting strategies
strategies = [
    "majority",     # Simple majority vote
    "weighted",     # Weighted by algorithm confidence
    "unanimous",    # All algorithms must agree
    "consensus",    # Configurable consensus threshold
    "adaptive",     # Adaptive weighting
    "hierarchical"  # Priority-based decision
]

for strategy in strategies:
    vad = create_adaptive_vad(voting_strategy=strategy)
    # Test strategy...
```

### Noise Floor Adaptation

```python
# Energy VAD with adaptive threshold
vad = create_vad("energy", threshold=0.02)
vad.initialize()

# Process quiet audio to establish noise floor
for quiet_chunk in quiet_audio:
    vad.process_stream(quiet_chunk)

# Automatically adjust threshold based on noise floor
if hasattr(vad, 'set_adaptive_threshold'):
    vad.set_adaptive_threshold(multiplier=3.0)
    print(f"Adaptive threshold: {vad.get_adaptive_threshold()}")
```

## Error Handling and Fallbacks

### Graceful Degradation

```python
from gianna.audio.vad import create_vad

def create_best_available_vad():
    """Create the best available VAD with fallbacks."""

    # Try algorithms in order of preference
    preferred_algorithms = ["silero", "webrtc", "spectral", "energy"]

    for algorithm in preferred_algorithms:
        try:
            vad = create_vad(algorithm)
            if vad.initialize():
                print(f"Using {algorithm} VAD")
                return vad
        except Exception as e:
            print(f"{algorithm} VAD unavailable: {e}")

    raise RuntimeError("No VAD algorithms available")

# Use the best available algorithm
vad = create_best_available_vad()
```

### Dependency Checking

```python
from gianna.audio.vad import get_available_algorithms

# Check what algorithms are available
available = get_available_algorithms()
print(f"Available algorithms: {available}")

# Check specific algorithm availability
if "silero" in available:
    print("Silero VAD is available (PyTorch installed)")
else:
    print("Silero VAD requires PyTorch")

if "webrtc" in available:
    print("WebRTC VAD is available")
else:
    print("WebRTC VAD requires webrtcvad library")
```

## Installation Requirements

### Core Requirements (Always Available)
```bash
pip install numpy scipy
```

### Optional Dependencies

#### For WebRTC VAD:
```bash
pip install webrtcvad
```

#### For Silero VAD:
```bash
pip install torch torchaudio
```

#### For GPU acceleration (Silero):
```bash
# CUDA version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or ROCm for AMD GPUs
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## Configuration Reference

### Common Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.02-0.5 | Voice activity threshold |
| `min_silence_duration` | float | 0.5-1.0 | Minimum silence duration (seconds) |
| `min_speech_duration` | float | 0.1 | Minimum speech duration (seconds) |
| `sample_rate` | int | 16000 | Audio sample rate (Hz) |
| `chunk_size` | int | 1024 | Audio chunk size (samples) |
| `channels` | int | 1 | Number of audio channels |

### Algorithm-Specific Parameters

#### Energy VAD
- `adaptive_threshold`: Enable automatic threshold adjustment
- `noise_adaptation_rate`: Rate of noise floor adaptation

#### Spectral VAD
- `fft_size`: FFT window size for spectral analysis
- `use_spectral_flux`: Enable spectral flux feature
- `noise_floor_adaptation`: Enable adaptive noise floor
- `spectral_subtraction`: Enable spectral subtraction

#### WebRTC VAD
- `aggressiveness`: Aggressiveness level (0-3)
- `frame_duration_ms`: Frame duration (10, 20, or 30ms)
- `confidence_smoothing`: Enable confidence smoothing

#### Silero VAD
- `speech_threshold`: Speech detection threshold
- `device`: Inference device ("cpu", "cuda", "auto")
- `model_name`: Silero model to use

#### Adaptive VAD
- `algorithms`: List of algorithms to combine
- `voting_strategy`: How to combine algorithm results
- `adaptation_mode`: How to adapt algorithm weights

## Best Practices

### 1. Choose the Right Algorithm

- **Energy VAD**: Use for quiet environments and when you need minimal resource usage
- **WebRTC VAD**: Use for real-time applications like VoIP or streaming
- **Spectral VAD**: Use when you need to reject music or handle noisy environments
- **Silero VAD**: Use when you need maximum accuracy and have sufficient computational resources
- **Adaptive VAD**: Use for production systems where robustness is critical

### 2. Optimize for Your Use Case

```python
# Low-latency real-time processing
vad = create_realtime_vad("webrtc", latency_mode="low")

# High-accuracy batch processing
vad = create_vad("silero", speech_threshold=0.6)

# Balanced performance for production
vad = create_adaptive_vad(algorithms=["webrtc", "spectral"])
```

### 3. Handle Resources Properly

```python
# Always use context managers or explicit cleanup
try:
    vad = create_vad("silero")
    vad.initialize()
    # Process audio...
finally:
    vad.cleanup()  # Important for GPU memory management
```

### 4. Monitor Performance

```python
# Regular performance monitoring
if len(vad._inference_times) > 100:
    avg_time = np.mean(vad._inference_times[-100:])
    if avg_time > target_latency:
        print("Performance degradation detected")
```

### 5. Implement Fallbacks

```python
# Always have a fallback strategy
primary_vad = create_vad("silero")
fallback_vad = create_vad("energy")

def detect_voice_activity(audio_chunk):
    try:
        return primary_vad.process_stream(audio_chunk)
    except Exception:
        return fallback_vad.process_stream(audio_chunk)
```

## Troubleshooting

### Common Issues

1. **"Algorithm not available" errors**: Check optional dependencies
2. **High CPU usage**: Consider using WebRTC or Energy VAD
3. **High memory usage**: Reduce buffer sizes and history lengths
4. **GPU out of memory**: Use CPU mode for Silero VAD
5. **Poor accuracy in noisy environments**: Use Spectral or Adaptive VAD

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed VAD logging
vad = create_vad("adaptive")
# Detailed logs will show algorithm decisions and performance
```

### Performance Profiling

```python
import time

def profile_vad_performance(vad, audio_chunks):
    """Profile VAD performance."""
    times = []
    for chunk in audio_chunks:
        start = time.time()
        result = vad.process_stream(chunk)
        times.append(time.time() - start)

    print(f"Average processing time: {np.mean(times)*1000:.2f}ms")
    print(f"Max processing time: {np.max(times)*1000:.2f}ms")
    print(f"Real-time factor: {np.mean(times) / (len(chunk) / 16000):.2f}x")
```

## Contributing

The Gianna VAD system is designed to be extensible. To add a new VAD algorithm:

1. Inherit from `BaseVAD`
2. Implement required abstract methods
3. Register the algorithm in `_VAD_REGISTRY`
4. Add optional dependency handling
5. Include comprehensive tests and documentation

See the existing implementations for reference patterns.

## License

MIT License - see LICENSE file for details.
