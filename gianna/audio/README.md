# Gianna Audio Streaming Module

This module provides real-time voice streaming capabilities for the Gianna assistant, enabling continuous voice conversations with integrated Voice Activity Detection (VAD), Speech-to-Text (STT), Large Language Model (LLM) processing, and Text-to-Speech (TTS).

## Architecture Overview

```
Audio Input → VAD → Buffer → STT → LLM → TTS → Audio Output
```

The streaming pipeline consists of:

1. **Audio Input**: Real-time microphone capture using PyAudio
2. **Voice Activity Detection (VAD)**: Detects when speech starts and ends
3. **Audio Buffer**: Thread-safe circular buffer for audio chunks
4. **Speech-to-Text**: Converts speech to text using Whisper
5. **LLM Processing**: Processes user input through configured language model
6. **Text-to-Speech**: Synthesizes and plays assistant responses

## Key Components

### StreamingVoicePipeline

The main class that orchestrates the entire voice conversation pipeline.

**Features:**
- Asynchronous operation with asyncio
- Thread-safe audio processing
- Configurable VAD parameters
- LLM model selection
- TTS engine configuration
- Event callbacks for pipeline states
- Real-time statistics and monitoring
- Error handling and recovery

**States:**
- `STOPPED`: Pipeline is inactive
- `LISTENING`: Actively listening for speech
- `PROCESSING`: Processing speech through STT and LLM
- `SPEAKING`: Playing TTS response
- `ERROR`: Error state requiring attention

### AudioBuffer

Thread-safe circular buffer for managing audio chunks.

**Features:**
- Fixed-size buffer with automatic overflow handling
- Thread-safe operations
- Audio chunk combination
- Statistics tracking

### Usage Examples

#### Basic Voice Assistant

```python
import asyncio
from gianna.audio.streaming import StreamingVoicePipeline

async def main():
    # Create callbacks
    def on_listening():
        print("👂 Listening...")

    def on_speech_detected(transcript):
        print(f"🎤 You said: {transcript}")

    def on_response(response):
        print(f"💬 Assistant: {response}")

    # Create pipeline
    pipeline = StreamingVoicePipeline(
        model_name="gpt35",
        system_prompt="You are a helpful assistant.",
        on_listening=on_listening,
        on_speech_detected=on_speech_detected,
        on_response=on_response
    )

    # Start listening
    await pipeline.start_listening()

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await pipeline.stop_listening()

# Run
asyncio.run(main())
```

#### Convenience Function

```python
from gianna.audio.streaming import create_voice_assistant

# Quick setup
assistant = await create_voice_assistant(
    model_name="gpt4",
    system_prompt="You are Gianna, a helpful voice assistant."
)

await assistant.start_listening()
```

#### Text-Only Mode (No Audio)

```python
# Process text input directly
response = await pipeline.process_text_input("Hello, how are you?")
print(f"Response: {response}")
```

## Configuration Options

### Audio Parameters
- `sample_rate`: Audio sample rate (default: 16000 Hz)
- `chunk_size`: Audio chunk size (default: 1024)
- `channels`: Number of audio channels (default: 1 for mono)

### VAD Parameters
- `vad_threshold`: Voice activity threshold 0.0-1.0 (default: 0.02)
- `min_silence_duration`: Minimum silence to end speech (default: 1.0s)

### LLM Configuration
- `model_name`: Registered model name (default: "gpt35")
- `system_prompt`: System prompt for the assistant

### TTS Configuration
- `tts_type`: TTS engine type (default: "google")
- `tts_language`: Language code (default: "en")
- `tts_voice`: Voice identifier (default: "default")

### Buffer Configuration
- `buffer_max_chunks`: Maximum audio chunks in buffer (default: 1000)

## Event Callbacks

The pipeline supports various event callbacks for monitoring and customization:

- `on_listening()`: Called when starting to listen for speech
- `on_speech_detected(transcript)`: Called when speech is transcribed
- `on_processing()`: Called when processing LLM request
- `on_response(response)`: Called when LLM response is received
- `on_speaking()`: Called when starting TTS playback
- `on_error(exception)`: Called when errors occur

## Runtime Control

### State Management
```python
# Check current state
status = pipeline.get_pipeline_status()
print(f"State: {status['state']}")

# Pause/resume listening
pipeline.pause_listening()
pipeline.resume_listening()
```

### VAD Adjustment
```python
# Update VAD settings during runtime
pipeline.update_vad_settings(
    threshold=0.03,
    min_silence_duration=1.5
)
```

### Statistics Monitoring
```python
status = pipeline.get_pipeline_status()
stats = status['stats']
print(f"Chunks processed: {stats['chunks_processed']}")
print(f"Speech detections: {stats['speech_detected_count']}")
print(f"LLM requests: {stats['llm_requests']}")
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Audio errors**: Stream reconnection and recovery
- **STT errors**: Fallback and retry mechanisms
- **LLM errors**: Request timeout and error reporting
- **TTS errors**: Synthesis failure handling

All errors trigger the `on_error` callback and are logged for debugging.

## Thread Safety

The streaming pipeline is designed to be thread-safe:

- Audio processing runs in separate thread
- VAD and buffer operations are protected with locks
- Async/await pattern for pipeline control
- Clean shutdown and resource cleanup

## Requirements

- Python 3.8+
- PyAudio for real-time audio capture
- NumPy for audio processing
- Existing Gianna components (VAD, STT, LLM, TTS)

## Performance Considerations

- **Latency**: Typical end-to-end latency of 2-5 seconds
- **Memory**: Audio buffer size affects memory usage
- **CPU**: VAD processing is lightweight, STT/LLM are compute-intensive
- **Concurrency**: Uses thread pool for parallel processing

## Troubleshooting

### Common Issues

1. **No audio input detected**
   - Check microphone permissions
   - Adjust VAD threshold (lower = more sensitive)
   - Verify PyAudio installation

2. **High latency**
   - Reduce chunk size for lower latency
   - Use faster LLM models
   - Optimize TTS settings

3. **Frequent false positives**
   - Increase VAD threshold
   - Increase minimum silence duration

4. **STT not working**
   - Check Whisper API configuration
   - Verify audio file format support
   - Check network connectivity

### Debug Mode

Enable verbose logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Create pipeline with debug info
pipeline = StreamingVoicePipeline(...)
```

## Demo Application

Run the interactive demo:

```bash
# Voice mode (requires microphone)
python examples/streaming_voice_demo.py

# Text-only mode
python examples/streaming_voice_demo.py --text
```

## Integration with Gianna

The streaming module integrates seamlessly with existing Gianna components:

- **Models**: Uses `gianna.assistants.models.factory_method`
- **STT**: Uses `gianna.assistants.audio.stt.factory_method`
- **TTS**: Uses `gianna.assistants.audio.tts.factory_method`
- **VAD**: Uses `gianna.assistants.audio.vad`

This ensures compatibility with all supported LLM providers, audio formats, and TTS engines.
