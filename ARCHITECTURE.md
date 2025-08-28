# Gianna Architecture

Gianna is built with a modular, component-based architecture that follows SOLID principles and uses several design patterns to ensure flexibility and extensibility.

## Design Patterns Used

### Factory Method Pattern

This pattern is used extensively throughout Gianna to create objects without specifying their concrete class. Examples:

- `get_chain_instance()` in `models/factory_method.py`
- `get_recorder()` in `audio/recorders/factory_method.py`
- `text_to_speech()` in `audio/tts/factory_method.py`

### Abstract Factory Pattern

Used to create families of related objects:

- `TextToSpeechFactory` for creating TTS engines
- `AudioPlayerFactory` for creating audio players
- `AudioRecorderFactory` for creating audio recorders

### Registry Pattern

Used for dynamic registration and discovery of components:

- `LLMRegister` in `models/registers.py`
- `CommandRegister` in `commands/register.py`

### Strategy Pattern

Different implementations of:

- Audio players (MP3, WAV, OGG, etc.)
- Audio recorders
- TTS engines
- LLM providers

### Singleton Pattern

Used in the `LLMRegister` to maintain a single registry of models.

## Main Components

### LLM Models

- **Abstract Classes**: `AbstractLLMFactory`, `AbstractBasicChain`
- **Registry**: `LLMRegister`
- **Implementations**: OpenAI, Google, NVIDIA, Groq, Ollama

### Audio Processing

#### Players
- **Abstract Class**: `AbstractAudioPlayer`
- **Factory**: `AudioPlayerFactory`
- **Implementations**: MP3, WAV, OGG, FLAC, M4A, AAC

#### Recorders
- **Abstract Class**: `AbstractAudioRecorder`
- **Factory**: `AudioRecorderFactory`
- **Implementations**: MP3, WAV, OGG, M4A

#### TTS (Text-to-Speech)
- **Abstract Class**: `AbstractTextToSpeech`
- **Factory**: `TextToSpeechFactory`
- **Implementations**: Google, ElevenLabs, Whisper

#### STT (Speech-to-Text)
- **Abstract Class**: `AbstractAudioLoader`, `AbstractSpeechToTextLoader`
- **Implementations**: Whisper (local and API)

### Commands

- **Abstract Classes**: `AbstractCommand`, `AbstractCommandFactory`
- **Registry**: `CommandRegister`
- **Implementations**: ShellCommand, SpeechCommand

## Folder Structure

- **gianna/**
  - **assistants/**
    - **models/**: LLM integration
    - **audio/**: Audio processing components
      - **players/**: Audio playback
      - **recorders/**: Audio recording
      - **tts/**: Text-to-Speech
      - **stt/**: Speech-to-Text
    - **commands/**: Command system
  - **resources/**: Static resources

## Key Interfaces

### LLM Integration

```python
class AbstractBasicChain(ABC):
    @abstractmethod
    def process(self, inputs):
        pass
```

### Audio Playing

```python
class AbstractAudioPlayer(ABC):
    @abstractmethod
    def play(self):
        pass

    @abstractmethod
    def wait_until_finished(self):
        pass
```

### Audio Recording

```python
class AbstractAudioRecorder(ABC):
    @abstractmethod
    def record(self):
        pass

    @abstractmethod
    def stop(self):
        pass
```

### Text-to-Speech

```python
class AbstractTextToSpeech(ABC):
    @abstractmethod
    def synthesize(self, text):
        pass
```

### Commands

```python
class AbstractCommand(ABC):
    @abstractmethod
    def execute(self, prompt, **kwargs):
        pass
```

## Dependency Flow

1. User or application code calls factory methods (`get_chain_instance()`, `text_to_speech()`, etc.)
2. Factory methods use registries or factories to create the appropriate implementation
3. Concrete implementations (which inherit from abstract classes) handle the specific tasks
4. Results flow back to the calling code

This architecture makes it easy to add new implementations of any component without modifying existing code.
