# Using Gianna Voice Assistant Framework

This document provides detailed examples of how to use the Gianna framework.

## Setup

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Create a `.env` file with your API keys:
   ```
   # Required for OpenAI models
   OPENAI_API_KEY=your_openai_api_key_here

   # Required for Google TTS and models
   GOOGLE_API_KEY=your_google_api_key_here

   # Required for ElevenLabs TTS
   ELEVEN_LABS_API_KEY=your_elevenlabs_api_key_here

   # Optional for additional LLM providers
   GROQ_API_KEY=your_groq_api_key_here
   NVIDIA_API_KEY=your_nvidia_api_key_here

   # Default model and TTS engine
   LLM_DEFAULT_MODEL=gpt35
   TTS_DEFAULT_TYPE=google
   ```

## Using LLM Models

```python
from gianna.assistants.models.factory_method import get_chain_instance

# Create a chain instance with a specific model
chain = get_chain_instance(
    model_registered_name="gpt35",  # Other options: gpt4, groq_mixtral, nvidia_mixtral, ollama_llama2, etc.
    prompt="You are a helpful assistant that provides concise answers."
)

# Process user input
response = chain.process({"input": "What is artificial intelligence?"})
print(response.output)

# List all available models
from gianna.assistants.models.registers import LLMRegister
models = sorted([model_name for model_name, _ in LLMRegister.list()])
print(models)
```

## Using Text-to-Speech

```python
from gianna.assistants.audio.tts import text_to_speech
from gianna.assistants.audio.tts.factories import TextToSpeechType

# Convert text to speech using Google TTS
text_to_speech(
    text="Hello, how can I help you today?",
    speech_type=TextToSpeechType.GOOGLE.value,
    lang='en',
    voice='default'
)

# Using ElevenLabs
text_to_speech(
    text="Hello, how can I help you today?",
    speech_type=TextToSpeechType.ELEVEN_LABS.value
)
```

## Playing Audio Files

```python
from gianna.assistants.audio.players import play_audio

# Play an audio file
play_audio("path/to/audio_file.mp3")
```

## Recording Audio

```python
from gianna.assistants.audio.recorders import get_recorder
import time

# Create a recorder for MP3
recorder = get_recorder("output.mp3")

# Start recording
recorder.start()

# Record for 5 seconds
time.sleep(5)

# Stop recording
recorder.stop()
```

## Converting Speech to Text

```python
from gianna.assistants.audio.stt import speech_to_text

# Convert a directory of MP3 files to text
documents = speech_to_text(
    audio_files_path="/path/to/audio/files",
    filetype="mp3",
    local=False  # Set to True to use local Whisper instead of API
)

# Print the transcriptions
for doc in documents:
    print(doc.page_content)
```

## Using Shell Commands

```python
from gianna.assistants.commands.factory_method import get_command
from gianna.assistants.commands.speech import SpeechCommand, SpeechType
import os

# Create a speech command for TTS feedback
speech_cmd = SpeechCommand()

# Create a shell command handler
shell_cmd = get_command(
    command_name="shell",
    name="Assistant",
    human_companion_name="User",
    text_to_speech=speech_cmd
)

# Execute a shell command
shell_cmd.execute("How do I list all files in the current directory?")
```

For more examples, check the notebook files in the `/notebooks` directory.
