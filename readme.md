# Gianna - Voice Assistant Framework

Gianna is a modular and extensible voice assistant framework built with Python, designed to integrate with various LLM providers, audio processing tools, and command systems.

## Features

- **LLM Integration**: Connect with OpenAI, Google, NVIDIA, Groq, and Ollama models through a unified interface
- **Audio Processing**:
  - Play various audio formats (MP3, WAV, OGG, FLAC, M4A, AAC)
  - Record audio in multiple formats
  - Speech-to-Text (STT) using Whisper
  - Text-to-Speech (TTS) using Google, ElevenLabs, and more
- **Command System**: Execute shell commands and other actions based on voice input

## Quick Start

1. Clone the repository
2. Create a `.env` file based on `.env.example` with your API keys
3. Install dependencies:
   ```
   pip install -e .
   ```
   or
   ```
   poetry install
   ```

4. Run the example:
   ```python
   from gianna.assistants.models.factory_method import get_chain_instance

   # Create a chain instance
   chain = get_chain_instance(
       model_registered_name="gpt35",
       prompt="You are a helpful assistant."
   )

   # Process user input
   response = chain.process({"input": "Tell me about coffee."})
   print(response.output)
   ```

## Environment Variables

Create a `.env` file with the following variables:

```
# LLM configuration
LLM_DEFAULT_MODEL=gpt35

# TTS configuration
TTS_DEFAULT_TYPE=google

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
ELEVEN_LABS_API_KEY=your_elevenlabs_api_key_here
```

## Architecture

Gianna uses several design patterns:

- **Factory Method**: Create objects without specifying their concrete class
- **Abstract Factory**: Create families of related objects
- **Registry Pattern**: Register and retrieve factories dynamically

Check the `/notebooks` directory for detailed examples.

## License

See the LICENSE file for details.
