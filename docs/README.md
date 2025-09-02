# Gianna Framework Documentation

Welcome to the comprehensive documentation for the Gianna framework - a modular, extensible Python framework for building voice-enabled AI assistants.

## 📚 Documentation Overview

This documentation provides complete coverage of the Gianna framework, from basic concepts to advanced implementation details. Whether you're a new developer getting started or an experienced contributor extending the framework, these guides will help you understand and work with Gianna effectively.

## 📋 Documentation Structure

### 🏗️ [Architecture Overview](ARCHITECTURE.md)
**Complete framework architecture and design patterns**
- System overview and core principles
- Component architecture and relationships
- Design patterns (Factory Method, Registry, Singleton)
- Data flow and processing pipelines
- Extension points and customization
- Performance considerations and security

### 🤖 [LLM Integration System](LLM_INTEGRATION.md)
**Comprehensive LLM provider integration and management**
- Multi-provider support (OpenAI, Anthropic, Google, Groq, NVIDIA, xAI, Cohere, Ollama)
- Factory method implementation for seamless provider switching
- LangChain integration patterns
- Error handling and resilience strategies
- Performance optimization and caching
- Adding new LLM providers

### 🎵 [Audio Processing System](AUDIO_SYSTEM.md)
**Complete audio pipeline documentation**
- Text-to-speech (TTS) with multiple services
- Speech-to-text (STT) with cloud and local options
- Multi-format audio support (MP3, WAV, M4A, FLAC, OGG, AAC)
- Real-time audio processing and streaming
- Voice activity detection
- Audio quality enhancement

### ⚡ [Command System](COMMAND_SYSTEM.md)
**Extensible command processing and execution**
- Command registration and discovery
- Voice-activated command execution
- Shell command generation with AI
- Multi-step workflow processing
- Custom command development
- Integration with LLM processing

### 🛠️ [Development Guide](DEVELOPMENT_GUIDE.md)
**Complete development setup and contribution guidelines**
- Environment setup and prerequisites
- Development workflow and Git practices
- Adding new components (LLM providers, audio formats, commands)
- Testing strategies and quality assurance
- Code standards and documentation
- Deployment and distribution

### 🚀 [Future Roadmap](FUTURE_ROADMAP.md)
**Strategic direction and enhancement plans**
- Current limitations analysis
- LangGraph integration strategy
- Multi-agent system development
- Enterprise features roadmap
- Performance and scalability improvements
- Technology integration priorities

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/gianna.git
cd gianna

# Install with Poetry
poetry install
poetry shell

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage
```python
from gianna.assistants.models.factory_method import get_chain_instance

# Initialize LLM chain
chain = get_chain_instance("openai", "You are a helpful assistant. {input}")

# Process user input
response = chain.invoke({"input": "What is Python?"})
print(response.output)
```

### Voice Interaction
```python
from gianna.assistants.audio.tts.factory_method import get_tts_instance
from gianna.assistants.audio.stt.factory_method import get_stt_loader

# Text-to-speech
tts = get_tts_instance("elevenlabs", voice="Rachel")
tts.synthesize("Hello, welcome to Gianna!")

# Speech-to-text
stt_loader = get_stt_loader("whisper", "audio_file.wav")
transcription = stt_loader.load()
print(transcription.docs[0].page_content)
```

## 🏛️ Framework Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                        Gianna Framework                      │
├─────────────────────────────────────────────────────────────┤
│                      Main Application                       │
│                      (main.py)                              │
├─────────────────────┬───────────────┬─────────────────────────┤
│     LLM Models      │ Audio System  │   Command System        │
│   (assistants/      │ (assistants/  │  (assistants/commands/) │
│    models/)         │  audio/)      │                         │
├─────────────────────┼───────────────┼─────────────────────────┤
│ • OpenAI           │ • TTS Services│ • Voice Commands        │
│ • Anthropic        │ • STT Services│ • Shell Integration     │
│ • Google           │ • Audio Players│ • Workflow Execution   │
│ • Groq/NVIDIA      │ • Recorders   │ • Custom Actions        │
│ • 8+ Providers     │ • Multi-format│ • AI-Powered Automation │
└─────────────────────┴───────────────┴─────────────────────────┘
```

## 🔧 Key Features

### 🤖 **Multi-LLM Support**
- 8+ LLM providers with unified interface
- Seamless provider switching
- Intelligent fallback mechanisms
- Cost and performance optimization

### 🎵 **Comprehensive Audio**
- Text-to-speech with premium voices
- Speech-to-text with high accuracy
- Multiple audio formats support
- Real-time processing capabilities

### ⚡ **Smart Commands**
- Voice-activated command execution
- Natural language to shell commands
- Extensible command framework
- AI-powered automation workflows

### 🏗️ **Modular Architecture**
- Plugin-based extension system
- Clean separation of concerns
- Factory pattern implementations
- Easy customization and testing

## 📖 Learning Path

### 🟢 **Beginners**
1. Start with [Architecture Overview](ARCHITECTURE.md) for framework understanding
2. Follow [Development Guide](DEVELOPMENT_GUIDE.md) for environment setup
3. Explore [LLM Integration](LLM_INTEGRATION.md) for basic AI integration
4. Try [Audio System](AUDIO_SYSTEM.md) for voice capabilities

### 🟡 **Intermediate**
1. Dive into [Command System](COMMAND_SYSTEM.md) for workflow automation
2. Study extension patterns in [Architecture](ARCHITECTURE.md)
3. Implement custom components using [Development Guide](DEVELOPMENT_GUIDE.md)
4. Explore advanced features in each system documentation

### 🔴 **Advanced**
1. Review [Future Roadmap](FUTURE_ROADMAP.md) for upcoming features
2. Contribute to LangGraph integration planning
3. Develop enterprise extensions and plugins
4. Participate in framework architecture discussions

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](DEVELOPMENT_GUIDE.md) for:
- Setting up development environment
- Code standards and practices
- Testing requirements
- Pull request process

### Areas for Contribution
- **LLM Providers**: Add new language model integrations
- **Audio Formats**: Extend audio processing capabilities
- **Commands**: Create specialized command implementations
- **Documentation**: Improve guides and examples
- **Testing**: Expand test coverage and validation

## 📞 Support and Community

### 📧 Getting Help
- **Documentation**: Check relevant sections above
- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for questions
- **Examples**: See `notebooks/` directory for usage examples

### 🏘️ Community Resources
- **GitHub Repository**: Main development hub
- **Wiki**: Community-maintained documentation
- **Examples Gallery**: Real-world implementation examples
- **Plugin Registry**: Community-developed extensions

## 📄 License

Gianna is released under the Apache 2.0 License. See [LICENSE](../LICENSE) for details.

## 🙏 Acknowledgments

- **LangChain**: Core LLM integration framework
- **LangGraph**: Advanced workflow orchestration (planned)
- **Audio Libraries**: PyDub, PyAudio for audio processing
- **Contributors**: All framework developers and contributors

---

**Ready to get started?** Begin with the [Architecture Overview](ARCHITECTURE.md) to understand the framework, then follow the [Development Guide](DEVELOPMENT_GUIDE.md) to set up your environment and start building with Gianna!
