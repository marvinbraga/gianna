# Gianna: Generative Intelligent Artificial Neural Network Assistant

Gianna is a voice assistant that utilizes CrewAI and Langchain to perform complex tasks. It provides an easy way to create chain instances for working with a simple prompt, a string parser, and various LLM models such as OpenAI, Google, NVIDIA, Groq, and Ollama.

## Installation

### Prerequisites

- Python 3.10 or higher
- Poetry (Python package manager)

```shell
sudo apt update -y 
```

```shell
sudo apt install --no-install-recommends -y \
   build-essential \
   libpq-dev \
   libgirepository1.0-dev \
   libcairo2-dev pkg-config python3-dev \
   python3-pyaudio portaudio19-dev \
   libportaudio2
```

```shell
sudo apt update -y && sudo apt upgrade -y
```

### Clone

To get started with Gianna, you'll need to have Poetry installed. Poetry is a dependency management and packaging tool for Python projects. If you don't have Poetry installed, you can follow the installation instructions from the [official Poetry documentation](https://python-poetry.org/docs/#installation).

Once you have Poetry installed, follow these steps to set up Gianna:

1. Clone the Gianna repository:
   ```
   git clone https://github.com/your-username/gianna.git
   ```

2. Navigate to the project directory:
   ```
   cd gianna
   ```

3. Install the project dependencies using Poetry:
   ```
   poetry install
   ```

4. Activate the virtual environment created by Poetry:
   ```
   poetry shell
   ```

5. Run Gianna:
   ```
   python main.py
   ```

## More Info

- [**LLMs factory** - assistants/models/readme.md](assistants/models/readme.md)
- [**Recorders** - assistants/audio/recorders/readme.md](assistants/audio/recorders/readme.md)
- [**SpeechToText** - assistants/audio/stt/readme.md](assistants/audio/stt/readme.md)
- [**TextToSpeech** - assistants/audio/tts/readme.md](assistants/audio/tts/readme.md)
- [**Commands** - assistants/commands/readme.md](assistants/commands/readme.md)

## Contributing

We welcome contributions from the community to make Gianna even better! If you'd like to contribute to the project, please follow these steps:

1. Fork the Gianna repository on GitHub.

2. Create a new branch for your feature or bug fix:
   ```
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and commit them with descriptive commit messages.

4. Push your changes to your forked repository:
   ```
   git push origin feature/your-feature-name
   ```

5. Open a pull request on the main Gianna repository, describing your changes and why they should be merged.

We appreciate your contributions and will review your pull request as soon as possible. Together, let's make Gianna an even more powerful and versatile voice assistant!

If you have any questions or need further assistance, please don't hesitate to reach out to our friendly community on [GitHub Discussions](https://github.com/marvinbraga/gianna/discussions) or [Discord](https://discord.gg/xXaqSaYS).

Happy coding!