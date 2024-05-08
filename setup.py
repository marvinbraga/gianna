from setuptools import setup, find_packages

setup(
    name="gianna",
    version="0.1.4",
    author="Marcus Braga",
    author_email="mvbraga@gmail.com",
    description="Generative Intelligent Artificial Neural Network Assistant",
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/marvinbraga/gianna",
    packages=find_packages(exclude=["notebooks", "notebooks.*"]),
    install_requires=[
        "crewai>=0.28.8",
        "crewai-tools>=0.1.7",
        "langchain-google-genai>=1.0.2",
        "langchain-nvidia-ai-endpoints>=0.0.6",
        "langchain-groq>=0.1.2",
        "pydub>=0.25.1",
        "pyaudio>=0.2.14",
        "gtts>=2.5.1",
        "openai>=1.20.0",
        "elevenlabs>=1.0.5",
        "pyperclip>=1.8.2",
        "loguru>=0.7.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="Apache License 2.0",
    keywords="voice-assistant AI CrewAI Langchain OpenAI Google NVIDIA Groq Ollama",
    python_requires='>=3.11,<=3.13',
)
