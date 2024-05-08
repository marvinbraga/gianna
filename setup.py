from setuptools import setup, find_packages

setup(
    name="gianna",
    version="0.1.0",
    author="Marcus Braga",
    author_email="mvbraga@gmail.com",
    description="Generative Intelligent Artificial Neural Network Assistant",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/your-username/gianna",
    packages=find_packages(),
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
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11,<=3.13',
)
