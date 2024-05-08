from gianna.assistants.commands.shell import register_shell_command

from gianna.assistants.models.google import register_google_chains
from gianna.assistants.models.groq import register_groq_chains
from gianna.assistants.models.nvidia import register_nvidia_chains
from gianna.assistants.models.ollama import register_ollama_chains
from gianna.assistants.models.openai import register_openai_chains

# LLMs
register_ollama_chains()
register_openai_chains()
register_google_chains()
register_nvidia_chains()
register_groq_chains()

# Commands
register_shell_command()
