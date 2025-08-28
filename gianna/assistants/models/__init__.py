"""
Models module for Gianna assistant.
"""

from gianna.assistants.models import (
    abstracts,
    anthropic,
    basics,
    cohere,
    factory_method,
    google,
    groq,
    nvidia,
    ollama,
    openai,
    registers,
    xai,
)
from gianna.assistants.models.anthropic import register_anthropic_chains
from gianna.assistants.models.cohere import register_cohere_chains
from gianna.assistants.models.factory_method import get_chain_instance
from gianna.assistants.models.google import register_google_chains
from gianna.assistants.models.groq import register_groq_chains
from gianna.assistants.models.nvidia import register_nvidia_chains
from gianna.assistants.models.ollama import register_ollama_chains
from gianna.assistants.models.openai import register_openai_chains
from gianna.assistants.models.registers import LLMRegister
from gianna.assistants.models.xai import register_xai_chains

# Register all LLM chains
register_ollama_chains()
register_openai_chains()
register_google_chains()
register_nvidia_chains()
register_groq_chains()
register_anthropic_chains()
register_xai_chains()
register_cohere_chains()

__all__ = [
    "abstracts",
    "basics",
    "factory_method",
    "google",
    "groq",
    "nvidia",
    "ollama",
    "openai",
    "anthropic",
    "xai",
    "cohere",
    "registers",
    "get_chain_instance",
    "LLMRegister",
    "register_google_chains",
    "register_groq_chains",
    "register_nvidia_chains",
    "register_ollama_chains",
    "register_openai_chains",
    "register_anthropic_chains",
    "register_xai_chains",
    "register_cohere_chains",
]
