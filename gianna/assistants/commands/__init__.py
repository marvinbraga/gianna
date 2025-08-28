"""
Commands module for executing various actions
"""

from gianna.assistants.commands.factory_method import get_command
from gianna.assistants.commands.register import CommandRegister
from gianna.assistants.commands.shell import register_shell_command
from gianna.assistants.commands.speech import SpeechCommand, SpeechType

# Register all commands
register_shell_command()

__all__ = ["get_command", "CommandRegister", "SpeechCommand", "SpeechType"]
