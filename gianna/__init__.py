"""
Gianna - Generative Intelligent Artificial Neural Network Assistant.
"""

from dotenv import find_dotenv, load_dotenv

# Import from assistants package to trigger registration
import gianna.assistants
from gianna.assistants.audio import players, recorders, stt, tts
from gianna.assistants.commands import factory_method as commands
from gianna.assistants.models import factory_method as models

# Load environment variables
load_dotenv(find_dotenv())

__version__ = "0.1.4"
