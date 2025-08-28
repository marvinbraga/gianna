"""
Text-to-Speech module for converting text to spoken audio
"""

from gianna.assistants.audio.tts.abstracts import AbstractTextToSpeech
from gianna.assistants.audio.tts.factories import TextToSpeechFactory, TextToSpeechType
from gianna.assistants.audio.tts.factory_method import text_to_speech

__all__ = [
    "text_to_speech",
    "TextToSpeechFactory",
    "TextToSpeechType",
    "AbstractTextToSpeech",
]
