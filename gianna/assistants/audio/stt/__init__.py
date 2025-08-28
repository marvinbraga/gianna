"""
Speech-to-Text module for converting spoken audio to text
"""

from gianna.assistants.audio.stt.abstract_loaders import AbstractAudioLoader
from gianna.assistants.audio.stt.factory_method import speech_to_text

__all__ = ["speech_to_text", "AbstractAudioLoader"]
