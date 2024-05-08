from enum import Enum

from gianna.assistants.audio.tts.elevenlabs import TextToSpeechElevenLabs
from gianna.assistants.audio.tts.google_api import TextToSpeechGoogle
from gianna.assistants.audio.tts.whisper import TextToSpeechWhisper


class TextToSpeechType(Enum):
    """
    Enum representing the available text-to-speech types.
    """
    GOOGLE = "google"
    WHISPER = "whisper"
    ELEVEN_LABS = "eleven_labs"


class TextToSpeechFactory:
    """
    Factory class for creating instances of text-to-speech classes.
    """

    def __init__(self, tts_classes=None):
        """
        Initialize the TextToSpeechFactory instance.

        Args:
            tts_classes (dict): Dictionary mapping text-to-speech types to their corresponding classes.
                If not provided, the default mapping will be used.
        """
        self._tts_classes = tts_classes or {
            TextToSpeechType.GOOGLE: TextToSpeechGoogle,
            TextToSpeechType.WHISPER: TextToSpeechWhisper,
            TextToSpeechType.ELEVEN_LABS: TextToSpeechElevenLabs,
        }

    def create_text_to_speech(self, tts_type: TextToSpeechType, **kwargs):
        """
        Create an instance of a text-to-speech class based on the specified type.

        Args:
            tts_type (TextToSpeechType): The type of text-to-speech to create.
            **kwargs: Additional keyword arguments to pass to the constructor of the text-to-speech class.

        Returns:
            TextToSpeech: An instance of the specified text-to-speech class.

        Raises:
            ValueError: If an invalid text-to-speech type is provided.
        """
        tts_class = self._tts_classes.get(tts_type)
        if tts_class:
            return tts_class(**kwargs)
        else:
            raise ValueError(f"Invalid text-to-speech type: {tts_type}")
