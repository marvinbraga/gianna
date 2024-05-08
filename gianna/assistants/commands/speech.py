from enum import Enum

from gianna.assistants.audio.tts.factory_method import text_to_speech
from gianna.assistants.commands.abstracts import AbstractCommand


class SpeechType(Enum):
    GOOGLE = "google"
    WHISPER = "whisper"
    ELEVEN_LABS = "eleven_labs"


class SpeechCommand(AbstractCommand):
    def execute(self, text: str, speech_type: SpeechType, lang="en", voice="default"):
        text_to_speech(text, speech_type.value.strip(), lang, voice)
        return self
