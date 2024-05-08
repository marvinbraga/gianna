from enum import Enum

from dotenv import load_dotenv, find_dotenv

from gianna.assistants.audio.players.factory_method import play_audio
from gianna.assistants.audio.tts.abstracts import AbstractTextToSpeech

load_dotenv(find_dotenv())
import openai


class WhisperVoices(Enum):
    """
    Enum representing the available voices for Whisper text-to-speech.
    """
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


class TextToSpeechWhisper(AbstractTextToSpeech):
    """
    Text-to-speech conversion using Whisper API.
    """

    def __init__(self, voice: str, **kwargs):
        """
        Initialize the TextToSpeechWhisper instance.

        Args:
            voice (WhisperVoices): The voice to be used for speech synthesis. Default is WhisperVoices.ALLOY.
        """
        self._voice = voice

    def synthesize(self, text, output_file="output.mp3", **kwargs):
        """
        Convert text to speech using Whisper API and save it to an MP3 file.

        Args:
            :param text: The text to be converted to speech.
            :param output_file: The path to save the generated MP3 file. Default is "output.mp3".
            :param voice: Select the voice to be used for speech synthesis. Default is "default".

        Returns:
            self: The instance of the TextToSpeechWhisper class.
        """
        voice = WhisperVoices.ALLOY
        if self._voice != "default":
            voice = WhisperVoices(self._voice)
        client = openai.OpenAI()
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice.value,
            input=text,
        )
        response.stream_to_file(output_file)
        play_audio(output_file)
        return self
