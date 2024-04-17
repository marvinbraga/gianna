import os

from dotenv import load_dotenv, find_dotenv
from elevenlabs import play, Voice
from elevenlabs.client import ElevenLabs

from assistants.audio.tts.abstracts import AbstractTextToSpeech

load_dotenv(find_dotenv())


class TextToSpeechElevenLabs(AbstractTextToSpeech):
    """
    Text-to-speech conversion using ElevenLabs API.
    """

    def __init__(self, voice: str, **kwargs):
        """
        Initialize the TextToSpeechElevenLabs instance.

        Args:
            voice (str): Select the voice you want to convert.
        """
        self._voice = None if voice == "default" else voice
        self._client = ElevenLabs(api_key=os.environ["ELEVEN_LABS_API_KEY"])
        if self._voice is not None:
            response = self.get_voices()
            self._voices = response.voices
            self._voice = self.find_voice(self._voice)

    def find_voice(self, text: str) -> Voice:
        voice = [v for v in self._voices if v.name == text]
        return self._voices[0] if len(voice) == 0 else voice[0]

    def get_voices(self) -> list[Voice]:
        return self._client.voices.get_all()

    def synthesize(self, text: str, output_file="output.mp3", **kwargs):
        """
        Convert text to speech using ElevenLabs API and play the generated audio.

        Args:
            :param text: The text to be converted to speech.
            :param output_file: File path to save the converted speech.

        Returns:
            self: The instance of the TextToSpeechElevenLabs class.
        """
        audio = self._client.generate(
            text=text,
            voice=self._voice,
            model="eleven_turbo_v2",
        )
        play(audio)
        return self
