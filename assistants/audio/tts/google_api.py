from gtts import gTTS

from assistants.audio.players.factory_method import play_audio
from assistants.audio.tts.abstracts import AbstractTextToSpeech


class TextToSpeechGoogle(AbstractTextToSpeech):
    """
    Text-to-speech conversion using Google Text-to-Speech API.
    """

    def __init__(self, language="pt", **kwargs):
        """
        Initialize the TextToSpeechGoogle instance.

        Args:
            language (str): The language code for the speech synthesis. Default is "pt" (Portuguese).
        """
        self.language = language

    def synthesize(self, text, output_file="output.mp3", **kwargs):
        """
        Convert text to speech using Google Text-to-Speech API and save it to an MP3 file.

        Args:
            text (str): The text to be converted to speech.
            output_file (str): The path to save the generated MP3 file. Default is "output.mp3".

        Returns:
            self: The instance of the TextToSpeechGoogle class.
        """
        transcript = gTTS(text, lang=self.language)
        transcript.save(output_file)
        play_audio(output_file)
        return self
