from abc import ABCMeta, abstractmethod


class AbstractTextToSpeech(metaclass=ABCMeta):
    """
    Abstract base class for text-to-speech conversion.
    """

    @abstractmethod
    def synthesize(self, text, output_file="output.mp3", **kwargs):
        """
        Convert text to speech and save it to an audio file.

        Args:
            text (str): The text to be converted to speech.
            output_file (str): The path to save the generated audio file. Default is "output.mp3".

        Returns:
            self: The instance of the TextToSpeech class.
        """
        pass
