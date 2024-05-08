from langchain.document_loaders.generic import GenericLoader

from gianna.assistants.audio.stt.parsers.whisper_local.parsers import WhisperCommandParser


class LocalWhisperSpeechToTextLoader:
    """
    Loader for converting speech to text using OpenAI's Whisper API.
    """

    def __init__(self, loader):
        """
        Initialize the WhisperSpeechToTextLoader.

        Args:
            loader: The underlying loader to use for loading the audio data.
        """
        self.loader = loader
        self._docs = []

    @property
    def docs(self):
        """
        Get the loaded documents.

        Returns:
            The loaded documents.
        """
        return self._docs

    def load(self):
        """
        Execute the speech-to-text conversion using OpenAI's Whisper API.

        Returns:
            self: The instance of WhisperSpeechToTextLoader.

        Raises:
            Exception: If an error occurs during the conversion process.
        """
        parser = WhisperCommandParser()

        loader = GenericLoader(self.loader, parser)
        try:
            self._docs = loader.load()
        except Exception as e:
            print(f"Error: {e}")
        return self
