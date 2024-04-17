from typing import Iterator

from langchain_community.document_loaders import Blob
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_core.documents import Document

from assistants.audio.stt.parsers.whisper_local.whisper_cmd import WhisperWrapperTranscriber


class WhisperCommandParser(BaseBlobParser):
    """
    A parser that uses the Whisper command-line tool to transcribe audio files.
    """

    def __init__(self, model="large-v2", device="cpu", output_dir=".", verbose=False, threads=20,
                 language=None, beam_size=None, temperature=None):
        """
        Initialize the WhisperCommandParser.

        Args:
            model (str): The Whisper model to use for transcription. Default is "large-v2".
            device (str): The device to run the model on (e.g., "cpu" or "cuda"). Default is "cpu".
            output_dir (str): The directory to save the output files. Default is the current directory.
            verbose (bool): Whether to enable verbose output. Default is False.
            threads (int): The number of threads to use for parallel processing. Default is 20.
            language (str): The language code for the audio. Default is None.
            beam_size (int): The beam size for beam search decoding. Default is None.
            temperature (float): The temperature for sampling. Default is None.
        """
        super().__init__()
        self.device = device
        self.output_dir = output_dir
        self.verbose = verbose
        self.threads = threads
        self.language = language
        self.beam_size = beam_size
        self.temperature = temperature
        self.model = WhisperWrapperTranscriber(
            model, self.device, self.output_dir, self.verbose, self.threads,
            self.language, self.beam_size, self.temperature,
        )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        Lazily parse the blob using the Whisper command-line tool.

        Args:
            blob (Blob): The blob to parse.

        Yields:
            Document: The parsed document containing the transcribed text and metadata.
        """
        prediction = self.model.transcribe(blob.path.as_posix())

        yield Document(
            page_content=prediction,
            metadata={"source": blob.source},
        )
