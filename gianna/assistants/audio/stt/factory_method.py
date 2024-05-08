from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document

from gianna.assistants.audio.stt.m4a import M4aAudioLoader
from gianna.assistants.audio.stt.mp3 import MP3AudioLoader
from gianna.assistants.audio.stt.whisper import WhisperSpeechToTextLoader
from gianna.assistants.audio.stt.whisper_local import LocalWhisperSpeechToTextLoader

load_dotenv(find_dotenv())


def speech_to_text(audio_files_path, filetype: str = "mp3", local: bool = False) -> [Document]:
    """
    Convert speech from an audio file to text using the specified file type loader and Whisper speech-to-text loader.

    Args:
        :param audio_files_path: The path to the directory containing the audio files.
        :param filetype: The file type of the audio files. Supported values are "mp3" and "m4a".
        :param local: If True execute the Whisper local speech-to-text loader instead.

    Returns:
        list[Document]: A list of Document objects containing the transcribed text from the audio files.

    Raises:
        ValueError: If an invalid file type is provided.

    Example:
        >>> audio_path = "path/to/your/audio/files"
        >>> filetype = "mp3"
        >>> documents = speech_to_text(audio_path, filetype)
        >>> for doc in documents:
        ...     print(doc.page_content)
    """
    loader_class = {
        "mp3": MP3AudioLoader,
        "m4a": M4aAudioLoader,
    }[filetype]
    if not loader_class:
        raise ValueError("Invalid file type.")

    whisper_class = LocalWhisperSpeechToTextLoader if local else WhisperSpeechToTextLoader
    loader = whisper_class(
        loader=loader_class(audio_files_path)
    )

    return loader.load().docs
