from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document

from assistants.audio.stt.m4a import M4aAudioLoader
from assistants.audio.stt.mp3 import MP3AudioLoader
from assistants.audio.stt.whisper import WhisperSpeechToTextLoader

load_dotenv(find_dotenv())


def speech_to_text(audio_files_path, filetype) -> [Document]:
    """
    Convert speech from an audio file to text using the specified file type loader and Whisper speech-to-text loader.

    Args:
        audio_files_path (str): The path to the directory containing the audio files.
        filetype (str): The file type of the audio files. Supported values are "mp3" and "m4a".

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

    loader = WhisperSpeechToTextLoader(
        loader=loader_class(audio_files_path)
    )

    return loader.load().docs
