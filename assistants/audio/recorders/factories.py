from pathlib import Path
from typing import Union

from assistants.audio.recorders.abstract_recorders import AbstractAudioRecorder
from assistants.audio.recorders.m4a import M4aRecorder
from assistants.audio.recorders.mp3 import Mp3Recorder
from assistants.audio.recorders.ogg import OggRecorder
from assistants.audio.recorders.wave import WaveRecorder


class AudioRecorderFactory:
    """
    A factory class for creating audio recorder instances based on the file type.
    """

    def __init__(self, audio_file: Union[str, Path]):
        """
        Initialize the AudioRecorderFactory.

        Args:
            audio_file (Union[str, Path]): The path to the audio file.
        """
        self._audio_file = audio_file
        self._recorder_classes = {
            ".mp3": Mp3Recorder,
            ".wav": WaveRecorder,
            ".m4a": M4aRecorder,
            ".ogg": OggRecorder,
        }

    def create_recorder(self) -> AbstractAudioRecorder:
        """
        Create an audio recorder instance based on the file type.

        Returns:
            AbstractAudioRecorder: An instance of the appropriate audio recorder class.

        Raises:
            ValueError: If the file type is not supported.
        """
        audio_file = Path(self._audio_file)
        file_extension = audio_file.suffix.lower()

        recorder_class = self._recorder_classes.get(file_extension)
        if recorder_class:
            return recorder_class()

        raise ValueError(f"Unsupported file type: {file_extension}")
