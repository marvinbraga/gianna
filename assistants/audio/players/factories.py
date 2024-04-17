from pathlib import Path
from typing import Union

from assistants.audio.players.aac import AACFilePlayer
from assistants.audio.players.abstract_players import AbstractAudioPlayer
from assistants.audio.players.flac import FLACFilePlayer
from assistants.audio.players.m4a import M4AFilePlayer
from assistants.audio.players.mp3 import MP3FilePlayer
from assistants.audio.players.ogg import OGGFilePlayer
from assistants.audio.players.wave import WaveFilePlayer


class AudioPlayerFactory:
    """
    A factory class for creating audio player instances based on the file type.
    """

    def __init__(self, audio_file: Union[str, Path]):
        """
        Initialize the AudioPlayerFactory.

        Args:
            audio_file (Union[str, Path]): The path to the audio file.
        """
        self._audio_file = audio_file
        self._player_classes = {
            ".mp3": MP3FilePlayer,
            ".wav": WaveFilePlayer,
            ".m4a": M4AFilePlayer,
            ".flac": FLACFilePlayer,
            ".ogg": OGGFilePlayer,
            ".aac": AACFilePlayer,
        }

    def create_player(self) -> AbstractAudioPlayer:
        """
        Create an audio player instance based on the file type.

        Returns:
            AbstractAudioPlayer: An instance of the appropriate audio player class.

        Raises:
            ValueError: If the file type is not supported.
        """
        audio_file = Path(self._audio_file)
        file_extension = audio_file.suffix.lower()

        player_class = self._player_classes.get(file_extension)
        if player_class:
            return player_class(audio_file)

        raise ValueError(f"Unsupported file type: {file_extension}")
