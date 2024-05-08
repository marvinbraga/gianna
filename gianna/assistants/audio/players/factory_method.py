from gianna.assistants.audio.players.factories import AudioPlayerFactory


def play_audio(audio_file):
    """
    Play the specified audio file using the appropriate audio player.

    Args:
        audio_file (Union[str, Path]): The path to the audio file.

    Raises:
        ValueError: If the file type is not supported.
    """
    player = AudioPlayerFactory(audio_file=audio_file).create_player()
    with player:
        player.wait_until_finished()
