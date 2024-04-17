from pathlib import Path

from assistants.audio.recorders.basics import AudioRecorder
from assistants.audio.recorders.factories import AudioRecorderFactory


def get_recorder(filename: str | Path) -> AudioRecorder:
    """
    Get an AudioRecorder instance configured with the appropriate audio recorder based on the file type.

    Args:
        filename (str | Path): The path or filename of the audio file to be recorded.

    Returns:
        AudioRecorder: An instance of the AudioRecorder class configured with the appropriate audio recorder.

    Example:
        >>> audio_file = "path/to/your/audio/file.mp3"
        >>> audio_recorder = get_recorder(audio_file)
        >>> audio_recorder.start()
        >>> # Perform any necessary actions while recording
        >>> audio_recorder.stop()
    """
    return AudioRecorder(
        AudioRecorderFactory(audio_file=filename).create_recorder()
    )
