from assistants.audio.loaders.abstract_loaders import AbstractAudioLoader


class MP3AudioLoader(AbstractAudioLoader):
    """
    Loader for MP3 audio files.
    """
    glob = "*.mp3"
