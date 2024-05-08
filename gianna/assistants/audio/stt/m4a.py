from gianna.assistants.audio.loaders.abstract_loaders import AbstractAudioLoader


class M4aAudioLoader(AbstractAudioLoader):
    """
    Loader for M4A audio files.
    """
    glob = "*.m4a"


