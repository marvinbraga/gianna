from gianna.assistants.audio.stt.abstract_loaders import AbstractAudioLoader


class M4aAudioLoader(AbstractAudioLoader):
    """
    Loader for M4A audio files.
    """

    glob = "*.m4a"
