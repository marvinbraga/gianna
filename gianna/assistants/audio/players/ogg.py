import threading

from pydub import AudioSegment
from pydub.playback import play

from gianna.assistants.audio.players.abstract_players import AbstractAudioPlayer


class OGGFilePlayer(AbstractAudioPlayer):
    """
    A class for playing OGG audio files.
    """

    def __init__(self, audio_file, *args, **kwargs):
        """
        Initialize the OGGFilePlayer.

        Args:
            audio_file (str): The path to the OGG audio file.
        """
        super().__init__(*args, **kwargs)
        self.audio_file = audio_file
        self._play_thread = None
        self._stop_event = threading.Event()

    def _play(self):
        """
        Play the OGG audio file.
        """
        try:
            audio = AudioSegment.from_file(self.audio_file, format="ogg")
            play(audio)
        except Exception as e:
            print(f"Error playing audio: {e}")
        finally:
            self._stop_event.set()

    def play(self):
        """
        Start playing the OGG audio file.
        """
        self._stop_event.clear()
        self._play_thread = threading.Thread(target=self._play)
        self._play_thread.start()

    def stop(self):
        """
        Stop playing the OGG audio file.
        """
        self._stop_event.set()
        if self._play_thread:
            self._play_thread.join()

    def is_playing(self):
        """
        Check if the audio is currently playing.

        Returns:
            bool: True if the audio is playing, False otherwise.
        """
        return not self._stop_event.is_set()

    def wait_until_finished(self):
        """
        Wait until the audio playback is finished.
        """
        if self._play_thread:
            self._play_thread.join()

    def __enter__(self):
        """
        Enter the context manager.
        """
        self.play()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.
        """
        self.stop()