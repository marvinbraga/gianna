import threading
from abc import ABCMeta, abstractmethod

import pyaudio


class AbstractAudioPlayer(metaclass=ABCMeta):
    """
    Abstract base class for audio players.
    """

    def __init__(self, audio_format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024, audio_interface=None):
        """
        Initialize the AbstractAudioPlayer.

        Args:
            audio_format (int): The audio format (default: pyaudio.paInt16).
            channels (int): The number of audio channels (default: 1).
            rate (int): The sample rate (default: 44100).
            chunk (int): The chunk size (default: 1024).
            audio_interface (pyaudio.PyAudio): The audio interface (default: None).
        """
        self._format = audio_format
        self._channels = channels
        self._rate = rate
        self._chunk = chunk
        self._audio_interface = audio_interface or pyaudio.PyAudio()
        self._playing = False
        self._stream = None
        self._play_thread = None

    def start_playing(self):
        """
        Start playing the audio.
        """
        self._playing = True
        self._stream = self._audio_interface.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            output=True,
            frames_per_buffer=self._chunk,
        )
        self._play_thread = threading.Thread(target=self._play)
        self._play_thread.start()

    def stop_playing(self):
        """
        Stop playing the audio.
        """
        self._playing = False
        if self._play_thread:
            self._play_thread.join()
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()

    @abstractmethod
    def _play(self):
        """
        Abstract method to play the audio.
        """
        pass

    def is_playing(self):
        """
        Check if the audio is currently playing.

        Returns:
            bool: True if the audio is playing, False otherwise.
        """
        return self._playing

    def wait_until_finished(self):
        """
        Wait until the audio playback is finished.
        """
        while self.is_playing():
            pass

    def __del__(self):
        """
        Clean up the resources when the object is destroyed.
        """
        self.stop_playing()
        self._audio_interface.terminate()
