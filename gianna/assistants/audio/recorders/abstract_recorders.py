import threading
from abc import ABCMeta, abstractmethod

# sudo apt install python3-pyaudio portaudio19-dev
import pyaudio


class AbstractAudioRecorder(metaclass=ABCMeta):
    """
    Abstract base class for audio recorders.
    """
    COMMAND_OUTPUT_FILENAME = None

    def __init__(self, audio_format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024, audio_interface=None):
        """
        Initialize the AbstractAudioRecorder.

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
        self._frames = []
        self._recording = False
        self._stream = None
        self._record_thread = None

    def start_recording(self):
        """
        Start the audio recording.
        """
        self._frames = []
        self._recording = True
        self._stream = self._audio_interface.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
        )
        self._record_thread = threading.Thread(target=self._record)
        self._record_thread.start()

    def stop_recording(self):
        """
        Stop the audio recording.
        """
        self._recording = False
        self._record_thread.join()
        self._stream.stop_stream()
        self._stream.close()

    def _record(self):
        """
        Record audio data in a separate thread.
        """
        while self._recording:
            data = self._stream.read(self._chunk)
            self._frames.append(data)

    @abstractmethod
    def _save(self):
        """
        Abstract method to save the recorded audio.
        """
        pass

    def save(self):
        """
        Save the recorded audio.

        Returns:
            self: The instance of AbstractAudioRecorder.

        Raises:
            AssertionError: If the 'COMMAND_OUTPUT_FILENAME' attribute is not set.
        """
        if self.COMMAND_OUTPUT_FILENAME is None:
            assert "You must provide a filename in the 'COMMAND_OUTPUT_FILENAME' attribute."
        self._save()
        return self
