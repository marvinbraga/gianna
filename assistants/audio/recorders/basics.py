from pathlib import Path

from assistants.audio.players.factory_method import play_audio
from assistants.audio.recorders.abstract_recorders import AbstractAudioRecorder


class AudioRecorder:
    """
    A class for recording audio using a specified audio recorder.
    """

    SOUND_BEEP = Path().absolute().parent / 'resources' / 'beep_short.mp3'

    def __init__(self, recorder: AbstractAudioRecorder):
        """
        Initialize the AudioRecorder.

        Args:
            recorder (AbstractAudioRecorder): The audio recorder to use for recording.
        """
        self._recorder = recorder
        self._is_recording = True

    @property
    def is_recording(self):
        """
        Get the recording status.

        Returns:
            bool: True if recording is in progress, False otherwise.
        """
        return self._is_recording

    @property
    def recorder(self):
        """
        Get the audio recorder.

        Returns:
            AbstractAudioRecorder: The audio recorder used for recording.
        """
        return self._recorder

    def start(self):
        """
        Start the audio recording.

        This method plays a beep sound, sets the recording status to True,
        and starts the audio recorder.

        Returns:
            self: The instance of the AudioRecorder class.
        """
        play_audio(self.SOUND_BEEP)
        self._is_recording = True
        self._recorder.start_recording()
        return self

    def stop(self):
        """
        Stop the audio recording.

        This method sets the recording status to False, stops the audio recorder,
        saves the recorded audio, and plays a beep sound.

        Returns:
            self: The instance of the AudioRecorder class.
        """
        self._is_recording = False
        self._recorder.stop_recording()
        self._recorder.save()
        play_audio(self.SOUND_BEEP)
        return self
