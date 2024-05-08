import os
import wave

from gianna.assistants.audio.recorders.abstract_recorders import AbstractAudioRecorder


class WaveRecorder(AbstractAudioRecorder):
    """
    A class for recording audio and saving it as a WAV file.
    """

    COMMAND_OUTPUT_FILENAME = "resources/command.wav"

    def _save(self):
        """
        Save the recorded audio as a WAV file.

        This method saves the recorded audio directly as a WAV file using the wave module.

        Returns:
            self: The instance of the WaveRecorder class.
        """
        with wave.open(os.path.normpath(self.COMMAND_OUTPUT_FILENAME), 'wb') as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(self._audio_interface.get_sample_size(self._format))
            wf.setframerate(self._rate)
            wf.writeframes(b''.join(self._frames))
        return self
