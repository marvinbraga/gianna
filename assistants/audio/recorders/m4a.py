import os.path
import wave

from pydub import AudioSegment

from assistants.audio.recorders.abstract_recorders import AbstractAudioRecorder


class M4aRecorder(AbstractAudioRecorder):
    """
    A class for recording audio and saving it as an M4A file.
    """

    COMMAND_OUTPUT_FILENAME = "resources/command.m4a"

    def _save(self):
        """
        Save the recorded audio as an M4A file.

        This method first saves the recorded audio as a temporary WAV file,
        then converts it to an M4A file using the pydub library.

        Returns:
            self: The instance of the M4aRecorder class.
        """
        temp_filename = os.path.normpath("resources/temp.wav")
        try:
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self._channels)
                wf.setsampwidth(self._audio_interface.get_sample_size(self._format))
                wf.setframerate(self._rate)
                wf.writeframes(b''.join(self._frames))

            sound = AudioSegment.from_wav(temp_filename)
            sound.export(os.path.normpath(self.COMMAND_OUTPUT_FILENAME), format="mp4", codec="aac")
        finally:
            os.remove(temp_filename)
        return self
