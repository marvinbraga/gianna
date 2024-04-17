import os
import tempfile
import wave

from pydub import AudioSegment

from assistants.audio.recorders.abstract_recorders import AbstractAudioRecorder


class OggRecorder(AbstractAudioRecorder):
    """
    A class for recording audio and saving it as an OGG file.
    """

    COMMAND_OUTPUT_FILENAME = "resources/command.ogg"

    def _save(self):
        """
        Save the recorded audio as an OGG file.

        This method first saves the recorded audio as a temporary WAV file,
        then converts it to an OGG file using the pydub library.

        Returns:
            self: The instance of the OggRecorder class.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name

            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self._channels)
                wf.setsampwidth(self._audio_interface.get_sample_size(self._format))
                wf.setframerate(self._rate)
                wf.writeframes(b''.join(self._frames))

            sound = AudioSegment.from_wav(temp_filename)
            sound.export(os.path.normpath(self.COMMAND_OUTPUT_FILENAME), format="ogg")

        os.remove(temp_filename)
        return self
