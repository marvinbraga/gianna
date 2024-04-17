import os
import subprocess


class WhisperWrapperTranscriber:
    """
    A wrapper class for the Whisper transcription and translation tool.
    """

    def __init__(self, model="large-v2", device="cpu", output_dir=".", verbose=False, threads=20,
                 language=None, beam_size=None, temperature=None):
        """
        Initialize the WhisperWrapperTranscriber.

        Args:
            model (str): The Whisper model to use for transcription/translation. Default is "large-v2".
            device (str): The device to run the model on (e.g., "cpu" or "cuda"). Default is "cpu".
            output_dir (str): The directory to save the output files. Default is the current directory.
            verbose (bool): Whether to enable verbose output. Default is False.
            threads (int): The number of threads to use for parallel processing. Default is 20.
            language (str): The language code for the audio. Default is None.
            beam_size (int): The beam size for beam search decoding. Default is None.
            temperature (float): The temperature for sampling. Default is None.
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.verbose = verbose
        self.threads = threads
        self.language = language
        self.beam_size = beam_size
        self.temperature = temperature

    def _execute(self, audio_path, command: list[str]):
        """
        Execute the Whisper command with the specified audio path and command arguments.

        Args:
            audio_path (str): The path to the audio file.
            command (list[str]): The command arguments for the Whisper tool.

        Returns:
            str: The transcribed or translated text.
        """
        if self.language is not None:
            command.extend(["--language", self.language])

        if self.beam_size is not None:
            command.extend(["--beam_size", str(self.beam_size)])

        if self.temperature is not None:
            command.extend(["--temperature", str(self.temperature)])

        try:
            subprocess.run(command, check=True)
        except Exception as e:
            print([f"ERROR: {e}"])

        output_file = os.path.splitext(os.path.basename(audio_path))[0] + ".txt"
        output_path = os.path.join(self.output_dir, output_file)

        with open(output_path, "r") as f:
            return f.read()

    def transcribe(self, audio_path):
        """
        Transcribe the audio file at the specified path.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            str: The transcribed text.
        """
        command = [
            "whisper",
            audio_path,
            "--model", self.model,
            "--device", self.device,
            "--output_dir", self.output_dir,
            "--verbose", str(self.verbose),
            "--threads", str(self.threads),
            "--task", "transcribe"
        ]
        return self._execute(audio_path, command)

    def translate(self, audio_path, language=None):
        """
        Translate the audio file at the specified path.

        Args:
            audio_path (str): The path to the audio file.
            language (str): The target language code for translation. If not provided, the default language will be used.

        Returns:
            str: The translated text.
        """
        self.language = language if language is not None else self.language
        command = [
            "whisper",
            audio_path,
            "--model", self.model,
            "--device", self.device,
            "--output_dir", self.output_dir,
            "--verbose", str(self.verbose),
            "--threads", str(self.threads),
            "--task", "translate"
        ]
        return self._execute(audio_path, command)


if __name__ == '__main__':
    transcriber = WhisperWrapperTranscriber()
    path = "/resource/"
    filename = "your_audio_file.m4a"
    result = transcriber.transcribe(os.path.join(path, filename))
    with open(filename + ".md", "w", encoding="utf-8") as file:
        file.writelines(result)
