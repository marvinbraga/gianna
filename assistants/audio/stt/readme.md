# Whisper Speech-to-Text Loader

This Python script demonstrates how to use the `WhisperSpeechToTextLoader` and `MP3AudioLoader` classes to convert MP3
audio files to text using OpenAI's Whisper API.

## Prerequisites

Before running the script, make sure you have the following:

- Python installed on your system
- Required dependencies
  installed (`pathlib`, `assistants.audio.stt.mp3`, `assistants.audio.stt.whisper`)
- M4A audio files located in the `resources` directory relative to the script's parent directory

## Usage

1. Place your M4A audio files in the `resources` directory. The directory structure should look like this:

    ```
    gianna/
        ├── your_script.py
        └── resources/
            ├── audio1.mp3
            ├── audio2.mp3
            └── ...
    ```

2. Open the script file and ensure that the `save_dir` variable is set correctly. By default, it assumes that
   the `resources` directory is located in the parent directory of the script.

    ```python
    save_dir = Path().parent.absolute() / "resources"
    ```

3. Run the script using Python:

    ```bash
    python your_script.py
    ```

4. The script will load the M4A audio files from the `resources` directory using the `MP3AudioLoader` class and pass
   them to the `WhisperSpeechToTextLoader` for speech-to-text conversion.

5. The resulting text documents will be stored in the `docs` variable. The script will print the `source` metadata of
   each document, which represents the path of the original audio file.

## Example Code

```python
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from assistants.audio.stt.mp3 import MP3AudioLoader
from assistants.audio.stt.whisper import WhisperSpeechToTextLoader

load_dotenv(find_dotenv())

save_dir = Path().absolute() / "resources"
print(save_dir, save_dir.exists(), save_dir.is_dir())

loader = WhisperSpeechToTextLoader(loader=MP3AudioLoader(save_dir))
docs = loader.load().docs

for doc in docs:
    print(doc.metadata["source"], doc.page_content)
```

## Customization

- If your M4A audio files are located in a different directory, update the `save_dir` variable accordingly.

    ```python
    save_dir = Path("/path/to/your/audio/files")
    ```

- You can modify the script to perform additional processing on the resulting text documents or save them to a file if
  needed.

## Classes

### `MP3AudioLoader`

The `MP3AudioLoader` class is responsible for loading MP3 audio files from a specified directory. It inherits from
the `AbstractAudioLoader` class.

### `WhisperSpeechToTextLoader`

The `WhisperSpeechToTextLoader` class is used to convert speech to text using OpenAI's Whisper API. It takes an instance
of the `MP3AudioLoader` class as input and performs the speech-to-text conversion on the loaded audio files.

## Note

Make sure you have the necessary permissions and API credentials to use OpenAI's Whisper API. Refer to the OpenAI
documentation for more information on setting up and using the API.

---

Feel free to customize the script and the README file based on your specific requirements and project structure.
