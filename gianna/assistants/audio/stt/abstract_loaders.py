from typing import Iterable

from langchain.document_loaders import BlobLoader, Blob, FileSystemBlobLoader


class AbstractAudioLoader(BlobLoader):
    """
    Abstract base class for audio loaders.
    """
    glob = None

    def __init__(self, save_dir: str):
        """
        Initialize the AbstractAudioLoader.

        Args:
            save_dir (str): The directory where the audio files are saved.
        """
        self.save_dir = save_dir

    def yield_blobs(self) -> Iterable[Blob]:
        """
        Yield blobs from the audio files.

        Returns:
            Iterable[Blob]: An iterable of blobs representing the audio files.

        Raises:
            AssertionError: If the 'glob' attribute of the class is not set.
        """
        if self.glob is None:
            assert "You must provide a value for the 'glob' attribute of your class."

        loader = FileSystemBlobLoader(self.save_dir, glob=self.glob)
        for blob in loader.yield_blobs():
            yield blob
