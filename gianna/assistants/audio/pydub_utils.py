"""
Pydub utility module with SyntaxWarning suppression.

This module suppresses the SyntaxWarnings that occur in pydub/utils.py due to
invalid escape sequences in regex patterns. This is a known issue in pydub 0.25.1
where regex patterns contain unescaped backslashes that should be raw strings.

The problematic patterns are found in:
- Line 235: re_stream variable
- Lines 300, 301, 310, 314: regex patterns in mediainfo_json function
- Line 341: regex pattern in mediainfo function
- Line 394: regex pattern in get_supported_codecs function

This wrapper provides a clean import interface for the rest of the application
while suppressing these specific warnings.

Usage:
    from gianna.assistants.audio.pydub_utils import AudioSegment, play

Instead of:
    from pydub import AudioSegment
    from pydub.playback import play
"""

import warnings

# Store original warnings filter state
_original_filters = warnings.filters[:]


def _suppress_pydub_warnings():
    """Suppress specific SyntaxWarnings from pydub.utils module."""
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub.utils")


def _restore_warnings():
    """Restore original warnings filter state."""
    warnings.filters[:] = _original_filters


# Suppress warnings before importing pydub
_suppress_pydub_warnings()

try:
    # Import pydub components
    from pydub import AudioSegment, effects
    from pydub.playback import play
    from pydub.utils import mediainfo, mediainfo_json, which

    # Export the imports for use by other modules
    __all__ = [
        "AudioSegment",
        "play",
        "effects",
        "mediainfo",
        "mediainfo_json",
        "which",
    ]

finally:
    # Restore warnings after importing
    _restore_warnings()
