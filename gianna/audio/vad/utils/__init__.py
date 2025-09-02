"""
Voice Activity Detection utilities.

This package provides utility functions and helpers for VAD operations,
including audio processing, signal analysis, and performance optimization.
"""

from .audio_utils import (
    calculate_audio_statistics,
    convert_audio_format,
    normalize_audio_chunk,
    resample_audio,
    validate_audio_chunk,
)
from .signal_processing import (
    apply_bandpass_filter,
    calculate_spectral_features,
    calculate_zero_crossing_rate,
    compute_mfcc_features,
    compute_spectral_centroid,
    detect_clipping,
    estimate_noise_floor,
    preemphasis_filter,
)

__all__ = [
    # Audio utilities
    "calculate_audio_statistics",
    "convert_audio_format",
    "normalize_audio_chunk",
    "resample_audio",
    "validate_audio_chunk",
    # Signal processing
    "apply_bandpass_filter",
    "calculate_spectral_features",
    "calculate_zero_crossing_rate",
    "compute_mfcc_features",
    "compute_spectral_centroid",
    "detect_clipping",
    "estimate_noise_floor",
    "preemphasis_filter",
]
