"""
Audio utility functions for VAD processing.

This module provides common audio processing utilities used across
different VAD implementations for format conversion, validation,
and preprocessing operations.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def validate_audio_chunk(
    audio_chunk: Union[np.ndarray, bytes],
    expected_dtype: Optional[np.dtype] = None,
    min_length: int = 0,
) -> bool:
    """
    Validate an audio chunk for processing.

    Args:
        audio_chunk: Audio data to validate.
        expected_dtype: Expected numpy dtype for the data.
        min_length: Minimum required length of the audio data.

    Returns:
        bool: True if audio chunk is valid for processing.
    """
    try:
        if audio_chunk is None:
            logger.warning("Audio chunk is None")
            return False

        # Convert to numpy array if bytes
        if isinstance(audio_chunk, bytes):
            if len(audio_chunk) == 0:
                logger.warning("Empty bytes audio chunk")
                return False
            data = np.frombuffer(audio_chunk, dtype=np.int16)
        elif isinstance(audio_chunk, np.ndarray):
            data = audio_chunk
        else:
            logger.warning(f"Unsupported audio chunk type: {type(audio_chunk)}")
            return False

        # Check minimum length
        if len(data) < min_length:
            logger.warning(f"Audio chunk too short: {len(data)} < {min_length}")
            return False

        # Check data type if specified
        if expected_dtype is not None and data.dtype != expected_dtype:
            logger.warning(
                f"Unexpected audio dtype: {data.dtype}, expected {expected_dtype}"
            )
            return False

        # Check for invalid values (NaN, infinity)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning("Audio chunk contains NaN or infinity values")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating audio chunk: {e}")
        return False


def convert_audio_format(
    audio_chunk: Union[np.ndarray, bytes],
    target_dtype: np.dtype = np.int16,
    source_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Convert audio data between different formats.

    Args:
        audio_chunk: Input audio data.
        target_dtype: Target numpy dtype for conversion.
        source_dtype: Source dtype (inferred if not provided).

    Returns:
        np.ndarray: Converted audio data.

    Raises:
        ValueError: If conversion is not supported.
    """
    try:
        # Convert bytes to numpy array
        if isinstance(audio_chunk, bytes):
            source_dtype = source_dtype or np.int16
            data = np.frombuffer(audio_chunk, dtype=source_dtype)
        else:
            data = audio_chunk.copy()
            if source_dtype is None:
                source_dtype = data.dtype

        # No conversion needed
        if data.dtype == target_dtype:
            return data

        # Convert between common audio formats
        if source_dtype == np.int16 and target_dtype == np.float32:
            # 16-bit int to 32-bit float (-1.0 to 1.0)
            return data.astype(np.float32) / 32768.0

        elif source_dtype == np.float32 and target_dtype == np.int16:
            # 32-bit float to 16-bit int
            clipped = np.clip(data, -1.0, 1.0)
            return (clipped * 32767.0).astype(np.int16)

        elif source_dtype == np.int32 and target_dtype == np.int16:
            # 32-bit int to 16-bit int
            return (data / 65536).astype(np.int16)

        elif source_dtype == np.int16 and target_dtype == np.int32:
            # 16-bit int to 32-bit int
            return data.astype(np.int32) * 65536

        elif source_dtype == np.float64 and target_dtype == np.float32:
            # 64-bit float to 32-bit float
            return data.astype(np.float32)

        elif source_dtype == np.float32 and target_dtype == np.float64:
            # 32-bit float to 64-bit float
            return data.astype(np.float64)

        else:
            # Generic conversion - may lose precision
            logger.warning(f"Generic conversion from {source_dtype} to {target_dtype}")
            if target_dtype in [np.float32, np.float64]:
                # Convert to float range
                if source_dtype in [np.int8, np.int16, np.int32, np.int64]:
                    # Normalize integer to float
                    info = np.iinfo(source_dtype)
                    normalized = data.astype(np.float64) / max(
                        abs(info.min), abs(info.max)
                    )
                    return normalized.astype(target_dtype)
                else:
                    return data.astype(target_dtype)
            else:
                # Convert to integer
                if source_dtype in [np.float32, np.float64]:
                    # Scale float to integer range
                    info = np.iinfo(target_dtype)
                    scaled = np.clip(data, -1.0, 1.0) * max(
                        abs(info.min), abs(info.max)
                    )
                    return scaled.astype(target_dtype)
                else:
                    return data.astype(target_dtype)

    except Exception as e:
        logger.error(
            f"Error converting audio format from {source_dtype} to {target_dtype}: {e}"
        )
        raise ValueError(
            f"Unsupported audio format conversion: {source_dtype} -> {target_dtype}"
        )


def normalize_audio_chunk(
    audio_chunk: Union[np.ndarray, bytes],
    target_range: Tuple[float, float] = (-1.0, 1.0),
    method: str = "peak",
) -> np.ndarray:
    """
    Normalize audio chunk to a specific range.

    Args:
        audio_chunk: Audio data to normalize.
        target_range: Target range for normalization (min, max).
        method: Normalization method ("peak", "rms", "lufs").

    Returns:
        np.ndarray: Normalized audio data as float32.
    """
    try:
        # Convert to numpy array
        if isinstance(audio_chunk, bytes):
            data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        else:
            data = audio_chunk.astype(np.float32)

        if len(data) == 0:
            return data

        target_min, target_max = target_range

        if method == "peak":
            # Peak normalization - scale by maximum absolute value
            peak = np.max(np.abs(data))
            if peak > 0:
                normalized = data / peak
            else:
                normalized = data

        elif method == "rms":
            # RMS normalization - scale by RMS energy
            rms = np.sqrt(np.mean(data**2))
            if rms > 0:
                normalized = data / (rms * np.sqrt(2))  # RMS to peak conversion
            else:
                normalized = data

        elif method == "lufs":
            # Simplified LUFS-inspired normalization
            # This is a basic approximation, not true LUFS
            squared = data**2
            mean_squared = np.mean(squared)
            if mean_squared > 0:
                loudness = -0.691 + 10 * np.log10(mean_squared)
                target_loudness = -23.0  # Target LUFS
                gain_db = target_loudness - loudness
                gain_linear = 10 ** (gain_db / 20)
                normalized = data * gain_linear
            else:
                normalized = data

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Scale to target range
        if target_min != -1.0 or target_max != 1.0:
            # Clip to [-1, 1] first
            normalized = np.clip(normalized, -1.0, 1.0)
            # Scale to target range
            normalized = (normalized + 1.0) / 2.0  # Map [-1,1] to [0,1]
            normalized = normalized * (target_max - target_min) + target_min

        return normalized

    except Exception as e:
        logger.error(f"Error normalizing audio chunk: {e}")
        # Return original data converted to float32 as fallback
        if isinstance(audio_chunk, bytes):
            return (
                np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            )
        else:
            return audio_chunk.astype(np.float32)


def resample_audio(
    audio_chunk: Union[np.ndarray, bytes],
    source_rate: int,
    target_rate: int,
    method: str = "linear",
) -> np.ndarray:
    """
    Resample audio data to a different sample rate.

    Args:
        audio_chunk: Audio data to resample.
        source_rate: Original sample rate in Hz.
        target_rate: Target sample rate in Hz.
        method: Resampling method ("linear", "nearest").

    Returns:
        np.ndarray: Resampled audio data.
    """
    try:
        # Convert to numpy array
        if isinstance(audio_chunk, bytes):
            data = np.frombuffer(audio_chunk, dtype=np.int16)
        else:
            data = audio_chunk.copy()

        if source_rate == target_rate:
            return data

        if len(data) == 0:
            return data

        # Calculate resampling ratio
        ratio = target_rate / source_rate
        new_length = int(len(data) * ratio)

        if method == "linear":
            # Linear interpolation resampling
            old_indices = np.arange(len(data))
            new_indices = np.linspace(0, len(data) - 1, new_length)
            resampled = np.interp(new_indices, old_indices, data)

        elif method == "nearest":
            # Nearest neighbor resampling
            old_indices = np.arange(len(data))
            new_indices = np.linspace(0, len(data) - 1, new_length)
            nearest_indices = np.round(new_indices).astype(int)
            nearest_indices = np.clip(nearest_indices, 0, len(data) - 1)
            resampled = data[nearest_indices]

        else:
            raise ValueError(f"Unknown resampling method: {method}")

        return resampled.astype(data.dtype)

    except Exception as e:
        logger.error(
            f"Error resampling audio from {source_rate}Hz to {target_rate}Hz: {e}"
        )
        # Return original data as fallback
        if isinstance(audio_chunk, bytes):
            return np.frombuffer(audio_chunk, dtype=np.int16)
        else:
            return audio_chunk


def detect_clipping(
    audio_chunk: Union[np.ndarray, bytes], threshold: float = 0.95
) -> dict:
    """
    Detect audio clipping in the chunk.

    Args:
        audio_chunk: Audio data to analyze.
        threshold: Clipping threshold as fraction of maximum value.

    Returns:
        dict: Clipping analysis results.
    """
    try:
        # Convert to numpy array and normalize to [-1, 1]
        if isinstance(audio_chunk, bytes):
            data = (
                np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            )
        else:
            data = audio_chunk.astype(np.float32)
            # Normalize if integer type
            if np.issubdtype(audio_chunk.dtype, np.integer):
                info = np.iinfo(audio_chunk.dtype)
                data = data / max(abs(info.min), abs(info.max))

        if len(data) == 0:
            return {
                "clipped_samples": 0,
                "clipping_ratio": 0.0,
                "max_amplitude": 0.0,
                "is_clipped": False,
            }

        # Detect clipped samples
        abs_data = np.abs(data)
        clipped_mask = abs_data >= threshold
        clipped_samples = np.sum(clipped_mask)
        clipping_ratio = clipped_samples / len(data)
        max_amplitude = np.max(abs_data)

        return {
            "clipped_samples": int(clipped_samples),
            "clipping_ratio": float(clipping_ratio),
            "max_amplitude": float(max_amplitude),
            "is_clipped": clipping_ratio > 0.01,  # Consider clipped if >1% of samples
        }

    except Exception as e:
        logger.error(f"Error detecting clipping: {e}")
        return {
            "clipped_samples": 0,
            "clipping_ratio": 0.0,
            "max_amplitude": 0.0,
            "is_clipped": False,
            "error": str(e),
        }


def calculate_audio_statistics(audio_chunk: Union[np.ndarray, bytes]) -> dict:
    """
    Calculate comprehensive audio statistics.

    Args:
        audio_chunk: Audio data to analyze.

    Returns:
        dict: Comprehensive audio statistics.
    """
    try:
        # Convert to numpy array
        if isinstance(audio_chunk, bytes):
            data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        else:
            data = audio_chunk.astype(np.float32)

        if len(data) == 0:
            return {
                "length": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "rms": 0.0,
                "peak": 0.0,
                "zero_crossings": 0,
                "dynamic_range": 0.0,
            }

        # Basic statistics
        mean = float(np.mean(data))
        std = float(np.std(data))
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))

        # Zero crossings
        zero_crossings = int(np.sum(np.diff(np.signbit(data))))

        # Dynamic range (difference between max and min in dB)
        if min_val < 0 and max_val > 0:
            dynamic_range = (
                20 * np.log10(max_val / abs(min_val)) if min_val != 0 else float("inf")
            )
        else:
            dynamic_range = 0.0

        return {
            "length": len(data),
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "rms": rms,
            "peak": peak,
            "zero_crossings": zero_crossings,
            "dynamic_range": (
                float(dynamic_range) if not np.isinf(dynamic_range) else 0.0
            ),
        }

    except Exception as e:
        logger.error(f"Error calculating audio statistics: {e}")
        return {
            "length": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "zero_crossings": 0,
            "dynamic_range": 0.0,
            "error": str(e),
        }
