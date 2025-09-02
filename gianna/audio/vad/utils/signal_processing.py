"""
Signal processing utilities for VAD operations.

This module provides advanced signal processing functions used for
voice activity detection, including frequency analysis, filtering,
and feature extraction.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def preemphasis_filter(
    audio_chunk: Union[np.ndarray, bytes], coeff: float = 0.97
) -> np.ndarray:
    """
    Apply pre-emphasis filter to audio signal.

    Pre-emphasis helps balance the frequency spectrum and improves
    the performance of subsequent processing steps.

    Args:
        audio_chunk: Audio data to filter.
        coeff: Pre-emphasis coefficient (typically 0.95-0.97).

    Returns:
        np.ndarray: Pre-emphasized audio signal.
    """
    try:
        # Convert to numpy array
        if isinstance(audio_chunk, bytes):
            data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        else:
            data = audio_chunk.astype(np.float32)

        if len(data) <= 1:
            return data

        # Apply pre-emphasis: y[n] = x[n] - coeff * x[n-1]
        emphasized = np.zeros_like(data)
        emphasized[0] = data[0]
        emphasized[1:] = data[1:] - coeff * data[:-1]

        return emphasized

    except Exception as e:
        logger.error(f"Error applying pre-emphasis filter: {e}")
        if isinstance(audio_chunk, bytes):
            return np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        else:
            return audio_chunk.astype(np.float32)


def apply_bandpass_filter(
    audio_chunk: Union[np.ndarray, bytes],
    sample_rate: int,
    low_freq: float = 300.0,
    high_freq: float = 3400.0,
    order: int = 2,
) -> np.ndarray:
    """
    Apply a bandpass filter to focus on speech frequencies.

    Args:
        audio_chunk: Audio data to filter.
        sample_rate: Sample rate of the audio in Hz.
        low_freq: Low cutoff frequency in Hz.
        high_freq: High cutoff frequency in Hz.
        order: Filter order (complexity).

    Returns:
        np.ndarray: Bandpass filtered audio.
    """
    try:
        # Convert to numpy array
        if isinstance(audio_chunk, bytes):
            data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        else:
            data = audio_chunk.astype(np.float32)

        if len(data) < order * 2:
            return data

        # Simple FIR bandpass filter using windowing method
        # This is a simplified implementation suitable for basic filtering

        # Design filter using frequency sampling
        fft_size = min(512, len(data))
        freqs = np.fft.fftfreq(fft_size, 1 / sample_rate)

        # Create ideal frequency response
        H = np.zeros(fft_size, dtype=complex)
        for i, freq in enumerate(freqs):
            abs_freq = abs(freq)
            if low_freq <= abs_freq <= high_freq:
                H[i] = 1.0

        # Convert to time domain to get impulse response
        h = np.fft.ifft(H).real

        # Apply Hamming window
        window = np.hamming(len(h))
        h = h * window

        # Apply filter using convolution
        if len(data) >= len(h):
            filtered = np.convolve(data, h, mode="same")
        else:
            filtered = data

        return filtered

    except Exception as e:
        logger.error(f"Error applying bandpass filter: {e}")
        if isinstance(audio_chunk, bytes):
            return np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        else:
            return audio_chunk.astype(np.float32)


def calculate_zero_crossing_rate(
    audio_chunk: Union[np.ndarray, bytes], frame_length: int = 512
) -> float:
    """
    Calculate zero crossing rate of audio signal.

    Zero crossing rate is useful for distinguishing between
    voiced and unvoiced speech segments.

    Args:
        audio_chunk: Audio data to analyze.
        frame_length: Length of analysis frame.

    Returns:
        float: Zero crossing rate (crossings per sample).
    """
    try:
        # Convert to numpy array
        if isinstance(audio_chunk, bytes):
            data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        else:
            data = audio_chunk.astype(np.float32)

        if len(data) <= 1:
            return 0.0

        # Calculate zero crossings
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(data))))

        # Normalize by signal length
        zcr = zero_crossings / (len(data) - 1)

        return float(zcr)

    except Exception as e:
        logger.error(f"Error calculating zero crossing rate: {e}")
        return 0.0


def compute_spectral_centroid(
    audio_chunk: Union[np.ndarray, bytes], sample_rate: int
) -> float:
    """
    Compute spectral centroid of audio signal.

    Spectral centroid indicates where the "center of mass" of the
    spectrum is located and is useful for voice activity detection.

    Args:
        audio_chunk: Audio data to analyze.
        sample_rate: Sample rate of the audio.

    Returns:
        float: Spectral centroid in Hz.
    """
    try:
        # Convert to numpy array
        if isinstance(audio_chunk, bytes):
            data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        else:
            data = audio_chunk.astype(np.float32)

        if len(data) <= 1:
            return 0.0

        # Apply window to reduce spectral leakage
        windowed = data * np.hamming(len(data))

        # Compute FFT
        fft = np.fft.fft(windowed)
        magnitude = np.abs(fft[: len(fft) // 2])  # Take only positive frequencies

        # Frequency bins
        freqs = np.fft.fftfreq(len(windowed), 1 / sample_rate)[: len(fft) // 2]

        # Calculate spectral centroid
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            centroid = 0.0

        return float(centroid)

    except Exception as e:
        logger.error(f"Error computing spectral centroid: {e}")
        return 0.0


def calculate_spectral_features(
    audio_chunk: Union[np.ndarray, bytes], sample_rate: int
) -> dict:
    """
    Calculate comprehensive spectral features.

    Args:
        audio_chunk: Audio data to analyze.
        sample_rate: Sample rate of the audio.

    Returns:
        dict: Dictionary containing various spectral features.
    """
    try:
        # Convert to numpy array
        if isinstance(audio_chunk, bytes):
            data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        else:
            data = audio_chunk.astype(np.float32)

        if len(data) <= 1:
            return {
                "spectral_centroid": 0.0,
                "spectral_bandwidth": 0.0,
                "spectral_rolloff": 0.0,
                "spectral_flatness": 0.0,
                "zero_crossing_rate": 0.0,
            }

        # Apply window
        windowed = data * np.hamming(len(data))

        # Compute FFT
        fft = np.fft.fft(windowed)
        magnitude = np.abs(fft[: len(fft) // 2])
        freqs = np.fft.fftfreq(len(windowed), 1 / sample_rate)[: len(fft) // 2]

        # Spectral centroid
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            centroid = 0.0

        # Spectral bandwidth
        if np.sum(magnitude) > 0:
            bandwidth = np.sqrt(
                np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude)
            )
        else:
            bandwidth = 0.0

        # Spectral rolloff (frequency below which 85% of energy lies)
        if np.sum(magnitude) > 0:
            cumsum = np.cumsum(magnitude)
            rolloff_threshold = 0.85 * cumsum[-1]
            rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
            if len(rolloff_idx) > 0:
                rolloff = freqs[rolloff_idx[0]]
            else:
                rolloff = freqs[-1]
        else:
            rolloff = 0.0

        # Spectral flatness (Wiener entropy)
        if len(magnitude) > 0 and np.all(magnitude > 0):
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            if arithmetic_mean > 0:
                flatness = geometric_mean / arithmetic_mean
            else:
                flatness = 0.0
        else:
            flatness = 0.0

        # Zero crossing rate
        zcr = calculate_zero_crossing_rate(data)

        return {
            "spectral_centroid": float(centroid),
            "spectral_bandwidth": float(bandwidth),
            "spectral_rolloff": float(rolloff),
            "spectral_flatness": float(flatness),
            "zero_crossing_rate": float(zcr),
        }

    except Exception as e:
        logger.error(f"Error calculating spectral features: {e}")
        return {
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_rolloff": 0.0,
            "spectral_flatness": 0.0,
            "zero_crossing_rate": 0.0,
            "error": str(e),
        }


def estimate_noise_floor(audio_chunks: list, percentile: float = 25.0) -> float:
    """
    Estimate noise floor from a collection of audio chunks.

    Args:
        audio_chunks: List of audio chunks to analyze.
        percentile: Percentile to use for noise floor estimation.

    Returns:
        float: Estimated noise floor energy level.
    """
    try:
        if not audio_chunks:
            return 0.0

        energies = []

        for chunk in audio_chunks:
            # Convert to numpy array
            if isinstance(chunk, bytes):
                data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            else:
                data = chunk.astype(np.float32)

            if len(data) > 0:
                # Calculate RMS energy
                rms = np.sqrt(np.mean(data**2))
                # Normalize assuming 16-bit audio
                normalized_rms = rms / 32768.0 if isinstance(chunk, bytes) else rms
                energies.append(normalized_rms)

        if not energies:
            return 0.0

        # Use percentile to estimate noise floor
        noise_floor = np.percentile(energies, percentile)

        return float(noise_floor)

    except Exception as e:
        logger.error(f"Error estimating noise floor: {e}")
        return 0.0


def compute_mfcc_features(
    audio_chunk: Union[np.ndarray, bytes],
    sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 512,
) -> np.ndarray:
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCC).

    This is a simplified MFCC implementation suitable for basic
    voice activity detection applications.

    Args:
        audio_chunk: Audio data to analyze.
        sample_rate: Sample rate of the audio.
        n_mfcc: Number of MFCC coefficients to compute.
        n_fft: FFT window size.

    Returns:
        np.ndarray: MFCC feature vector.
    """
    try:
        # Convert to numpy array
        if isinstance(audio_chunk, bytes):
            data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        else:
            data = audio_chunk.astype(np.float32)

        if len(data) < n_fft:
            return np.zeros(n_mfcc)

        # Apply pre-emphasis
        emphasized = preemphasis_filter(data)

        # Apply window and compute FFT
        windowed = emphasized[:n_fft] * np.hamming(n_fft)
        fft = np.fft.fft(windowed, n_fft)
        magnitude = np.abs(fft[: n_fft // 2])

        # Mel filter bank (simplified)
        n_filters = 26
        low_freq = 0
        high_freq = sample_rate // 2

        # Convert to mel scale
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel filter banks
        mel_points = np.linspace(
            hz_to_mel(low_freq), hz_to_mel(high_freq), n_filters + 2
        )
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        # Create filter bank matrix
        filter_banks = np.zeros((n_filters, n_fft // 2))
        for i in range(1, n_filters + 1):
            left = int(bin_points[i - 1])
            center = int(bin_points[i])
            right = int(bin_points[i + 1])

            for j in range(left, center):
                if center != left:
                    filter_banks[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right != center:
                    filter_banks[i - 1, j] = (right - j) / (right - center)

        # Apply filter banks
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        mel_energies = np.dot(magnitude, filter_banks.T)
        mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)

        # Log and DCT
        log_mel_energies = np.log(mel_energies)

        # Simple DCT implementation
        mfcc = np.zeros(n_mfcc)
        for i in range(n_mfcc):
            for j in range(n_filters):
                mfcc[i] += log_mel_energies[j] * np.cos(
                    np.pi * i * (j + 0.5) / n_filters
                )

        return mfcc

    except Exception as e:
        logger.error(f"Error computing MFCC features: {e}")
        return np.zeros(n_mfcc)


def detect_clipping(
    audio_chunk: Union[np.ndarray, bytes], threshold: float = 0.95
) -> dict:
    """
    Detect audio clipping and saturation.

    Args:
        audio_chunk: Audio data to analyze.
        threshold: Clipping threshold as fraction of maximum value.

    Returns:
        dict: Clipping detection results.
    """
    try:
        # Convert to numpy array and normalize
        if isinstance(audio_chunk, bytes):
            data = (
                np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            )
        else:
            data = audio_chunk.astype(np.float32)
            if np.issubdtype(audio_chunk.dtype, np.integer):
                info = np.iinfo(audio_chunk.dtype)
                data = data / max(abs(info.min), abs(info.max))

        if len(data) == 0:
            return {
                "is_clipped": False,
                "clipping_ratio": 0.0,
                "max_amplitude": 0.0,
                "clipped_samples": 0,
            }

        # Detect clipping
        abs_data = np.abs(data)
        clipped_mask = abs_data >= threshold
        clipped_samples = np.sum(clipped_mask)
        clipping_ratio = clipped_samples / len(data)
        max_amplitude = np.max(abs_data)

        return {
            "is_clipped": clipping_ratio > 0.01,  # >1% clipped samples
            "clipping_ratio": float(clipping_ratio),
            "max_amplitude": float(max_amplitude),
            "clipped_samples": int(clipped_samples),
        }

    except Exception as e:
        logger.error(f"Error detecting clipping: {e}")
        return {
            "is_clipped": False,
            "clipping_ratio": 0.0,
            "max_amplitude": 0.0,
            "clipped_samples": 0,
            "error": str(e),
        }
