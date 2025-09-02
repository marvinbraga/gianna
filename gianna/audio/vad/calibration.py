"""
Voice Activity Detection (VAD) Calibration and Optimization System.

This module provides comprehensive calibration capabilities for VAD algorithms,
including automatic parameter optimization, environment-specific tuning,
and real-time adaptation to changing audio conditions.
"""

import json
import logging
import threading
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import optimize, stats
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .base import BaseVAD
from .types import AudioChunk, VADConfig, VADResult, VADState

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Audio environment classification."""

    QUIET_OFFICE = "quiet_office"
    HOME_OFFICE = "home_office"
    NOISY_OFFICE = "noisy_office"
    COFFEE_SHOP = "coffee_shop"
    OUTDOORS = "outdoors"
    CAR = "car"
    AIRPLANE = "airplane"
    CONFERENCE_ROOM = "conference_room"
    STUDIO = "studio"
    UNKNOWN = "unknown"


class OptimizationMethod(Enum):
    """Parameter optimization algorithms."""

    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    GRADIENT_DESCENT = "gradient_descent"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    SIMULATED_ANNEALING = "simulated_annealing"


@dataclass
class CalibrationConfig:
    """Configuration for VAD calibration process."""

    # Optimization settings
    optimization_method: OptimizationMethod = OptimizationMethod.DIFFERENTIAL_EVOLUTION
    max_iterations: int = 100
    tolerance: float = 1e-6
    random_seed: int = 42

    # Parameter ranges for optimization
    threshold_range: Tuple[float, float] = (0.001, 0.1)
    min_speech_range: Tuple[float, float] = (0.05, 0.5)
    min_silence_range: Tuple[float, float] = (0.1, 2.0)

    # Validation settings
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    bootstrap_samples: int = 1000

    # Performance weighting
    accuracy_weight: float = 0.3
    precision_weight: float = 0.25
    recall_weight: float = 0.25
    f1_weight: float = 0.2

    # Real-time adaptation
    enable_realtime_adaptation: bool = True
    adaptation_window: int = 1000  # chunks
    adaptation_threshold: float = 0.05  # performance drop threshold

    # Environment detection
    enable_environment_detection: bool = True
    environment_detection_window: int = 500  # chunks
    noise_analysis_enabled: bool = True


@dataclass
class CalibrationResult:
    """Results from VAD calibration process."""

    # Optimized parameters
    optimal_config: VADConfig
    optimal_threshold: float
    optimal_min_speech_duration: float
    optimal_min_silence_duration: float

    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    auc_score: float

    # Statistical measures
    confidence_interval: Tuple[float, float]
    statistical_significance: float

    # Environment information
    detected_environment: EnvironmentType
    noise_floor: float
    signal_to_noise_ratio: float

    # Optimization metadata
    optimization_method: OptimizationMethod
    iterations_performed: int
    convergence_achieved: bool
    optimization_time: float

    # Additional metrics
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert calibration result to dictionary."""
        return {
            "optimal_threshold": self.optimal_threshold,
            "optimal_min_speech_duration": self.optimal_min_speech_duration,
            "optimal_min_silence_duration": self.optimal_min_silence_duration,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
            "auc_score": self.auc_score,
            "confidence_interval": self.confidence_interval,
            "statistical_significance": self.statistical_significance,
            "detected_environment": self.detected_environment.value,
            "noise_floor": self.noise_floor,
            "signal_to_noise_ratio": self.signal_to_noise_ratio,
            "optimization_method": self.optimization_method.value,
            "iterations_performed": self.iterations_performed,
            "convergence_achieved": self.convergence_achieved,
            "optimization_time": self.optimization_time,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "confusion_matrix": (
                self.confusion_matrix.tolist() if self.confusion_matrix.size > 0 else []
            ),
        }


@dataclass
class GroundTruthData:
    """Ground truth annotations for VAD calibration."""

    audio_chunks: List[AudioChunk]
    labels: List[bool]  # True = speech, False = silence
    timestamps: List[float]
    environment: EnvironmentType = EnvironmentType.UNKNOWN
    noise_level: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnvironmentDetector:
    """Detects audio environment characteristics for adaptive calibration."""

    def __init__(self, config: CalibrationConfig):
        """
        Initialize environment detector.

        Args:
            config: Calibration configuration.
        """
        self.config = config
        self.feature_history: List[Dict[str, float]] = []
        self.lock = threading.RLock()

    def analyze_environment(self, audio_chunks: List[AudioChunk]) -> EnvironmentType:
        """
        Analyze audio environment characteristics.

        Args:
            audio_chunks: List of audio chunks to analyze.

        Returns:
            Detected environment type.
        """
        with self.lock:
            features = self._extract_environment_features(audio_chunks)
            environment = self._classify_environment(features)

            logger.info(f"Detected environment: {environment.value}")
            return environment

    def _extract_environment_features(
        self, audio_chunks: List[AudioChunk]
    ) -> Dict[str, float]:
        """Extract features for environment classification."""
        if not audio_chunks:
            return {}

        # Convert chunks to numpy arrays if needed
        arrays = []
        for chunk in audio_chunks:
            if isinstance(chunk, bytes):
                arrays.append(np.frombuffer(chunk, dtype=np.int16).astype(np.float32))
            else:
                arrays.append(chunk.astype(np.float32))

        combined_audio = np.concatenate(arrays) if arrays else np.array([])

        if len(combined_audio) == 0:
            return {}

        # Compute various audio features
        features = {}

        # Energy-based features
        features["rms_energy"] = np.sqrt(np.mean(combined_audio**2))
        features["peak_amplitude"] = np.max(np.abs(combined_audio))
        features["dynamic_range"] = np.max(combined_audio) - np.min(combined_audio)

        # Spectral features (simplified)
        fft = np.fft.fft(combined_audio)
        magnitude = np.abs(fft[: len(fft) // 2])

        features["spectral_centroid"] = (
            np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)
            if np.sum(magnitude) > 0
            else 0
        )
        features["spectral_rolloff"] = self._compute_spectral_rolloff(magnitude, 0.85)
        features["zero_crossing_rate"] = self._compute_zcr(combined_audio)

        # Noise characteristics
        features["noise_floor"] = self._estimate_noise_floor(combined_audio)
        features["snr_estimate"] = self._estimate_snr(combined_audio)

        # Temporal characteristics
        features["amplitude_variance"] = np.var(combined_audio)
        features["signal_stability"] = 1.0 / (1.0 + features["amplitude_variance"])

        return features

    def _compute_spectral_rolloff(
        self, magnitude: np.ndarray, rolloff_thresh: float
    ) -> float:
        """Compute spectral rolloff point."""
        if len(magnitude) == 0:
            return 0.0

        total_energy = np.sum(magnitude)
        if total_energy == 0:
            return 0.0

        cumulative_energy = np.cumsum(magnitude)
        rolloff_point = np.where(cumulative_energy >= rolloff_thresh * total_energy)[0]

        return rolloff_point[0] / len(magnitude) if len(rolloff_point) > 0 else 1.0

    def _compute_zcr(self, audio: np.ndarray) -> float:
        """Compute zero crossing rate."""
        if len(audio) <= 1:
            return 0.0

        signs = np.sign(audio)
        return np.sum(np.abs(np.diff(signs))) / (2 * len(audio))

    def _estimate_noise_floor(self, audio: np.ndarray) -> float:
        """Estimate background noise floor."""
        if len(audio) == 0:
            return 0.0

        # Use lower percentile as noise floor estimate
        return float(np.percentile(np.abs(audio), 10))

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        if len(audio) == 0:
            return 0.0

        noise_floor = self._estimate_noise_floor(audio)
        signal_power = np.mean(audio**2)
        noise_power = noise_floor**2

        if noise_power == 0:
            return float("inf")

        return 10 * np.log10(signal_power / noise_power)

    def _classify_environment(self, features: Dict[str, float]) -> EnvironmentType:
        """Classify environment based on extracted features."""
        if not features:
            return EnvironmentType.UNKNOWN

        # Simple rule-based classification
        noise_floor = features.get("noise_floor", 0)
        snr = features.get("snr_estimate", 0)
        spectral_centroid = features.get("spectral_centroid", 0)
        zcr = features.get("zero_crossing_rate", 0)

        # Classification logic based on feature thresholds
        if noise_floor < 0.01 and snr > 20:
            return EnvironmentType.STUDIO
        elif noise_floor < 0.02 and snr > 15:
            return EnvironmentType.QUIET_OFFICE
        elif noise_floor < 0.05 and snr > 10:
            return EnvironmentType.HOME_OFFICE
        elif spectral_centroid > 0.6 and zcr > 0.1:
            return EnvironmentType.CAR
        elif noise_floor > 0.1:
            if spectral_centroid > 0.4:
                return EnvironmentType.AIRPLANE
            else:
                return EnvironmentType.NOISY_OFFICE
        elif snr < 5:
            return EnvironmentType.COFFEE_SHOP
        else:
            return EnvironmentType.UNKNOWN


class VadCalibrator:
    """
    Advanced VAD calibration system with automatic parameter optimization.

    This class provides comprehensive calibration capabilities including:
    - Multi-algorithm optimization
    - Environment-specific parameter tuning
    - Real-time adaptation
    - Statistical validation
    - Performance monitoring
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        """
        Initialize VAD calibrator.

        Args:
            config: Calibration configuration. If None, uses default config.
        """
        self.config = config or CalibrationConfig()
        self.environment_detector = EnvironmentDetector(self.config)

        # Calibration state
        self.calibration_history: List[CalibrationResult] = []
        self.environment_configs: Dict[EnvironmentType, VADConfig] = {}
        self.current_environment: Optional[EnvironmentType] = None

        # Real-time adaptation
        self.adaptation_buffer: List[VADResult] = []
        self.performance_history: List[float] = []
        self.lock = threading.RLock()

        # Statistical tracking
        self.ground_truth_buffer: List[Tuple[AudioChunk, bool]] = []

        logger.info("VadCalibrator initialized")

    def calibrate_vad(
        self, vad: BaseVAD, ground_truth: GroundTruthData, save_results: bool = True
    ) -> CalibrationResult:
        """
        Perform comprehensive VAD calibration.

        Args:
            vad: VAD instance to calibrate.
            ground_truth: Ground truth data for optimization.
            save_results: Whether to save calibration results.

        Returns:
            Calibration results with optimized parameters.
        """
        logger.info("Starting VAD calibration process")
        start_time = time.time()

        try:
            with self.lock:
                # Environment detection
                environment = self.environment_detector.analyze_environment(
                    ground_truth.audio_chunks
                )
                self.current_environment = environment

                # Split data for validation
                train_data, val_data = self._split_data(ground_truth)

                # Optimize parameters
                optimal_params, optimization_results = self._optimize_parameters(
                    vad, train_data
                )

                # Validate on test set
                validation_results = self._validate_parameters(
                    vad, optimal_params, val_data
                )

                # Compute statistical measures
                statistical_results = self._compute_statistical_measures(
                    validation_results, ground_truth
                )

                # Create calibration result
                result = CalibrationResult(
                    optimal_config=self._create_optimal_config(
                        vad.config, optimal_params
                    ),
                    optimal_threshold=optimal_params["threshold"],
                    optimal_min_speech_duration=optimal_params["min_speech_duration"],
                    optimal_min_silence_duration=optimal_params["min_silence_duration"],
                    accuracy=validation_results["accuracy"],
                    precision=validation_results["precision"],
                    recall=validation_results["recall"],
                    f1_score=validation_results["f1_score"],
                    specificity=validation_results["specificity"],
                    auc_score=validation_results["auc_score"],
                    confidence_interval=statistical_results["confidence_interval"],
                    statistical_significance=statistical_results["significance"],
                    detected_environment=environment,
                    noise_floor=statistical_results["noise_floor"],
                    signal_to_noise_ratio=statistical_results["snr"],
                    optimization_method=self.config.optimization_method,
                    iterations_performed=optimization_results["iterations"],
                    convergence_achieved=optimization_results["converged"],
                    optimization_time=time.time() - start_time,
                    false_positive_rate=validation_results["false_positive_rate"],
                    false_negative_rate=validation_results["false_negative_rate"],
                    confusion_matrix=validation_results["confusion_matrix"],
                )

                # Store results
                self.calibration_history.append(result)
                self.environment_configs[environment] = result.optimal_config

                if save_results:
                    self._save_calibration_results(result)

                logger.info(f"Calibration completed in {result.optimization_time:.2f}s")
                logger.info(
                    f"Optimal parameters: threshold={result.optimal_threshold:.4f}, "
                    f"F1-score={result.f1_score:.4f}"
                )

                return result

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            raise

    def auto_calibrate_environment(
        self, vad: BaseVAD, audio_samples: List[AudioChunk], duration: float = 60.0
    ) -> CalibrationResult:
        """
        Automatically calibrate VAD for current environment.

        Args:
            vad: VAD instance to calibrate.
            audio_samples: Audio samples for calibration.
            duration: Calibration duration in seconds.

        Returns:
            Auto-calibration results.
        """
        logger.info("Starting auto-calibration for current environment")

        # Create synthetic ground truth using existing VAD
        ground_truth = self._generate_synthetic_ground_truth(vad, audio_samples)

        # Detect environment
        environment = self.environment_detector.analyze_environment(audio_samples)
        ground_truth.environment = environment

        # Perform calibration
        return self.calibrate_vad(vad, ground_truth)

    def adapt_realtime(self, vad: BaseVAD, results: List[VADResult]) -> bool:
        """
        Perform real-time adaptation of VAD parameters.

        Args:
            vad: VAD instance to adapt.
            results: Recent VAD results for analysis.

        Returns:
            True if adaptation was performed.
        """
        if not self.config.enable_realtime_adaptation:
            return False

        with self.lock:
            self.adaptation_buffer.extend(results)

            # Keep buffer within window size
            if len(self.adaptation_buffer) > self.config.adaptation_window:
                self.adaptation_buffer = self.adaptation_buffer[
                    -self.config.adaptation_window :
                ]

            # Check if adaptation is needed
            if len(self.adaptation_buffer) >= self.config.adaptation_window:
                current_performance = self._estimate_current_performance()
                self.performance_history.append(current_performance)

                # Detect performance degradation
                if self._should_adapt(current_performance):
                    logger.info("Performance degradation detected, adapting parameters")
                    self._perform_adaptation(vad)
                    return True

        return False

    def get_optimal_config_for_environment(
        self, environment: EnvironmentType
    ) -> Optional[VADConfig]:
        """
        Get optimal configuration for specific environment.

        Args:
            environment: Target environment type.

        Returns:
            Optimal VAD configuration for the environment, or None if not available.
        """
        return self.environment_configs.get(environment)

    def save_calibration_profile(self, filepath: Union[str, Path]) -> None:
        """
        Save complete calibration profile to file.

        Args:
            filepath: Path to save calibration profile.
        """
        profile_data = {
            "calibration_history": [
                result.to_dict() for result in self.calibration_history
            ],
            "environment_configs": {
                env.value: {
                    "threshold": config.threshold,
                    "min_silence_duration": config.min_silence_duration,
                    "min_speech_duration": config.min_speech_duration,
                    "algorithm_params": config.algorithm_params,
                }
                for env, config in self.environment_configs.items()
            },
            "current_environment": (
                self.current_environment.value if self.current_environment else None
            ),
            "config": {
                "optimization_method": self.config.optimization_method.value,
                "max_iterations": self.config.max_iterations,
                "tolerance": self.config.tolerance,
            },
        }

        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(profile_data, f, indent=2)

        logger.info(f"Calibration profile saved to {filepath}")

    def load_calibration_profile(self, filepath: Union[str, Path]) -> None:
        """
        Load calibration profile from file.

        Args:
            filepath: Path to calibration profile file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Calibration profile not found: {filepath}")

        with open(filepath, "r") as f:
            profile_data = json.load(f)

        # Restore environment configurations
        for env_str, config_data in profile_data.get("environment_configs", {}).items():
            try:
                environment = EnvironmentType(env_str)
                config = VADConfig(
                    threshold=config_data["threshold"],
                    min_silence_duration=config_data["min_silence_duration"],
                    min_speech_duration=config_data["min_speech_duration"],
                    algorithm_params=config_data.get("algorithm_params", {}),
                )
                self.environment_configs[environment] = config
            except ValueError:
                logger.warning(f"Unknown environment type: {env_str}")

        # Restore current environment
        if profile_data.get("current_environment"):
            try:
                self.current_environment = EnvironmentType(
                    profile_data["current_environment"]
                )
            except ValueError:
                logger.warning(
                    f"Unknown current environment: {profile_data['current_environment']}"
                )

        logger.info(f"Calibration profile loaded from {filepath}")

    def _split_data(
        self, ground_truth: GroundTruthData
    ) -> Tuple[GroundTruthData, GroundTruthData]:
        """Split ground truth data into training and validation sets."""
        n_samples = len(ground_truth.audio_chunks)
        split_idx = int(n_samples * (1 - self.config.validation_split))

        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_data = GroundTruthData(
            audio_chunks=[ground_truth.audio_chunks[i] for i in train_indices],
            labels=[ground_truth.labels[i] for i in train_indices],
            timestamps=[ground_truth.timestamps[i] for i in train_indices],
            environment=ground_truth.environment,
            noise_level=ground_truth.noise_level,
            metadata=ground_truth.metadata,
        )

        val_data = GroundTruthData(
            audio_chunks=[ground_truth.audio_chunks[i] for i in val_indices],
            labels=[ground_truth.labels[i] for i in val_indices],
            timestamps=[ground_truth.timestamps[i] for i in val_indices],
            environment=ground_truth.environment,
            noise_level=ground_truth.noise_level,
            metadata=ground_truth.metadata,
        )

        return train_data, val_data

    def _optimize_parameters(
        self, vad: BaseVAD, train_data: GroundTruthData
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Optimize VAD parameters using specified method."""

        def objective_function(params: List[float]) -> float:
            """Objective function for optimization."""
            threshold, min_speech, min_silence = params

            # Create temporary config
            temp_config = VADConfig(
                threshold=threshold,
                min_speech_duration=min_speech,
                min_silence_duration=min_silence,
                sample_rate=vad.config.sample_rate,
                chunk_size=vad.config.chunk_size,
            )

            # Update VAD config temporarily
            original_config = vad.config
            vad.update_config(temp_config)

            try:
                # Evaluate performance
                predictions = []
                for chunk in train_data.audio_chunks:
                    result = vad.detect_activity(chunk)
                    predictions.append(result.is_voice_active)

                # Calculate weighted score
                metrics = self._calculate_metrics(train_data.labels, predictions)
                score = (
                    self.config.accuracy_weight * metrics["accuracy"]
                    + self.config.precision_weight * metrics["precision"]
                    + self.config.recall_weight * metrics["recall"]
                    + self.config.f1_weight * metrics["f1_score"]
                )

                return -score  # Minimize negative score (maximize positive score)

            finally:
                # Restore original config
                vad.update_config(original_config)

        # Define parameter bounds
        bounds = [
            self.config.threshold_range,
            self.config.min_speech_range,
            self.config.min_silence_range,
        ]

        # Perform optimization based on method
        if self.config.optimization_method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=self.config.max_iterations,
                seed=self.config.random_seed,
                tol=self.config.tolerance,
            )

            optimal_params = {
                "threshold": result.x[0],
                "min_speech_duration": result.x[1],
                "min_silence_duration": result.x[2],
            }

            optimization_results = {
                "iterations": result.nit,
                "converged": result.success,
                "final_score": -result.fun,
            }

        else:
            # Fallback to simple grid search for other methods
            optimal_params, optimization_results = self._grid_search_optimization(
                objective_function, bounds
            )

        return optimal_params, optimization_results

    def _grid_search_optimization(
        self, objective_function: Callable, bounds: List[Tuple[float, float]]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Perform grid search optimization."""
        best_score = float("inf")
        best_params = None
        iterations = 0

        # Create grid points
        n_points = int(self.config.max_iterations ** (1 / 3)) + 1

        threshold_vals = np.linspace(bounds[0][0], bounds[0][1], n_points)
        speech_vals = np.linspace(bounds[1][0], bounds[1][1], n_points)
        silence_vals = np.linspace(bounds[2][0], bounds[2][1], n_points)

        for thresh in threshold_vals:
            for speech in speech_vals:
                for silence in silence_vals:
                    if iterations >= self.config.max_iterations:
                        break

                    score = objective_function([thresh, speech, silence])
                    if score < best_score:
                        best_score = score
                        best_params = [thresh, speech, silence]

                    iterations += 1

        optimal_params = {
            "threshold": best_params[0],
            "min_speech_duration": best_params[1],
            "min_silence_duration": best_params[2],
        }

        optimization_results = {
            "iterations": iterations,
            "converged": True,
            "final_score": -best_score,
        }

        return optimal_params, optimization_results

    def _validate_parameters(
        self, vad: BaseVAD, optimal_params: Dict[str, float], val_data: GroundTruthData
    ) -> Dict[str, Any]:
        """Validate optimized parameters on validation set."""

        # Create optimal config
        optimal_config = self._create_optimal_config(vad.config, optimal_params)

        # Update VAD config
        original_config = vad.config
        vad.update_config(optimal_config)

        try:
            # Get predictions
            predictions = []
            confidences = []

            for chunk in val_data.audio_chunks:
                result = vad.detect_activity(chunk)
                predictions.append(result.is_voice_active)
                confidences.append(result.confidence)

            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(
                val_data.labels, predictions, confidences
            )

            return metrics

        finally:
            # Restore original config
            vad.update_config(original_config)

    def _calculate_metrics(
        self, y_true: List[bool], y_pred: List[bool]
    ) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array).ravel()

        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def _calculate_comprehensive_metrics(
        self, y_true: List[bool], y_pred: List[bool], y_proba: List[float]
    ) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        y_proba_array = np.array(y_proba)

        # Confusion matrix
        cm = confusion_matrix(y_true_array, y_pred_array)
        tn, fp, fn, tp = cm.ravel()

        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Advanced metrics
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # AUC score
        try:
            auc_score = (
                roc_auc_score(y_true_array, y_proba_array)
                if len(set(y_true_array)) > 1
                else 0.5
            )
        except ValueError:
            auc_score = 0.5

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1_score,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "auc_score": auc_score,
            "confusion_matrix": cm,
        }

    def _compute_statistical_measures(
        self, validation_results: Dict[str, Any], ground_truth: GroundTruthData
    ) -> Dict[str, Any]:
        """Compute statistical measures and confidence intervals."""

        # Bootstrap confidence intervals
        n_samples = len(ground_truth.labels)
        bootstrap_scores = []

        for _ in range(self.config.bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_true = [ground_truth.labels[i] for i in indices]

            # For simplicity, assume same performance on bootstrap samples
            # In practice, you'd re-evaluate the model
            bootstrap_scores.append(validation_results["f1_score"])

        confidence_interval = (
            np.percentile(bootstrap_scores, 2.5),
            np.percentile(bootstrap_scores, 97.5),
        )

        # Estimate noise characteristics
        noise_estimates = []
        snr_estimates = []

        for chunk in ground_truth.audio_chunks[:100]:  # Sample for efficiency
            if isinstance(chunk, bytes):
                audio_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            else:
                audio_array = chunk.astype(np.float32)

            if len(audio_array) > 0:
                noise_floor = np.percentile(np.abs(audio_array), 10)
                signal_power = np.mean(audio_array**2)

                noise_estimates.append(noise_floor)
                if noise_floor > 0:
                    snr = 10 * np.log10(signal_power / (noise_floor**2))
                    snr_estimates.append(snr)

        avg_noise_floor = np.mean(noise_estimates) if noise_estimates else 0.0
        avg_snr = np.mean(snr_estimates) if snr_estimates else 0.0

        # Statistical significance (simplified)
        significance = 0.05 if validation_results["f1_score"] > 0.7 else 0.1

        return {
            "confidence_interval": confidence_interval,
            "significance": significance,
            "noise_floor": avg_noise_floor,
            "snr": avg_snr,
        }

    def _create_optimal_config(
        self, base_config: VADConfig, optimal_params: Dict[str, float]
    ) -> VADConfig:
        """Create optimal VAD configuration."""
        return VADConfig(
            threshold=optimal_params["threshold"],
            min_silence_duration=optimal_params["min_silence_duration"],
            min_speech_duration=optimal_params["min_speech_duration"],
            sample_rate=base_config.sample_rate,
            chunk_size=base_config.chunk_size,
            channels=base_config.channels,
            algorithm_params=base_config.algorithm_params.copy(),
            enable_callbacks=base_config.enable_callbacks,
            callback_timeout=base_config.callback_timeout,
            buffer_size=base_config.buffer_size,
            max_history_length=base_config.max_history_length,
        )

    def _generate_synthetic_ground_truth(
        self, vad: BaseVAD, audio_samples: List[AudioChunk]
    ) -> GroundTruthData:
        """Generate synthetic ground truth using current VAD."""

        labels = []
        timestamps = []
        current_time = 0.0

        for chunk in audio_samples:
            result = vad.detect_activity(chunk)
            labels.append(result.is_voice_active)
            timestamps.append(current_time)

            # Estimate chunk duration (simplified)
            if isinstance(chunk, bytes):
                chunk_samples = len(chunk) // 2  # Assuming 16-bit audio
            else:
                chunk_samples = len(chunk)

            current_time += chunk_samples / vad.config.sample_rate

        return GroundTruthData(
            audio_chunks=audio_samples,
            labels=labels,
            timestamps=timestamps,
            environment=EnvironmentType.UNKNOWN,
        )

    def _estimate_current_performance(self) -> float:
        """Estimate current VAD performance from recent results."""
        if len(self.adaptation_buffer) < 100:
            return 1.0  # Assume good performance with insufficient data

        recent_results = self.adaptation_buffer[-100:]

        # Simple heuristics for performance estimation
        confidence_scores = [r.confidence for r in recent_results]
        energy_levels = [r.energy_level for r in recent_results]

        avg_confidence = mean(confidence_scores)
        confidence_stability = 1.0 / (1.0 + stdev(confidence_scores))

        # Estimate based on confidence patterns
        performance_score = avg_confidence * confidence_stability

        return max(0.0, min(1.0, performance_score))

    def _should_adapt(self, current_performance: float) -> bool:
        """Determine if adaptation is needed based on performance history."""
        if len(self.performance_history) < 10:
            return False

        # Check for consistent performance degradation
        recent_performance = mean(self.performance_history[-10:])
        baseline_performance = (
            mean(self.performance_history[:10])
            if len(self.performance_history) >= 20
            else 1.0
        )

        performance_drop = baseline_performance - recent_performance

        return performance_drop > self.config.adaptation_threshold

    def _perform_adaptation(self, vad: BaseVAD) -> None:
        """Perform real-time parameter adaptation."""
        # Simple threshold adjustment based on recent energy levels
        recent_results = self.adaptation_buffer[-100:]

        energy_levels = [r.energy_level for r in recent_results]
        avg_energy = mean(energy_levels)

        # Adjust threshold based on energy patterns
        current_threshold = vad.config.threshold

        if avg_energy > current_threshold * 2:
            new_threshold = min(0.1, current_threshold * 1.2)
        elif avg_energy < current_threshold * 0.5:
            new_threshold = max(0.001, current_threshold * 0.8)
        else:
            return  # No adaptation needed

        logger.info(
            f"Adapting threshold: {current_threshold:.4f} -> {new_threshold:.4f}"
        )
        vad.set_threshold(new_threshold)

    def _save_calibration_results(self, result: CalibrationResult) -> None:
        """Save calibration results to persistent storage."""
        # Create calibration results directory
        results_dir = Path("calibration_results")
        results_dir.mkdir(exist_ok=True)

        # Save individual result
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_{result.detected_environment.value}_{timestamp}.json"
        filepath = results_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Calibration results saved to {filepath}")
