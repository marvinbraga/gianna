"""
Voice Activity Detection (VAD) Metrics and Performance Monitoring System.

This module provides comprehensive metrics collection, analysis, and monitoring
capabilities for VAD systems, including real-time performance tracking,
statistical analysis, and visualization support.
"""

import json
import logging
import threading
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting functionality disabled.")

from .base import BaseVAD
from .calibration import EnvironmentType, GroundTruthData
from .types import VADEventType, VADResult, VADState

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics for VAD evaluation."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SPECIFICITY = "specificity"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    FALSE_NEGATIVE_RATE = "false_negative_rate"
    AUC_SCORE = "auc_score"
    AVERAGE_PRECISION = "average_precision"
    PROCESSING_TIME = "processing_time"
    ENERGY_LEVEL = "energy_level"
    CONFIDENCE = "confidence"
    SNR = "signal_to_noise_ratio"


class AlertLevel(Enum):
    """Alert levels for performance monitoring."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance monitoring alert."""

    level: AlertLevel
    metric: MetricType
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricThreshold:
    """Threshold configuration for performance monitoring."""

    metric: MetricType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None
    enabled: bool = True


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""

    # Basic metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float

    # Advanced metrics
    auc_score: float
    average_precision: float
    false_positive_rate: float
    false_negative_rate: float

    # Performance metrics
    avg_processing_time: float
    max_processing_time: float
    min_processing_time: float

    # Signal quality metrics
    avg_energy_level: float
    avg_confidence: float
    avg_snr: float

    # Statistical measures
    confidence_interval_95: Tuple[float, float]
    statistical_significance: float

    # Confusion matrix
    confusion_matrix: np.ndarray

    # Environment information
    environment: EnvironmentType
    evaluation_period: Tuple[float, float]

    # Sample information
    total_samples: int
    positive_samples: int
    negative_samples: int

    # Quality indicators
    data_quality_score: float
    reliability_score: float

    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "specificity": self.specificity,
            "auc_score": self.auc_score,
            "average_precision": self.average_precision,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "avg_processing_time": self.avg_processing_time,
            "max_processing_time": self.max_processing_time,
            "min_processing_time": self.min_processing_time,
            "avg_energy_level": self.avg_energy_level,
            "avg_confidence": self.avg_confidence,
            "avg_snr": self.avg_snr,
            "confidence_interval_95": self.confidence_interval_95,
            "statistical_significance": self.statistical_significance,
            "confusion_matrix": (
                self.confusion_matrix.tolist()
                if hasattr(self.confusion_matrix, "tolist")
                else []
            ),
            "environment": self.environment.value,
            "evaluation_period": self.evaluation_period,
            "total_samples": self.total_samples,
            "positive_samples": self.positive_samples,
            "negative_samples": self.negative_samples,
            "data_quality_score": self.data_quality_score,
            "reliability_score": self.reliability_score,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class PerformanceMonitor:
    """Real-time performance monitoring system."""

    def __init__(self, window_size: int = 1000, alert_cooldown: float = 60.0):
        """
        Initialize performance monitor.

        Args:
            window_size: Size of sliding window for metrics.
            alert_cooldown: Minimum time between duplicate alerts (seconds).
        """
        self.window_size = window_size
        self.alert_cooldown = alert_cooldown

        # Monitoring state
        self.metrics_buffer = deque(maxlen=window_size)
        self.alerts: List[PerformanceAlert] = []
        self.last_alert_time: Dict[str, float] = {}

        # Configuration
        self.thresholds: Dict[MetricType, MetricThreshold] = {}
        self.monitoring_enabled = True

        # Thread safety
        self.lock = threading.RLock()

        # Initialize default thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self) -> None:
        """Setup default performance thresholds."""
        self.thresholds = {
            MetricType.ACCURACY: MetricThreshold(
                metric=MetricType.ACCURACY, warning_threshold=0.8, error_threshold=0.7
            ),
            MetricType.F1_SCORE: MetricThreshold(
                metric=MetricType.F1_SCORE, warning_threshold=0.75, error_threshold=0.6
            ),
            MetricType.PROCESSING_TIME: MetricThreshold(
                metric=MetricType.PROCESSING_TIME,
                warning_threshold=0.05,  # 50ms
                error_threshold=0.1,  # 100ms
            ),
            MetricType.CONFIDENCE: MetricThreshold(
                metric=MetricType.CONFIDENCE, warning_threshold=0.6, error_threshold=0.4
            ),
        }

    def update_metrics(
        self, result: VADResult, ground_truth: Optional[bool] = None
    ) -> None:
        """
        Update monitoring metrics with new VAD result.

        Args:
            result: VAD processing result.
            ground_truth: True label for accuracy calculation.
        """
        if not self.monitoring_enabled:
            return

        with self.lock:
            # Store metrics data point
            metrics_point = {
                "timestamp": result.timestamp,
                "is_voice_active": result.is_voice_active,
                "confidence": result.confidence,
                "energy_level": result.energy_level,
                "processing_time": result.processing_time,
                "snr": result.signal_to_noise_ratio or 0.0,
                "ground_truth": ground_truth,
            }

            self.metrics_buffer.append(metrics_point)

            # Check for alerts if we have sufficient data
            if len(self.metrics_buffer) >= 100:
                self._check_alerts()

    def _check_alerts(self) -> None:
        """Check current metrics against thresholds and generate alerts."""
        current_metrics = self._calculate_current_metrics()

        for metric_type, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue

            metric_value = current_metrics.get(metric_type.value)
            if metric_value is None:
                continue

            alert = self._evaluate_threshold(metric_type, metric_value, threshold)
            if alert:
                self._emit_alert(alert)

    def _calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics from buffer."""
        if not self.metrics_buffer:
            return {}

        # Extract data
        recent_data = list(self.metrics_buffer)[-100:]  # Last 100 samples

        confidences = [d["confidence"] for d in recent_data]
        processing_times = [d["processing_time"] for d in recent_data]
        energy_levels = [d["energy_level"] for d in recent_data]

        # Calculate metrics with ground truth if available
        predictions = [d["is_voice_active"] for d in recent_data]
        ground_truths = [
            d["ground_truth"] for d in recent_data if d["ground_truth"] is not None
        ]

        metrics = {
            "confidence": mean(confidences) if confidences else 0.0,
            "processing_time": mean(processing_times) if processing_times else 0.0,
            "energy_level": mean(energy_levels) if energy_levels else 0.0,
        }

        # Add accuracy-based metrics if ground truth is available
        if len(ground_truths) >= len(predictions) and len(ground_truths) > 0:
            metrics["accuracy"] = accuracy_score(ground_truths, predictions)
            if len(set(ground_truths)) > 1:  # Both classes present
                metrics["f1_score"] = f1_score(ground_truths, predictions)
                metrics["precision"] = precision_score(
                    ground_truths, predictions, zero_division=0
                )
                metrics["recall"] = recall_score(
                    ground_truths, predictions, zero_division=0
                )

        return metrics

    def _evaluate_threshold(
        self, metric_type: MetricType, value: float, threshold: MetricThreshold
    ) -> Optional[PerformanceAlert]:
        """Evaluate metric value against threshold."""

        if threshold.error_threshold is not None:
            if (
                threshold.min_value is not None and value < threshold.error_threshold
            ) or (
                threshold.max_value is not None and value > threshold.error_threshold
            ):
                return PerformanceAlert(
                    level=AlertLevel.ERROR,
                    metric=metric_type,
                    message=f"{metric_type.value} ({value:.4f}) exceeded error threshold ({threshold.error_threshold:.4f})",
                    value=value,
                    threshold=threshold.error_threshold,
                )

        if threshold.warning_threshold is not None:
            if (
                threshold.min_value is not None and value < threshold.warning_threshold
            ) or (
                threshold.max_value is not None and value > threshold.warning_threshold
            ):
                return PerformanceAlert(
                    level=AlertLevel.WARNING,
                    metric=metric_type,
                    message=f"{metric_type.value} ({value:.4f}) exceeded warning threshold ({threshold.warning_threshold:.4f})",
                    value=value,
                    threshold=threshold.warning_threshold,
                )

        return None

    def _emit_alert(self, alert: PerformanceAlert) -> None:
        """Emit performance alert with cooldown."""
        alert_key = f"{alert.metric.value}_{alert.level.value}"
        current_time = time.time()

        # Check cooldown
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.alert_cooldown:
                return

        self.alerts.append(alert)
        self.last_alert_time[alert_key] = current_time

        # Log alert
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }.get(alert.level, logger.info)

        log_func(f"Performance Alert: {alert.message}")

    def get_recent_alerts(self, hours: float = 24.0) -> List[PerformanceAlert]:
        """Get recent alerts within specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        with self.lock:
            self.alerts.clear()
            self.last_alert_time.clear()


class VadMetrics:
    """
    Comprehensive VAD metrics and performance evaluation system.

    Provides detailed performance analysis, statistical evaluation,
    and visualization capabilities for VAD algorithms.
    """

    def __init__(self, enable_monitoring: bool = True, window_size: int = 10000):
        """
        Initialize VAD metrics system.

        Args:
            enable_monitoring: Enable real-time performance monitoring.
            window_size: Size of metrics collection window.
        """
        self.enable_monitoring = enable_monitoring
        self.window_size = window_size

        # Metrics storage
        self.results_history: List[VADResult] = []
        self.ground_truth_history: List[bool] = []
        self.environment_history: List[EnvironmentType] = []

        # Performance monitoring
        self.monitor = PerformanceMonitor(window_size) if enable_monitoring else None

        # Statistical tracking
        self.performance_reports: List[PerformanceReport] = []
        self.benchmark_results: Dict[str, Dict[str, float]] = {}

        # Thread safety
        self.lock = threading.RLock()

        logger.info("VadMetrics initialized")

    def record_result(
        self,
        result: VADResult,
        ground_truth: Optional[bool] = None,
        environment: Optional[EnvironmentType] = None,
    ) -> None:
        """
        Record a VAD result for metrics tracking.

        Args:
            result: VAD processing result.
            ground_truth: True label for the result.
            environment: Current audio environment.
        """
        with self.lock:
            self.results_history.append(result)

            if ground_truth is not None:
                self.ground_truth_history.append(ground_truth)

            if environment is not None:
                self.environment_history.append(environment)

            # Maintain window size
            if len(self.results_history) > self.window_size:
                self.results_history = self.results_history[-self.window_size :]
            if len(self.ground_truth_history) > self.window_size:
                self.ground_truth_history = self.ground_truth_history[
                    -self.window_size :
                ]
            if len(self.environment_history) > self.window_size:
                self.environment_history = self.environment_history[-self.window_size :]

            # Update monitoring
            if self.monitor:
                self.monitor.update_metrics(result, ground_truth)

    def evaluate_performance(
        self,
        ground_truth_data: Optional[GroundTruthData] = None,
        environment: Optional[EnvironmentType] = None,
    ) -> PerformanceReport:
        """
        Evaluate comprehensive VAD performance.

        Args:
            ground_truth_data: Ground truth data for evaluation.
            environment: Current environment for context.

        Returns:
            Comprehensive performance report.
        """
        logger.info("Evaluating VAD performance")

        with self.lock:
            # Use provided data or stored history
            if ground_truth_data is not None:
                y_true = ground_truth_data.labels
                y_pred = [
                    r.is_voice_active for r in self.results_history[-len(y_true) :]
                ]
                y_proba = [r.confidence for r in self.results_history[-len(y_true) :]]
                env = ground_truth_data.environment
            else:
                if not self.ground_truth_history:
                    raise ValueError("No ground truth data available for evaluation")

                y_true = self.ground_truth_history
                y_pred = [
                    r.is_voice_active for r in self.results_history[-len(y_true) :]
                ]
                y_proba = [r.confidence for r in self.results_history[-len(y_true) :]]
                env = environment or EnvironmentType.UNKNOWN

            # Ensure equal lengths
            min_length = min(len(y_true), len(y_pred), len(y_proba))
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]
            y_proba = y_proba[:min_length]

            if min_length == 0:
                raise ValueError("No data available for evaluation")

            # Calculate metrics
            report = self._calculate_comprehensive_metrics(y_true, y_pred, y_proba, env)

            # Store report
            self.performance_reports.append(report)

            logger.info(
                f"Performance evaluation completed: F1={report.f1_score:.4f}, "
                f"Accuracy={report.accuracy:.4f}"
            )

            return report

    def _calculate_comprehensive_metrics(
        self,
        y_true: List[bool],
        y_pred: List[bool],
        y_proba: List[float],
        environment: EnvironmentType,
    ) -> PerformanceReport:
        """Calculate comprehensive performance metrics."""

        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        y_proba_array = np.array(y_proba)

        # Basic classification metrics
        accuracy = accuracy_score(y_true_array, y_pred_array)

        # Handle case where only one class is present
        if len(set(y_true)) == 1:
            precision = 1.0 if y_true[0] == y_pred[0] else 0.0
            recall = 1.0 if y_true[0] == y_pred[0] else 0.0
            f1 = 1.0 if y_true[0] == y_pred[0] else 0.0
            specificity = 1.0
            auc = 0.5
            avg_precision = 1.0 if y_true[0] else 0.0
        else:
            precision = precision_score(y_true_array, y_pred_array, zero_division=0)
            recall = recall_score(y_true_array, y_pred_array, zero_division=0)
            f1 = f1_score(y_true_array, y_pred_array, zero_division=0)

            # Confusion matrix for specificity
            cm = confusion_matrix(y_true_array, y_pred_array)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                specificity = 0.0

            # AUC and average precision
            try:
                auc = roc_auc_score(y_true_array, y_proba_array)
                avg_precision = average_precision_score(y_true_array, y_proba_array)
            except ValueError:
                auc = 0.5
                avg_precision = sum(y_true) / len(y_true) if y_true else 0.0

        # False positive/negative rates
        cm = confusion_matrix(y_true_array, y_pred_array)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        else:
            fpr = 0.0
            fnr = 0.0

        # Performance metrics from recent results
        recent_results = self.results_history[-len(y_true) :]
        processing_times = [
            r.processing_time for r in recent_results if r.processing_time > 0
        ]
        energy_levels = [r.energy_level for r in recent_results]
        confidences = [r.confidence for r in recent_results]
        snrs = [
            r.signal_to_noise_ratio
            for r in recent_results
            if r.signal_to_noise_ratio is not None
        ]

        avg_processing_time = mean(processing_times) if processing_times else 0.0
        max_processing_time = max(processing_times) if processing_times else 0.0
        min_processing_time = min(processing_times) if processing_times else 0.0

        avg_energy = mean(energy_levels) if energy_levels else 0.0
        avg_confidence = mean(confidences) if confidences else 0.0
        avg_snr = mean(snrs) if snrs else 0.0

        # Statistical measures
        confidence_interval = self._calculate_confidence_interval(y_true, y_pred)
        significance = self._calculate_statistical_significance(y_true, y_pred)

        # Quality scores
        data_quality = self._assess_data_quality(y_true, y_proba)
        reliability = self._assess_reliability(recent_results)

        return PerformanceReport(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            auc_score=auc,
            average_precision=avg_precision,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            avg_processing_time=avg_processing_time,
            max_processing_time=max_processing_time,
            min_processing_time=min_processing_time,
            avg_energy_level=avg_energy,
            avg_confidence=avg_confidence,
            avg_snr=avg_snr,
            confidence_interval_95=confidence_interval,
            statistical_significance=significance,
            confusion_matrix=cm,
            environment=environment,
            evaluation_period=(
                (recent_results[0].timestamp, recent_results[-1].timestamp)
                if recent_results
                else (0, 0)
            ),
            total_samples=len(y_true),
            positive_samples=sum(y_true),
            negative_samples=len(y_true) - sum(y_true),
            data_quality_score=data_quality,
            reliability_score=reliability,
        )

    def _calculate_confidence_interval(
        self, y_true: List[bool], y_pred: List[bool], confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for accuracy."""
        accuracy = accuracy_score(y_true, y_pred)
        n = len(y_true)

        if n == 0:
            return (0.0, 0.0)

        # Wilson score interval
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        p_hat = accuracy

        denominator = 1 + z**2 / n
        centre_adjusted_probability = p_hat + z**2 / (2 * n)
        adjusted_standard_deviation = np.sqrt(
            (p_hat * (1 - p_hat) + z**2 / (4 * n)) / n
        )

        lower_bound = (
            centre_adjusted_probability - z * adjusted_standard_deviation
        ) / denominator
        upper_bound = (
            centre_adjusted_probability + z * adjusted_standard_deviation
        ) / denominator

        return (max(0.0, lower_bound), min(1.0, upper_bound))

    def _calculate_statistical_significance(
        self, y_true: List[bool], y_pred: List[bool]
    ) -> float:
        """Calculate statistical significance of results."""
        if len(y_true) < 30:  # Insufficient data for significance testing
            return 0.5

        # McNemar's test for paired binary data
        try:
            # Create contingency table
            correct_pred = [t == p for t, p in zip(y_true, y_pred)]
            baseline_pred = [True] * len(y_true)  # Naive baseline
            correct_baseline = [t == b for t, b in zip(y_true, baseline_pred)]

            # Count disagreements
            b01 = sum(1 for c, b in zip(correct_pred, correct_baseline) if not c and b)
            b10 = sum(1 for c, b in zip(correct_pred, correct_baseline) if c and not b)

            if b01 + b10 == 0:
                return 0.05  # Perfect agreement, highly significant

            # McNemar's statistic
            chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
            p_value = 1 - stats.chi2.cdf(chi2, 1)

            return p_value

        except Exception:
            return 0.5  # Default to moderate significance

    def _assess_data_quality(self, y_true: List[bool], y_proba: List[float]) -> float:
        """Assess quality of the data used for evaluation."""
        if not y_true or not y_proba:
            return 0.0

        # Balance score (closer to 0.5 is better for binary classification)
        positive_ratio = sum(y_true) / len(y_true)
        balance_score = 1.0 - abs(positive_ratio - 0.5) * 2

        # Confidence distribution score (prefer varied confidences)
        if len(set(y_proba)) > 1:
            confidence_variance = np.var(y_proba)
            confidence_score = min(1.0, confidence_variance * 4)  # Scale to 0-1
        else:
            confidence_score = 0.0

        # Size score (more data is better)
        size_score = min(1.0, len(y_true) / 1000.0)

        # Weighted average
        quality_score = balance_score * 0.4 + confidence_score * 0.3 + size_score * 0.3

        return quality_score

    def _assess_reliability(self, results: List[VADResult]) -> float:
        """Assess reliability of VAD results."""
        if not results:
            return 0.0

        # Consistency in confidence scores
        confidences = [r.confidence for r in results]
        if len(set(confidences)) > 1:
            confidence_stability = 1.0 / (1.0 + np.std(confidences))
        else:
            confidence_stability = 1.0

        # Processing time stability
        times = [r.processing_time for r in results if r.processing_time > 0]
        if len(times) > 1:
            time_stability = 1.0 / (1.0 + np.std(times) / np.mean(times))
        else:
            time_stability = 1.0

        # State transition smoothness (fewer abrupt changes is better)
        state_changes = 0
        for i in range(1, len(results)):
            if results[i].is_voice_active != results[i - 1].is_voice_active:
                state_changes += 1

        if len(results) > 1:
            transition_smoothness = 1.0 - min(1.0, state_changes / (len(results) * 0.2))
        else:
            transition_smoothness = 1.0

        reliability_score = (
            confidence_stability * 0.4
            + time_stability * 0.3
            + transition_smoothness * 0.3
        )

        return reliability_score

    def compare_algorithms(
        self, vad_instances: Dict[str, BaseVAD], ground_truth_data: GroundTruthData
    ) -> Dict[str, PerformanceReport]:
        """
        Compare performance of multiple VAD algorithms.

        Args:
            vad_instances: Dictionary of VAD instances to compare.
            ground_truth_data: Ground truth data for evaluation.

        Returns:
            Dictionary of performance reports for each algorithm.
        """
        logger.info(f"Comparing {len(vad_instances)} VAD algorithms")

        comparison_results = {}

        for name, vad_instance in vad_instances.items():
            logger.info(f"Evaluating algorithm: {name}")

            # Reset metrics for this algorithm
            original_history = self.results_history.copy()
            self.results_history.clear()

            try:
                # Process all audio chunks
                for chunk in ground_truth_data.audio_chunks:
                    result = vad_instance.detect_activity(chunk)
                    self.results_history.append(result)

                # Evaluate performance
                report = self.evaluate_performance(
                    ground_truth_data, ground_truth_data.environment
                )
                comparison_results[name] = report

                # Store benchmark results
                self.benchmark_results[name] = {
                    "accuracy": report.accuracy,
                    "precision": report.precision,
                    "recall": report.recall,
                    "f1_score": report.f1_score,
                    "auc_score": report.auc_score,
                    "processing_time": report.avg_processing_time,
                }

            except Exception as e:
                logger.error(f"Error evaluating algorithm {name}: {e}")
                continue
            finally:
                # Restore original history
                self.results_history = original_history

        logger.info("Algorithm comparison completed")
        return comparison_results

    def generate_roc_curve(
        self, ground_truth_data: Optional[GroundTruthData] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate ROC curve data.

        Args:
            ground_truth_data: Ground truth data for ROC calculation.

        Returns:
            Tuple of (fpr, tpr, auc_score) or None if plotting not available.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot generate ROC curve.")
            return None

        # Get data
        if ground_truth_data is not None:
            y_true = ground_truth_data.labels
            y_proba = [r.confidence for r in self.results_history[-len(y_true) :]]
        else:
            if not self.ground_truth_history:
                logger.error("No ground truth data available for ROC curve")
                return None
            y_true = self.ground_truth_history
            y_proba = [r.confidence for r in self.results_history[-len(y_true) :]]

        # Ensure equal lengths
        min_length = min(len(y_true), len(y_proba))
        y_true = y_true[:min_length]
        y_proba = y_proba[:min_length]

        if len(set(y_true)) < 2:
            logger.warning("Need both positive and negative samples for ROC curve")
            return None

        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = roc_auc_score(y_true, y_proba)
            return fpr, tpr, auc_score
        except Exception as e:
            logger.error(f"Error generating ROC curve: {e}")
            return None

    def generate_precision_recall_curve(
        self, ground_truth_data: Optional[GroundTruthData] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Generate Precision-Recall curve data.

        Args:
            ground_truth_data: Ground truth data for PR calculation.

        Returns:
            Tuple of (precision, recall, avg_precision) or None if not available.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot generate PR curve.")
            return None

        # Get data
        if ground_truth_data is not None:
            y_true = ground_truth_data.labels
            y_proba = [r.confidence for r in self.results_history[-len(y_true) :]]
        else:
            if not self.ground_truth_history:
                logger.error("No ground truth data available for PR curve")
                return None
            y_true = self.ground_truth_history
            y_proba = [r.confidence for r in self.results_history[-len(y_true) :]]

        # Ensure equal lengths
        min_length = min(len(y_true), len(y_proba))
        y_true = y_true[:min_length]
        y_proba = y_proba[:min_length]

        if sum(y_true) == 0:
            logger.warning("Need positive samples for PR curve")
            return None

        try:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            avg_precision = average_precision_score(y_true, y_proba)
            return precision, recall, avg_precision
        except Exception as e:
            logger.error(f"Error generating PR curve: {e}")
            return None

    def plot_performance_metrics(
        self, save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot comprehensive performance metrics.

        Args:
            save_path: Optional path to save the plot.
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot plot metrics.")
            return

        if not self.results_history:
            logger.warning("No data available for plotting")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("VAD Performance Metrics", fontsize=16)

        # Extract data for plotting
        timestamps = [
            r.timestamp for r in self.results_history[-1000:]
        ]  # Last 1000 results
        confidences = [r.confidence for r in self.results_history[-1000:]]
        energy_levels = [r.energy_level for r in self.results_history[-1000:]]
        processing_times = [
            r.processing_time * 1000 for r in self.results_history[-1000:]
        ]  # Convert to ms

        # Confidence over time
        axes[0, 0].plot(timestamps, confidences, alpha=0.7)
        axes[0, 0].set_title("Confidence Over Time")
        axes[0, 0].set_xlabel("Timestamp")
        axes[0, 0].set_ylabel("Confidence")
        axes[0, 0].grid(True, alpha=0.3)

        # Energy levels over time
        axes[0, 1].plot(timestamps, energy_levels, alpha=0.7, color="orange")
        axes[0, 1].set_title("Energy Levels Over Time")
        axes[0, 1].set_xlabel("Timestamp")
        axes[0, 1].set_ylabel("Energy Level")
        axes[0, 1].grid(True, alpha=0.3)

        # Processing time distribution
        axes[0, 2].hist(processing_times, bins=30, alpha=0.7, color="green")
        axes[0, 2].set_title("Processing Time Distribution")
        axes[0, 2].set_xlabel("Processing Time (ms)")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].grid(True, alpha=0.3)

        # ROC Curve
        roc_data = self.generate_roc_curve()
        if roc_data:
            fpr, tpr, auc_score = roc_data
            axes[1, 0].plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
            axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[1, 0].set_title("ROC Curve")
            axes[1, 0].set_xlabel("False Positive Rate")
            axes[1, 0].set_ylabel("True Positive Rate")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "ROC Curve\nNot Available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("ROC Curve")

        # Precision-Recall Curve
        pr_data = self.generate_precision_recall_curve()
        if pr_data:
            precision, recall, avg_precision = pr_data
            axes[1, 1].plot(
                recall, precision, label=f"PR Curve (AP = {avg_precision:.3f})"
            )
            axes[1, 1].set_title("Precision-Recall Curve")
            axes[1, 1].set_xlabel("Recall")
            axes[1, 1].set_ylabel("Precision")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "PR Curve\nNot Available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Precision-Recall Curve")

        # Performance summary
        if self.performance_reports:
            latest_report = self.performance_reports[-1]
            metrics_text = f"""
            Accuracy: {latest_report.accuracy:.3f}
            Precision: {latest_report.precision:.3f}
            Recall: {latest_report.recall:.3f}
            F1-Score: {latest_report.f1_score:.3f}
            AUC: {latest_report.auc_score:.3f}
            Avg Processing Time: {latest_report.avg_processing_time*1000:.2f}ms
            """
            axes[1, 2].text(
                0.1,
                0.9,
                metrics_text,
                transform=axes[1, 2].transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
            )
            axes[1, 2].set_title("Performance Summary")
            axes[1, 2].axis("off")
        else:
            axes[1, 2].text(
                0.5,
                0.5,
                "Performance Summary\nNot Available",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
            )
            axes[1, 2].set_title("Performance Summary")
            axes[1, 2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Performance plot saved to {save_path}")
        else:
            plt.show()

    def export_metrics_report(self, filepath: Union[str, Path]) -> None:
        """
        Export comprehensive metrics report to JSON.

        Args:
            filepath: Path to save the metrics report.
        """
        filepath = Path(filepath)

        # Compile comprehensive report
        report_data = {
            "summary": {
                "total_results": len(self.results_history),
                "total_ground_truth": len(self.ground_truth_history),
                "evaluation_period": {
                    "start": (
                        min(r.timestamp for r in self.results_history)
                        if self.results_history
                        else 0
                    ),
                    "end": (
                        max(r.timestamp for r in self.results_history)
                        if self.results_history
                        else 0
                    ),
                },
            },
            "performance_reports": [
                report.to_dict() for report in self.performance_reports
            ],
            "benchmark_results": self.benchmark_results,
            "recent_alerts": [
                {
                    "level": alert.level.value,
                    "metric": alert.metric.value,
                    "message": alert.message,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp,
                }
                for alert in (self.monitor.get_recent_alerts() if self.monitor else [])
            ],
            "export_timestamp": time.time(),
        }

        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"Metrics report exported to {filepath}")

    def reset_metrics(self) -> None:
        """Reset all collected metrics and history."""
        with self.lock:
            self.results_history.clear()
            self.ground_truth_history.clear()
            self.environment_history.clear()
            self.performance_reports.clear()
            self.benchmark_results.clear()

            if self.monitor:
                self.monitor.clear_alerts()

        logger.info("Metrics reset completed")
