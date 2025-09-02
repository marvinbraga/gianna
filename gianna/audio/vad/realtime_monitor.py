"""
Real-time VAD Performance Monitor and Adaptive Calibration System.

This module provides continuous monitoring and automatic adaptation capabilities
for VAD systems in production environments, with real-time performance tracking,
alert generation, and automatic parameter adjustment.
"""

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseVAD
from .calibration import CalibrationConfig, EnvironmentType, VadCalibrator
from .metrics import AlertLevel, PerformanceAlert, PerformanceMonitor, VadMetrics
from .types import VADConfig, VADResult

logger = logging.getLogger(__name__)


class MonitoringState(Enum):
    """States of the real-time monitoring system."""

    STOPPED = "stopped"
    STARTING = "starting"
    MONITORING = "monitoring"
    CALIBRATING = "calibrating"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class AdaptationTrigger(Enum):
    """Triggers for automatic adaptation."""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    ENVIRONMENT_CHANGE = "environment_change"
    MANUAL_REQUEST = "manual_request"
    SCHEDULED = "scheduled"
    ALERT_THRESHOLD = "alert_threshold"


@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring system."""

    # Monitoring intervals
    monitoring_interval: float = 1.0  # seconds
    calibration_interval: float = 300.0  # 5 minutes
    alert_check_interval: float = 10.0  # seconds

    # Performance thresholds
    min_accuracy_threshold: float = 0.7
    max_processing_time: float = 0.05  # 50ms
    performance_window_size: int = 1000

    # Adaptation settings
    enable_auto_adaptation: bool = True
    adaptation_cooldown: float = 60.0  # 1 minute
    min_samples_for_adaptation: int = 500

    # Environment detection
    enable_environment_detection: bool = True
    environment_change_threshold: float = 0.3  # 30% change in metrics

    # Logging and storage
    save_monitoring_data: bool = True
    log_level: str = "INFO"
    data_retention_hours: float = 24.0

    # Alert configuration
    enable_email_alerts: bool = False
    email_recipients: List[str] = field(default_factory=list)
    alert_cooldown: float = 300.0  # 5 minutes


@dataclass
class MonitoringSnapshot:
    """Snapshot of system performance at a point in time."""

    timestamp: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    avg_processing_time: float = 0.0
    avg_confidence: float = 0.0
    avg_energy_level: float = 0.0

    detected_environment: Optional[EnvironmentType] = None
    noise_level: float = 0.0

    total_samples: int = 0
    speech_ratio: float = 0.0

    alerts_count: int = 0
    state: MonitoringState = MonitoringState.MONITORING

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "timestamp": self.timestamp,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "avg_processing_time": self.avg_processing_time,
            "avg_confidence": self.avg_confidence,
            "avg_energy_level": self.avg_energy_level,
            "detected_environment": (
                self.detected_environment.value if self.detected_environment else None
            ),
            "noise_level": self.noise_level,
            "total_samples": self.total_samples,
            "speech_ratio": self.speech_ratio,
            "alerts_count": self.alerts_count,
            "state": self.state.value,
        }


class RealtimeMonitor:
    """
    Real-time VAD monitoring and adaptive calibration system.

    Provides continuous monitoring, performance tracking, automatic adaptation,
    and alert generation for VAD systems in production environments.
    """

    def __init__(
        self, vad_instance: BaseVAD, config: Optional[MonitoringConfig] = None
    ):
        """
        Initialize real-time monitor.

        Args:
            vad_instance: VAD instance to monitor.
            config: Monitoring configuration.
        """
        self.vad = vad_instance
        self.config = config or MonitoringConfig()

        # Components
        self.metrics = VadMetrics(enable_monitoring=True)
        self.calibrator = VadCalibrator()
        self.performance_monitor = PerformanceMonitor()

        # Monitoring state
        self.state = MonitoringState.STOPPED
        self.monitoring_thread: Optional[threading.Thread] = None
        self.calibration_thread: Optional[threading.Thread] = None
        self.alert_thread: Optional[threading.Thread] = None

        # Data storage
        self.result_queue = queue.Queue(maxsize=10000)
        self.ground_truth_queue = queue.Queue(maxsize=10000)
        self.snapshots: deque = deque(maxlen=1000)  # Keep last 1000 snapshots

        # Adaptation tracking
        self.last_adaptation_time = 0.0
        self.adaptation_history: List[Dict[str, Any]] = []
        self.current_environment: Optional[EnvironmentType] = None

        # Thread synchronization
        self.stop_event = threading.Event()
        self.lock = threading.RLock()

        # Callbacks
        self.adaptation_callbacks: List[
            Callable[[AdaptationTrigger, Dict[str, Any]], None]
        ] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []

        logger.info("RealtimeMonitor initialized")

    def start_monitoring(self) -> bool:
        """
        Start real-time monitoring.

        Returns:
            True if monitoring started successfully.
        """
        with self.lock:
            if self.state != MonitoringState.STOPPED:
                logger.warning("Monitor already running")
                return False

            logger.info("Starting real-time VAD monitoring")
            self.state = MonitoringState.STARTING
            self.stop_event.clear()

            try:
                # Start monitoring thread
                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop, name="VAD-Monitor", daemon=True
                )
                self.monitoring_thread.start()

                # Start calibration thread
                self.calibration_thread = threading.Thread(
                    target=self._calibration_loop, name="VAD-Calibration", daemon=True
                )
                self.calibration_thread.start()

                # Start alert thread
                self.alert_thread = threading.Thread(
                    target=self._alert_loop, name="VAD-Alerts", daemon=True
                )
                self.alert_thread.start()

                self.state = MonitoringState.MONITORING
                logger.info("Real-time monitoring started successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to start monitoring: {e}")
                self.state = MonitoringState.ERROR
                return False

    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        with self.lock:
            if self.state == MonitoringState.STOPPED:
                return

            logger.info("Stopping real-time monitoring")
            self.state = MonitoringState.SHUTTING_DOWN

            # Signal threads to stop
            self.stop_event.set()

            # Wait for threads to finish
            threads = [
                self.monitoring_thread,
                self.calibration_thread,
                self.alert_thread,
            ]

            for thread in threads:
                if thread and thread.is_alive():
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.warning(f"Thread {thread.name} did not stop gracefully")

            self.state = MonitoringState.STOPPED
            logger.info("Real-time monitoring stopped")

    def record_result(
        self, result: VADResult, ground_truth: Optional[bool] = None
    ) -> None:
        """
        Record a VAD result for monitoring.

        Args:
            result: VAD processing result.
            ground_truth: True label if available.
        """
        try:
            if not self.result_queue.full():
                self.result_queue.put((result, time.time()), block=False)

            if ground_truth is not None and not self.ground_truth_queue.full():
                self.ground_truth_queue.put((ground_truth, time.time()), block=False)

            # Update metrics
            self.metrics.record_result(result, ground_truth, self.current_environment)

        except queue.Full:
            logger.warning("Result queue is full, dropping sample")
        except Exception as e:
            logger.error(f"Error recording result: {e}")

    def trigger_calibration(
        self, trigger: AdaptationTrigger = AdaptationTrigger.MANUAL_REQUEST
    ) -> bool:
        """
        Trigger immediate calibration.

        Args:
            trigger: Reason for triggering calibration.

        Returns:
            True if calibration was triggered successfully.
        """
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_adaptation_time < self.config.adaptation_cooldown:
            logger.info("Calibration cooldown active, skipping")
            return False

        logger.info(f"Triggering calibration: {trigger.value}")

        # Queue calibration request
        try:
            if hasattr(self, "_calibration_queue"):
                self._calibration_queue.put(trigger, block=False)
            else:
                # Direct calibration if queue not available
                self._perform_calibration(trigger)
            return True
        except Exception as e:
            logger.error(f"Failed to trigger calibration: {e}")
            return False

    def get_current_snapshot(self) -> MonitoringSnapshot:
        """
        Get current system performance snapshot.

        Returns:
            Current performance snapshot.
        """
        with self.lock:
            # Collect recent results
            recent_results = []
            recent_ground_truth = []

            # Drain queues to get recent data
            temp_results = []
            temp_ground_truth = []

            try:
                while not self.result_queue.empty():
                    temp_results.append(self.result_queue.get_nowait())
            except queue.Empty:
                pass

            try:
                while not self.ground_truth_queue.empty():
                    temp_ground_truth.append(self.ground_truth_queue.get_nowait())
            except queue.Empty:
                pass

            # Get last N results for snapshot
            n_samples = min(100, len(temp_results))
            if n_samples > 0:
                recent_data = temp_results[-n_samples:]
                recent_results = [item[0] for item in recent_data]

                # Match ground truth data
                if temp_ground_truth:
                    recent_gt_data = temp_ground_truth[-n_samples:]
                    recent_ground_truth = [item[0] for item in recent_gt_data]

            # Calculate metrics
            snapshot = MonitoringSnapshot(timestamp=time.time(), state=self.state)

            if recent_results:
                snapshot.avg_processing_time = mean(
                    [r.processing_time for r in recent_results]
                )
                snapshot.avg_confidence = mean([r.confidence for r in recent_results])
                snapshot.avg_energy_level = mean(
                    [r.energy_level for r in recent_results]
                )

                predictions = [r.is_voice_active for r in recent_results]
                snapshot.speech_ratio = sum(predictions) / len(predictions)
                snapshot.total_samples = len(recent_results)

                # Calculate accuracy if ground truth available
                if recent_ground_truth and len(recent_ground_truth) == len(predictions):
                    from sklearn.metrics import (
                        accuracy_score,
                        f1_score,
                        precision_score,
                        recall_score,
                    )

                    snapshot.accuracy = accuracy_score(recent_ground_truth, predictions)

                    if len(set(recent_ground_truth)) > 1:
                        snapshot.precision = precision_score(
                            recent_ground_truth, predictions, zero_division=0
                        )
                        snapshot.recall = recall_score(
                            recent_ground_truth, predictions, zero_division=0
                        )
                        snapshot.f1_score = f1_score(
                            recent_ground_truth, predictions, zero_division=0
                        )

            # Get alert count
            snapshot.alerts_count = len(
                self.performance_monitor.get_recent_alerts(hours=1.0)
            )

            # Environment detection
            if self.config.enable_environment_detection and recent_results:
                audio_chunks = (
                    []
                )  # Would need actual audio data for environment detection
                # For now, use heuristics based on signal characteristics
                avg_energy = snapshot.avg_energy_level
                if avg_energy < 0.01:
                    snapshot.detected_environment = EnvironmentType.STUDIO
                elif avg_energy < 0.05:
                    snapshot.detected_environment = EnvironmentType.QUIET_OFFICE
                else:
                    snapshot.detected_environment = EnvironmentType.NOISY_OFFICE

                self.current_environment = snapshot.detected_environment

            return snapshot

    def get_performance_history(self, hours: float = 1.0) -> List[MonitoringSnapshot]:
        """
        Get performance history for specified time period.

        Args:
            hours: Number of hours of history to return.

        Returns:
            List of performance snapshots.
        """
        cutoff_time = time.time() - (hours * 3600)
        return [
            snapshot for snapshot in self.snapshots if snapshot.timestamp >= cutoff_time
        ]

    def add_adaptation_callback(
        self, callback: Callable[[AdaptationTrigger, Dict[str, Any]], None]
    ) -> None:
        """Add callback for adaptation events."""
        self.adaptation_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback for alert events."""
        self.alert_callbacks.append(callback)

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Monitoring loop started")

        while not self.stop_event.wait(self.config.monitoring_interval):
            try:
                # Create performance snapshot
                snapshot = self.get_current_snapshot()
                self.snapshots.append(snapshot)

                # Check for environment changes
                if self.config.enable_environment_detection:
                    self._check_environment_change(snapshot)

                # Log performance summary
                if snapshot.total_samples > 0:
                    logger.debug(
                        f"Performance: Samples={snapshot.total_samples}, "
                        f"AvgTime={snapshot.avg_processing_time*1000:.2f}ms, "
                        f"Confidence={snapshot.avg_confidence:.3f}, "
                        f"SpeechRatio={snapshot.speech_ratio:.3f}"
                    )

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.state = MonitoringState.ERROR
                time.sleep(1.0)  # Brief pause before retrying

        logger.info("Monitoring loop stopped")

    def _calibration_loop(self) -> None:
        """Calibration management loop."""
        logger.info("Calibration loop started")

        # Create calibration request queue
        self._calibration_queue = queue.Queue()

        last_scheduled_calibration = time.time()

        while not self.stop_event.wait(self.config.calibration_interval):
            try:
                # Check for manual calibration requests
                try:
                    trigger = self._calibration_queue.get_nowait()
                    self._perform_calibration(trigger)
                    continue
                except queue.Empty:
                    pass

                # Scheduled calibration
                current_time = time.time()
                if (
                    current_time - last_scheduled_calibration
                    >= self.config.calibration_interval
                ):
                    if self._should_perform_calibration():
                        self._perform_calibration(AdaptationTrigger.SCHEDULED)
                        last_scheduled_calibration = current_time

            except Exception as e:
                logger.error(f"Error in calibration loop: {e}")

        logger.info("Calibration loop stopped")

    def _alert_loop(self) -> None:
        """Alert monitoring and processing loop."""
        logger.info("Alert loop started")

        while not self.stop_event.wait(self.config.alert_check_interval):
            try:
                # Check for new alerts
                recent_alerts = self.performance_monitor.get_recent_alerts(
                    hours=0.1
                )  # Last 6 minutes

                for alert in recent_alerts:
                    # Process alert
                    self._process_alert(alert)

                    # Execute alert callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")

            except Exception as e:
                logger.error(f"Error in alert loop: {e}")

        logger.info("Alert loop stopped")

    def _check_environment_change(self, snapshot: MonitoringSnapshot) -> None:
        """Check for significant environment changes."""
        if not snapshot.detected_environment or not self.snapshots:
            return

        # Compare with recent snapshots
        recent_snapshots = list(self.snapshots)[-10:]  # Last 10 snapshots
        recent_environments = [
            s.detected_environment for s in recent_snapshots if s.detected_environment
        ]

        if not recent_environments:
            return

        # Check for consistent environment change
        if (
            len(set(recent_environments)) == 1
            and recent_environments[0] != snapshot.detected_environment
        ):
            # Environment has changed
            logger.info(
                f"Environment change detected: {recent_environments[0]} -> {snapshot.detected_environment}"
            )

            if self.config.enable_auto_adaptation:
                self.trigger_calibration(AdaptationTrigger.ENVIRONMENT_CHANGE)

    def _should_perform_calibration(self) -> bool:
        """Determine if calibration should be performed."""
        # Check if enough samples available
        if len(self.snapshots) < 10:
            return False

        # Check recent performance
        recent_snapshots = list(self.snapshots)[-10:]

        # Look for performance degradation
        accuracy_scores = [
            s.accuracy for s in recent_snapshots if s.accuracy is not None
        ]
        if accuracy_scores:
            avg_accuracy = mean(accuracy_scores)
            if avg_accuracy < self.config.min_accuracy_threshold:
                logger.info(
                    f"Performance degradation detected: accuracy={avg_accuracy:.3f}"
                )
                return True

        # Check processing time issues
        processing_times = [
            s.avg_processing_time for s in recent_snapshots if s.avg_processing_time > 0
        ]
        if processing_times:
            avg_time = mean(processing_times)
            if avg_time > self.config.max_processing_time:
                logger.info(
                    f"Processing time issue detected: avg_time={avg_time*1000:.2f}ms"
                )
                return True

        return False

    def _perform_calibration(self, trigger: AdaptationTrigger) -> None:
        """Perform automatic calibration."""
        if self.state == MonitoringState.CALIBRATING:
            logger.info("Calibration already in progress")
            return

        logger.info(f"Starting calibration triggered by: {trigger.value}")
        self.state = MonitoringState.CALIBRATING

        try:
            # Collect recent data for calibration
            recent_audio_chunks = []  # Would need actual audio data

            # For now, use auto-calibration with synthetic data
            # In practice, you'd collect real audio data
            if hasattr(self.vad, "get_recent_audio_data"):
                recent_audio_chunks = self.vad.get_recent_audio_data()

            if recent_audio_chunks:
                result = self.calibrator.auto_calibrate_environment(
                    self.vad, recent_audio_chunks
                )

                # Apply optimized configuration
                self.vad.update_config(result.optimal_config)

                # Record adaptation
                adaptation_record = {
                    "timestamp": time.time(),
                    "trigger": trigger.value,
                    "old_threshold": self.vad.config.threshold,
                    "new_threshold": result.optimal_threshold,
                    "performance_improvement": result.f1_score,
                    "environment": result.detected_environment.value,
                }

                self.adaptation_history.append(adaptation_record)
                self.last_adaptation_time = time.time()

                logger.info(
                    f"Calibration completed: F1={result.f1_score:.4f}, "
                    f"Threshold={result.optimal_threshold:.4f}"
                )

                # Execute adaptation callbacks
                for callback in self.adaptation_callbacks:
                    try:
                        callback(trigger, adaptation_record)
                    except Exception as e:
                        logger.error(f"Error in adaptation callback: {e}")

            else:
                logger.warning("No audio data available for calibration")

        except Exception as e:
            logger.error(f"Calibration failed: {e}")

        finally:
            self.state = MonitoringState.MONITORING

    def _process_alert(self, alert: PerformanceAlert) -> None:
        """Process performance alert."""
        logger.warning(f"Performance Alert: {alert.message}")

        # Check if alert should trigger calibration
        if (
            alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]
            and self.config.enable_auto_adaptation
        ):

            # Trigger calibration for critical alerts
            self.trigger_calibration(AdaptationTrigger.ALERT_THRESHOLD)

        # Send email alert if configured
        if self.config.enable_email_alerts and self.config.email_recipients:
            self._send_email_alert(alert)

    def _send_email_alert(self, alert: PerformanceAlert) -> None:
        """Send email alert (placeholder implementation)."""
        # This would integrate with actual email service
        logger.info(f"Email alert sent: {alert.message}")

    def export_monitoring_data(self, filepath: Union[str, Path]) -> None:
        """
        Export monitoring data to file.

        Args:
            filepath: Path to save monitoring data.
        """
        filepath = Path(filepath)

        data = {
            "config": {
                "monitoring_interval": self.config.monitoring_interval,
                "calibration_interval": self.config.calibration_interval,
                "min_accuracy_threshold": self.config.min_accuracy_threshold,
                "max_processing_time": self.config.max_processing_time,
            },
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "adaptation_history": self.adaptation_history,
            "current_state": self.state.value,
            "export_timestamp": time.time(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Monitoring data exported to {filepath}")

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
