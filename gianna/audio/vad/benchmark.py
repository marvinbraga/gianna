"""
Voice Activity Detection (VAD) Benchmarking Suite.

This module provides comprehensive benchmarking capabilities for VAD algorithms,
including standardized test datasets, performance comparison tools, and
statistical analysis for algorithm validation.
"""

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from .base import BaseVAD
from .calibration import EnvironmentType, GroundTruthData, VadCalibrator
from .metrics import PerformanceReport, VadMetrics
from .types import VADConfig, VADResult

logger = logging.getLogger(__name__)


class BenchmarkDataset(Enum):
    """Standard benchmark datasets for VAD evaluation."""

    CLEAN_SPEECH = "clean_speech"
    NOISY_OFFICE = "noisy_office"
    CONVERSATION = "conversation"
    MIXED_ENVIRONMENT = "mixed_environment"
    SYNTHETIC = "synthetic"
    CUSTOM = "custom"


class BenchmarkCategory(Enum):
    """Categories of benchmarking tests."""

    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    REAL_TIME = "real_time"
    ENVIRONMENT_ADAPTATION = "environment_adaptation"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking tests."""

    # Test selection
    categories: List[BenchmarkCategory] = field(
        default_factory=lambda: list(BenchmarkCategory)
    )
    datasets: List[BenchmarkDataset] = field(
        default_factory=lambda: [BenchmarkDataset.CLEAN_SPEECH]
    )

    # Statistical settings
    num_trials: int = 5
    confidence_level: float = 0.95
    statistical_tests: bool = True

    # Performance settings
    max_processing_time: float = 0.1  # 100ms
    real_time_factor: float = 1.0  # 1x real-time

    # Environment settings
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.5])
    snr_levels: List[float] = field(default_factory=lambda: [20, 10, 5, 0, -5])  # dB

    # Output settings
    generate_plots: bool = True
    save_results: bool = True
    detailed_analysis: bool = True


@dataclass
class BenchmarkResult:
    """Results from a single algorithm benchmark."""

    algorithm_name: str
    algorithm_config: VADConfig

    # Performance metrics
    accuracy_scores: List[float] = field(default_factory=list)
    precision_scores: List[float] = field(default_factory=list)
    recall_scores: List[float] = field(default_factory=list)
    f1_scores: List[float] = field(default_factory=list)
    auc_scores: List[float] = field(default_factory=list)

    # Efficiency metrics
    processing_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)

    # Robustness metrics
    noise_performance: Dict[float, float] = field(default_factory=dict)
    snr_performance: Dict[float, float] = field(default_factory=dict)
    environment_performance: Dict[str, float] = field(default_factory=dict)

    # Statistical measures
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # Overall scores
    overall_score: float = 0.0
    ranking_score: float = 0.0

    # Metadata
    test_duration: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.accuracy_scores:
            self.mean_accuracy = mean(self.accuracy_scores)
            if len(self.accuracy_scores) > 1:
                self.std_accuracy = stdev(self.accuracy_scores)

                # Calculate confidence interval
                n = len(self.accuracy_scores)
                t_critical = stats.t.ppf(0.975, n - 1)  # 95% CI
                margin_error = t_critical * self.std_accuracy / np.sqrt(n)
                self.confidence_interval = (
                    self.mean_accuracy - margin_error,
                    self.mean_accuracy + margin_error,
                )


class DatasetGenerator:
    """Generate synthetic datasets for benchmarking."""

    @staticmethod
    def generate_clean_speech_dataset(
        duration: float = 60.0, sample_rate: int = 16000, speech_ratio: float = 0.6
    ) -> GroundTruthData:
        """
        Generate clean speech dataset.

        Args:
            duration: Dataset duration in seconds.
            sample_rate: Audio sample rate.
            speech_ratio: Ratio of speech to silence.

        Returns:
            Generated ground truth data.
        """
        chunk_size = 1024
        chunks_per_second = sample_rate // chunk_size
        total_chunks = int(duration * chunks_per_second)
        speech_chunks = int(total_chunks * speech_ratio)

        # Generate audio chunks and labels
        audio_chunks = []
        labels = []
        timestamps = []

        # Create pattern: alternating speech and silence periods
        current_time = 0.0
        chunk_duration = chunk_size / sample_rate

        for i in range(total_chunks):
            # Determine if chunk should be speech or silence
            is_speech = (i % 10) < (speech_ratio * 10)  # Pattern-based

            # Generate synthetic audio
            if is_speech:
                # Generate speech-like signal with higher energy
                amplitude = np.random.normal(0, 0.1, chunk_size)
                noise = np.random.normal(0, 0.01, chunk_size)
                chunk = amplitude + noise
            else:
                # Generate silence with low-level noise
                chunk = np.random.normal(0, 0.005, chunk_size)

            audio_chunks.append(chunk.astype(np.float32))
            labels.append(is_speech)
            timestamps.append(current_time)
            current_time += chunk_duration

        return GroundTruthData(
            audio_chunks=audio_chunks,
            labels=labels,
            timestamps=timestamps,
            environment=EnvironmentType.STUDIO,
            metadata={"dataset_type": "clean_speech", "synthetic": True},
        )

    @staticmethod
    def generate_noisy_dataset(
        base_dataset: GroundTruthData,
        noise_level: float = 0.1,
        noise_type: str = "gaussian",
    ) -> GroundTruthData:
        """
        Add noise to existing dataset.

        Args:
            base_dataset: Clean dataset to add noise to.
            noise_level: Noise level (0.0 to 1.0).
            noise_type: Type of noise ('gaussian', 'uniform', 'pink').

        Returns:
            Noisy ground truth data.
        """
        noisy_chunks = []

        for chunk in base_dataset.audio_chunks:
            chunk_array = (
                chunk
                if isinstance(chunk, np.ndarray)
                else np.frombuffer(chunk, dtype=np.float32)
            )

            if noise_type == "gaussian":
                noise = np.random.normal(0, noise_level, len(chunk_array))
            elif noise_type == "uniform":
                noise = np.random.uniform(-noise_level, noise_level, len(chunk_array))
            elif noise_type == "pink":
                # Simplified pink noise approximation
                white_noise = np.random.randn(len(chunk_array))
                noise = np.cumsum(white_noise) * noise_level / 10
            else:
                noise = np.zeros_like(chunk_array)

            noisy_chunk = chunk_array + noise
            noisy_chunks.append(noisy_chunk.astype(np.float32))

        return GroundTruthData(
            audio_chunks=noisy_chunks,
            labels=base_dataset.labels.copy(),
            timestamps=base_dataset.timestamps.copy(),
            environment=EnvironmentType.NOISY_OFFICE,
            noise_level=noise_level,
            metadata={
                **base_dataset.metadata,
                "noise_type": noise_type,
                "noise_level": noise_level,
            },
        )


class VadBenchmark:
    """
    Comprehensive VAD benchmarking and comparison system.

    Provides standardized benchmarking capabilities for VAD algorithms
    including accuracy, performance, robustness, and efficiency testing.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize VAD benchmark suite.

        Args:
            config: Benchmark configuration.
        """
        self.config = config or BenchmarkConfig()
        self.results: Dict[str, BenchmarkResult] = {}
        self.datasets: Dict[BenchmarkDataset, GroundTruthData] = {}

        # Components
        self.metrics = VadMetrics(enable_monitoring=False)
        self.calibrator = VadCalibrator()

        # Initialize datasets
        self._initialize_datasets()

        logger.info("VadBenchmark initialized")

    def _initialize_datasets(self) -> None:
        """Initialize benchmark datasets."""
        # Generate synthetic datasets
        if BenchmarkDataset.CLEAN_SPEECH in self.config.datasets:
            self.datasets[BenchmarkDataset.CLEAN_SPEECH] = (
                DatasetGenerator.generate_clean_speech_dataset()
            )

        if BenchmarkDataset.NOISY_OFFICE in self.config.datasets:
            clean_data = self.datasets.get(BenchmarkDataset.CLEAN_SPEECH)
            if not clean_data:
                clean_data = DatasetGenerator.generate_clean_speech_dataset()

            self.datasets[BenchmarkDataset.NOISY_OFFICE] = (
                DatasetGenerator.generate_noisy_dataset(clean_data, noise_level=0.2)
            )

    def benchmark_algorithm(
        self,
        algorithm_name: str,
        vad_instance: BaseVAD,
        custom_datasets: Optional[Dict[str, GroundTruthData]] = None,
    ) -> BenchmarkResult:
        """
        Benchmark a single VAD algorithm across all configured tests.

        Args:
            algorithm_name: Name identifier for the algorithm.
            vad_instance: VAD instance to benchmark.
            custom_datasets: Optional custom datasets for testing.

        Returns:
            Comprehensive benchmark results.
        """
        logger.info(f"Starting benchmark for algorithm: {algorithm_name}")
        start_time = time.time()

        result = BenchmarkResult(
            algorithm_name=algorithm_name, algorithm_config=vad_instance.config
        )

        # Use custom datasets if provided
        test_datasets = custom_datasets or self.datasets

        try:
            # Accuracy benchmarking
            if BenchmarkCategory.ACCURACY in self.config.categories:
                self._benchmark_accuracy(vad_instance, result, test_datasets)

            # Performance benchmarking
            if BenchmarkCategory.PERFORMANCE in self.config.categories:
                self._benchmark_performance(vad_instance, result, test_datasets)

            # Robustness benchmarking
            if BenchmarkCategory.ROBUSTNESS in self.config.categories:
                self._benchmark_robustness(vad_instance, result, test_datasets)

            # Efficiency benchmarking
            if BenchmarkCategory.EFFICIENCY in self.config.categories:
                self._benchmark_efficiency(vad_instance, result, test_datasets)

            # Real-time benchmarking
            if BenchmarkCategory.REAL_TIME in self.config.categories:
                self._benchmark_real_time(vad_instance, result, test_datasets)

            # Environment adaptation benchmarking
            if BenchmarkCategory.ENVIRONMENT_ADAPTATION in self.config.categories:
                self._benchmark_environment_adaptation(
                    vad_instance, result, test_datasets
                )

            # Calculate overall scores
            result.overall_score = self._calculate_overall_score(result)
            result.test_duration = time.time() - start_time

            # Store results
            self.results[algorithm_name] = result

            logger.info(
                f"Benchmark completed for {algorithm_name}: "
                f"Overall Score = {result.overall_score:.4f}, "
                f"Duration = {result.test_duration:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Benchmark failed for {algorithm_name}: {e}")
            raise

    def _benchmark_accuracy(
        self,
        vad: BaseVAD,
        result: BenchmarkResult,
        datasets: Dict[str, GroundTruthData],
    ) -> None:
        """Benchmark accuracy across datasets."""
        logger.debug("Running accuracy benchmarks")

        for dataset_name, dataset in datasets.items():
            for trial in range(self.config.num_trials):
                # Reset VAD state
                vad.reset_state()

                # Process dataset
                predictions = []
                for chunk in dataset.audio_chunks:
                    vad_result = vad.detect_activity(chunk)
                    predictions.append(vad_result.is_voice_active)

                # Calculate metrics
                from sklearn.metrics import (
                    accuracy_score,
                    f1_score,
                    precision_score,
                    recall_score,
                    roc_auc_score,
                )

                # Get confidences for AUC
                confidences = []
                vad.reset_state()
                for chunk in dataset.audio_chunks:
                    vad_result = vad.detect_activity(chunk)
                    confidences.append(vad_result.confidence)

                accuracy = accuracy_score(dataset.labels, predictions)

                # Handle case where only one class is present
                if len(set(dataset.labels)) > 1:
                    precision = precision_score(
                        dataset.labels, predictions, zero_division=0
                    )
                    recall = recall_score(dataset.labels, predictions, zero_division=0)
                    f1 = f1_score(dataset.labels, predictions, zero_division=0)

                    try:
                        auc = roc_auc_score(dataset.labels, confidences)
                    except ValueError:
                        auc = 0.5
                else:
                    precision = recall = f1 = auc = 1.0 if accuracy == 1.0 else 0.0

                result.accuracy_scores.append(accuracy)
                result.precision_scores.append(precision)
                result.recall_scores.append(recall)
                result.f1_scores.append(f1)
                result.auc_scores.append(auc)

    def _benchmark_performance(
        self,
        vad: BaseVAD,
        result: BenchmarkResult,
        datasets: Dict[str, GroundTruthData],
    ) -> None:
        """Benchmark processing performance."""
        logger.debug("Running performance benchmarks")

        # Use first dataset for performance testing
        dataset = next(iter(datasets.values()))

        processing_times = []

        for trial in range(self.config.num_trials):
            vad.reset_state()

            trial_times = []
            for chunk in dataset.audio_chunks[:100]:  # Limit for performance testing
                start_time = time.time()
                vad.detect_activity(chunk)
                processing_time = time.time() - start_time
                trial_times.append(processing_time)

            processing_times.extend(trial_times)

        result.processing_times = processing_times

    def _benchmark_robustness(
        self,
        vad: BaseVAD,
        result: BenchmarkResult,
        datasets: Dict[str, GroundTruthData],
    ) -> None:
        """Benchmark robustness to noise and environmental conditions."""
        logger.debug("Running robustness benchmarks")

        # Get base clean dataset
        base_dataset = next(iter(datasets.values()))

        # Test different noise levels
        for noise_level in self.config.noise_levels:
            if noise_level == 0.0:
                test_dataset = base_dataset
            else:
                test_dataset = DatasetGenerator.generate_noisy_dataset(
                    base_dataset, noise_level
                )

            # Run accuracy test on noisy data
            vad.reset_state()
            predictions = []
            for chunk in test_dataset.audio_chunks:
                vad_result = vad.detect_activity(chunk)
                predictions.append(vad_result.is_voice_active)

            accuracy = accuracy_score(test_dataset.labels, predictions)
            result.noise_performance[noise_level] = accuracy

        # Test different SNR levels (simplified)
        for snr_db in self.config.snr_levels:
            # Convert dB to linear scale for noise level approximation
            noise_level = 0.1 * (10 ** (-snr_db / 20))
            noise_level = min(1.0, max(0.0, noise_level))

            test_dataset = DatasetGenerator.generate_noisy_dataset(
                base_dataset, noise_level
            )

            vad.reset_state()
            predictions = []
            for chunk in test_dataset.audio_chunks:
                vad_result = vad.detect_activity(chunk)
                predictions.append(vad_result.is_voice_active)

            accuracy = accuracy_score(test_dataset.labels, predictions)
            result.snr_performance[snr_db] = accuracy

    def _benchmark_efficiency(
        self,
        vad: BaseVAD,
        result: BenchmarkResult,
        datasets: Dict[str, GroundTruthData],
    ) -> None:
        """Benchmark computational efficiency."""
        logger.debug("Running efficiency benchmarks")

        import os

        import psutil

        dataset = next(iter(datasets.values()))

        # Memory usage tracking
        process = psutil.Process(os.getpid())
        memory_usage = []
        cpu_usage = []

        for trial in range(min(3, self.config.num_trials)):  # Limit for efficiency
            vad.reset_state()

            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            initial_cpu = process.cpu_percent()

            # Process chunks
            for chunk in dataset.audio_chunks[:500]:  # Limit for efficiency testing
                vad.detect_activity(chunk)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = process.cpu_percent()

            memory_usage.append(final_memory - initial_memory)
            cpu_usage.append(final_cpu - initial_cpu)

        result.memory_usage = memory_usage
        result.cpu_usage = cpu_usage

    def _benchmark_real_time(
        self,
        vad: BaseVAD,
        result: BenchmarkResult,
        datasets: Dict[str, GroundTruthData],
    ) -> None:
        """Benchmark real-time performance capabilities."""
        logger.debug("Running real-time benchmarks")

        dataset = next(iter(datasets.values()))

        # Calculate real-time constraints
        chunk_size = vad.config.chunk_size
        sample_rate = vad.config.sample_rate
        real_time_limit = (chunk_size / sample_rate) * self.config.real_time_factor

        vad.reset_state()
        real_time_violations = 0
        total_chunks = 0

        for chunk in dataset.audio_chunks:
            start_time = time.time()
            vad.detect_activity(chunk)
            processing_time = time.time() - start_time

            if processing_time > real_time_limit:
                real_time_violations += 1
            total_chunks += 1

        real_time_ratio = (
            1.0 - (real_time_violations / total_chunks) if total_chunks > 0 else 0.0
        )
        result.metadata["real_time_ratio"] = real_time_ratio
        result.metadata["real_time_violations"] = real_time_violations
        result.metadata["total_chunks_tested"] = total_chunks

    def _benchmark_environment_adaptation(
        self,
        vad: BaseVAD,
        result: BenchmarkResult,
        datasets: Dict[str, GroundTruthData],
    ) -> None:
        """Benchmark environment adaptation capabilities."""
        logger.debug("Running environment adaptation benchmarks")

        # Test different environments
        environments = [
            (EnvironmentType.STUDIO, 0.0),
            (EnvironmentType.QUIET_OFFICE, 0.05),
            (EnvironmentType.HOME_OFFICE, 0.1),
            (EnvironmentType.NOISY_OFFICE, 0.2),
            (EnvironmentType.CAR, 0.3),
        ]

        base_dataset = next(iter(datasets.values()))

        for env_type, noise_level in environments:
            if noise_level == 0.0:
                test_dataset = base_dataset
            else:
                test_dataset = DatasetGenerator.generate_noisy_dataset(
                    base_dataset, noise_level
                )
                test_dataset.environment = env_type

            vad.reset_state()
            predictions = []
            for chunk in test_dataset.audio_chunks:
                vad_result = vad.detect_activity(chunk)
                predictions.append(vad_result.is_voice_active)

            accuracy = accuracy_score(test_dataset.labels, predictions)
            result.environment_performance[env_type.value] = accuracy

    def _calculate_overall_score(self, result: BenchmarkResult) -> float:
        """Calculate overall performance score."""
        scores = []
        weights = []

        # Accuracy score (40% weight)
        if result.accuracy_scores:
            scores.append(mean(result.accuracy_scores))
            weights.append(0.4)

        # Processing time score (20% weight)
        if result.processing_times:
            avg_time = mean(result.processing_times)
            # Score inversely related to processing time
            time_score = max(
                0.0,
                min(
                    1.0,
                    (self.config.max_processing_time - avg_time)
                    / self.config.max_processing_time,
                ),
            )
            scores.append(time_score)
            weights.append(0.2)

        # Robustness score (25% weight)
        if result.noise_performance:
            robustness_score = mean(result.noise_performance.values())
            scores.append(robustness_score)
            weights.append(0.25)

        # Real-time performance (15% weight)
        real_time_ratio = result.metadata.get("real_time_ratio", 0.0)
        scores.append(real_time_ratio)
        weights.append(0.15)

        # Calculate weighted average
        if scores and weights:
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return weighted_score

        return 0.0

    def compare_algorithms(
        self,
        algorithms: Dict[str, BaseVAD],
        custom_datasets: Optional[Dict[str, GroundTruthData]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple VAD algorithms.

        Args:
            algorithms: Dictionary of algorithm name -> VAD instance.
            custom_datasets: Optional custom datasets for comparison.

        Returns:
            Dictionary of benchmark results for each algorithm.
        """
        logger.info(f"Comparing {len(algorithms)} algorithms")

        comparison_results = {}

        for name, vad_instance in algorithms.items():
            try:
                result = self.benchmark_algorithm(name, vad_instance, custom_datasets)
                comparison_results[name] = result
            except Exception as e:
                logger.error(f"Failed to benchmark {name}: {e}")
                continue

        # Calculate ranking scores
        self._calculate_ranking_scores(comparison_results)

        # Generate comparison report
        if self.config.save_results:
            self._save_comparison_report(comparison_results)

        if self.config.generate_plots and PLOTTING_AVAILABLE:
            self._generate_comparison_plots(comparison_results)

        logger.info("Algorithm comparison completed")
        return comparison_results

    def _calculate_ranking_scores(self, results: Dict[str, BenchmarkResult]) -> None:
        """Calculate ranking scores for algorithm comparison."""
        if not results:
            return

        # Get all overall scores
        scores = [result.overall_score for result in results.values()]

        if not scores:
            return

        # Normalize scores to 0-100 scale
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        for result in results.values():
            normalized_score = ((result.overall_score - min_score) / score_range) * 100
            result.ranking_score = normalized_score

    def _save_comparison_report(self, results: Dict[str, BenchmarkResult]) -> None:
        """Save comprehensive comparison report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"vad_benchmark_report_{timestamp}.json")

        # Compile report data
        report_data = {
            "benchmark_config": {
                "categories": [cat.value for cat in self.config.categories],
                "datasets": [ds.value for ds in self.config.datasets],
                "num_trials": self.config.num_trials,
                "confidence_level": self.config.confidence_level,
            },
            "results": {},
            "summary": {
                "total_algorithms": len(results),
                "best_algorithm": None,
                "best_score": 0.0,
                "timestamp": time.time(),
            },
        }

        # Add individual results
        best_algo = None
        best_score = 0.0

        for name, result in results.items():
            report_data["results"][name] = {
                "overall_score": result.overall_score,
                "ranking_score": result.ranking_score,
                "mean_accuracy": result.mean_accuracy,
                "std_accuracy": result.std_accuracy,
                "confidence_interval": result.confidence_interval,
                "avg_processing_time": (
                    mean(result.processing_times) if result.processing_times else 0.0
                ),
                "noise_performance": result.noise_performance,
                "environment_performance": result.environment_performance,
                "test_duration": result.test_duration,
                "metadata": result.metadata,
            }

            if result.overall_score > best_score:
                best_score = result.overall_score
                best_algo = name

        report_data["summary"]["best_algorithm"] = best_algo
        report_data["summary"]["best_score"] = best_score

        # Save report
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"Benchmark report saved to {report_path}")

    def _generate_comparison_plots(self, results: Dict[str, BenchmarkResult]) -> None:
        """Generate comparison plots."""
        if not results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("VAD Algorithm Comparison", fontsize=16)

        algorithms = list(results.keys())

        # Overall scores comparison
        overall_scores = [results[algo].overall_score for algo in algorithms]
        axes[0, 0].bar(algorithms, overall_scores)
        axes[0, 0].set_title("Overall Performance Scores")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Accuracy comparison with confidence intervals
        accuracies = [results[algo].mean_accuracy for algo in algorithms]
        errors = [results[algo].std_accuracy for algo in algorithms]
        axes[0, 1].bar(algorithms, accuracies, yerr=errors, capsize=5)
        axes[0, 1].set_title("Accuracy Comparison")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Processing time comparison
        processing_times = [
            mean(results[algo].processing_times) * 1000  # Convert to ms
            for algo in algorithms
            if results[algo].processing_times
        ]
        active_algos = [algo for algo in algorithms if results[algo].processing_times]
        if processing_times:
            axes[1, 0].bar(active_algos, processing_times)
            axes[1, 0].set_title("Processing Time Comparison")
            axes[1, 0].set_ylabel("Time (ms)")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # Robustness to noise
        if any(results[algo].noise_performance for algo in algorithms):
            noise_levels = list(next(iter(results.values())).noise_performance.keys())
            for algo in algorithms:
                if results[algo].noise_performance:
                    noise_scores = [
                        results[algo].noise_performance[level] for level in noise_levels
                    ]
                    axes[1, 1].plot(noise_levels, noise_scores, marker="o", label=algo)

            axes[1, 1].set_title("Robustness to Noise")
            axes[1, 1].set_xlabel("Noise Level")
            axes[1, 1].set_ylabel("Accuracy")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = f"vad_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Comparison plot saved to {plot_path}")

        plt.show()

    def generate_detailed_report(
        self, algorithm_name: str, filepath: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Generate detailed analysis report for a specific algorithm.

        Args:
            algorithm_name: Name of the algorithm to analyze.
            filepath: Optional path to save the report.
        """
        if algorithm_name not in self.results:
            logger.error(f"No results found for algorithm: {algorithm_name}")
            return

        result = self.results[algorithm_name]

        if not filepath:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"detailed_report_{algorithm_name}_{timestamp}.json"

        # Compile detailed report
        detailed_data = {
            "algorithm_info": {
                "name": result.algorithm_name,
                "config": {
                    "threshold": result.algorithm_config.threshold,
                    "min_speech_duration": result.algorithm_config.min_speech_duration,
                    "min_silence_duration": result.algorithm_config.min_silence_duration,
                    "sample_rate": result.algorithm_config.sample_rate,
                    "chunk_size": result.algorithm_config.chunk_size,
                },
            },
            "performance_metrics": {
                "accuracy": {
                    "scores": result.accuracy_scores,
                    "mean": result.mean_accuracy,
                    "std": result.std_accuracy,
                    "confidence_interval": result.confidence_interval,
                },
                "precision": {
                    "scores": result.precision_scores,
                    "mean": (
                        mean(result.precision_scores)
                        if result.precision_scores
                        else 0.0
                    ),
                    "std": (
                        stdev(result.precision_scores)
                        if len(result.precision_scores) > 1
                        else 0.0
                    ),
                },
                "recall": {
                    "scores": result.recall_scores,
                    "mean": mean(result.recall_scores) if result.recall_scores else 0.0,
                    "std": (
                        stdev(result.recall_scores)
                        if len(result.recall_scores) > 1
                        else 0.0
                    ),
                },
                "f1_score": {
                    "scores": result.f1_scores,
                    "mean": mean(result.f1_scores) if result.f1_scores else 0.0,
                    "std": (
                        stdev(result.f1_scores) if len(result.f1_scores) > 1 else 0.0
                    ),
                },
                "auc_score": {
                    "scores": result.auc_scores,
                    "mean": mean(result.auc_scores) if result.auc_scores else 0.0,
                    "std": (
                        stdev(result.auc_scores) if len(result.auc_scores) > 1 else 0.0
                    ),
                },
            },
            "efficiency_metrics": {
                "processing_time": {
                    "times": result.processing_times,
                    "mean_ms": (
                        mean(result.processing_times) * 1000
                        if result.processing_times
                        else 0.0
                    ),
                    "max_ms": (
                        max(result.processing_times) * 1000
                        if result.processing_times
                        else 0.0
                    ),
                    "min_ms": (
                        min(result.processing_times) * 1000
                        if result.processing_times
                        else 0.0
                    ),
                    "std_ms": (
                        stdev(result.processing_times) * 1000
                        if len(result.processing_times) > 1
                        else 0.0
                    ),
                },
                "memory_usage_mb": result.memory_usage,
                "cpu_usage_percent": result.cpu_usage,
            },
            "robustness_analysis": {
                "noise_performance": result.noise_performance,
                "snr_performance": result.snr_performance,
                "environment_performance": result.environment_performance,
            },
            "overall_assessment": {
                "overall_score": result.overall_score,
                "ranking_score": result.ranking_score,
                "test_duration": result.test_duration,
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
            },
            "metadata": result.metadata,
            "report_timestamp": time.time(),
        }

        # Add analysis insights
        self._add_analysis_insights(detailed_data, result)

        # Save report
        with open(filepath, "w") as f:
            json.dump(detailed_data, f, indent=2, default=str)

        logger.info(f"Detailed report saved to {filepath}")

    def _add_analysis_insights(
        self, report_data: Dict[str, Any], result: BenchmarkResult
    ) -> None:
        """Add analysis insights to detailed report."""
        strengths = []
        weaknesses = []
        recommendations = []

        # Accuracy analysis
        if result.mean_accuracy > 0.9:
            strengths.append("Excellent accuracy performance")
        elif result.mean_accuracy < 0.7:
            weaknesses.append("Below-average accuracy")
            recommendations.append(
                "Consider parameter tuning or algorithm optimization"
            )

        # Processing time analysis
        if result.processing_times:
            avg_time = mean(result.processing_times)
            if avg_time < 0.01:  # 10ms
                strengths.append("Very fast processing time")
            elif avg_time > 0.05:  # 50ms
                weaknesses.append("Slow processing time")
                recommendations.append("Optimize algorithm for real-time performance")

        # Robustness analysis
        if result.noise_performance:
            noise_performance = list(result.noise_performance.values())
            if min(noise_performance) > 0.8:
                strengths.append("Excellent noise robustness")
            elif min(noise_performance) < 0.6:
                weaknesses.append("Poor noise robustness")
                recommendations.append(
                    "Implement noise reduction or adaptive thresholding"
                )

        # Update report
        report_data["overall_assessment"]["strengths"] = strengths
        report_data["overall_assessment"]["weaknesses"] = weaknesses
        report_data["overall_assessment"]["recommendations"] = recommendations
