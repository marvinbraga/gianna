#!/usr/bin/env python3
"""
Comprehensive VAD Calibration and Metrics System Demonstration.

This script demonstrates the full capabilities of the new VAD calibration,
metrics, benchmarking, and real-time monitoring systems.

Features demonstrated:
- Automatic parameter optimization for different environments
- Real-time performance monitoring and adaptation
- Algorithm benchmarking and comparison
- Statistical analysis with ROC curves and confidence intervals
- Environment detection and adaptive thresholds
- Noise floor estimation and SNR analysis
- Production-ready VAD deployment

Requirements:
- numpy
- scipy
- scikit-learn
- matplotlib (optional, for plots)
"""

import logging
import time
import warnings
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    """Demonstrate VAD calibration and metrics system."""

    print("=" * 80)
    print("VAD Calibration and Metrics System Demonstration")
    print("=" * 80)

    try:
        # Import VAD components
        from gianna.audio.vad import (
            BenchmarkCategory,
            BenchmarkConfig,
            CalibrationConfig,
            DatasetGenerator,
            EnvironmentType,
            GroundTruthData,
            OptimizationMethod,
            RealtimeMonitor,
            VadBenchmark,
            VadCalibrator,
            VadMetrics,
            create_monitored_vad,
            create_production_vad,
            create_vad,
        )

        print("\n✓ All VAD components loaded successfully")

        # Demonstrate basic calibration
        demonstrate_basic_calibration()

        # Demonstrate metrics collection
        demonstrate_metrics_collection()

        # Demonstrate benchmarking
        demonstrate_benchmarking()

        # Demonstrate real-time monitoring
        demonstrate_realtime_monitoring()

        # Demonstrate production setup
        demonstrate_production_setup()

        print("\n" + "=" * 80)
        print("✓ All demonstrations completed successfully!")
        print("=" * 80)

    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("\nTo run this demo, install required packages:")
        print("pip install scipy scikit-learn matplotlib")
        return False

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False

    return True


def demonstrate_basic_calibration():
    """Demonstrate basic VAD calibration functionality."""

    print("\n" + "-" * 60)
    print("1. BASIC VAD CALIBRATION")
    print("-" * 60)

    from gianna.audio.vad import (
        CalibrationConfig,
        DatasetGenerator,
        EnvironmentType,
        GroundTruthData,
        OptimizationMethod,
        VadCalibrator,
        create_vad,
    )

    # Create a VAD instance to calibrate
    vad = create_vad("energy", threshold=0.05)  # Intentionally poor threshold
    print(f"✓ Created VAD with initial threshold: {vad.config.threshold}")

    # Create calibrator
    config = CalibrationConfig(
        optimization_method=OptimizationMethod.DIFFERENTIAL_EVOLUTION,
        max_iterations=50,  # Reduced for demo speed
        validation_split=0.3,
    )
    calibrator = VadCalibrator(config)
    print("✓ Created calibrator with differential evolution optimization")

    # Generate synthetic ground truth data
    ground_truth = DatasetGenerator.generate_clean_speech_dataset(
        duration=30.0, speech_ratio=0.6  # 30 seconds of audio
    )
    print(f"✓ Generated synthetic dataset: {len(ground_truth.audio_chunks)} chunks")

    # Perform calibration
    print("\n  Calibrating VAD parameters...")
    start_time = time.time()

    result = calibrator.calibrate_vad(vad, ground_truth)

    calibration_time = time.time() - start_time
    print(f"  ⏱️  Calibration completed in {calibration_time:.2f} seconds")

    # Display results
    print(f"\n  📊 CALIBRATION RESULTS:")
    print(f"     Environment: {result.detected_environment.value}")
    print(f"     Optimal threshold: {result.optimal_threshold:.4f}")
    print(f"     Accuracy: {result.accuracy:.3f}")
    print(f"     Precision: {result.precision:.3f}")
    print(f"     Recall: {result.recall:.3f}")
    print(f"     F1-Score: {result.f1_score:.3f}")
    print(f"     AUC Score: {result.auc_score:.3f}")
    print(
        f"     Confidence Interval: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]"
    )
    print(f"     Noise Floor: {result.noise_floor:.4f}")
    print(f"     SNR: {result.signal_to_noise_ratio:.1f} dB")

    # Test auto-calibration for different environments
    print("\n  🌍 Testing environment-specific calibration:")

    # Generate noisy dataset
    noisy_data = DatasetGenerator.generate_noisy_dataset(
        ground_truth, noise_level=0.2, noise_type="gaussian"
    )

    noisy_result = calibrator.auto_calibrate_environment(
        vad, noisy_data.audio_chunks[:100], duration=10.0
    )

    print(f"     Noisy environment threshold: {noisy_result.optimal_threshold:.4f}")
    print(f"     Performance in noise: F1={noisy_result.f1_score:.3f}")

    # Save calibration profile
    calibrator.save_calibration_profile("vad_calibration_profile.json")
    print("✓ Calibration profile saved")


def demonstrate_metrics_collection():
    """Demonstrate metrics collection and analysis."""

    print("\n" + "-" * 60)
    print("2. PERFORMANCE METRICS AND ANALYSIS")
    print("-" * 60)

    from gianna.audio.vad import (
        DatasetGenerator,
        EnvironmentType,
        GroundTruthData,
        VadMetrics,
        create_vad,
    )

    # Create VAD and metrics system
    vad = create_vad("energy", threshold=0.025)
    metrics = VadMetrics(enable_monitoring=True)
    print("✓ Created VAD and metrics system")

    # Generate test data
    test_data = DatasetGenerator.generate_clean_speech_dataset(
        duration=20.0, speech_ratio=0.7
    )
    print(f"✓ Generated test dataset: {len(test_data.audio_chunks)} chunks")

    # Process audio and collect metrics
    print("\n  📈 Processing audio and collecting metrics...")

    predictions = []
    processing_times = []

    for i, chunk in enumerate(test_data.audio_chunks):
        start_time = time.time()
        result = vad.detect_activity(chunk)
        processing_time = time.time() - start_time

        # Record metrics
        ground_truth = test_data.labels[i] if i < len(test_data.labels) else None
        metrics.record_result(result, ground_truth, EnvironmentType.STUDIO)

        predictions.append(result.is_voice_active)
        processing_times.append(processing_time)

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(test_data.audio_chunks)} chunks")

    # Generate performance report
    print("\n  📊 Generating performance report...")
    report = metrics.evaluate_performance(test_data, EnvironmentType.STUDIO)

    print(f"\n  📋 PERFORMANCE REPORT:")
    print(f"     Total samples: {report.total_samples}")
    print(f"     Accuracy: {report.accuracy:.3f}")
    print(f"     Precision: {report.precision:.3f}")
    print(f"     Recall: {report.recall:.3f}")
    print(f"     F1-Score: {report.f1_score:.3f}")
    print(f"     Specificity: {report.specificity:.3f}")
    print(f"     AUC Score: {report.auc_score:.3f}")
    print(f"     False Positive Rate: {report.false_positive_rate:.3f}")
    print(f"     False Negative Rate: {report.false_negative_rate:.3f}")
    print(f"     Avg Processing Time: {report.avg_processing_time*1000:.2f} ms")
    print(f"     Max Processing Time: {report.max_processing_time*1000:.2f} ms")
    print(f"     Data Quality Score: {report.data_quality_score:.3f}")
    print(f"     Reliability Score: {report.reliability_score:.3f}")

    # Generate ROC curve data
    roc_data = metrics.generate_roc_curve(test_data)
    if roc_data:
        fpr, tpr, auc_score = roc_data
        print(f"     ROC AUC from curve: {auc_score:.3f}")

    # Generate Precision-Recall curve data
    pr_data = metrics.generate_precision_recall_curve(test_data)
    if pr_data:
        precision, recall, avg_precision = pr_data
        print(f"     Average Precision: {avg_precision:.3f}")

    # Export comprehensive report
    metrics.export_metrics_report("vad_metrics_report.json")
    print("✓ Metrics report exported")

    # Plot performance metrics if matplotlib is available
    try:
        metrics.plot_performance_metrics("vad_performance_plots.png")
        print("✓ Performance plots saved")
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")


def demonstrate_benchmarking():
    """Demonstrate algorithm benchmarking and comparison."""

    print("\n" + "-" * 60)
    print("3. ALGORITHM BENCHMARKING")
    print("-" * 60)

    from gianna.audio.vad import (
        BenchmarkCategory,
        BenchmarkConfig,
        BenchmarkDataset,
        DatasetGenerator,
        VadBenchmark,
        create_vad,
        get_available_algorithms,
    )

    # Get available algorithms
    available_algorithms = get_available_algorithms()
    print(f"✓ Available algorithms: {', '.join(available_algorithms)}")

    # Create benchmark configuration
    config = BenchmarkConfig(
        categories=[
            BenchmarkCategory.ACCURACY,
            BenchmarkCategory.PERFORMANCE,
            BenchmarkCategory.ROBUSTNESS,
        ],
        datasets=[BenchmarkDataset.CLEAN_SPEECH, BenchmarkDataset.NOISY_OFFICE],
        num_trials=3,  # Reduced for demo speed
        generate_plots=True,
        save_results=True,
    )

    benchmark = VadBenchmark(config)
    print("✓ Created benchmark suite")

    # Create algorithm instances to compare
    algorithms = {}
    for algo_name in available_algorithms:
        try:
            if algo_name == "energy":
                algorithms[f"{algo_name}_conservative"] = create_vad(
                    algo_name, threshold=0.015
                )
                algorithms[f"{algo_name}_balanced"] = create_vad(
                    algo_name, threshold=0.025
                )
                algorithms[f"{algo_name}_aggressive"] = create_vad(
                    algo_name, threshold=0.04
                )
            else:
                algorithms[algo_name] = create_vad(algo_name)
        except Exception as e:
            print(f"⚠️  Could not create {algo_name}: {e}")

    print(f"✓ Created {len(algorithms)} algorithm instances for comparison")

    # Run benchmarks (limiting to prevent long runtime)
    if len(algorithms) > 0:
        print("\n  🏃 Running benchmarks...")

        # Limit to first few algorithms for demo
        limited_algorithms = dict(list(algorithms.items())[: min(3, len(algorithms))])

        start_time = time.time()
        results = benchmark.compare_algorithms(limited_algorithms)
        benchmark_time = time.time() - start_time

        print(f"  ⏱️  Benchmarking completed in {benchmark_time:.2f} seconds")

        # Display results
        print(f"\n  🏆 BENCHMARK RESULTS:")

        # Sort by overall score
        sorted_results = sorted(
            results.items(), key=lambda x: x[1].overall_score, reverse=True
        )

        for rank, (name, result) in enumerate(sorted_results, 1):
            print(f"     #{rank} {name}:")
            print(f"         Overall Score: {result.overall_score:.3f}")
            print(f"         Ranking Score: {result.ranking_score:.1f}/100")
            print(
                f"         Mean Accuracy: {result.mean_accuracy:.3f} ± {result.std_accuracy:.3f}"
            )
            if result.processing_times:
                avg_time = np.mean(result.processing_times) * 1000
                print(f"         Avg Processing Time: {avg_time:.2f} ms")
            print(f"         Test Duration: {result.test_duration:.2f}s")

        # Generate detailed report for best algorithm
        best_algo_name = sorted_results[0][0]
        benchmark.generate_detailed_report(
            best_algo_name, f"detailed_report_{best_algo_name}.json"
        )
        print(f"\n✓ Detailed report generated for best algorithm: {best_algo_name}")

    else:
        print("⚠️  No algorithms available for benchmarking")


def demonstrate_realtime_monitoring():
    """Demonstrate real-time monitoring and adaptation."""

    print("\n" + "-" * 60)
    print("4. REAL-TIME MONITORING")
    print("-" * 60)

    from gianna.audio.vad import (
        DatasetGenerator,
        EnvironmentType,
        MonitoringConfig,
        RealtimeMonitor,
        create_vad,
    )

    # Create VAD and monitor
    vad = create_vad("energy", threshold=0.03)

    monitor_config = MonitoringConfig(
        monitoring_interval=0.1,  # Fast monitoring for demo
        enable_auto_adaptation=True,
        min_accuracy_threshold=0.8,
        max_processing_time=0.02,  # 20ms max
    )

    monitor = RealtimeMonitor(vad, monitor_config)
    print("✓ Created real-time monitor")

    # Generate test data with changing conditions
    clean_data = DatasetGenerator.generate_clean_speech_dataset(
        duration=10.0, speech_ratio=0.6
    )
    noisy_data = DatasetGenerator.generate_noisy_dataset(clean_data, noise_level=0.3)

    print("✓ Generated test datasets with varying conditions")

    # Start monitoring
    print("\n  📡 Starting real-time monitoring...")

    if not monitor.start_monitoring():
        print("❌ Failed to start monitoring")
        return

    try:
        # Simulate real-time processing
        print("     Processing clean audio...")

        # Process clean data
        for i, chunk in enumerate(clean_data.audio_chunks[:50]):  # Limited for demo
            result = vad.detect_activity(chunk)
            ground_truth = clean_data.labels[i] if i < len(clean_data.labels) else None
            monitor.record_result(result, ground_truth)

            time.sleep(0.01)  # Simulate real-time processing

        # Get first snapshot
        snapshot1 = monitor.get_current_snapshot()
        print(
            f"     Clean audio metrics: Accuracy={snapshot1.accuracy:.3f if snapshot1.accuracy else 'N/A'}, "
            f"Avg Time={snapshot1.avg_processing_time*1000:.2f}ms"
        )

        print("     Processing noisy audio (should trigger adaptation)...")

        # Process noisy data (should trigger adaptation)
        for i, chunk in enumerate(noisy_data.audio_chunks[:50]):  # Limited for demo
            result = vad.detect_activity(chunk)
            ground_truth = noisy_data.labels[i] if i < len(noisy_data.labels) else None
            monitor.record_result(result, ground_truth)

            time.sleep(0.01)  # Simulate real-time processing

        # Allow time for monitoring to process
        time.sleep(0.5)

        # Get second snapshot
        snapshot2 = monitor.get_current_snapshot()
        print(
            f"     Noisy audio metrics: Accuracy={snapshot2.accuracy:.3f if snapshot2.accuracy else 'N/A'}, "
            f"Avg Time={snapshot2.avg_processing_time*1000:.2f}ms"
        )

        # Check for adaptations
        adaptations = len(monitor.adaptation_history)
        if adaptations > 0:
            print(f"     ✓ {adaptations} adaptations performed")
            latest = monitor.adaptation_history[-1]
            print(
                f"       Latest: {latest['trigger']} -> threshold {latest['new_threshold']:.4f}"
            )
        else:
            print("     ⚠️  No adaptations triggered (may need longer runtime)")

        # Get performance history
        history = monitor.get_performance_history(hours=1.0)
        print(f"     📊 {len(history)} performance snapshots collected")

        # Export monitoring data
        monitor.export_monitoring_data("realtime_monitoring_data.json")
        print("✓ Monitoring data exported")

    finally:
        monitor.stop_monitoring()
        print("✓ Monitoring stopped")


def demonstrate_production_setup():
    """Demonstrate production-ready VAD setup."""

    print("\n" + "-" * 60)
    print("5. PRODUCTION VAD SETUP")
    print("-" * 60)

    from gianna.audio.vad import create_monitored_vad, create_production_vad

    # Create production VAD for office environment
    print("  🏢 Setting up VAD for office environment...")

    vad, monitor, calibrator = create_production_vad(
        algorithm="energy", environment="office", sample_rate=16000, chunk_size=1024
    )

    print(f"✓ Created production VAD:")
    print(f"   Algorithm: {vad.algorithm.algorithm_id}")
    print(f"   Threshold: {vad.config.threshold}")
    print(f"   Sample Rate: {vad.config.sample_rate} Hz")
    print(f"   Chunk Size: {vad.config.chunk_size}")
    print(f"   Monitoring: {'Enabled' if monitor else 'Disabled'}")
    print(f"   Calibration: {'Available' if calibrator else 'Not Available'}")

    # Demonstrate different environment setups
    environments = ["studio", "noisy", "car"]

    print(f"\n  🌍 Environment-specific configurations:")

    for env in environments:
        try:
            env_vad, env_monitor, env_calibrator = create_production_vad(
                algorithm="energy", environment=env
            )

            print(
                f"     {env.capitalize()}: threshold={env_vad.config.threshold:.3f}, "
                f"min_silence={env_vad.config.min_silence_duration:.1f}s"
            )

            # Cleanup
            if env_monitor:
                env_monitor.stop_monitoring()

        except Exception as e:
            print(f"     {env.capitalize()}: Error - {e}")

    # Demonstrate monitored VAD
    print(f"\n  📊 Creating monitored VAD with custom settings...")

    monitored_vad, monitor_instance = create_monitored_vad(
        algorithm="energy",
        enable_calibration=True,
        enable_realtime_monitor=True,
        threshold=0.02,
        monitoring_interval=1.0,
        monitoring_min_accuracy_threshold=0.75,
    )

    if monitor_instance:
        print("✓ Monitored VAD created with real-time adaptation")

        # Test a few samples
        print("  Testing monitored VAD with sample audio...")

        # Generate quick test
        from gianna.audio.vad import DatasetGenerator

        test_data = DatasetGenerator.generate_clean_speech_dataset(duration=5.0)

        monitor_instance.start_monitoring()

        try:
            for i, chunk in enumerate(test_data.audio_chunks[:20]):
                result = monitored_vad.detect_activity(chunk)
                ground_truth = (
                    test_data.labels[i] if i < len(test_data.labels) else None
                )
                monitor_instance.record_result(result, ground_truth)

                time.sleep(0.01)

            # Get final snapshot
            final_snapshot = monitor_instance.get_current_snapshot()
            print(
                f"  Final metrics: Samples={final_snapshot.total_samples}, "
                f"Speech Ratio={final_snapshot.speech_ratio:.2f}"
            )

        finally:
            monitor_instance.stop_monitoring()

    else:
        print("⚠️  Real-time monitoring not available (missing dependencies)")

    # Cleanup main monitor
    if monitor:
        monitor.stop_monitoring()

    print("✓ Production setup demonstration completed")


def print_summary():
    """Print a summary of created files and capabilities."""

    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)

    print("\n📁 Files Created:")
    files = [
        "vad_calibration_profile.json",
        "vad_metrics_report.json",
        "vad_performance_plots.png",
        "detailed_report_*.json",
        "realtime_monitoring_data.json",
    ]

    for filename in files:
        filepath = Path(filename.replace("*", "best_algorithm"))
        if filepath.exists() or any(Path(".").glob(filename)):
            print(f"  ✓ {filename}")
        else:
            print(f"  - {filename} (not created)")

    print(f"\n🚀 Key Capabilities Demonstrated:")
    capabilities = [
        "Automatic parameter optimization with multiple algorithms",
        "Environment-specific calibration and adaptation",
        "Real-time performance monitoring and alerting",
        "Statistical analysis with confidence intervals",
        "Algorithm benchmarking and comparison",
        "ROC curves and precision-recall analysis",
        "Production-ready deployment configurations",
        "Noise floor estimation and SNR analysis",
        "Multi-environment support and detection",
    ]

    for capability in capabilities:
        print(f"  ✓ {capability}")

    print(f"\n💡 Next Steps:")
    next_steps = [
        "Integrate with real audio streams in your application",
        "Set up continuous monitoring in production environments",
        "Use the calibration system to optimize for your specific audio conditions",
        "Implement custom environment detection based on your use case",
        "Set up alerting for performance degradation",
        "Use benchmarking to select the best algorithm for your needs",
    ]

    for step in next_steps:
        print(f"  → {step}")


if __name__ == "__main__":
    success = main()

    if success:
        print_summary()
    else:
        print("\n❌ Demonstration failed. Check the error messages above.")
        exit(1)
