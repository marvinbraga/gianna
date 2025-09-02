# VAD Calibration and Metrics System

A comprehensive calibration, metrics, and monitoring system for Voice Activity Detection (VAD) algorithms, providing automatic parameter optimization, real-time performance tracking, and statistical analysis capabilities.

## 🚀 Features

### Core Capabilities
- **Automatic Parameter Optimization**: Multi-algorithm optimization with environment-specific tuning
- **Real-time Performance Monitoring**: Continuous tracking with automatic adaptation
- **Comprehensive Benchmarking**: Algorithm comparison with statistical validation
- **Environment Detection**: Automatic audio environment classification and adaptation
- **Statistical Analysis**: ROC curves, confidence intervals, and significance testing
- **Production-ready Deployment**: Integrated monitoring and calibration for production use

### Advanced Features
- **Multi-environment Support**: Studio, office, noisy, car, airplane environments
- **Noise Floor Estimation**: Automatic background noise level detection
- **SNR Analysis**: Signal-to-noise ratio calculation and optimization
- **Alert System**: Performance degradation detection with customizable thresholds
- **Ground Truth Validation**: Algorithm validation against labeled datasets
- **Export Capabilities**: JSON reports, performance plots, and calibration profiles

## 📦 Components

### VadCalibrator
Automatic parameter optimization system with multiple optimization algorithms:

```python
from gianna.audio.vad import VadCalibrator, CalibrationConfig, GroundTruthData

# Create calibrator with custom configuration
config = CalibrationConfig(
    optimization_method=OptimizationMethod.DIFFERENTIAL_EVOLUTION,
    max_iterations=100,
    validation_split=0.2
)
calibrator = VadCalibrator(config)

# Calibrate VAD with ground truth data
result = calibrator.calibrate_vad(vad_instance, ground_truth_data)
print(f"Optimal threshold: {result.optimal_threshold}")
print(f"F1-Score: {result.f1_score}")
```

### VadMetrics
Performance monitoring and evaluation system:

```python
from gianna.audio.vad import VadMetrics

# Create metrics system
metrics = VadMetrics(enable_monitoring=True)

# Record VAD results
for audio_chunk, true_label in test_data:
    result = vad.detect_activity(audio_chunk)
    metrics.record_result(result, true_label)

# Generate comprehensive report
report = metrics.evaluate_performance()
print(f"Accuracy: {report.accuracy:.3f}")
print(f"AUC: {report.auc_score:.3f}")
```

### VadBenchmark
Algorithm comparison and benchmarking suite:

```python
from gianna.audio.vad import VadBenchmark, BenchmarkConfig, BenchmarkCategory

# Configure benchmarking
config = BenchmarkConfig(
    categories=[BenchmarkCategory.ACCURACY, BenchmarkCategory.PERFORMANCE],
    num_trials=5
)
benchmark = VadBenchmark(config)

# Compare algorithms
algorithms = {
    'energy': create_vad('energy'),
    'spectral': create_vad('spectral'),
    'webrtc': create_vad('webrtc')
}
results = benchmark.compare_algorithms(algorithms)
```

### RealtimeMonitor
Continuous monitoring with automatic adaptation:

```python
from gianna.audio.vad import RealtimeMonitor, MonitoringConfig

# Create monitor
config = MonitoringConfig(
    enable_auto_adaptation=True,
    min_accuracy_threshold=0.8,
    adaptation_cooldown=60.0
)
monitor = RealtimeMonitor(vad_instance, config)

# Start monitoring
with monitor:
    # Process audio stream
    for audio_chunk in audio_stream:
        result = vad.detect_activity(audio_chunk)
        monitor.record_result(result, ground_truth)

        # Monitor automatically adapts parameters when needed
```

## 🎯 Quick Start

### Basic Usage

```python
from gianna.audio.vad import create_production_vad

# Create production-ready VAD with monitoring
vad, monitor, calibrator = create_production_vad(
    algorithm="energy",
    environment="office",  # Optimized for office environment
    sample_rate=16000
)

# Process audio with automatic monitoring
with monitor:
    for audio_chunk in audio_stream:
        result = vad.detect_activity(audio_chunk)
        # Monitor tracks performance and adapts automatically
```

### Advanced Calibration

```python
from gianna.audio.vad import (
    VadCalibrator, GroundTruthData, EnvironmentType,
    CalibrationConfig, OptimizationMethod
)

# Create calibrator with advanced settings
config = CalibrationConfig(
    optimization_method=OptimizationMethod.BAYESIAN,
    max_iterations=200,
    enable_realtime_adaptation=True,
    enable_environment_detection=True
)
calibrator = VadCalibrator(config)

# Prepare ground truth data
ground_truth = GroundTruthData(
    audio_chunks=your_audio_chunks,
    labels=your_true_labels,  # True for speech, False for silence
    environment=EnvironmentType.NOISY_OFFICE
)

# Perform comprehensive calibration
result = calibrator.calibrate_vad(vad, ground_truth)

# Apply optimized settings
vad.update_config(result.optimal_config)

# Save calibration profile for later use
calibrator.save_calibration_profile("production_profile.json")
```

## 📊 Environment Types and Optimization

The system supports automatic detection and optimization for various audio environments:

| Environment | Characteristics | Typical Threshold | Use Cases |
|-------------|-----------------|-------------------|-----------|
| `STUDIO` | Very quiet, controlled | 0.010-0.015 | Recording studios, professional audio |
| `QUIET_OFFICE` | Low background noise | 0.015-0.025 | Home office, quiet rooms |
| `HOME_OFFICE` | Moderate background | 0.020-0.030 | Typical home environment |
| `NOISY_OFFICE` | High background noise | 0.030-0.050 | Open office, air conditioning |
| `COFFEE_SHOP` | Variable noise levels | 0.040-0.060 | Public spaces, cafes |
| `CAR` | Road noise, engine | 0.050-0.080 | Vehicle-based applications |
| `AIRPLANE` | Constant engine noise | 0.060-0.100 | Air travel environments |

## 🔧 Configuration Options

### CalibrationConfig
```python
config = CalibrationConfig(
    # Optimization method
    optimization_method=OptimizationMethod.DIFFERENTIAL_EVOLUTION,
    max_iterations=100,
    tolerance=1e-6,

    # Parameter ranges
    threshold_range=(0.001, 0.1),
    min_speech_range=(0.05, 0.5),
    min_silence_range=(0.1, 2.0),

    # Validation settings
    validation_split=0.2,
    cross_validation_folds=5,
    bootstrap_samples=1000,

    # Performance weighting
    accuracy_weight=0.3,
    precision_weight=0.25,
    recall_weight=0.25,
    f1_weight=0.2,

    # Real-time adaptation
    enable_realtime_adaptation=True,
    adaptation_threshold=0.05,

    # Environment detection
    enable_environment_detection=True,
    noise_analysis_enabled=True
)
```

### MonitoringConfig
```python
config = MonitoringConfig(
    # Monitoring intervals
    monitoring_interval=1.0,  # seconds
    calibration_interval=300.0,  # 5 minutes

    # Performance thresholds
    min_accuracy_threshold=0.7,
    max_processing_time=0.05,  # 50ms

    # Adaptation settings
    enable_auto_adaptation=True,
    adaptation_cooldown=60.0,

    # Environment detection
    enable_environment_detection=True,
    environment_change_threshold=0.3,

    # Alerting
    enable_email_alerts=False,
    alert_cooldown=300.0
)
```

## 📈 Performance Metrics

The system provides comprehensive performance analysis:

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: Speech detection accuracy
- **Recall**: Speech detection completeness
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: Silence detection accuracy
- **AUC Score**: Area under ROC curve

### Operational Metrics
- **Processing Time**: Average, max, min processing time
- **Memory Usage**: Memory consumption tracking
- **CPU Usage**: Processor utilization
- **Real-time Ratio**: Real-time performance compliance

### Quality Metrics
- **Confidence Levels**: Prediction confidence analysis
- **Energy Levels**: Signal energy statistics
- **SNR Analysis**: Signal-to-noise ratio calculation
- **Noise Floor**: Background noise level estimation

## 🔄 Real-time Adaptation

The system automatically adapts to changing conditions:

### Adaptation Triggers
- **Performance Degradation**: Accuracy drops below threshold
- **Environment Change**: Different acoustic characteristics detected
- **Processing Issues**: Excessive processing time or resource usage
- **Manual Request**: Explicit calibration request
- **Scheduled**: Regular recalibration intervals

### Adaptation Process
1. **Detection**: Monitor performance metrics continuously
2. **Analysis**: Identify performance issues or environment changes
3. **Optimization**: Run parameter optimization on recent data
4. **Validation**: Test new parameters against validation set
5. **Application**: Apply optimized parameters if improvement is significant
6. **Monitoring**: Continue monitoring with new parameters

## 📋 Benchmarking System

Comprehensive algorithm comparison and validation:

### Benchmark Categories
- **Accuracy**: Classification performance metrics
- **Performance**: Processing speed and efficiency
- **Robustness**: Performance under noise and varying conditions
- **Efficiency**: Memory and CPU usage
- **Real-time**: Real-time processing capability
- **Environment Adaptation**: Multi-environment performance

### Dataset Types
- **Clean Speech**: High-quality audio recordings
- **Noisy Office**: Office environment with background noise
- **Conversation**: Multi-speaker dialogue
- **Mixed Environment**: Various acoustic conditions
- **Synthetic**: Generated test datasets
- **Custom**: User-provided datasets

## 🛡️ Production Deployment

### Production Checklist
- [ ] Environment-specific calibration completed
- [ ] Performance thresholds configured
- [ ] Monitoring and alerting set up
- [ ] Fallback mechanisms implemented
- [ ] Resource limits configured
- [ ] Logging and metrics collection enabled

### Best Practices
1. **Calibration**: Perform initial calibration with representative data
2. **Monitoring**: Enable continuous performance monitoring
3. **Thresholds**: Set appropriate performance thresholds for your use case
4. **Adaptation**: Allow automatic adaptation but monitor changes
5. **Validation**: Regularly validate performance with ground truth data
6. **Backup**: Maintain backup configurations for critical applications

## 🔧 Dependencies

### Required
- `numpy`: Numerical computations
- `scipy`: Scientific computing and optimization
- `scikit-learn`: Machine learning metrics and analysis

### Optional
- `matplotlib`: Performance plotting and visualization
- `seaborn`: Enhanced statistical plotting
- `torch`: For Silero VAD algorithm
- `webrtcvad`: For WebRTC VAD algorithm

Install all dependencies:
```bash
pip install numpy scipy scikit-learn matplotlib seaborn
```

## 📚 Examples

### Complete Example
```python
#!/usr/bin/env python3
import numpy as np
from gianna.audio.vad import (
    create_vad, VadCalibrator, VadMetrics, RealtimeMonitor,
    GroundTruthData, CalibrationConfig, MonitoringConfig,
    EnvironmentType, OptimizationMethod
)

# Create VAD instance
vad = create_vad("energy", threshold=0.03)

# Create calibration system
cal_config = CalibrationConfig(
    optimization_method=OptimizationMethod.DIFFERENTIAL_EVOLUTION,
    max_iterations=50,
    enable_realtime_adaptation=True
)
calibrator = VadCalibrator(cal_config)

# Create monitoring system
mon_config = MonitoringConfig(
    enable_auto_adaptation=True,
    min_accuracy_threshold=0.8
)
monitor = RealtimeMonitor(vad, mon_config)

# Create metrics system
metrics = VadMetrics(enable_monitoring=True)

# Prepare ground truth data (your audio and labels)
ground_truth = GroundTruthData(
    audio_chunks=your_audio_chunks,  # List of numpy arrays
    labels=your_labels,  # List of boolean values
    environment=EnvironmentType.OFFICE
)

# Perform initial calibration
print("Calibrating VAD...")
result = calibrator.calibrate_vad(vad, ground_truth)
print(f"Optimal threshold: {result.optimal_threshold:.4f}")
print(f"Expected accuracy: {result.accuracy:.3f}")

# Start real-time monitoring
with monitor:
    print("Processing audio stream...")

    # Process your audio stream
    for audio_chunk, true_label in your_audio_stream:
        # Detect voice activity
        vad_result = vad.detect_activity(audio_chunk)

        # Record for monitoring and metrics
        monitor.record_result(vad_result, true_label)
        metrics.record_result(vad_result, true_label)

        # Your application logic here
        if vad_result.is_voice_active:
            print(f"Voice detected! Confidence: {vad_result.confidence:.3f}")

# Generate final performance report
report = metrics.evaluate_performance()
print(f"\nFinal Performance:")
print(f"Accuracy: {report.accuracy:.3f}")
print(f"F1-Score: {report.f1_score:.3f}")
print(f"AUC: {report.auc_score:.3f}")

# Export results
calibrator.save_calibration_profile("production_calibration.json")
metrics.export_metrics_report("performance_metrics.json")
monitor.export_monitoring_data("monitoring_log.json")
```

## 🚀 Getting Started

1. **Run the Demo**: Try the comprehensive demonstration:
   ```bash
   python examples/vad_calibration_demo.py
   ```

2. **Basic Integration**: Start with a simple monitored VAD:
   ```python
   from gianna.audio.vad import create_monitored_vad

   vad, monitor = create_monitored_vad("energy", enable_calibration=True)
   ```

3. **Production Setup**: Use the production factory:
   ```python
   from gianna.audio.vad import create_production_vad

   vad, monitor, calibrator = create_production_vad(
       algorithm="energy",
       environment="office"
   )
   ```

## 📖 API Reference

For detailed API documentation, see the docstrings in each module:
- `calibration.py`: VadCalibrator, EnvironmentDetector, CalibrationConfig
- `metrics.py`: VadMetrics, PerformanceMonitor, PerformanceReport
- `benchmark.py`: VadBenchmark, BenchmarkConfig, DatasetGenerator
- `realtime_monitor.py`: RealtimeMonitor, MonitoringConfig, MonitoringSnapshot

## 🤝 Contributing

The calibration and metrics system is designed to be extensible:

1. **Custom Optimization Methods**: Implement new optimization algorithms
2. **Environment Types**: Add support for new acoustic environments
3. **Metrics**: Contribute additional performance metrics
4. **Benchmarks**: Add new benchmark categories or datasets

## 📄 License

This calibration and metrics system is part of the Gianna VAD module and follows the same license as the main project.
