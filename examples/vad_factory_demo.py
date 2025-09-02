#!/usr/bin/env python3
"""
VAD Factory Pattern Demo

This example demonstrates the unified factory pattern implementation for
Voice Activity Detection in the Gianna assistant framework.

Features demonstrated:
- Creating VAD instances with different algorithms
- Using configuration presets and custom configurations
- Integrating VAD with streaming voice pipeline
- Backward compatibility with legacy interfaces
- Error handling and fallback mechanisms
"""

import asyncio
import logging
import time
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_basic_vad_usage():
    """Demonstrate basic VAD factory usage."""
    print("=" * 60)
    print("1. Basic VAD Factory Usage")
    print("=" * 60)

    from gianna.audio.vad import create_vad, get_available_algorithms

    # Show available algorithms
    algorithms = get_available_algorithms()
    print(f"Available VAD algorithms: {algorithms}")

    # Create basic energy VAD
    vad = create_vad("energy")
    print(f"Created basic VAD: {vad}")

    # Create VAD with custom parameters
    vad = create_vad("energy", threshold=0.03, min_silence_duration=1.5)
    print(f"Created custom VAD: threshold={vad.config.threshold}")

    print()


def demo_advanced_vad_configuration():
    """Demonstrate advanced VAD configuration."""
    print("=" * 60)
    print("2. Advanced VAD Configuration")
    print("=" * 60)

    from gianna.audio.vad import create_realtime_vad, create_vad, create_vad_config

    # Create configuration presets
    for preset in ["fast", "balanced", "accurate"]:
        config = create_vad_config("energy", preset=preset)
        vad = create_vad("energy", config=config)
        print(
            f"{preset:>8} preset: threshold={config.threshold}, silence={config.min_silence_duration}s"
        )

    # Create realtime-optimized VAD
    realtime_vad = create_realtime_vad("energy", latency_mode="low")
    print(f"Realtime VAD: chunk_size={realtime_vad.config.chunk_size}")

    print()


def demo_vad_processing():
    """Demonstrate VAD processing with synthetic audio."""
    print("=" * 60)
    print("3. VAD Processing Demo")
    print("=" * 60)

    from gianna.audio.vad import create_vad

    # Create and initialize VAD
    vad = create_vad("energy", threshold=0.02)
    vad.initialize()

    print(f"VAD initialized: {vad}")
    print("Processing synthetic audio samples...")

    # Parameters
    sample_rate = 16000
    chunk_duration = 0.05  # 50ms chunks
    samples_per_chunk = int(sample_rate * chunk_duration)

    # Simulate different audio scenarios
    scenarios = [
        ("Silence", np.random.normal(0, 50, samples_per_chunk)),
        ("Background noise", np.random.normal(0, 200, samples_per_chunk)),
        ("Quiet speech", np.random.normal(0, 1000, samples_per_chunk)),
        ("Normal speech", np.random.normal(0, 3000, samples_per_chunk)),
        ("Loud speech", np.random.normal(0, 8000, samples_per_chunk)),
    ]

    for name, audio_data in scenarios:
        audio_chunk = audio_data.astype(np.int16)
        result = vad.process_stream(audio_chunk)

        print(
            f"{name:>15}: voice_active={result.is_voice_active:>5}, energy={result.energy_level:.4f}, confidence={result.confidence:.3f}"
        )

    # Show statistics
    stats = vad.statistics
    print(f"\nProcessing statistics:")
    print(f"  Total chunks: {stats.total_chunks_processed}")
    print(f"  Speech chunks: {stats.speech_chunks_detected}")
    print(f"  Detection ratio: {stats.speech_detection_ratio:.2%}")

    print()


def demo_streaming_integration():
    """Demonstrate integration with streaming voice pipeline."""
    print("=" * 60)
    print("4. Streaming Pipeline Integration")
    print("=" * 60)

    from gianna.audio.streaming import StreamingVoicePipeline
    from gianna.audio.vad import create_vad_config

    print("Creating streaming pipelines with different VAD configurations...")

    # Basic integration
    pipeline1 = StreamingVoicePipeline(
        model_name="gpt35",
        vad_algorithm="energy",
        vad_threshold=0.025,
    )
    print(f"Basic pipeline: VAD algorithm via parameter")

    # Advanced integration with config
    config = create_vad_config("energy", preset="balanced", threshold=0.03)
    pipeline2 = StreamingVoicePipeline(
        model_name="gpt35",
        vad_config=config,
    )
    print(f"Advanced pipeline: VAD via configuration object")

    # Show pipeline status
    status1 = pipeline1.get_pipeline_status()
    status2 = pipeline2.get_pipeline_status()

    print(f"Pipeline 1 VAD info: {status1.get('vad_info', {})}")
    print(f"Pipeline 2 VAD info: {status2.get('vad_info', {})}")

    print()


def demo_advanced_features():
    """Demonstrate advanced VAD features."""
    print("=" * 60)
    print("5. Advanced VAD Features")
    print("=" * 60)

    from gianna.audio.vad import (
        create_vad,
        create_vad_pipeline,
        get_algorithm_info,
        is_algorithm_available,
    )

    # Algorithm information
    print("Algorithm information:")
    for algorithm in ["energy", "spectral", "webrtc", "silero", "adaptive"]:
        available = is_algorithm_available(algorithm)
        info = get_algorithm_info(algorithm)
        print(
            f"  {algorithm:>8}: {'✅' if available else '❌'} {info.get('reason', 'Available')}"
        )

    # VAD pipeline with fallbacks
    print("\nCreating VAD with fallback algorithms...")
    try:
        vad = create_vad_pipeline(
            algorithms=["silero", "webrtc", "spectral", "energy"],
            fallback_algorithm="energy",
        )
        algorithm_name = (
            vad.algorithm.algorithm_id if hasattr(vad, "algorithm") else "legacy"
        )
        print(f"Created VAD pipeline: using {algorithm_name} algorithm")
    except Exception as e:
        print(f"VAD pipeline creation failed: {e}")

    print()


def demo_backward_compatibility():
    """Demonstrate backward compatibility features."""
    print("=" * 60)
    print("6. Backward Compatibility")
    print("=" * 60)

    import warnings

    # Capture deprecation warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test legacy imports and functions
        from gianna.audio.vad import (
            VoiceActivityDetector,
            create_vad_detector,
            get_vad_instance,
        )

        # Create VAD using legacy functions
        vad1 = create_vad_detector(threshold=0.02)
        vad2 = get_vad_instance("energy", threshold=0.03)

        print(f"Legacy VAD 1: {type(vad1).__name__}")
        print(f"Legacy VAD 2: {type(vad2).__name__}")

        # Show deprecation warnings
        if w:
            print(f"Deprecation warnings captured: {len(w)}")
            for warning in w:
                print(f"  Warning: {warning.message}")
        else:
            print("No deprecation warnings (functions may not be deprecated yet)")

    print()


def demo_error_handling():
    """Demonstrate error handling and recovery."""
    print("=" * 60)
    print("7. Error Handling and Recovery")
    print("=" * 60)

    from gianna.audio.vad import create_vad, create_vad_config

    # Test invalid algorithm
    try:
        vad = create_vad("nonexistent_algorithm")
    except ValueError as e:
        print(f"✅ Correctly handled invalid algorithm: {e}")

    # Test invalid configuration
    try:
        config = create_vad_config("energy", threshold=2.0)  # Invalid threshold
        vad = create_vad("energy", config=config)
    except (ValueError, RuntimeError) as e:
        print(f"✅ Correctly handled invalid config: {e}")

    # Test recovery with valid parameters
    try:
        vad = create_vad("energy", threshold=0.02)
        print(f"✅ Successfully created VAD after error recovery")
    except Exception as e:
        print(f"❌ Failed to recover: {e}")

    print()


async def demo_async_usage():
    """Demonstrate asynchronous VAD usage with voice assistant."""
    print("=" * 60)
    print("8. Async Voice Assistant Demo")
    print("=" * 60)

    from gianna.audio import create_streaming_voice_assistant, create_voice_assistant

    print("Creating voice assistants with different VAD configurations...")

    # Create basic voice assistant
    assistant1 = await create_voice_assistant(
        model_name="gpt35", vad_algorithm="energy", vad_preset="balanced"
    )
    print(f"Created voice assistant: {type(assistant1).__name__}")

    # Create streaming-optimized assistant
    assistant2 = await create_streaming_voice_assistant(
        model_name="gpt35",
        vad_algorithm="energy",
        sample_rate=16000,
        chunk_size=512,  # Smaller chunks for lower latency
    )
    print(f"Created streaming assistant: {type(assistant2).__name__}")

    # Show configuration differences
    status1 = assistant1.get_pipeline_status()
    status2 = assistant2.get_pipeline_status()

    print("Configuration comparison:")
    print(f"  Basic assistant VAD:     {status1.get('vad_info', {})}")
    print(f"  Streaming assistant VAD: {status2.get('vad_info', {})}")

    print()


async def main():
    """Run all demonstrations."""
    print("🎤 VAD Factory Pattern Demonstration")
    print("=" * 60)
    print("This demo showcases the unified VAD factory system in Gianna")
    print()

    # Run synchronous demos
    demo_basic_vad_usage()
    demo_advanced_vad_configuration()
    demo_vad_processing()
    demo_streaming_integration()
    demo_advanced_features()
    demo_backward_compatibility()
    demo_error_handling()

    # Run asynchronous demo
    await demo_async_usage()

    print("=" * 60)
    print("🎉 VAD Factory Pattern Demo Complete!")
    print("=" * 60)
    print("Key takeaways:")
    print("• Unified factory pattern provides consistent interface")
    print("• Multiple algorithms with graceful fallbacks")
    print("• Configuration presets for common use cases")
    print("• Seamless streaming pipeline integration")
    print("• Full backward compatibility maintained")
    print("• Comprehensive error handling and recovery")


if __name__ == "__main__":
    asyncio.run(main())
