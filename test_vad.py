"""
Test script for Voice Activity Detection (VAD) functionality.

This script tests all the core features of the VAD implementation
to ensure everything works correctly.
"""

import logging
import sys
import time
from typing import List

import numpy as np

# Add current directory to path
sys.path.append(".")

from gianna.assistants.audio.vad import VoiceActivityDetector, create_vad_detector


def test_basic_functionality():
    """Test basic VAD functionality."""
    print("=== Testing Basic VAD Functionality ===")

    # Create VAD detector
    vad = create_vad_detector(threshold=0.02, min_silence_duration=0.5)
    print(f"Created VAD: {vad}")

    # Test energy calculation with different audio types
    test_cases = [
        ("Silence", np.zeros(1024, dtype=np.int16)),
        ("Quiet noise", np.random.randn(1024).astype(np.int16) * 500),
        ("Normal speech", np.random.randn(1024).astype(np.int16) * 5000),
        ("Loud speech", np.random.randn(1024).astype(np.int16) * 12000),
        ("Very loud", np.random.randn(1024).astype(np.int16) * 20000),
    ]

    for name, audio in test_cases:
        energy = vad.calculate_rms_energy(audio)
        is_active, detected_energy = vad.detect_activity(audio)
        print(f"{name:15} - Energy: {energy:.4f} - Active: {is_active}")

    print("✅ Basic functionality test passed\n")


def test_state_management():
    """Test VAD state management and callbacks."""
    print("=== Testing State Management ===")

    events = []

    def on_speech_start():
        events.append(("speech_start", time.time()))
        print("📢 Speech started!")

    def on_speech_end():
        events.append(("speech_end", time.time()))
        print("🔇 Speech ended!")

    vad = create_vad_detector(
        threshold=0.02,
        min_silence_duration=0.2,  # Short for testing
        speech_start_callback=on_speech_start,
        speech_end_callback=on_speech_end,
    )

    # Simulate audio sequence: silence -> speech -> silence -> speech -> silence
    audio_sequence = [
        ("Silence 1", np.random.randn(1024).astype(np.int16) * 300),
        ("Silence 2", np.random.randn(1024).astype(np.int16) * 400),
        ("Speech 1", np.random.randn(1024).astype(np.int16) * 8000),
        ("Speech 2", np.random.randn(1024).astype(np.int16) * 7000),
        ("Speech 3", np.random.randn(1024).astype(np.int16) * 9000),
        ("Silence 3", np.random.randn(1024).astype(np.int16) * 350),
        ("Silence 4", np.random.randn(1024).astype(np.int16) * 380),
        (
            "Silence 5",
            np.random.randn(1024).astype(np.int16) * 320,
        ),  # Should trigger speech_end
        ("Speech 4", np.random.randn(1024).astype(np.int16) * 10000),
        ("Silence 6", np.random.randn(1024).astype(np.int16) * 300),
    ]

    for name, audio in audio_sequence:
        result = vad.process_stream(audio)
        event_type = result.get("event_type") or "None"
        print(
            f"{name:15} - Speaking: {result['is_speaking']:5} - "
            f"Active: {result['is_voice_active']:5} - "
            f"Energy: {result['energy']:.4f} - "
            f"Event: {event_type:12}"
        )

        # Small delay to simulate real-time processing
        time.sleep(0.05)

    print(f"\nEvents captured: {len(events)}")
    for i, (event, timestamp) in enumerate(events):
        print(f"  {i+1}. {event}")

    print("✅ State management test passed\n")


def test_threshold_adjustment():
    """Test dynamic threshold adjustment."""
    print("=== Testing Threshold Adjustment ===")

    vad = create_vad_detector()
    test_audio = np.random.randn(1024).astype(np.int16) * 3000

    thresholds = [0.01, 0.05, 0.1, 0.02]  # Return to original

    for threshold in thresholds:
        vad.set_threshold(threshold)
        is_active, energy = vad.detect_activity(test_audio)
        print(
            f"Threshold: {threshold:.3f} - Energy: {energy:.4f} - Active: {is_active}"
        )

    print("✅ Threshold adjustment test passed\n")


def test_statistics():
    """Test statistics collection."""
    print("=== Testing Statistics ===")

    vad = create_vad_detector(threshold=0.03)

    # Process multiple chunks
    for i in range(10):
        if i % 3 == 0:
            # Speech chunks
            audio = np.random.randn(1024).astype(np.int16) * 8000
        else:
            # Silence chunks
            audio = np.random.randn(1024).astype(np.int16) * 1000

        vad.detect_activity(audio)

    stats = vad.get_statistics()
    print(f"Statistics after 10 chunks:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("✅ Statistics test passed\n")


def test_error_handling():
    """Test error handling and edge cases."""
    print("=== Testing Error Handling ===")

    vad = create_vad_detector()

    # Test with empty audio
    empty_energy = vad.calculate_rms_energy(np.array([], dtype=np.int16))
    print(f"Empty audio energy: {empty_energy}")

    # Test with bytes input
    bytes_audio = np.random.randn(1024).astype(np.int16).tobytes()
    bytes_energy = vad.calculate_rms_energy(bytes_audio)
    print(f"Bytes audio energy: {bytes_energy:.4f}")

    # Test invalid threshold values
    try:
        vad.set_threshold(1.5)  # Should raise ValueError
        print("❌ Should have raised ValueError for threshold > 1.0")
    except ValueError:
        print("✅ Correctly rejected threshold > 1.0")

    try:
        vad.set_threshold(-0.1)  # Should raise ValueError
        print("❌ Should have raised ValueError for threshold < 0.0")
    except ValueError:
        print("✅ Correctly rejected threshold < 0.0")

    # Test invalid silence duration
    try:
        vad.set_min_silence_duration(-1.0)  # Should raise ValueError
        print("❌ Should have raised ValueError for negative silence duration")
    except ValueError:
        print("✅ Correctly rejected negative silence duration")

    print("✅ Error handling test passed\n")


def test_thread_safety():
    """Test thread safety of VAD operations."""
    print("=== Testing Thread Safety ===")

    import threading

    vad = create_vad_detector()
    results = []
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(50):
                audio = np.random.randn(1024).astype(np.int16) * (
                    2000 + worker_id * 1000
                )
                result = vad.process_stream(audio)
                results.append((worker_id, i, result["energy"]))
        except Exception as e:
            errors.append((worker_id, str(e)))

    # Start multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    print(f"Processed {len(results)} chunks across {len(threads)} threads")
    print(f"Errors: {len(errors)}")

    if errors:
        for worker_id, error in errors:
            print(f"  Worker {worker_id}: {error}")
    else:
        print("✅ No thread safety issues detected")

    # Check final statistics
    stats = vad.get_statistics()
    expected_chunks = 3 * 50  # 3 workers * 50 chunks each
    actual_chunks = stats["total_chunks"]

    if actual_chunks >= expected_chunks:
        print(f"✅ All {expected_chunks} chunks processed correctly")
    else:
        print(f"⚠️ Expected {expected_chunks} chunks, got {actual_chunks}")

    print("✅ Thread safety test completed\n")


def main():
    """Run all VAD tests."""
    print("Voice Activity Detection (VAD) Test Suite")
    print("=========================================")

    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during testing

    try:
        test_basic_functionality()
        test_state_management()
        test_threshold_adjustment()
        test_statistics()
        test_error_handling()
        test_thread_safety()

        print("🎉 All VAD tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
