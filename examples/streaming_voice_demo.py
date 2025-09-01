#!/usr/bin/env python3
"""
Streaming Voice Assistant Demo

This example demonstrates how to use the StreamingVoicePipeline
for real-time voice conversations with the Gianna assistant.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to path - must be before other imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gianna.audio.streaming import PipelineState, StreamingVoicePipeline  # noqa: E402


class VoiceAssistantDemo:
    """Demo application for streaming voice assistant."""

    def __init__(self):
        self.pipeline = None
        self.running = True

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            print(f"\n📡 Received signal {signum}, shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def create_callbacks(self):
        """Create callback functions for pipeline events."""

        def on_listening():
            print("👂 Listening for speech...")

        def on_speech_detected(transcript):
            print(f"🎤 You said: '{transcript}'")

        def on_processing():
            print("🧠 Processing your request...")

        def on_response(response):
            print(f"💬 Assistant: '{response}'")

        def on_speaking():
            print("🔊 Speaking response...")

        def on_error(error):
            print(f"❌ Error occurred: {error}")
            # Optionally restart pipeline or handle specific errors

        return {
            "on_listening": on_listening,
            "on_speech_detected": on_speech_detected,
            "on_processing": on_processing,
            "on_response": on_response,
            "on_speaking": on_speaking,
            "on_error": on_error,
        }

    async def run_interactive_demo(self):
        """Run interactive voice assistant demo."""

        print("🎯 Gianna Streaming Voice Assistant Demo")
        print("=" * 50)

        # Create callbacks
        callbacks = self.create_callbacks()

        # Create pipeline with configuration
        self.pipeline = StreamingVoicePipeline(
            # LLM Configuration
            model_name="gpt35",  # Use GPT-3.5 for faster responses
            system_prompt=(
                "You are Gianna, a helpful voice assistant. "
                "Keep your responses concise and conversational. "
                "Respond in a friendly, natural manner."
            ),
            # Audio Configuration
            sample_rate=16000,
            chunk_size=1024,
            # VAD Configuration (adjust based on your environment)
            vad_threshold=0.02,  # Lower = more sensitive
            min_silence_duration=1.5,  # Seconds to wait for speech end
            # TTS Configuration
            tts_type="google",
            tts_language="en",
            tts_voice="default",
            # Callbacks
            **callbacks,
        )

        try:
            print("🚀 Starting voice assistant...")
            print("💡 Speak naturally and wait for responses")
            print("💡 Press Ctrl+C to stop")
            print("-" * 50)

            # Start the pipeline
            await self.pipeline.start_listening()

            # Run until interrupted
            while self.running:
                # Check pipeline status periodically
                status = self.pipeline.get_pipeline_status()

                # Handle error states
                if status["state"] == PipelineState.ERROR.value:
                    print("⚠️  Pipeline encountered an error, restarting...")
                    await self.pipeline.stop_listening()
                    await asyncio.sleep(1)
                    await self.pipeline.start_listening()

                # Print statistics every 30 seconds
                if hasattr(self, "_last_stats_time"):
                    if asyncio.get_event_loop().time() - self._last_stats_time > 30:
                        self._print_statistics(status)
                        self._last_stats_time = asyncio.get_event_loop().time()
                else:
                    self._last_stats_time = asyncio.get_event_loop().time()

                await asyncio.sleep(0.1)  # Small sleep to prevent busy waiting

        except Exception as e:
            print(f"❌ Demo error: {e}")
        finally:
            if self.pipeline:
                print("🛑 Stopping voice assistant...")
                await self.pipeline.stop_listening()

    def _print_statistics(self, status):
        """Print pipeline statistics."""
        stats = status["stats"]
        print("\n📊 Pipeline Statistics:")
        print(f"   Chunks processed: {stats['chunks_processed']}")
        print(f"   Speech detections: {stats['speech_detected_count']}")
        print(f"   LLM requests: {stats['llm_requests']}")
        print(f"   TTS synthesized: {stats['tts_synthesized']}")
        print(f"   Current state: {status['state']}")
        print("-" * 50)

    async def run_text_demo(self):
        """Run text-only demo (for testing without microphone)."""

        print("📝 Text-only Demo Mode")
        print("=" * 50)

        # Create pipeline without audio callbacks
        self.pipeline = StreamingVoicePipeline(
            model_name="gpt35",
            system_prompt="You are Gianna, a helpful assistant.",
            on_speech_detected=lambda x: print(f"🎤 Input: {x}"),
            on_response=lambda x: print(f"💬 Response: {x}"),
            on_processing=lambda: print("🧠 Processing..."),
        )

        try:
            print("💡 Type messages to chat with Gianna")
            print("💡 Type 'quit' to exit")
            print("-" * 50)

            while self.running:
                try:
                    user_input = input("\nYou: ").strip()

                    if user_input.lower() in ["quit", "exit", "bye"]:
                        break

                    if user_input:
                        response = await self.pipeline.process_text_input(user_input)
                        if not response:
                            print("❌ No response received")

                except (EOFError, KeyboardInterrupt):
                    break

        except Exception as e:
            print(f"❌ Demo error: {e}")


async def main():
    """Main demo function."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Reduce some noisy loggers
    logging.getLogger("pyaudio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    demo = VoiceAssistantDemo()
    demo.setup_signal_handlers()

    # Check if we should run text-only mode
    if len(sys.argv) > 1 and sys.argv[1] == "--text":
        await demo.run_text_demo()
    else:
        await demo.run_interactive_demo()

    print("👋 Demo finished")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
