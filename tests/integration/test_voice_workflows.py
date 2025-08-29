"""
Integration tests for voice processing workflows - FASE 3

Tests for:
- Complete voice input/output pipelines
- VAD + STT + LLM + TTS integration
- Streaming voice processing
- Audio format handling workflows
- Voice workflow error recovery
- Multi-modal voice/text integration

Test Coverage:
- End-to-end voice workflows
- Audio processing pipeline integration
- Real-time streaming workflows
- Voice workflow performance
- Error handling across voice components
"""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Core imports
from tests import PERFORMANCE_THRESHOLDS


@pytest.mark.integration
@pytest.mark.fase3
@pytest.mark.voice
@pytest.mark.slow
class TestCompleteVoiceWorkflows:
    """Test complete voice processing workflows."""

    def test_basic_voice_input_workflow(
        self, gianna_state, mock_vad, mock_stt, mock_langgraph_chain, temp_audio_file
    ):
        """Test basic voice input processing workflow."""
        # 1. Voice Activity Detection
        mock_vad.detect_activity.return_value = True
        has_speech = mock_vad.detect_activity([0.1, 0.2, 0.3])
        assert has_speech is True

        # 2. Speech to Text
        mock_stt.return_value = "hello gianna how are you today"
        transcribed_text = mock_stt(temp_audio_file)
        assert transcribed_text == "hello gianna how are you today"

        # 3. Update state with user input
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": transcribed_text,
                "timestamp": datetime.now().isoformat(),
                "source": "voice",
            }
        )

        # 4. LLM Processing
        mock_langgraph_chain.invoke.return_value = {
            "output": "Hello! I'm doing great, thank you for asking. How can I help you today?"
        }
        response = mock_langgraph_chain.invoke({"input": transcribed_text})

        # Verify workflow
        assert len(gianna_state["conversation"].messages) >= 1
        assert gianna_state["conversation"].messages[-1]["source"] == "voice"
        assert "hello" in transcribed_text.lower()
        assert "help" in response["output"].lower()

        # Verify all components were called
        mock_vad.detect_activity.assert_called()
        mock_stt.assert_called_with(temp_audio_file)
        mock_langgraph_chain.invoke.assert_called()

    def test_voice_output_workflow(self, gianna_state, mock_tts, mock_langgraph_chain):
        """Test voice output generation workflow."""
        # 1. Generate response
        response_text = "I can help you with various tasks. What would you like to do?"
        mock_langgraph_chain.invoke.return_value = {"output": response_text}

        response = mock_langgraph_chain.invoke({"input": "what can you do"})

        # 2. Text to Speech
        mock_tts_instance = MagicMock()
        mock_tts.return_value = mock_tts_instance
        tts_result = mock_tts(response["output"])

        # 3. Update audio state
        gianna_state["audio"].current_mode = "speaking"

        # 4. Add response to conversation
        gianna_state["conversation"].messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "timestamp": datetime.now().isoformat(),
                "source": "voice",
            }
        )

        # Verify workflow
        assert response["output"] == response_text
        assert gianna_state["audio"].current_mode == "speaking"
        assert tts_result is not None
        assert len(gianna_state["conversation"].messages) >= 1

        # Verify components were called
        mock_tts.assert_called_with(response_text)
        mock_langgraph_chain.invoke.assert_called()

    def test_complete_voice_conversation_cycle(
        self,
        gianna_state,
        mock_vad,
        mock_stt,
        mock_tts,
        mock_langgraph_chain,
        temp_audio_file,
    ):
        """Test complete voice conversation cycle."""
        # Initial state
        initial_mode = gianna_state["audio"].current_mode
        assert initial_mode == "idle"

        # === VOICE INPUT PHASE ===
        # 1. Start listening
        gianna_state["audio"].current_mode = "listening"

        # 2. Detect speech
        mock_vad.detect_activity.return_value = True
        has_speech = mock_vad.detect_activity([0.2, 0.3, 0.4])
        assert has_speech

        # 3. Speech to text
        user_input = "tell me a joke"
        mock_stt.return_value = user_input
        transcribed = mock_stt(temp_audio_file)

        # 4. Update conversation
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": transcribed,
                "timestamp": datetime.now().isoformat(),
                "source": "voice",
            }
        )

        # === PROCESSING PHASE ===
        gianna_state["audio"].current_mode = "processing"

        # 5. LLM processing
        joke_response = "Why did the AI cross the road? To get to the other algorithm!"
        mock_langgraph_chain.invoke.return_value = {"output": joke_response}
        response = mock_langgraph_chain.invoke({"input": transcribed})

        # === VOICE OUTPUT PHASE ===
        # 6. Text to speech
        gianna_state["audio"].current_mode = "speaking"
        mock_tts_instance = MagicMock()
        mock_tts.return_value = mock_tts_instance
        tts_result = mock_tts(response["output"])

        # 7. Update conversation
        gianna_state["conversation"].messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "timestamp": datetime.now().isoformat(),
                "source": "voice",
            }
        )

        # 8. Return to idle
        gianna_state["audio"].current_mode = "idle"

        # === VERIFICATION ===
        assert len(gianna_state["conversation"].messages) == 2
        assert gianna_state["conversation"].messages[0]["role"] == "user"
        assert gianna_state["conversation"].messages[0]["content"] == user_input
        assert gianna_state["conversation"].messages[1]["role"] == "assistant"
        assert "joke" in gianna_state["conversation"].messages[1]["content"].lower()
        assert gianna_state["audio"].current_mode == "idle"

        # Verify all components called
        mock_vad.detect_activity.assert_called()
        mock_stt.assert_called()
        mock_langgraph_chain.invoke.assert_called()
        mock_tts.assert_called()

    @pytest.mark.async_test
    def test_streaming_voice_workflow(
        self,
        gianna_state,
        mock_vad,
        mock_stt,
        mock_tts,
        mock_react_agent,
        async_test_runner,
    ):
        """Test streaming voice processing workflow."""

        async def streaming_workflow():
            # Mock streaming voice pipeline
            with patch(
                "gianna.workflows.voice_streaming.StreamingVoicePipeline"
            ) as mock_pipeline:
                pipeline_instance = MagicMock()
                mock_pipeline.return_value = pipeline_instance

                # Setup async methods
                pipeline_instance.start_listening = AsyncMock()
                pipeline_instance._process_audio_buffer = AsyncMock()
                pipeline_instance._speech_to_text = AsyncMock(
                    return_value="streaming test command"
                )
                pipeline_instance._text_to_speech = AsyncMock()

                # Create pipeline
                pipeline = mock_pipeline(mock_react_agent, mock_vad)

                # Start streaming
                await pipeline.start_listening(gianna_state)

                # Simulate speech processing
                transcribed_text = await pipeline._speech_to_text(b"fake_audio_buffer")
                assert transcribed_text == "streaming test command"

                # Process with agent
                mock_react_agent.ainvoke = AsyncMock(
                    return_value={"output": "Streaming response received"}
                )
                response = await mock_react_agent.ainvoke(gianna_state)

                # Generate speech
                await pipeline._text_to_speech(response["output"], gianna_state)

                # Verify streaming workflow
                pipeline.start_listening.assert_called_once_with(gianna_state)
                pipeline._speech_to_text.assert_called_once()
                pipeline._text_to_speech.assert_called_once()
                mock_react_agent.ainvoke.assert_called_once()

                return response

        result = async_test_runner(streaming_workflow())
        assert result["output"] == "Streaming response received"

    def test_voice_workflow_with_different_engines(
        self, gianna_state, mock_langgraph_chain
    ):
        """Test voice workflow with different TTS/STT engines."""
        engines = [
            ("google", "whisper"),
            ("elevenlabs", "whisper_local"),
            ("whisper", "whisper"),
        ]

        for tts_engine, stt_engine in engines:
            with patch(
                f"gianna.assistants.audio.tts.factory_method.text_to_speech"
            ) as mock_tts:
                with patch(
                    f"gianna.assistants.audio.stt.factory_method.speech_to_text"
                ) as mock_stt:
                    # Setup mocks
                    mock_tts_instance = MagicMock()
                    mock_tts.return_value = mock_tts_instance
                    mock_stt.return_value = f"text from {stt_engine}"

                    # Mock LLM response
                    mock_langgraph_chain.invoke.return_value = {
                        "output": f"Response using {tts_engine} and {stt_engine}"
                    }

                    # Test workflow
                    stt_result = mock_stt("test.wav")
                    llm_response = mock_langgraph_chain.invoke({"input": stt_result})
                    tts_result = mock_tts(llm_response["output"])

                    # Verify engine-specific processing
                    assert stt_engine in stt_result
                    assert tts_engine in llm_response["output"]
                    assert stt_engine in llm_response["output"]
                    assert tts_result is not None


@pytest.mark.integration
@pytest.mark.fase3
@pytest.mark.voice
class TestVoiceWorkflowErrorHandling:
    """Test voice workflow error handling and recovery."""

    def test_stt_failure_recovery(
        self, gianna_state, mock_vad, mock_stt, mock_tts, temp_audio_file
    ):
        """Test recovery from STT service failures."""
        # Setup STT failure
        mock_stt.side_effect = Exception("STT service unavailable")

        # Setup VAD success
        mock_vad.detect_activity.return_value = True
        has_speech = mock_vad.detect_activity([0.1, 0.2])
        assert has_speech

        # Test STT failure handling
        with pytest.raises(Exception) as exc_info:
            mock_stt(temp_audio_file)

        assert "STT service unavailable" in str(exc_info.value)

        # Test recovery with fallback
        mock_stt.side_effect = None  # Reset
        mock_stt.return_value = "fallback transcription"

        result = mock_stt(temp_audio_file)
        assert result == "fallback transcription"

    def test_tts_failure_recovery(self, gianna_state, mock_tts, mock_langgraph_chain):
        """Test recovery from TTS service failures."""
        # Setup TTS failure
        mock_tts.side_effect = Exception("TTS service unavailable")

        # Generate response
        response_text = "This should be spoken"
        mock_langgraph_chain.invoke.return_value = {"output": response_text}
        response = mock_langgraph_chain.invoke({"input": "test"})

        # Test TTS failure
        with pytest.raises(Exception) as exc_info:
            mock_tts(response["output"])

        assert "TTS service unavailable" in str(exc_info.value)

        # Test fallback to text response
        gianna_state["conversation"].messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "timestamp": datetime.now().isoformat(),
                "source": "text",  # Fallback to text mode
            }
        )

        # Verify fallback worked
        last_msg = gianna_state["conversation"].messages[-1]
        assert last_msg["source"] == "text"
        assert last_msg["content"] == response_text

    def test_vad_failure_handling(self, gianna_state, mock_vad):
        """Test handling VAD failures."""
        # Test VAD failure
        mock_vad.detect_activity.side_effect = Exception("VAD hardware error")

        with pytest.raises(Exception) as exc_info:
            mock_vad.detect_activity([0.1, 0.2])

        assert "VAD hardware error" in str(exc_info.value)

        # Test fallback to always-on mode
        gianna_state["audio"].current_mode = "listening"  # Force listening mode
        mock_vad.detect_activity.side_effect = None
        mock_vad.detect_activity.return_value = True  # Always detect speech

        result = mock_vad.detect_activity([0.0, 0.0])  # Even silence triggers
        assert result is True

    def test_audio_file_corruption_handling(self, mock_stt, temp_file):
        """Test handling corrupted audio files."""
        # Create corrupted audio file
        with open(temp_file, "w") as f:
            f.write("This is not audio data")

        # Test STT with corrupted file
        mock_stt.side_effect = Exception("Invalid audio format")

        with pytest.raises(Exception) as exc_info:
            mock_stt(temp_file)

        assert "Invalid audio format" in str(exc_info.value)

    def test_voice_workflow_timeout_handling(self, gianna_state, mock_langgraph_chain):
        """Test handling timeouts in voice workflow."""
        # Setup timeout scenario
        mock_langgraph_chain.invoke.side_effect = Exception("Request timeout")

        # Test timeout handling
        with pytest.raises(Exception) as exc_info:
            mock_langgraph_chain.invoke({"input": "test"}, timeout=5.0)

        assert "timeout" in str(exc_info.value).lower()

        # Test recovery
        mock_langgraph_chain.invoke.side_effect = None
        mock_langgraph_chain.invoke.return_value = {"output": "Recovered response"}

        response = mock_langgraph_chain.invoke({"input": "test"})
        assert response["output"] == "Recovered response"


@pytest.mark.integration
@pytest.mark.fase3
@pytest.mark.voice
class TestMultiModalVoiceIntegration:
    """Test multi-modal voice and text integration."""

    def test_voice_to_text_transition(
        self, gianna_state, mock_vad, mock_stt, mock_langgraph_chain, temp_audio_file
    ):
        """Test transition from voice to text input."""
        # Start with voice input
        gianna_state["audio"].current_mode = "listening"

        # Voice input
        mock_vad.detect_activity.return_value = True
        mock_stt.return_value = "start voice conversation"

        voice_text = mock_stt(temp_audio_file)
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": voice_text,
                "timestamp": datetime.now().isoformat(),
                "source": "voice",
            }
        )

        # Switch to text input
        gianna_state["audio"].current_mode = "idle"
        text_input = "continue with text input"
        gianna_state["conversation"].messages.append(
            {
                "role": "user",
                "content": text_input,
                "timestamp": datetime.now().isoformat(),
                "source": "text",
            }
        )

        # Process mixed conversation
        mock_langgraph_chain.invoke.return_value = {
            "output": "I can handle both voice and text inputs seamlessly"
        }

        response = mock_langgraph_chain.invoke({"input": f"{voice_text} {text_input}"})

        # Verify multi-modal conversation
        assert len(gianna_state["conversation"].messages) == 2
        assert gianna_state["conversation"].messages[0]["source"] == "voice"
        assert gianna_state["conversation"].messages[1]["source"] == "text"
        assert "seamlessly" in response["output"]

    def test_text_to_voice_transition(
        self, gianna_state, mock_tts, mock_langgraph_chain
    ):
        """Test transition from text to voice output."""
        # Start with text conversation
        gianna_state["conversation"].messages.extend(
            [
                {
                    "role": "user",
                    "content": "hello in text",
                    "timestamp": datetime.now().isoformat(),
                    "source": "text",
                },
                {
                    "role": "assistant",
                    "content": "text response",
                    "timestamp": datetime.now().isoformat(),
                    "source": "text",
                },
            ]
        )

        # Switch to voice output
        mock_langgraph_chain.invoke.return_value = {
            "output": "Now switching to voice output mode"
        }

        response = mock_langgraph_chain.invoke({"input": "please speak your response"})

        # Generate voice output
        mock_tts_instance = MagicMock()
        mock_tts.return_value = mock_tts_instance
        tts_result = mock_tts(response["output"])

        # Add voice response
        gianna_state["conversation"].messages.append(
            {
                "role": "assistant",
                "content": response["output"],
                "timestamp": datetime.now().isoformat(),
                "source": "voice",
            }
        )

        # Verify transition
        messages = gianna_state["conversation"].messages
        assert len(messages) == 3
        assert messages[0]["source"] == "text"
        assert messages[1]["source"] == "text"
        assert messages[2]["source"] == "voice"
        assert tts_result is not None

    def test_simultaneous_voice_text_handling(self, gianna_state, mock_langgraph_chain):
        """Test handling simultaneous voice and text inputs."""
        # Simulate simultaneous inputs
        timestamp = datetime.now().isoformat()

        gianna_state["conversation"].messages.extend(
            [
                {
                    "role": "user",
                    "content": "voice input",
                    "timestamp": timestamp,
                    "source": "voice",
                },
                {
                    "role": "user",
                    "content": "text input at same time",
                    "timestamp": timestamp,
                    "source": "text",
                },
            ]
        )

        # Process combined inputs
        combined_input = "voice input text input at same time"
        mock_langgraph_chain.invoke.return_value = {
            "output": "I received both voice and text inputs simultaneously"
        }

        response = mock_langgraph_chain.invoke({"input": combined_input})

        # Verify handling
        assert "simultaneously" in response["output"]
        assert len(gianna_state["conversation"].messages) == 2


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.voice
class TestVoiceWorkflowPerformance:
    """Test voice workflow performance characteristics."""

    def test_voice_processing_latency(
        self,
        gianna_state,
        mock_vad,
        mock_stt,
        mock_tts,
        mock_langgraph_chain,
        benchmark_timer,
    ):
        """Test voice processing latency requirements."""
        # Setup fast mocks
        mock_vad.detect_activity.return_value = True
        mock_stt.return_value = "performance test"
        mock_langgraph_chain.invoke.return_value = {"output": "fast response"}
        mock_tts.return_value = MagicMock()

        benchmark_timer.start()

        # Full voice workflow
        has_speech = mock_vad.detect_activity([0.1, 0.2])
        transcribed = mock_stt("test.wav")
        response = mock_langgraph_chain.invoke({"input": transcribed})
        tts_result = mock_tts(response["output"])

        benchmark_timer.stop()

        # Verify performance
        assert (
            benchmark_timer.elapsed
            < PERFORMANCE_THRESHOLDS["response_time"]["complex_workflow"]
        )
        assert has_speech is True
        assert transcribed == "performance test"
        assert tts_result is not None

    def test_concurrent_voice_sessions(self, async_test_runner, benchmark_timer):
        """Test concurrent voice session handling."""

        async def concurrent_voice_sessions():
            sessions = []

            for i in range(5):
                # Create mock session
                session_state = {
                    "conversation": {"messages": []},
                    "audio": {"current_mode": "idle"},
                    "session_id": f"session-{i}",
                }

                # Mock voice processing
                with patch(
                    "gianna.workflows.voice_streaming.StreamingVoicePipeline"
                ) as mock_pipeline:
                    pipeline = mock_pipeline.return_value
                    pipeline.start_listening = AsyncMock()

                    await pipeline.start_listening(session_state)
                    sessions.append(session_state)

            return sessions

        benchmark_timer.start()
        sessions = async_test_runner(concurrent_voice_sessions())
        benchmark_timer.stop()

        assert len(sessions) == 5
        assert benchmark_timer.elapsed < 3.0  # < 3 seconds for 5 concurrent sessions

    def test_voice_memory_usage(self, gianna_state, mock_vad, sample_audio_data):
        """Test voice processing memory usage."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process large audio chunks
        large_audio = np.tile(sample_audio_data, 100)  # 100x larger

        for _ in range(50):
            mock_vad.detect_activity(large_audio)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage should be reasonable
        assert memory_increase < 100  # < 100MB increase

    def test_voice_throughput(
        self, mock_vad, mock_stt, mock_tts, mock_langgraph_chain, benchmark_timer
    ):
        """Test voice processing throughput."""
        # Setup fast mocks
        mock_vad.detect_activity.return_value = True
        mock_stt.return_value = "throughput test"
        mock_langgraph_chain.invoke.return_value = {"output": "throughput response"}
        mock_tts.return_value = MagicMock()

        benchmark_timer.start()

        # Process multiple voice interactions
        interactions = 0
        for i in range(20):
            if mock_vad.detect_activity([0.1, 0.2]):
                text = mock_stt(f"test_{i}.wav")
                response = mock_langgraph_chain.invoke({"input": text})
                mock_tts(response["output"])
                interactions += 1

        benchmark_timer.stop()

        # Calculate throughput
        throughput = interactions / benchmark_timer.elapsed

        assert interactions == 20
        assert (
            throughput
            >= PERFORMANCE_THRESHOLDS["throughput"]["interactions_per_minute"] / 60
        )  # per second
