"""
Unit tests for audio processing components - FASE 3

Tests for:
- Audio players and recorders (MP3, WAV, M4A, FLAC, OGG, AAC)
- Text-to-speech engines (Google, ElevenLabs, Whisper)
- Speech-to-text processing (Whisper, local variants)
- Voice Activity Detection (VAD)
- Audio format conversion and processing

Test Coverage:
- All audio format handlers
- TTS/STT engine functionality
- Audio file operations
- Format conversion
- VAD algorithms
- Error handling and fallbacks
"""

import tempfile
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

# Audio system imports
from gianna.assistants.audio.pydub_utils import suppress_pydub_warnings


@pytest.mark.unit
@pytest.mark.fase3
@pytest.mark.voice
class TestAudioPlayers:
    """Test audio player implementations."""

    def test_mp3_player_import(self):
        """Test MP3 player can be imported."""
        try:
            from gianna.assistants.audio.players.mp3 import MP3Player

            assert MP3Player is not None
        except ImportError as e:
            pytest.skip(f"MP3Player not available: {e}")

    def test_wav_player_import(self):
        """Test WAV player can be imported."""
        try:
            from gianna.assistants.audio.players.wave import WAVPlayer

            assert WAVPlayer is not None
        except ImportError as e:
            pytest.skip(f"WAVPlayer not available: {e}")

    def test_m4a_player_import(self):
        """Test M4A player can be imported."""
        try:
            from gianna.assistants.audio.players.m4a import M4APlayer

            assert M4APlayer is not None
        except ImportError as e:
            pytest.skip(f"M4APlayer not available: {e}")

    def test_flac_player_import(self):
        """Test FLAC player can be imported."""
        try:
            from gianna.assistants.audio.players.flac import FLACPlayer

            assert FLACPlayer is not None
        except ImportError as e:
            pytest.skip(f"FLACPlayer not available: {e}")

    def test_ogg_player_import(self):
        """Test OGG player can be imported."""
        try:
            from gianna.assistants.audio.players.ogg import OGGPlayer

            assert OGGPlayer is not None
        except ImportError as e:
            pytest.skip(f"OGGPlayer not available: {e}")

    def test_aac_player_import(self):
        """Test AAC player can be imported."""
        try:
            from gianna.assistants.audio.players.aac import AACPlayer

            assert AACPlayer is not None
        except ImportError as e:
            pytest.skip(f"AACPlayer not available: {e}")

    def test_audio_player_factory(self):
        """Test audio player factory method."""
        try:
            from gianna.assistants.audio.players.factory_method import get_audio_player

            # Test different formats
            formats = ["mp3", "wav", "m4a", "flac", "ogg", "aac"]

            for fmt in formats:
                try:
                    player = get_audio_player(fmt)
                    assert player is not None
                    assert hasattr(player, "play")
                except (ImportError, ValueError):
                    # Some formats might not be available
                    pass

        except ImportError as e:
            pytest.skip(f"Audio player factory not available: {e}")

    @patch("gianna.assistants.audio.players.mp3.pygame")
    def test_mp3_player_functionality(self, mock_pygame):
        """Test MP3 player functionality with mocked pygame."""
        try:
            from gianna.assistants.audio.players.mp3 import MP3Player

            # Mock pygame components
            mock_mixer = MagicMock()
            mock_pygame.mixer = mock_mixer

            player = MP3Player()
            player.play("test.mp3")

            # Verify pygame was called appropriately
            mock_mixer.init.assert_called()
            mock_mixer.music.load.assert_called_with("test.mp3")
            mock_mixer.music.play.assert_called()

        except ImportError:
            pytest.skip("MP3Player not available")

    def test_wav_player_functionality(self, temp_audio_file):
        """Test WAV player functionality with real audio file."""
        try:
            from gianna.assistants.audio.players.wave import WAVPlayer

            with patch("gianna.assistants.audio.players.wave.pyaudio") as mock_pyaudio:
                mock_stream = MagicMock()
                mock_audio = MagicMock()
                mock_audio.open.return_value = mock_stream
                mock_pyaudio.PyAudio.return_value = mock_audio

                player = WAVPlayer()
                player.play(temp_audio_file)

                # Verify pyaudio was used
                mock_audio.open.assert_called()
                mock_stream.write.assert_called()
                mock_stream.close.assert_called()

        except ImportError:
            pytest.skip("WAVPlayer not available")


@pytest.mark.unit
@pytest.mark.fase3
@pytest.mark.voice
class TestAudioRecorders:
    """Test audio recorder implementations."""

    def test_wav_recorder_import(self):
        """Test WAV recorder can be imported."""
        try:
            from gianna.assistants.audio.recorders.wave import WAVRecorder

            assert WAVRecorder is not None
        except ImportError as e:
            pytest.skip(f"WAVRecorder not available: {e}")

    def test_mp3_recorder_import(self):
        """Test MP3 recorder can be imported."""
        try:
            from gianna.assistants.audio.recorders.mp3 import MP3Recorder

            assert MP3Recorder is not None
        except ImportError as e:
            pytest.skip(f"MP3Recorder not available: {e}")

    def test_m4a_recorder_import(self):
        """Test M4A recorder can be imported."""
        try:
            from gianna.assistants.audio.recorders.m4a import M4ARecorder

            assert M4ARecorder is not None
        except ImportError as e:
            pytest.skip(f"M4ARecorder not available: {e}")

    def test_audio_recorder_factory(self):
        """Test audio recorder factory method."""
        try:
            from gianna.assistants.audio.recorders.factory_method import audio_record

            with patch("gianna.assistants.audio.recorders.factory_method.pyaudio"):
                recorder = audio_record("wav", "test.wav", duration=1.0)
                assert recorder is not None

        except ImportError as e:
            pytest.skip(f"Audio recorder factory not available: {e}")

    @patch("gianna.assistants.audio.recorders.wave.pyaudio")
    def test_wav_recorder_functionality(self, mock_pyaudio):
        """Test WAV recorder functionality."""
        try:
            from gianna.assistants.audio.recorders.wave import WAVRecorder

            # Mock pyaudio components
            mock_stream = MagicMock()
            mock_audio = MagicMock()
            mock_audio.open.return_value = mock_stream
            mock_stream.read.return_value = b"audio_data" * 100
            mock_pyaudio.PyAudio.return_value = mock_audio

            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
                recorder = WAVRecorder()
                recorder.record(temp_file.name, duration=1.0)

                # Verify recording process
                mock_audio.open.assert_called()
                mock_stream.read.assert_called()

        except ImportError:
            pytest.skip("WAVRecorder not available")


@pytest.mark.unit
@pytest.mark.fase3
@pytest.mark.voice
class TestTextToSpeech:
    """Test text-to-speech engines."""

    def test_google_tts_import(self):
        """Test Google TTS can be imported."""
        try:
            from gianna.assistants.audio.tts.google_api import GoogleTTS

            assert GoogleTTS is not None
        except ImportError as e:
            pytest.skip(f"GoogleTTS not available: {e}")

    def test_elevenlabs_tts_import(self):
        """Test ElevenLabs TTS can be imported."""
        try:
            from gianna.assistants.audio.tts.elevenlabs import ElevenLabsTTS

            assert ElevenLabsTTS is not None
        except ImportError as e:
            pytest.skip(f"ElevenLabsTTS not available: {e}")

    def test_whisper_tts_import(self):
        """Test Whisper TTS can be imported."""
        try:
            from gianna.assistants.audio.tts.whisper import WhisperTTS

            assert WhisperTTS is not None
        except ImportError as e:
            pytest.skip(f"WhisperTTS not available: {e}")

    def test_tts_factory_method(self, tts_engine):
        """Test TTS factory method with different engines."""
        try:
            from gianna.assistants.audio.tts.factory_method import text_to_speech

            with patch(f"gianna.assistants.audio.tts.{tts_engine}") as mock_tts:
                mock_instance = MagicMock()
                mock_tts.return_value = mock_instance

                result = text_to_speech(tts_engine, "Hello world")

                # Factory should return a TTS instance
                assert result is not None

        except ImportError:
            pytest.skip(f"TTS factory not available for {tts_engine}")

    @patch("gianna.assistants.audio.tts.google_api.gTTS")
    def test_google_tts_functionality(self, mock_gtts):
        """Test Google TTS functionality."""
        try:
            from gianna.assistants.audio.tts.google_api import GoogleTTS

            # Mock gTTS
            mock_tts_instance = MagicMock()
            mock_gtts.return_value = mock_tts_instance

            tts = GoogleTTS()
            tts.synthesize("Hello world", "output.mp3")

            # Verify gTTS was used
            mock_gtts.assert_called_with(text="Hello world", lang="pt-br")
            mock_tts_instance.save.assert_called_with("output.mp3")

        except ImportError:
            pytest.skip("Google TTS not available")

    def test_elevenlabs_tts_functionality(self):
        """Test ElevenLabs TTS functionality."""
        try:
            from gianna.assistants.audio.tts.elevenlabs import ElevenLabsTTS

            with patch("gianna.assistants.audio.tts.elevenlabs.elevenlabs") as mock_el:
                mock_el.generate.return_value = b"fake_audio_data"

                tts = ElevenLabsTTS()
                result = tts.synthesize("Hello world", "output.mp3")

                # Verify ElevenLabs was called
                mock_el.generate.assert_called()
                assert result is not None

        except ImportError:
            pytest.skip("ElevenLabs TTS not available")


@pytest.mark.unit
@pytest.mark.fase3
@pytest.mark.voice
class TestSpeechToText:
    """Test speech-to-text engines."""

    def test_whisper_stt_import(self):
        """Test Whisper STT can be imported."""
        try:
            from gianna.assistants.audio.stt.whisper import WhisperSTT

            assert WhisperSTT is not None
        except ImportError as e:
            pytest.skip(f"WhisperSTT not available: {e}")

    def test_whisper_local_stt_import(self):
        """Test local Whisper STT can be imported."""
        try:
            from gianna.assistants.audio.stt.whisper_local import WhisperLocalSTT

            assert WhisperLocalSTT is not None
        except ImportError as e:
            pytest.skip(f"WhisperLocalSTT not available: {e}")

    def test_stt_factory_method(self, stt_engine):
        """Test STT factory method with different engines."""
        try:
            from gianna.assistants.audio.stt.factory_method import speech_to_text

            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
                with patch(f"gianna.assistants.audio.stt.{stt_engine}") as mock_stt:
                    mock_stt.transcribe.return_value = "Transcribed text"

                    result = speech_to_text(stt_engine, temp_file.name)

                    assert isinstance(result, str)
                    assert len(result) > 0

        except ImportError:
            pytest.skip(f"STT factory not available for {stt_engine}")

    def test_whisper_stt_functionality(self, temp_audio_file):
        """Test Whisper STT functionality."""
        try:
            from gianna.assistants.audio.stt.whisper import WhisperSTT

            with patch("gianna.assistants.audio.stt.whisper.openai") as mock_openai:
                mock_client = MagicMock()
                mock_transcription = MagicMock()
                mock_transcription.text = "Hello world transcription"
                mock_client.audio.transcriptions.create.return_value = (
                    mock_transcription
                )
                mock_openai.OpenAI.return_value = mock_client

                stt = WhisperSTT()
                result = stt.transcribe(temp_audio_file)

                assert result == "Hello world transcription"
                mock_client.audio.transcriptions.create.assert_called()

        except ImportError:
            pytest.skip("Whisper STT not available")

    def test_whisper_local_functionality(self, temp_audio_file):
        """Test local Whisper functionality."""
        try:
            from gianna.assistants.audio.stt.whisper_local import WhisperLocalSTT

            with patch(
                "gianna.assistants.audio.stt.whisper_local.whisper"
            ) as mock_whisper:
                mock_model = MagicMock()
                mock_result = {"text": "Local whisper transcription"}
                mock_model.transcribe.return_value = mock_result
                mock_whisper.load_model.return_value = mock_model

                stt = WhisperLocalSTT()
                result = stt.transcribe(temp_audio_file)

                assert result == "Local whisper transcription"
                mock_whisper.load_model.assert_called()
                mock_model.transcribe.assert_called()

        except ImportError:
            pytest.skip("Local Whisper not available")


@pytest.mark.unit
@pytest.mark.fase3
@pytest.mark.voice
class TestVoiceActivityDetection:
    """Test Voice Activity Detection functionality."""

    def test_vad_import(self):
        """Test VAD can be imported."""
        try:
            from gianna.audio.streaming import VoiceActivityDetector

            assert VoiceActivityDetector is not None
        except ImportError:
            # Try alternative import path
            try:
                from gianna.assistants.audio.vad import VoiceActivityDetector

                assert VoiceActivityDetector is not None
            except ImportError as e:
                pytest.skip(f"VoiceActivityDetector not available: {e}")

    def test_vad_creation(self):
        """Test VAD instance creation."""
        try:
            from gianna.audio.streaming import VoiceActivityDetector

            vad = VoiceActivityDetector(threshold=0.02, min_silence_duration=1.0)

            assert vad.threshold == 0.02
            assert vad.min_silence_duration == 1.0
            assert vad.is_speech_active is False

        except ImportError:
            pytest.skip("VoiceActivityDetector not available")

    def test_vad_speech_detection(self, sample_audio_data):
        """Test VAD speech detection with sample data."""
        try:
            from gianna.audio.streaming import VoiceActivityDetector

            vad = VoiceActivityDetector(threshold=0.01)  # Lower threshold for test data

            # Test with active speech (sine wave should trigger detection)
            has_speech = vad.detect_activity(sample_audio_data)
            assert isinstance(has_speech, bool)

            # Test with silence (zeros should not trigger detection)
            silence = np.zeros(1000, dtype=np.float32)
            no_speech = vad.detect_activity(silence)
            assert no_speech is False

        except ImportError:
            pytest.skip("VoiceActivityDetector not available")

    def test_vad_rms_calculation(self, sample_audio_data):
        """Test VAD RMS calculation accuracy."""
        try:
            from gianna.audio.streaming import VoiceActivityDetector

            vad = VoiceActivityDetector()

            # Calculate expected RMS manually
            expected_rms = np.sqrt(np.mean(sample_audio_data**2))

            # Test VAD calculation (if method is accessible)
            if hasattr(vad, "_calculate_rms"):
                calculated_rms = vad._calculate_rms(sample_audio_data)
                np.testing.assert_almost_equal(calculated_rms, expected_rms, decimal=5)

        except ImportError:
            pytest.skip("VoiceActivityDetector not available")

    def test_vad_streaming_callback(self):
        """Test VAD streaming with callbacks."""
        try:
            from gianna.audio.streaming import VoiceActivityDetector

            vad = VoiceActivityDetector()

            # Mock callback function
            callbacks = {"speech_start": MagicMock(), "speech_end": MagicMock()}

            # Test callback registration and invocation
            if hasattr(vad, "set_callbacks"):
                vad.set_callbacks(callbacks)

                # Simulate speech start
                vad.is_speech_active = False
                if hasattr(vad, "_trigger_callback"):
                    vad._trigger_callback("speech_start")
                    callbacks["speech_start"].assert_called_once()

        except ImportError:
            pytest.skip("VoiceActivityDetector not available")


@pytest.mark.unit
@pytest.mark.fase3
@pytest.mark.voice
class TestAudioUtilities:
    """Test audio utility functions."""

    def test_pydub_warnings_suppression(self):
        """Test PyDub warnings suppression utility."""
        with suppress_pydub_warnings():
            # This should run without warnings
            pass

        # Verify context manager works
        assert True  # If we reach here, context manager worked

    def test_audio_format_conversion(self):
        """Test audio format conversion utilities."""
        try:
            from gianna.assistants.audio.pydub_utils import convert_audio_format

            with tempfile.NamedTemporaryFile(suffix=".wav") as input_file:
                with tempfile.NamedTemporaryFile(suffix=".mp3") as output_file:
                    # Create minimal WAV file
                    with wave.open(input_file.name, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(16000)
                        wav_file.writeframes(b"\x00\x00" * 1000)

                    # Test conversion
                    result = convert_audio_format(
                        input_file.name, output_file.name, "mp3"
                    )
                    assert result is not None

        except ImportError:
            pytest.skip("Audio format conversion not available")

    def test_audio_duration_calculation(self):
        """Test audio duration calculation."""
        try:
            from gianna.assistants.audio.pydub_utils import get_audio_duration

            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
                # Create 1-second WAV file
                sample_rate = 16000
                with wave.open(temp_file.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(b"\x00\x00" * sample_rate)  # 1 second

                duration = get_audio_duration(temp_file.name)
                assert abs(duration - 1.0) < 0.1  # Should be approximately 1 second

        except ImportError:
            pytest.skip("Audio duration calculation not available")


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.voice
class TestAudioPerformance:
    """Test audio processing performance."""

    def test_vad_performance(self, sample_audio_data, benchmark_timer):
        """Test VAD processing performance."""
        try:
            from gianna.audio.streaming import VoiceActivityDetector

            vad = VoiceActivityDetector()

            benchmark_timer.start()

            # Process multiple audio chunks
            results = []
            for _ in range(100):
                result = vad.detect_activity(sample_audio_data)
                results.append(result)

            benchmark_timer.stop()

            assert len(results) == 100
            assert benchmark_timer.elapsed < 1.0  # < 1 second for 100 VAD operations

        except ImportError:
            pytest.skip("VoiceActivityDetector not available")

    def test_audio_loading_performance(self, temp_audio_file, benchmark_timer):
        """Test audio file loading performance."""
        try:
            from pydub import AudioSegment

            benchmark_timer.start()

            # Load audio file multiple times
            segments = []
            for _ in range(50):
                segment = AudioSegment.from_wav(temp_audio_file)
                segments.append(segment)

            benchmark_timer.stop()

            assert len(segments) == 50
            assert benchmark_timer.elapsed < 2.0  # < 2 seconds for 50 loads

        except ImportError:
            pytest.skip("PyDub not available")

    def test_audio_conversion_performance(self, temp_audio_file, benchmark_timer):
        """Test audio format conversion performance."""
        try:
            from pydub import AudioSegment

            # Load source file
            audio = AudioSegment.from_wav(temp_audio_file)

            benchmark_timer.start()

            # Convert to different formats
            formats = ["mp3", "flv", "ogg"]  # Available formats in pydub
            conversions = []

            for fmt in formats:
                with tempfile.NamedTemporaryFile(suffix=f".{fmt}") as temp_file:
                    audio.export(temp_file.name, format=fmt)
                    conversions.append(fmt)

            benchmark_timer.stop()

            assert len(conversions) == len(formats)
            assert benchmark_timer.elapsed < 5.0  # < 5 seconds for conversions

        except ImportError:
            pytest.skip("PyDub not available")
