"""
Audio processing tools that integrate with Gianna's audio system.

This module provides LangChain-compatible tools for text-to-speech,
speech-to-text, and audio processing operations using Gianna's existing
audio infrastructure.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from langchain.tools import BaseTool
from loguru import logger
from pydantic import Field

from gianna.assistants.audio.players import play_audio
from gianna.assistants.audio.stt import speech_to_text
from gianna.assistants.audio.tts import text_to_speech
from gianna.assistants.commands.speech import SpeechType


class AudioProcessorTool(BaseTool):
    """
    General audio processing tool for file operations and format conversions.

    Integrates with Gianna's audio system for comprehensive audio handling.
    """

    name: str = "audio_processor"
    description: str = """Process audio files with format conversion, playback, and analysis.
    Input: JSON with 'action' (play|info|convert), 'file_path', and optional parameters
    Output: JSON with processing results and file information"""

    def _run(self, input_data: str) -> str:
        """
        Process audio files with various operations.

        Args:
            input_data: JSON string with action and parameters

        Returns:
            JSON string with processing results
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data

            action = data.get("action", "").lower()
            file_path = data.get("file_path", "")

            if not file_path or not os.path.exists(file_path):
                return json.dumps(
                    {"error": f"File not found: {file_path}", "success": False}
                )

            file_path = Path(file_path)

            if action == "play":
                return self._play_audio(file_path, data.get("options", {}))
            elif action == "info":
                return self._get_audio_info(file_path)
            elif action == "convert":
                return self._convert_audio(file_path, data.get("target_format", "mp3"))
            else:
                return json.dumps(
                    {
                        "error": f"Unknown action: {action}. Use 'play', 'info', or 'convert'",
                        "success": False,
                    }
                )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            return json.dumps(
                {"error": f"Invalid JSON input: {str(e)}", "success": False}
            )
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return json.dumps(
                {"error": f"Audio processing failed: {str(e)}", "success": False}
            )

    def _play_audio(self, file_path: Path, options: dict) -> str:
        """Play audio file using Gianna's audio system."""
        try:
            # Use Gianna's play_audio function
            result = play_audio(str(file_path))

            return json.dumps(
                {
                    "success": True,
                    "action": "play",
                    "file_path": str(file_path),
                    "message": f"Playing audio file: {file_path.name}",
                    "result": result,
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Playback failed: {str(e)}",
                    "success": False,
                    "file_path": str(file_path),
                }
            )

    def _get_audio_info(self, file_path: Path) -> str:
        """Get audio file information."""
        try:
            import pydub

            audio = pydub.AudioSegment.from_file(str(file_path))

            info = {
                "success": True,
                "action": "info",
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "format": file_path.suffix.lower().lstrip("."),
                "duration_seconds": len(audio) / 1000.0,
                "sample_rate": audio.frame_rate,
                "channels": audio.channels,
                "frame_width": audio.frame_width,
                "duration_formatted": f"{len(audio) // 60000:02d}:{(len(audio) % 60000) // 1000:02d}",
            }

            return json.dumps(info, indent=2)

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Could not get audio info: {str(e)}",
                    "success": False,
                    "file_path": str(file_path),
                }
            )

    def _convert_audio(self, file_path: Path, target_format: str) -> str:
        """Convert audio file to different format."""
        try:
            import pydub

            audio = pydub.AudioSegment.from_file(str(file_path))
            output_path = file_path.with_suffix(f".{target_format.lower()}")

            # Export in target format
            audio.export(str(output_path), format=target_format.lower())

            return json.dumps(
                {
                    "success": True,
                    "action": "convert",
                    "input_file": str(file_path),
                    "output_file": str(output_path),
                    "target_format": target_format,
                    "message": f"Converted {file_path.name} to {output_path.name}",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Conversion failed: {str(e)}",
                    "success": False,
                    "input_file": str(file_path),
                }
            )

    async def _arun(self, input_data: str) -> str:
        """Async version - delegates to sync version for now."""
        return self._run(input_data)


class TTSTool(BaseTool):
    """
    Text-to-Speech tool using Gianna's TTS system.

    Converts text to speech audio files using various TTS engines.
    """

    name: str = "text_to_speech"
    description: str = """Convert text to speech audio file.
    Input: JSON with 'text', optional 'voice_type' (google|elevenlabs), 'language', 'output_file'
    Output: JSON with generated audio file path and TTS details"""

    default_voice_type: str = Field(default="google", description="Default TTS engine")
    default_language: str = Field(default="pt-br", description="Default language")

    def _run(self, input_data: str) -> str:
        """
        Convert text to speech using Gianna's TTS system.

        Args:
            input_data: JSON string with text and TTS parameters

        Returns:
            JSON string with generated audio file information
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data

            text = data.get("text", "")
            if not text.strip():
                return json.dumps(
                    {"error": "No text provided for TTS conversion", "success": False}
                )

            # Get TTS parameters
            voice_type = data.get("voice_type", self.default_voice_type)
            language = data.get("language", self.default_language)
            output_file = data.get("output_file")

            # Create temporary file if no output specified
            if not output_file:
                temp_dir = tempfile.mkdtemp()
                output_file = os.path.join(temp_dir, "tts_output.mp3")

            logger.info(f"Generating TTS for text: {text[:50]}...")

            # Use Gianna's TTS system
            speech_type = SpeechType(voice_type)
            result = text_to_speech(
                text=text,
                speech_type=speech_type,
                language=language,
                output_file=output_file,
            )

            # Verify file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                return json.dumps(
                    {
                        "success": True,
                        "text": text,
                        "voice_type": voice_type,
                        "language": language,
                        "output_file": output_file,
                        "file_size": file_size,
                        "message": f"TTS generated successfully: {os.path.basename(output_file)}",
                        "result": result,
                    }
                )
            else:
                return json.dumps(
                    {
                        "error": "TTS file was not created",
                        "success": False,
                        "text": text,
                        "voice_type": voice_type,
                        "expected_file": output_file,
                    }
                )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            return json.dumps(
                {"error": f"Invalid JSON input: {str(e)}", "success": False}
            )
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return json.dumps(
                {
                    "error": f"TTS generation failed: {str(e)}",
                    "success": False,
                    "text": data.get("text", "") if "data" in locals() else "",
                }
            )

    async def _arun(self, input_data: str) -> str:
        """Async version - delegates to sync version for now."""
        return self._run(input_data)


class STTTool(BaseTool):
    """
    Speech-to-Text tool using Gianna's STT system.

    Converts audio files to text using various STT engines.
    """

    name: str = "speech_to_text"
    description: str = """Convert speech audio file to text.
    Input: JSON with 'audio_file', optional 'language', 'engine' (whisper|google)
    Output: JSON with transcribed text and STT details"""

    default_language: str = Field(
        default="pt-br", description="Default language for STT"
    )

    def _run(self, input_data: str) -> str:
        """
        Convert speech to text using Gianna's STT system.

        Args:
            input_data: JSON string with audio file path and STT parameters

        Returns:
            JSON string with transcription results
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data

            audio_file = data.get("audio_file", "")
            if not audio_file or not os.path.exists(audio_file):
                return json.dumps(
                    {"error": f"Audio file not found: {audio_file}", "success": False}
                )

            # Get STT parameters
            language = data.get("language", self.default_language)
            engine = data.get("engine", "whisper")

            logger.info(f"Transcribing audio file: {audio_file}")

            # Use Gianna's STT system
            result = speech_to_text(audio_file=audio_file, language=language)

            # Extract text from result
            if isinstance(result, str):
                transcription = result
            elif isinstance(result, dict):
                transcription = result.get("text", result.get("transcription", ""))
            else:
                transcription = str(result)

            return json.dumps(
                {
                    "success": True,
                    "audio_file": audio_file,
                    "transcription": transcription,
                    "language": language,
                    "engine": engine,
                    "file_size": os.path.getsize(audio_file),
                    "message": f"Successfully transcribed {os.path.basename(audio_file)}",
                    "result": result,
                }
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            return json.dumps(
                {"error": f"Invalid JSON input: {str(e)}", "success": False}
            )
        except Exception as e:
            logger.error(f"STT error: {e}")
            return json.dumps(
                {
                    "error": f"Speech transcription failed: {str(e)}",
                    "success": False,
                    "audio_file": (
                        data.get("audio_file", "") if "data" in locals() else ""
                    ),
                }
            )

    async def _arun(self, input_data: str) -> str:
        """Async version - delegates to sync version for now."""
        return self._run(input_data)
