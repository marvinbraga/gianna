"""
Tool Integration Layer for Gianna Assistant - FASE 2

This module provides LangChain-compatible tools that integrate with Gianna's
existing command and audio systems. All tools inherit from BaseTool and
provide structured JSON responses for reliable agent integration.
"""

from .audio_tools import AudioProcessorTool, STTTool, TTSTool
from .filesystem_tools import FileSystemTool
from .memory_tools import MemoryTool
from .shell_tools import ShellExecutorTool

__all__ = [
    "ShellExecutorTool",
    "AudioProcessorTool",
    "TTSTool",
    "STTTool",
    "MemoryTool",
    "FileSystemTool",
]
