#!/usr/bin/env python3
"""
Tool Integration Demo for Gianna Assistant - FASE 2

This example demonstrates how to use Gianna's integrated tool system
with LangChain-compatible tools for shell execution, file operations,
audio processing, and memory management.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gianna.tools import (
    AudioProcessorTool,
    FileSystemTool,
    MemoryTool,
    ShellExecutorTool,
    STTTool,
    TTSTool,
)


def demo_shell_tool():
    """Demonstrate shell tool usage"""
    print("=== Shell Tool Demo ===")

    shell_tool = ShellExecutorTool()

    # Execute a safe command
    result = shell_tool._run("ls -la | head -5")
    data = json.loads(result)

    print(f"Command: ls -la | head -5")
    print(f"Success: {data['success']}")
    print(f"Output:\n{data['stdout']}")
    print()


def demo_filesystem_tool():
    """Demonstrate filesystem tool usage"""
    print("=== FileSystem Tool Demo ===")

    fs_tool = FileSystemTool()

    # Create a demo file
    content = """# Demo File
This is a demonstration of the FileSystem tool.
Created by: Gianna Tool Integration Layer
"""

    # Write file
    write_result = fs_tool._run(
        json.dumps({"action": "write", "path": "demo_file.md", "content": content})
    )
    print("Write result:", json.loads(write_result)["message"])

    # Read file back
    read_result = fs_tool._run(json.dumps({"action": "read", "path": "demo_file.md"}))
    read_data = json.loads(read_result)
    print(f"Read {read_data['size']} bytes")

    # Get file info
    info_result = fs_tool._run(json.dumps({"action": "info", "path": "demo_file.md"}))
    info_data = json.loads(info_result)["info"]
    print(f"File info: {info_data['name']} ({info_data['size']} bytes)")

    # Clean up
    fs_tool._run(
        json.dumps({"action": "delete", "path": "demo_file.md", "confirm": True})
    )
    print("Demo file cleaned up")
    print()


def demo_audio_tools():
    """Demonstrate audio tool usage"""
    print("=== Audio Tools Demo ===")

    # TTS Tool
    tts_tool = TTSTool()
    print(f"TTS Tool: {tts_tool.name}")
    print(f"Description: {tts_tool.description}")

    # STT Tool
    stt_tool = STTTool()
    print(f"STT Tool: {stt_tool.name}")
    print(f"Description: {stt_tool.description}")

    # Audio Processor Tool
    audio_tool = AudioProcessorTool()
    print(f"Audio Processor: {audio_tool.name}")
    print(f"Description: {audio_tool.description}")
    print()


def demo_memory_tool():
    """Demonstrate memory tool usage"""
    print("=== Memory Tool Demo ===")

    memory_tool = MemoryTool()

    # Get session statistics
    stats_result = memory_tool._run(
        json.dumps({"action": "get_stats", "session_id": "demo_session"})
    )
    stats_data = json.loads(stats_result)
    print("Session stats:", stats_data["message"])

    # List all sessions
    list_result = memory_tool._run(json.dumps({"action": "list_sessions"}))
    list_data = json.loads(list_result)
    print(f"Total sessions: {list_data['total_sessions']}")
    print()


def demo_tool_descriptions():
    """Show all tool descriptions for LangChain integration"""
    print("=== Tool Descriptions for LangChain Integration ===")

    tools = [
        ShellExecutorTool(),
        FileSystemTool(),
        AudioProcessorTool(),
        TTSTool(),
        STTTool(),
        MemoryTool(),
    ]

    for tool in tools:
        print(f"Tool: {tool.name}")
        print(f"Description: {tool.description}")
        print("-" * 50)


def main():
    """Run all demonstrations"""
    print("🚀 Gianna Tool Integration Layer - Demo\n")

    try:
        demo_shell_tool()
        demo_filesystem_tool()
        demo_audio_tools()
        demo_memory_tool()
        demo_tool_descriptions()

        print("✅ All demonstrations completed successfully!")
        print("\n💡 These tools are ready for integration with LangChain agents.")
        print(
            "   Each tool provides structured JSON responses and proper error handling."
        )

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
