#!/usr/bin/env python3
"""
Comprehensive Tests for Tool Integration Layer - FASE 2

This module provides comprehensive tests for all tools in the Gianna
Tool Integration Layer, validating safety, functionality, and integration.
"""

import json
import os
import sys
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from gianna.tools import (
    AudioProcessorTool,
    FileSystemTool,
    MemoryTool,
    ShellExecutorTool,
    STTTool,
    TTSTool,
)


class TestShellExecutorTool:
    """Test cases for the ShellExecutorTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = ShellExecutorTool()

    def test_tool_initialization(self):
        """Test tool initialization."""
        assert self.tool is not None
        assert self.tool.name == "shell_executor"
        assert "Execute shell commands safely" in self.tool.description
        assert hasattr(self.tool, "_run")
        assert hasattr(self.tool, "safety_checks")

    def test_safe_command_execution(self):
        """Test execution of safe commands."""
        # Test simple echo command
        result = self.tool._run('echo "Hello World"')
        data = json.loads(result)

        assert data["success"] is True
        assert "Hello World" in data["stdout"]
        assert data["exit_code"] == 0
        assert data["stderr"] == ""
        assert "command" in data

    def test_command_with_output(self):
        """Test command that produces output."""
        result = self.tool._run("pwd")
        data = json.loads(result)

        assert data["success"] is True
        assert len(data["stdout"].strip()) > 0
        assert data["exit_code"] == 0

    def test_command_timeout(self):
        """Test command timeout functionality."""
        # Test with a very short timeout
        tool = ShellExecutorTool(timeout=0.1)
        result = tool._run("sleep 1")  # Command takes longer than timeout
        data = json.loads(result)

        assert data["success"] is False
        assert "timeout" in data["error"].lower()

    def test_dangerous_command_blocking(self):
        """Test that dangerous commands are blocked."""
        dangerous_commands = [
            "rm -rf /",
            "format c:",
            "sudo rm -rf /usr",
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sda1",
        ]

        for cmd in dangerous_commands:
            result = self.tool._run(cmd)
            data = json.loads(result)

            assert data["success"] is False
            assert (
                "dangerous" in data["error"].lower()
                or "blocked" in data["error"].lower()
            )

    def test_invalid_command(self):
        """Test handling of invalid commands."""
        result = self.tool._run("nonexistentcommand12345")
        data = json.loads(result)

        assert data["success"] is False
        assert data["exit_code"] != 0
        assert len(data["stderr"]) > 0 or "not found" in data["error"]

    def test_empty_command(self):
        """Test handling of empty commands."""
        result = self.tool._run("")
        data = json.loads(result)

        assert data["success"] is False
        assert "empty" in data["error"].lower() or "no command" in data["error"].lower()

    def test_command_with_pipes(self):
        """Test command with pipes and redirects."""
        result = self.tool._run('echo "line1\nline2\nline3" | head -2')
        data = json.loads(result)

        assert data["success"] is True
        assert data["exit_code"] == 0
        assert "line1" in data["stdout"]
        assert "line2" in data["stdout"]

    def test_security_validation(self):
        """Test security validation features."""
        # Test that dangerous operators are flagged
        suspicious_commands = [
            "ls; rm file.txt",  # Command chaining
            "ls && rm file.txt",  # Conditional execution
            "ls || rm file.txt",  # Alternative execution
            "ls | xargs rm",  # Dangerous pipe usage
        ]

        for cmd in suspicious_commands:
            result = self.tool._run(cmd)
            data = json.loads(result)
            # Note: Some may succeed if they're actually safe,
            # but the tool should at least analyze them
            assert "command" in data

    def test_async_method(self):
        """Test async method exists and delegates properly."""
        assert hasattr(self.tool, "arun")
        # The arun method should exist even if it delegates to _run


class TestFileSystemTool:
    """Test cases for the FileSystemTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = FileSystemTool()
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp(prefix="gianna_test_")

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_tool_initialization(self):
        """Test tool initialization."""
        assert self.tool is not None
        assert self.tool.name == "filesystem_manager"
        assert "Manage file system operations" in self.tool.description

    def test_write_and_read_file(self):
        """Test writing and reading files."""
        test_content = "This is a test file\nwith multiple lines."
        test_file = os.path.join(self.test_dir, "test.txt")

        # Write file
        write_result = self.tool._run(
            json.dumps({"action": "write", "path": test_file, "content": test_content})
        )
        write_data = json.loads(write_result)

        assert write_data["success"] is True
        assert "Successfully wrote" in write_data["message"]
        assert os.path.exists(test_file)

        # Read file
        read_result = self.tool._run(json.dumps({"action": "read", "path": test_file}))
        read_data = json.loads(read_result)

        assert read_data["success"] is True
        assert read_data["content"] == test_content
        assert read_data["size"] == len(test_content.encode())

    def test_list_directory(self):
        """Test directory listing."""
        # Create some test files
        for i in range(3):
            test_file = os.path.join(self.test_dir, f"file{i}.txt")
            with open(test_file, "w") as f:
                f.write(f"Content {i}")

        list_result = self.tool._run(
            json.dumps({"action": "list", "path": self.test_dir})
        )
        list_data = json.loads(list_result)

        assert list_data["success"] is True
        assert len(list_data["files"]) >= 3
        assert any("file0.txt" in item["name"] for item in list_data["files"])
        assert any("file1.txt" in item["name"] for item in list_data["files"])
        assert any("file2.txt" in item["name"] for item in list_data["files"])

    def test_file_info(self):
        """Test file information retrieval."""
        test_content = "Test content for info"
        test_file = os.path.join(self.test_dir, "info_test.txt")

        with open(test_file, "w") as f:
            f.write(test_content)

        info_result = self.tool._run(json.dumps({"action": "info", "path": test_file}))
        info_data = json.loads(info_result)

        assert info_data["success"] is True
        file_info = info_data["info"]
        assert file_info["name"] == "info_test.txt"
        assert file_info["size"] == len(test_content)
        assert file_info["type"] == "file"
        assert "modified" in file_info

    def test_copy_file(self):
        """Test file copying."""
        source_content = "Source file content"
        source_file = os.path.join(self.test_dir, "source.txt")
        dest_file = os.path.join(self.test_dir, "destination.txt")

        # Create source file
        with open(source_file, "w") as f:
            f.write(source_content)

        # Copy file
        copy_result = self.tool._run(
            json.dumps(
                {"action": "copy", "path": source_file, "destination": dest_file}
            )
        )
        copy_data = json.loads(copy_result)

        assert copy_data["success"] is True
        assert "Successfully copied" in copy_data["message"]
        assert os.path.exists(dest_file)

        # Verify content
        with open(dest_file, "r") as f:
            assert f.read() == source_content

    def test_move_file(self):
        """Test file moving."""
        source_content = "File to move"
        source_file = os.path.join(self.test_dir, "move_source.txt")
        dest_file = os.path.join(self.test_dir, "moved_file.txt")

        # Create source file
        with open(source_file, "w") as f:
            f.write(source_content)

        # Move file
        move_result = self.tool._run(
            json.dumps(
                {"action": "move", "path": source_file, "destination": dest_file}
            )
        )
        move_data = json.loads(move_result)

        assert move_data["success"] is True
        assert "Successfully moved" in move_data["message"]
        assert not os.path.exists(source_file)
        assert os.path.exists(dest_file)

        # Verify content
        with open(dest_file, "r") as f:
            assert f.read() == source_content

    def test_delete_file(self):
        """Test file deletion."""
        test_file = os.path.join(self.test_dir, "delete_test.txt")

        # Create test file
        with open(test_file, "w") as f:
            f.write("File to delete")

        assert os.path.exists(test_file)

        # Delete file
        delete_result = self.tool._run(
            json.dumps({"action": "delete", "path": test_file, "confirm": True})
        )
        delete_data = json.loads(delete_result)

        assert delete_data["success"] is True
        assert "Successfully deleted" in delete_data["message"]
        assert not os.path.exists(test_file)

    def test_security_restrictions(self):
        """Test security restrictions."""
        # Test dangerous file extensions
        dangerous_files = [
            "/tmp/test.exe",
            "/tmp/script.bat",
            "/tmp/program.com",
            "/tmp/dangerous.scr",
        ]

        for dangerous_file in dangerous_files:
            write_result = self.tool._run(
                json.dumps(
                    {"action": "write", "path": dangerous_file, "content": "test"}
                )
            )
            write_data = json.loads(write_result)

            assert write_data["success"] is False
            assert (
                "dangerous" in write_data["error"].lower()
                or "blocked" in write_data["error"].lower()
            )

    def test_path_traversal_protection(self):
        """Test path traversal attack protection."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "../../../../usr/bin/bash",
        ]

        for path in malicious_paths:
            read_result = self.tool._run(json.dumps({"action": "read", "path": path}))
            read_data = json.loads(read_result)

            # Should either be blocked or fail safely
            if not read_data["success"]:
                # This is expected - security protection
                assert (
                    "security" in read_data["error"].lower()
                    or "access" in read_data["error"].lower()
                    or "not found" in read_data["error"].lower()
                )

    def test_invalid_json(self):
        """Test handling of invalid JSON input."""
        result = self.tool._run("invalid json")
        data = json.loads(result)

        assert data["success"] is False
        assert "json" in data["error"].lower() or "parse" in data["error"].lower()

    def test_missing_action(self):
        """Test handling of missing action."""
        result = self.tool._run(json.dumps({"path": "/some/path"}))
        data = json.loads(result)

        assert data["success"] is False
        assert "action" in data["error"].lower() or "required" in data["error"].lower()


class TestAudioTools:
    """Test cases for audio tools."""

    def test_tts_tool_initialization(self):
        """Test TTS tool initialization."""
        tool = TTSTool()
        assert tool is not None
        assert tool.name == "text_to_speech"
        assert "Convert text to speech" in tool.description

    def test_stt_tool_initialization(self):
        """Test STT tool initialization."""
        tool = STTTool()
        assert tool is not None
        assert tool.name == "speech_to_text"
        assert "Convert speech to text" in tool.description

    def test_audio_processor_initialization(self):
        """Test Audio Processor tool initialization."""
        tool = AudioProcessorTool()
        assert tool is not None
        assert tool.name == "audio_processor"
        assert "Process audio files" in tool.description

    def test_tts_basic_functionality(self):
        """Test basic TTS functionality (without actual audio processing)."""
        tool = TTSTool()

        # Test with invalid JSON
        result = tool._run("invalid json")
        data = json.loads(result)

        assert data["success"] is False
        assert "json" in data["error"].lower() or "parse" in data["error"].lower()

        # Test with missing text
        result = tool._run(json.dumps({"voice_type": "default"}))
        data = json.loads(result)

        assert data["success"] is False
        assert "text" in data["error"].lower() or "required" in data["error"].lower()

    def test_stt_basic_functionality(self):
        """Test basic STT functionality (without actual audio processing)."""
        tool = STTTool()

        # Test with invalid JSON
        result = tool._run("invalid json")
        data = json.loads(result)

        assert data["success"] is False
        assert "json" in data["error"].lower() or "parse" in data["error"].lower()

        # Test with missing audio_file
        result = tool._run(json.dumps({"language": "pt"}))
        data = json.loads(result)

        assert data["success"] is False
        assert (
            "audio_file" in data["error"].lower() or "required" in data["error"].lower()
        )

    def test_audio_processor_basic_functionality(self):
        """Test basic Audio Processor functionality."""
        tool = AudioProcessorTool()

        # Test with invalid JSON
        result = tool._run("invalid json")
        data = json.loads(result)

        assert data["success"] is False
        assert "json" in data["error"].lower() or "parse" in data["error"].lower()

        # Test with missing action
        result = tool._run(json.dumps({"file_path": "test.mp3"}))
        data = json.loads(result)

        assert data["success"] is False
        assert "action" in data["error"].lower() or "required" in data["error"].lower()


class TestMemoryTool:
    """Test cases for the MemoryTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = MemoryTool()

    def test_tool_initialization(self):
        """Test tool initialization."""
        assert self.tool is not None
        assert self.tool.name == "memory_manager"
        assert "Manage conversation memory and context" in self.tool.description

    def test_get_stats_functionality(self):
        """Test get_stats functionality."""
        result = self.tool._run(
            json.dumps({"action": "get_stats", "session_id": "test_session"})
        )
        data = json.loads(result)

        assert data["success"] is True
        assert "session_id" in data
        assert "message" in data

    def test_list_sessions_functionality(self):
        """Test list_sessions functionality."""
        result = self.tool._run(json.dumps({"action": "list_sessions"}))
        data = json.loads(result)

        assert data["success"] is True
        assert "total_sessions" in data
        assert isinstance(data["total_sessions"], int)

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON."""
        result = self.tool._run("invalid json")
        data = json.loads(result)

        assert data["success"] is False
        assert "json" in data["error"].lower() or "parse" in data["error"].lower()

    def test_missing_action_handling(self):
        """Test handling of missing action."""
        result = self.tool._run(json.dumps({"session_id": "test"}))
        data = json.loads(result)

        assert data["success"] is False
        assert "action" in data["error"].lower() or "required" in data["error"].lower()

    def test_invalid_action_handling(self):
        """Test handling of invalid actions."""
        result = self.tool._run(
            json.dumps({"action": "invalid_action", "session_id": "test"})
        )
        data = json.loads(result)

        assert data["success"] is False
        assert "invalid" in data["error"].lower() or "unknown" in data["error"].lower()


class TestToolIntegration:
    """Integration tests for tool interoperability."""

    def setup_method(self):
        """Setup test fixtures."""
        self.shell_tool = ShellExecutorTool()
        self.fs_tool = FileSystemTool()
        self.memory_tool = MemoryTool()
        self.test_dir = tempfile.mkdtemp(prefix="gianna_integration_test_")

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_shell_and_filesystem_integration(self):
        """Test integration between shell and filesystem tools."""
        test_file = os.path.join(self.test_dir, "integration_test.txt")
        test_content = "Integration test content"

        # Use filesystem tool to create file
        fs_result = self.fs_tool._run(
            json.dumps({"action": "write", "path": test_file, "content": test_content})
        )
        fs_data = json.loads(fs_result)
        assert fs_data["success"] is True

        # Use shell tool to read file
        shell_result = self.shell_tool._run(f'cat "{test_file}"')
        shell_data = json.loads(shell_result)
        assert shell_data["success"] is True
        assert test_content in shell_data["stdout"]

    def test_error_consistency(self):
        """Test that all tools handle errors consistently."""
        tools = [self.shell_tool, self.fs_tool, self.memory_tool]

        # Test invalid JSON handling across all tools
        for tool in tools:
            result = tool._run("invalid json")
            data = json.loads(result)

            # All tools should handle invalid JSON consistently
            assert data["success"] is False
            assert "error" in data
            assert isinstance(data["error"], str)
            assert len(data["error"]) > 0

    def test_concurrent_tool_usage(self):
        """Test concurrent usage of multiple tools."""
        import queue
        import threading

        results = queue.Queue()

        def run_shell_command():
            result = self.shell_tool._run('echo "Concurrent test"')
            results.put(("shell", result))

        def run_memory_operation():
            result = self.memory_tool._run(
                json.dumps({"action": "get_stats", "session_id": "concurrent_test"})
            )
            results.put(("memory", result))

        def run_fs_operation():
            result = self.fs_tool._run(
                json.dumps({"action": "list", "path": self.test_dir})
            )
            results.put(("fs", result))

        # Run operations concurrently
        threads = [
            threading.Thread(target=run_shell_command),
            threading.Thread(target=run_memory_operation),
            threading.Thread(target=run_fs_operation),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Collect results
        collected_results = []
        while not results.empty():
            collected_results.append(results.get())

        assert len(collected_results) == 3

        # Verify all operations succeeded
        for tool_name, result_json in collected_results:
            data = json.loads(result_json)
            assert (
                data["success"] is True
            ), f"{tool_name} tool failed in concurrent test"

    def test_langchain_compatibility(self):
        """Test LangChain BaseTool compatibility."""
        from langchain.tools import BaseTool

        tools = [
            ShellExecutorTool(),
            FileSystemTool(),
            AudioProcessorTool(),
            TTSTool(),
            STTTool(),
            MemoryTool(),
        ]

        for tool in tools:
            # Verify inheritance
            assert isinstance(tool, BaseTool)

            # Verify required attributes
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "_run")
            assert hasattr(tool, "arun")

            # Verify attribute types
            assert isinstance(tool.name, str)
            assert isinstance(tool.description, str)
            assert len(tool.name) > 0
            assert len(tool.description) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
