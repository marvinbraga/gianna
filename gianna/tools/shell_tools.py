"""
Shell execution tools that integrate with Gianna's command system.

This module provides safe shell command execution with timeout controls,
validation, and integration with the existing Gianna command infrastructure.
"""

import json
import os
import shlex
import subprocess
from typing import Any, List, Optional

from langchain.tools import BaseTool
from loguru import logger
from pydantic import Field


class ShellExecutorTool(BaseTool):
    """
    Tool for safe shell command execution with timeout and validation.

    Integrates with Gianna's existing shell command system while providing
    LangChain-compatible tool interface for agent use.
    """

    name: str = "shell_executor"
    description: str = """Execute shell commands safely with timeout and validation.
    Input: Command string to execute (will be validated for safety)
    Output: JSON with exit_code, stdout, stderr, and success status"""

    # Tool configuration
    timeout: int = Field(default=30, description="Command timeout in seconds")
    allowed_commands: Optional[List[str]] = Field(
        default=None, description="List of allowed command prefixes for security"
    )
    dangerous_commands: List[str] = Field(
        default_factory=lambda: [
            "rm -rf",
            "format",
            "mkfs",
            "dd if=",
            ":(){ :|:& };:",
            "sudo rm",
            "chmod -R 777",
            "> /dev/",
            "cat /dev/random",
        ],
        description="List of dangerous command patterns to block",
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Set default allowed commands if none specified
        if self.allowed_commands is None:
            self.allowed_commands = [
                "ls",
                "pwd",
                "whoami",
                "date",
                "echo",
                "cat",
                "grep",
                "find",
                "which",
                "python",
                "pip",
                "git",
                "curl",
                "wget",
                "mkdir",
                "touch",
                "cp",
                "mv",
                "chmod",
                "chown",
            ]

    def _run(self, command: str) -> str:
        """
        Execute shell command with safety checks and timeout.

        Args:
            command: Shell command to execute

        Returns:
            JSON string with execution results
        """
        try:
            # Validate command safety
            validation_result = self._validate_command(command)
            if not validation_result["safe"]:
                return json.dumps(
                    {
                        "error": f"Command blocked for security: {validation_result['reason']}",
                        "success": False,
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": validation_result["reason"],
                    }
                )

            logger.info(f"Executing shell command: {command}")

            # Execute command with timeout
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.getcwd(),
            )

            response = {
                "exit_code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "success": result.returncode == 0,
                "command": command,
            }

            logger.info(f"Command executed with exit code: {result.returncode}")
            return json.dumps(response, indent=2)

        except subprocess.TimeoutExpired:
            error_msg = f"Command timeout after {self.timeout} seconds"
            logger.warning(error_msg)
            return json.dumps(
                {
                    "error": error_msg,
                    "success": False,
                    "exit_code": -2,
                    "stdout": "",
                    "stderr": error_msg,
                    "command": command,
                }
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            return json.dumps(
                {
                    "error": str(e),
                    "success": False,
                    "exit_code": e.returncode,
                    "stdout": e.stdout if e.stdout else "",
                    "stderr": e.stderr if e.stderr else str(e),
                    "command": command,
                }
            )

        except Exception as e:
            logger.error(f"Unexpected error executing command: {e}")
            return json.dumps(
                {
                    "error": f"Unexpected error: {str(e)}",
                    "success": False,
                    "exit_code": -3,
                    "stdout": "",
                    "stderr": str(e),
                    "command": command,
                }
            )

    def _validate_command(self, command: str) -> dict:
        """
        Validate command for security and safety.

        Args:
            command: Command to validate

        Returns:
            Dict with 'safe' boolean and 'reason' if unsafe
        """
        # Check for dangerous patterns
        command_lower = command.lower()
        for dangerous in self.dangerous_commands:
            if dangerous in command_lower:
                return {
                    "safe": False,
                    "reason": f"Contains dangerous pattern: {dangerous}",
                }

        # Check if command starts with allowed prefix
        if self.allowed_commands:
            command_parts = shlex.split(command)
            if command_parts:
                base_command = command_parts[0].split("/")[
                    -1
                ]  # Get command name without path
                if not any(
                    base_command.startswith(allowed)
                    for allowed in self.allowed_commands
                ):
                    return {
                        "safe": False,
                        "reason": f"Command '{base_command}' not in allowed list",
                    }

        # Additional safety checks
        if any(char in command for char in [";", "|", "&", "`", "$("]):
            # Allow some safe uses of these characters
            safe_patterns = ["echo ", "grep ", "find "]
            if not any(pattern in command for pattern in safe_patterns):
                return {
                    "safe": False,
                    "reason": "Contains potentially dangerous shell operators",
                }

        return {"safe": True, "reason": ""}

    async def _arun(self, command: str) -> str:
        """Async version - delegates to sync version for now."""
        return self._run(command)
