"""
File system operations tools with security controls.

This module provides safe file system operations with proper validation,
sandboxing, and integration with Gianna's security model.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from langchain.tools import BaseTool
from loguru import logger
from pydantic import Field


class FileSystemTool(BaseTool):
    """
    Safe file system operations tool with security controls.

    Provides controlled file system access with validation, sandboxing,
    and comprehensive error handling for agent use.
    """

    name: str = "filesystem_manager"
    description: str = """Perform safe file system operations.
    Input: JSON with 'action' (read|write|list|create_dir|delete|copy|move|info), 'path', optional 'content', 'target_path'
    Output: JSON with operation results and file information"""

    # Security configuration
    allowed_extensions: Optional[List[str]] = Field(
        default=None, description="List of allowed file extensions"
    )
    blocked_extensions: List[str] = Field(
        default_factory=lambda: [
            ".exe",
            ".bat",
            ".cmd",
            ".com",
            ".pif",
            ".scr",
            ".vbs",
            ".js",
            ".jar",
            ".msi",
            ".dll",
            ".sys",
            ".drv",
        ],
        description="List of blocked file extensions for security",
    )
    max_file_size: int = Field(
        default=100 * 1024 * 1024, description="Maximum file size in bytes"  # 100MB
    )
    sandbox_path: Optional[str] = Field(
        default=None, description="Sandbox directory path for operations"
    )

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Set default allowed extensions if none specified
        if self.allowed_extensions is None:
            self.allowed_extensions = [
                ".txt",
                ".md",
                ".py",
                ".js",
                ".json",
                ".yaml",
                ".yml",
                ".csv",
                ".log",
                ".conf",
                ".cfg",
                ".ini",
                ".xml",
                ".html",
                ".css",
                ".sql",
                ".sh",
                ".dockerfile",
                ".gitignore",
            ]

        # Set default sandbox to current working directory if none specified
        if self.sandbox_path is None:
            self.sandbox_path = os.getcwd()

    def _run(self, input_data: str) -> str:
        """
        Perform file system operations with security validation.

        Args:
            input_data: JSON string with operation details

        Returns:
            JSON string with operation results
        """
        try:
            # Parse input
            if isinstance(input_data, str):
                data = json.loads(input_data)
            else:
                data = input_data

            action = data.get("action", "").lower()
            file_path = data.get("path", "")

            # Validate and normalize path
            validation_result = self._validate_path(file_path)
            if not validation_result["valid"]:
                return json.dumps(
                    {
                        "error": f"Path validation failed: {validation_result['reason']}",
                        "success": False,
                        "path": file_path,
                    }
                )

            normalized_path = validation_result["normalized_path"]

            # Route to appropriate action
            if action == "read":
                return self._read_file(normalized_path)
            elif action == "write":
                content = data.get("content", "")
                return self._write_file(
                    normalized_path, content, data.get("append", False)
                )
            elif action == "list":
                return self._list_directory(
                    normalized_path, data.get("show_hidden", False)
                )
            elif action == "create_dir":
                return self._create_directory(normalized_path)
            elif action == "delete":
                return self._delete_path(normalized_path, data.get("confirm", False))
            elif action == "copy":
                target_path = data.get("target_path", "")
                return self._copy_path(normalized_path, target_path)
            elif action == "move":
                target_path = data.get("target_path", "")
                return self._move_path(normalized_path, target_path)
            elif action == "info":
                return self._get_path_info(normalized_path)
            elif action == "search":
                pattern = data.get("pattern", "")
                return self._search_files(normalized_path, pattern)
            else:
                return json.dumps(
                    {
                        "error": f"Unknown action: {action}. Available: read, write, list, create_dir, delete, copy, move, info, search",
                        "success": False,
                    }
                )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {e}")
            return json.dumps(
                {"error": f"Invalid JSON input: {str(e)}", "success": False}
            )
        except Exception as e:
            logger.error(f"File system operation error: {e}")
            return json.dumps(
                {"error": f"File system operation failed: {str(e)}", "success": False}
            )

    def _validate_path(self, file_path: str) -> dict:
        """
        Validate file path for security and accessibility.

        Args:
            file_path: Path to validate

        Returns:
            Dict with validation results
        """
        if not file_path:
            return {"valid": False, "reason": "Empty path provided"}

        try:
            # Resolve path and check if it's within sandbox
            path = Path(file_path).resolve()
            sandbox = Path(self.sandbox_path).resolve()

            # Check if path is within sandbox
            try:
                path.relative_to(sandbox)
            except ValueError:
                return {"valid": False, "reason": f"Path outside sandbox: {file_path}"}

            # Check file extension
            if path.is_file() or (not path.exists() and path.suffix):
                extension = path.suffix.lower()
                if extension in self.blocked_extensions:
                    return {
                        "valid": False,
                        "reason": f"Blocked file extension: {extension}",
                    }

                if self.allowed_extensions and extension not in self.allowed_extensions:
                    return {
                        "valid": False,
                        "reason": f"Extension not in allowed list: {extension}",
                    }

            return {
                "valid": True,
                "normalized_path": str(path),
                "relative_path": str(path.relative_to(sandbox)),
            }

        except Exception as e:
            return {"valid": False, "reason": f"Path validation error: {str(e)}"}

    def _read_file(self, file_path: str) -> str:
        """Read file content with size validation."""
        try:
            path = Path(file_path)

            if not path.exists():
                return json.dumps(
                    {"error": f"File does not exist: {file_path}", "success": False}
                )

            if not path.is_file():
                return json.dumps(
                    {"error": f"Path is not a file: {file_path}", "success": False}
                )

            # Check file size
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                return json.dumps(
                    {
                        "error": f"File too large: {file_size} bytes (max: {self.max_file_size})",
                        "success": False,
                    }
                )

            # Read file content
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            return json.dumps(
                {
                    "success": True,
                    "action": "read",
                    "path": file_path,
                    "content": content,
                    "size": file_size,
                    "encoding": "utf-8",
                    "message": f"Successfully read {path.name} ({file_size} bytes)",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to read file: {str(e)}",
                    "success": False,
                    "path": file_path,
                }
            )

    def _write_file(self, file_path: str, content: str, append: bool = False) -> str:
        """Write content to file with validation."""
        try:
            path = Path(file_path)

            # Check content size
            content_size = len(content.encode("utf-8"))
            if content_size > self.max_file_size:
                return json.dumps(
                    {
                        "error": f"Content too large: {content_size} bytes (max: {self.max_file_size})",
                        "success": False,
                    }
                )

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            mode = "a" if append else "w"
            with open(path, mode, encoding="utf-8") as f:
                f.write(content)

            # Get final file size
            final_size = path.stat().st_size

            return json.dumps(
                {
                    "success": True,
                    "action": "write",
                    "path": file_path,
                    "mode": "append" if append else "overwrite",
                    "content_size": content_size,
                    "file_size": final_size,
                    "message": f"Successfully {'appended to' if append else 'wrote'} {path.name}",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to write file: {str(e)}",
                    "success": False,
                    "path": file_path,
                }
            )

    def _list_directory(self, dir_path: str, show_hidden: bool = False) -> str:
        """List directory contents."""
        try:
            path = Path(dir_path)

            if not path.exists():
                return json.dumps(
                    {"error": f"Directory does not exist: {dir_path}", "success": False}
                )

            if not path.is_dir():
                return json.dumps(
                    {"error": f"Path is not a directory: {dir_path}", "success": False}
                )

            # List directory contents
            items = []
            for item in path.iterdir():
                # Skip hidden files unless requested
                if not show_hidden and item.name.startswith("."):
                    continue

                try:
                    stat = item.stat()
                    item_info = {
                        "name": item.name,
                        "path": str(item),
                        "type": "directory" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "permissions": oct(stat.st_mode)[-3:],
                        "extension": item.suffix.lower() if item.is_file() else "",
                    }
                    items.append(item_info)
                except (OSError, PermissionError) as e:
                    # Skip items we can't access
                    logger.warning(f"Cannot access {item}: {e}")
                    continue

            # Sort items: directories first, then files
            items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))

            return json.dumps(
                {
                    "success": True,
                    "action": "list",
                    "path": dir_path,
                    "items": items,
                    "total_items": len(items),
                    "show_hidden": show_hidden,
                    "message": f"Listed {len(items)} items in {path.name}",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to list directory: {str(e)}",
                    "success": False,
                    "path": dir_path,
                }
            )

    def _create_directory(self, dir_path: str) -> str:
        """Create directory with parents."""
        try:
            path = Path(dir_path)

            if path.exists():
                if path.is_dir():
                    return json.dumps(
                        {
                            "success": True,
                            "action": "create_dir",
                            "path": dir_path,
                            "message": f"Directory already exists: {path.name}",
                            "already_existed": True,
                        }
                    )
                else:
                    return json.dumps(
                        {
                            "error": f"Path exists but is not a directory: {dir_path}",
                            "success": False,
                        }
                    )

            # Create directory
            path.mkdir(parents=True, exist_ok=True)

            return json.dumps(
                {
                    "success": True,
                    "action": "create_dir",
                    "path": dir_path,
                    "message": f"Successfully created directory: {path.name}",
                    "already_existed": False,
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to create directory: {str(e)}",
                    "success": False,
                    "path": dir_path,
                }
            )

    def _delete_path(self, file_path: str, confirm: bool = False) -> str:
        """Delete file or directory with confirmation."""
        try:
            if not confirm:
                return json.dumps(
                    {
                        "error": "Deletion requires confirmation. Set 'confirm': true",
                        "success": False,
                        "path": file_path,
                        "warning": "This action will permanently delete the file/directory",
                    }
                )

            path = Path(file_path)

            if not path.exists():
                return json.dumps(
                    {
                        "success": True,
                        "action": "delete",
                        "path": file_path,
                        "message": f"Path does not exist: {path.name}",
                        "was_deleted": False,
                    }
                )

            # Delete file or directory
            if path.is_file():
                path.unlink()
                item_type = "file"
            elif path.is_dir():
                shutil.rmtree(path)
                item_type = "directory"
            else:
                return json.dumps(
                    {
                        "error": f"Cannot delete special file: {file_path}",
                        "success": False,
                    }
                )

            return json.dumps(
                {
                    "success": True,
                    "action": "delete",
                    "path": file_path,
                    "item_type": item_type,
                    "message": f"Successfully deleted {item_type}: {path.name}",
                    "was_deleted": True,
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to delete: {str(e)}",
                    "success": False,
                    "path": file_path,
                }
            )

    def _copy_path(self, source_path: str, target_path: str) -> str:
        """Copy file or directory to target location."""
        try:
            source = Path(source_path)

            # Validate target path
            target_validation = self._validate_path(target_path)
            if not target_validation["valid"]:
                return json.dumps(
                    {
                        "error": f"Target path validation failed: {target_validation['reason']}",
                        "success": False,
                    }
                )

            target = Path(target_validation["normalized_path"])

            if not source.exists():
                return json.dumps(
                    {"error": f"Source does not exist: {source_path}", "success": False}
                )

            # Perform copy
            if source.is_file():
                shutil.copy2(source, target)
                item_type = "file"
            elif source.is_dir():
                shutil.copytree(source, target)
                item_type = "directory"
            else:
                return json.dumps(
                    {
                        "error": f"Cannot copy special file: {source_path}",
                        "success": False,
                    }
                )

            return json.dumps(
                {
                    "success": True,
                    "action": "copy",
                    "source_path": source_path,
                    "target_path": str(target),
                    "item_type": item_type,
                    "message": f"Successfully copied {item_type}: {source.name} -> {target.name}",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to copy: {str(e)}",
                    "success": False,
                    "source_path": source_path,
                    "target_path": target_path,
                }
            )

    def _move_path(self, source_path: str, target_path: str) -> str:
        """Move file or directory to target location."""
        try:
            source = Path(source_path)

            # Validate target path
            target_validation = self._validate_path(target_path)
            if not target_validation["valid"]:
                return json.dumps(
                    {
                        "error": f"Target path validation failed: {target_validation['reason']}",
                        "success": False,
                    }
                )

            target = Path(target_validation["normalized_path"])

            if not source.exists():
                return json.dumps(
                    {"error": f"Source does not exist: {source_path}", "success": False}
                )

            # Perform move
            shutil.move(str(source), str(target))
            item_type = "directory" if target.is_dir() else "file"

            return json.dumps(
                {
                    "success": True,
                    "action": "move",
                    "source_path": source_path,
                    "target_path": str(target),
                    "item_type": item_type,
                    "message": f"Successfully moved {item_type}: {source.name} -> {target.name}",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to move: {str(e)}",
                    "success": False,
                    "source_path": source_path,
                    "target_path": target_path,
                }
            )

    def _get_path_info(self, file_path: str) -> str:
        """Get detailed information about a path."""
        try:
            path = Path(file_path)

            if not path.exists():
                return json.dumps(
                    {
                        "success": True,
                        "action": "info",
                        "path": file_path,
                        "exists": False,
                        "message": f"Path does not exist: {file_path}",
                    }
                )

            stat = path.stat()

            info = {
                "path": file_path,
                "name": path.name,
                "parent": str(path.parent),
                "exists": True,
                "type": "directory" if path.is_dir() else "file",
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
                "is_readable": os.access(path, os.R_OK),
                "is_writable": os.access(path, os.W_OK),
                "is_executable": os.access(path, os.X_OK),
            }

            if path.is_file():
                info["extension"] = path.suffix.lower()
                info["stem"] = path.stem
            elif path.is_dir():
                try:
                    info["item_count"] = len(list(path.iterdir()))
                except (OSError, PermissionError):
                    info["item_count"] = "Permission denied"

            return json.dumps(
                {
                    "success": True,
                    "action": "info",
                    "info": info,
                    "message": f"Retrieved information for: {path.name}",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Failed to get path info: {str(e)}",
                    "success": False,
                    "path": file_path,
                }
            )

    def _search_files(self, search_path: str, pattern: str) -> str:
        """Search for files matching pattern."""
        try:
            path = Path(search_path)

            if not path.exists():
                return json.dumps(
                    {
                        "error": f"Search path does not exist: {search_path}",
                        "success": False,
                    }
                )

            if not path.is_dir():
                return json.dumps(
                    {
                        "error": f"Search path is not a directory: {search_path}",
                        "success": False,
                    }
                )

            if not pattern.strip():
                return json.dumps(
                    {"error": "Search pattern cannot be empty", "success": False}
                )

            # Search for files matching pattern
            matches = []
            try:
                for item in path.rglob(pattern):
                    if item.is_file():
                        try:
                            stat = item.stat()
                            match_info = {
                                "path": str(item),
                                "name": item.name,
                                "relative_path": str(item.relative_to(path)),
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(
                                    stat.st_mtime
                                ).isoformat(),
                                "extension": item.suffix.lower(),
                            }
                            matches.append(match_info)
                        except (OSError, PermissionError):
                            continue
            except (OSError, PermissionError) as e:
                return json.dumps(
                    {
                        "error": f"Permission denied during search: {str(e)}",
                        "success": False,
                    }
                )

            return json.dumps(
                {
                    "success": True,
                    "action": "search",
                    "search_path": search_path,
                    "pattern": pattern,
                    "matches": matches,
                    "total_matches": len(matches),
                    "message": f"Found {len(matches)} files matching '{pattern}'",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Search failed: {str(e)}",
                    "success": False,
                    "search_path": search_path,
                    "pattern": pattern,
                }
            )

    async def _arun(self, input_data: str) -> str:
        """Async version - delegates to sync version for now."""
        return self._run(input_data)
