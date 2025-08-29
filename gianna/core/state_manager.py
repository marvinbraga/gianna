"""
State Manager for Gianna AI Assistant

This module provides persistent state management using SQLite and LangGraph checkpointer.
It handles session persistence, state serialization, and database operations.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    # Try to import SQLite checkpointer (may require separate package)
    from langgraph_checkpoint_sqlite import SqliteSaver
except ImportError:
    # Fallback to memory checkpointer for now
    from langgraph.checkpoint.memory import MemorySaver as SqliteSaver

from loguru import logger

from .state import AudioState, CommandState, ConversationState, GiannaState


class StateManager:
    """
    Manages persistent state storage and retrieval for Gianna sessions.

    This class provides a unified interface for state persistence using SQLite
    and integrates with LangGraph's checkpointer system for workflow state management.
    """

    def __init__(self, db_path: str = "gianna_state.db"):
        """
        Initialize the StateManager with database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)

        # Initialize checkpointer based on available implementation
        try:
            # Try to use SQLite checkpointer if available
            if hasattr(SqliteSaver, "from_conn_string"):
                self.checkpointer = SqliteSaver.from_conn_string(str(self.db_path))
                self._checkpointer_type = "sqlite"
            else:
                # Fallback to memory checkpointer
                self.checkpointer = SqliteSaver()
                self._checkpointer_type = "memory"
                logger.warning(
                    "Using memory checkpointer as fallback - state will not persist between restarts"
                )
        except Exception as e:
            # Final fallback to memory checkpointer
            self.checkpointer = SqliteSaver()
            self._checkpointer_type = "memory"
            logger.warning(
                f"Failed to initialize SQLite checkpointer ({e}), using memory fallback"
            )

        self._init_database()
        logger.info(
            f"StateManager initialized with database: {self.db_path} (checkpointer: {self._checkpointer_type})"
        )

    def _init_database(self) -> None:
        """
        Initialize the database schema with required tables.

        Creates tables for user sessions, conversation history, and system metadata
        if they don't already exist.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # User sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_data JSON,
                    user_preferences JSON,
                    context_summary TEXT
                )
            """
            )

            # Conversation messages table for better querying
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON,
                    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
                )
            """
            )

            # Audio state history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    mode TEXT,
                    speech_type TEXT,
                    language TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settings JSON,
                    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
                )
            """
            )

            # Command execution history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS command_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    command TEXT,
                    result JSON,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN,
                    execution_time REAL,
                    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id)
                )
            """
            )

            conn.commit()
            logger.info("Database schema initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
        finally:
            conn.close()

    def get_config(self, session_id: str) -> Dict[str, Any]:
        """
        Get LangGraph configuration for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Dict containing LangGraph configuration
        """
        return {"configurable": {"thread_id": session_id}}

    def create_session(self, user_preferences: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new session with optional user preferences.

        Args:
            user_preferences: Optional dictionary of user preferences

        Returns:
            str: New session ID
        """
        session_id = str(uuid4())

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO user_sessions (session_id, session_data, user_preferences)
                VALUES (?, ?, ?)
            """,
                (session_id, json.dumps({}), json.dumps(user_preferences or {})),
            )

            conn.commit()
            logger.info(f"Created new session: {session_id}")

        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")
            raise
        finally:
            conn.close()

        return session_id

    def save_state(self, session_id: str, state: GiannaState) -> None:
        """
        Save complete session state to database.

        Args:
            session_id: Session identifier
            state: Complete GiannaState to save
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Update session data
            cursor.execute(
                """
                UPDATE user_sessions
                SET last_activity = CURRENT_TIMESTAMP,
                    session_data = ?,
                    user_preferences = ?,
                    context_summary = ?
                WHERE session_id = ?
            """,
                (
                    json.dumps(
                        {
                            "conversation": state["conversation"].model_dump(),
                            "audio": state["audio"].model_dump(),
                            "commands": state["commands"].model_dump(),
                            "metadata": state["metadata"],
                        }
                    ),
                    json.dumps(state["conversation"].user_preferences),
                    state["conversation"].context_summary,
                    session_id,
                ),
            )

            # Save conversation messages
            for message in state["conversation"].messages:
                cursor.execute(
                    """
                    INSERT INTO conversation_messages
                    (session_id, role, content, metadata)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        session_id,
                        message.get("role", ""),
                        message.get("content", ""),
                        json.dumps(
                            {
                                k: v
                                for k, v in message.items()
                                if k not in ["role", "content"]
                            }
                        ),
                    ),
                )

            # Save audio state
            cursor.execute(
                """
                INSERT INTO audio_history
                (session_id, mode, speech_type, language, settings)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    state["audio"].current_mode,
                    state["audio"].speech_type,
                    state["audio"].language,
                    json.dumps(state["audio"].voice_settings),
                ),
            )

            conn.commit()
            logger.info(f"Saved state for session: {session_id}")

        except Exception as e:
            logger.error(f"Error saving state for session {session_id}: {e}")
            raise
        finally:
            conn.close()

    def load_state(self, session_id: str) -> Optional[GiannaState]:
        """
        Load complete session state from database.

        Args:
            session_id: Session identifier

        Returns:
            Optional[GiannaState]: Loaded state or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Load session data
            cursor.execute(
                """
                SELECT session_data, user_preferences, context_summary
                FROM user_sessions
                WHERE session_id = ?
            """,
                (session_id,),
            )

            row = cursor.fetchone()
            if not row:
                logger.warning(f"Session not found: {session_id}")
                return None

            session_data, user_preferences, context_summary = row

            # Parse session data
            if session_data:
                data = json.loads(session_data)
            else:
                data = {}

            # Load conversation messages
            cursor.execute(
                """
                SELECT role, content, metadata, timestamp
                FROM conversation_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """,
                (session_id,),
            )

            messages = []
            for role, content, metadata_json, timestamp in cursor.fetchall():
                message = {"role": role, "content": content, "timestamp": timestamp}
                if metadata_json:
                    message.update(json.loads(metadata_json))
                messages.append(message)

            # Construct state
            state = GiannaState(
                conversation=ConversationState(
                    messages=messages,
                    session_id=session_id,
                    user_preferences=json.loads(user_preferences or "{}"),
                    context_summary=context_summary or "",
                ),
                audio=AudioState(**data.get("audio", {})),
                commands=CommandState(**data.get("commands", {})),
                metadata=data.get("metadata", {}),
            )

            logger.info(f"Loaded state for session: {session_id}")
            return state

        except Exception as e:
            logger.error(f"Error loading state for session {session_id}: {e}")
            return None
        finally:
            conn.close()

    def update_last_activity(self, session_id: str) -> None:
        """
        Update the last activity timestamp for a session.

        Args:
            session_id: Session identifier
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE user_sessions
                SET last_activity = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """,
                (session_id,),
            )

            conn.commit()

        except Exception as e:
            logger.error(f"Error updating last activity for session {session_id}: {e}")
        finally:
            conn.close()

    def get_session_list(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get a list of recent sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session information dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT session_id, created_at, last_activity, context_summary
                FROM user_sessions
                ORDER BY last_activity DESC
                LIMIT ?
            """,
                (limit,),
            )

            sessions = []
            for row in cursor.fetchall():
                sessions.append(
                    {
                        "session_id": row[0],
                        "created_at": row[1],
                        "last_activity": row[2],
                        "context_summary": row[3] or "",
                    }
                )

            return sessions

        except Exception as e:
            logger.error(f"Error getting session list: {e}")
            return []
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all related data.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session was deleted, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete from all tables
            cursor.execute(
                "DELETE FROM command_history WHERE session_id = ?", (session_id,)
            )
            cursor.execute(
                "DELETE FROM audio_history WHERE session_id = ?", (session_id,)
            )
            cursor.execute(
                "DELETE FROM conversation_messages WHERE session_id = ?", (session_id,)
            )
            cursor.execute(
                "DELETE FROM user_sessions WHERE session_id = ?", (session_id,)
            )

            deleted = cursor.rowcount > 0
            conn.commit()

            if deleted:
                logger.info(f"Deleted session: {session_id}")
            else:
                logger.warning(f"Session not found for deletion: {session_id}")

            return deleted

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
        finally:
            conn.close()

    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up sessions older than specified days.

        Args:
            days_old: Number of days after which sessions are considered old

        Returns:
            int: Number of sessions deleted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get old session IDs
            cursor.execute(
                """
                SELECT session_id FROM user_sessions
                WHERE last_activity < datetime('now', '-{} days')
            """.format(
                    days_old
                )
            )

            old_sessions = [row[0] for row in cursor.fetchall()]

            # Delete old sessions
            deleted_count = 0
            for session_id in old_sessions:
                if self.delete_session(session_id):
                    deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} old sessions")
            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return 0
        finally:
            conn.close()
