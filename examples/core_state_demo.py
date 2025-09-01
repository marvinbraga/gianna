#!/usr/bin/env python3
"""
Gianna Core State Management Demo

This script demonstrates the core state management system implementation
for FASE 1 of the Gianna AI Assistant project.
"""

import os
import tempfile
from datetime import datetime

from gianna.core import (
    StateManager,
    create_initial_state,
    state_to_dict,
    validate_state,
)


def main():
    """Demonstrate core state management functionality."""
    print("🤖 Gianna Core State Management Demo")
    print("=" * 50)

    # Create a temporary database for demo
    with tempfile.NamedTemporaryFile(delete=False, suffix="_demo.db") as tmp:
        db_path = tmp.name

    try:
        # Initialize StateManager
        print("\n1. Initializing StateManager...")
        state_manager = StateManager(db_path)
        print(
            f"   ✅ StateManager initialized with {state_manager._checkpointer_type} checkpointer"
        )

        # Create a new session
        print("\n2. Creating new session...")
        session_id = state_manager.create_session(
            {
                "language": "pt-br",
                "voice_preference": "female",
                "response_style": "friendly",
            }
        )
        print(f"   ✅ Created session: {session_id[:8]}...")

        # Create initial state
        print("\n3. Creating initial state...")
        state = create_initial_state(session_id)
        print(f"   ✅ Initial state created with {len(state.keys())} components")

        # Modify state to simulate usage
        print("\n4. Simulating conversation...")
        state["conversation"].messages.extend(
            [
                {
                    "role": "user",
                    "content": "Olá, Gianna! Como você está?",
                    "timestamp": datetime.now().isoformat(),
                    "source": "voice",
                },
                {
                    "role": "assistant",
                    "content": "Olá! Estou funcionando perfeitamente e pronta para ajudar!",
                    "timestamp": datetime.now().isoformat(),
                    "source": "text",
                },
            ]
        )

        state["audio"].current_mode = "speaking"
        state["commands"].execution_history.append(
            {
                "command": "greeting",
                "result": "success",
                "timestamp": datetime.now().isoformat(),
            }
        )

        print(f"   ✅ Added {len(state['conversation'].messages)} messages")
        print(f"   ✅ Audio mode: {state['audio'].current_mode}")
        print(f"   ✅ Commands executed: {len(state['commands'].execution_history)}")

        # Save state
        print("\n5. Persisting state...")
        state_manager.save_state(session_id, state)
        print("   ✅ State saved to database")

        # Load state back
        print("\n6. Loading state from database...")
        loaded_state = state_manager.load_state(session_id)
        print("   ✅ State loaded successfully")
        print(f"   📝 Messages: {len(loaded_state['conversation'].messages)}")
        print(f"   🔊 Audio mode: {loaded_state['audio'].current_mode}")

        # Test state validation
        print("\n7. Testing state validation...")
        state_dict = state_to_dict(state)
        validated_state = validate_state(state_dict)
        print("   ✅ State validation passed")

        # Test LangGraph integration
        print("\n8. Testing LangGraph integration...")
        config = state_manager.get_config(session_id)
        print(f"   ✅ LangGraph config: {config}")
        print(f"   🔧 Checkpointer ready: {state_manager.checkpointer is not None}")

        # Show session management
        print("\n9. Session management...")
        sessions = state_manager.get_session_list()
        print(f"   ✅ Total sessions: {len(sessions)}")

        if sessions:
            session = sessions[0]
            print(f"   📅 Created: {session['created_at']}")
            print(f"   🕐 Last activity: {session['last_activity']}")

        print("\n" + "=" * 50)
        print("🎉 All core state management features working correctly!")
        print(f"💾 Database file: {db_path}")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        raise
    finally:
        # Clean up (optional - comment out to inspect the database)
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"🧹 Cleaned up database file")


if __name__ == "__main__":
    main()
