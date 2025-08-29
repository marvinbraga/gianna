#!/usr/bin/env python3
"""
Comprehensive Tests for ReAct Agents System - FASE 2

This module provides comprehensive tests for all ReAct agents in the Gianna
system, validating agent creation, execution, lifecycle management, and
integration with tools and state management.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from gianna.agents.base_agent import AgentConfig
from gianna.agents.react_agents import (
    AudioAgent,
    CommandAgent,
    ConversationAgent,
    GiannaReActAgent,
    MemoryAgent,
)
from gianna.core.state import GiannaState, create_initial_state
from gianna.tools import (
    AudioProcessorTool,
    FileSystemTool,
    MemoryTool,
    ShellExecutorTool,
    STTTool,
    TTSTool,
)


class MockLanguageModel:
    """Mock language model for testing."""

    def __init__(self, response: str = "Mock LLM response"):
        self.response = response
        self.call_count = 0
        self.last_input = None

    def invoke(self, input_data: Any) -> str:
        """Mock invoke method."""
        self.call_count += 1
        self.last_input = input_data
        return self.response

    def __call__(self, *args, **kwargs):
        """Allow direct calling."""
        return self.invoke(*args, **kwargs)


class TestGiannaReActAgent:
    """Test cases for the base GiannaReActAgent class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = MockLanguageModel("Base agent response")
        self.mock_tools = [Mock(), Mock()]
        self.config = AgentConfig(
            name="test_agent", description="Test agent", max_iterations=10
        )

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = GiannaReActAgent(
            name="test_agent",
            llm=self.mock_llm,
            tools=self.mock_tools,
            config=self.config,
        )

        assert agent.name == "test_agent"
        assert agent.llm == self.mock_llm
        assert agent.tools == self.mock_tools
        assert agent.config == self.config
        assert agent.current_iteration == 0

    def test_system_message_property(self):
        """Test system message property."""
        agent = GiannaReActAgent(
            name="test_agent",
            llm=self.mock_llm,
            tools=self.mock_tools,
            config=self.config,
        )

        system_msg = agent.system_message
        assert isinstance(system_msg, str)
        assert len(system_msg) > 0
        assert "ReAct" in system_msg
        assert "THOUGHT" in system_msg
        assert "ACTION" in system_msg
        assert "OBSERVATION" in system_msg

    def test_prepare_agent_state(self):
        """Test state preparation for agent execution."""
        agent = GiannaReActAgent(
            name="test_agent",
            llm=self.mock_llm,
            tools=self.mock_tools,
            config=self.config,
        )

        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Test message"}
        ]

        prepared_state = agent._prepare_agent_state(state)

        assert isinstance(prepared_state, dict)
        assert "conversation_history" in prepared_state
        assert "session_id" in prepared_state
        assert "user_preferences" in prepared_state
        assert "audio_mode" in prepared_state
        assert "language" in prepared_state
        assert "command_history" in prepared_state
        assert "metadata" in prepared_state

        # Verify data extraction
        assert prepared_state["session_id"] == "test_session"
        assert len(prepared_state["conversation_history"]) == 1
        assert prepared_state["conversation_history"][0]["content"] == "Test message"

    def test_process_agent_output(self):
        """Test processing of agent output."""
        agent = GiannaReActAgent(
            name="test_agent",
            llm=self.mock_llm,
            tools=self.mock_tools,
            config=self.config,
        )

        state = create_initial_state("test_session")

        # Test with dict output containing messages
        mock_output = {
            "messages": [
                {"role": "user", "content": "Test input"},
                {"role": "assistant", "content": "Test response"},
            ]
        }

        processed = agent._process_agent_output(mock_output, state)

        assert isinstance(processed, dict)
        assert "agent_name" in processed
        assert "content" in processed
        assert "messages" in processed
        assert "raw_output" in processed
        assert "success" in processed

        assert processed["agent_name"] == "test_agent"
        assert processed["success"] is True
        assert processed["content"] == "Test response"
        assert processed["messages"] == mock_output["messages"]

    @patch("gianna.agents.react_agents.LANGGRAPH_AVAILABLE", False)
    def test_fallback_execution(self):
        """Test fallback execution when LangGraph is not available."""
        agent = GiannaReActAgent(
            name="test_agent",
            llm=self.mock_llm,
            tools=self.mock_tools,
            config=self.config,
        )

        state = create_initial_state("test_session")
        prepared_state = agent._prepare_agent_state(state)

        result = agent._execute_fallback("Test input", prepared_state)

        assert isinstance(result, dict)
        assert "messages" in result
        assert "fallback_used" in result
        assert result["fallback_used"] is True
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

        # Verify LLM was called
        assert self.mock_llm.call_count == 1
        assert "Test input" in str(self.mock_llm.last_input)

    def test_agent_execution_interface(self):
        """Test the agent execution interface."""
        agent = GiannaReActAgent(
            name="test_agent",
            llm=self.mock_llm,
            tools=self.mock_tools,
            config=self.config,
        )

        state = create_initial_state("test_session")

        # Test that execute method exists and is callable
        assert hasattr(agent, "execute")
        assert callable(agent.execute)

        # Note: We don't test actual execution here as it requires
        # LangGraph integration which is mocked


class TestCommandAgent:
    """Test cases for the CommandAgent."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = MockLanguageModel("Command executed successfully")

    def test_agent_initialization(self):
        """Test CommandAgent initialization."""
        agent = CommandAgent(llm=self.mock_llm)

        assert agent.name == "command_agent"
        assert agent.llm == self.mock_llm
        assert len(agent.tools) == 2  # ShellExecutorTool, FileSystemTool
        assert agent.config.name == "command_agent"
        assert agent.config.max_iterations == 15
        assert agent.config.safety_checks is True
        assert agent.config.validate_inputs is True

        # Verify tools are correct types
        tool_types = [type(tool).__name__ for tool in agent.tools]
        assert "ShellExecutorTool" in tool_types
        assert "FileSystemTool" in tool_types

    def test_custom_config(self):
        """Test CommandAgent with custom configuration."""
        custom_config = AgentConfig(
            name="custom_command_agent",
            description="Custom command agent",
            max_iterations=20,
            safety_checks=False,
        )

        agent = CommandAgent(llm=self.mock_llm, config=custom_config)

        assert agent.config == custom_config
        assert agent.config.max_iterations == 20
        assert agent.config.safety_checks is False

    def test_system_message(self):
        """Test CommandAgent system message."""
        agent = CommandAgent(llm=self.mock_llm)

        system_msg = agent.system_message
        assert isinstance(system_msg, str)
        assert "shell" in system_msg.lower()
        assert "comando" in system_msg.lower()  # Portuguese
        assert "segurança" in system_msg.lower() or "REGRAS DE SEGURANÇA" in system_msg
        assert "ShellExecutorTool" in system_msg
        assert "FileSystemTool" in system_msg
        assert "ReAct" in system_msg

    def test_prepare_agent_state(self):
        """Test CommandAgent state preparation."""
        agent = CommandAgent(llm=self.mock_llm)

        state = create_initial_state("test_session")
        state["commands"]["pending_operations"] = ["ls", "pwd"]
        state["commands"]["execution_history"] = [
            {"command": "echo test", "success": True}
        ]

        prepared_state = agent._prepare_agent_state(state)

        # Check base state preparation
        assert "conversation_history" in prepared_state
        assert "session_id" in prepared_state

        # Check command-specific additions
        assert "command_mode" in prepared_state
        assert "safety_level" in prepared_state
        assert "execution_context" in prepared_state

        execution_context = prepared_state["execution_context"]
        assert "pending_operations" in execution_context
        assert "recent_commands" in execution_context
        assert "current_directory" in execution_context

        assert execution_context["pending_operations"] == ["ls", "pwd"]
        assert len(execution_context["recent_commands"]) == 1
        assert execution_context["recent_commands"][0]["command"] == "echo test"

        assert prepared_state["command_mode"] == "interactive"
        assert prepared_state["safety_level"] == "high"

    def test_agent_tools_integration(self):
        """Test integration with command-specific tools."""
        agent = CommandAgent(llm=self.mock_llm)

        # Verify tools are properly initialized and accessible
        shell_tool = None
        fs_tool = None

        for tool in agent.tools:
            if tool.name == "shell_executor":
                shell_tool = tool
            elif tool.name == "filesystem_manager":
                fs_tool = tool

        assert shell_tool is not None
        assert fs_tool is not None

        # Test tool functionality integration
        assert hasattr(shell_tool, "_run")
        assert hasattr(fs_tool, "_run")
        assert callable(shell_tool._run)
        assert callable(fs_tool._run)


class TestAudioAgent:
    """Test cases for the AudioAgent."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = MockLanguageModel("Audio processing completed")

    def test_agent_initialization(self):
        """Test AudioAgent initialization."""
        agent = AudioAgent(llm=self.mock_llm)

        assert agent.name == "audio_agent"
        assert agent.llm == self.mock_llm
        assert len(agent.tools) == 3  # AudioProcessorTool, TTSTool, STTTool
        assert agent.config.name == "audio_agent"
        assert agent.config.max_iterations == 10
        assert agent.config.conversation_memory is True

        # Verify tools are correct types
        tool_types = [type(tool).__name__ for tool in agent.tools]
        assert "AudioProcessorTool" in tool_types
        assert "TTSTool" in tool_types
        assert "STTTool" in tool_types

    def test_system_message(self):
        """Test AudioAgent system message."""
        agent = AudioAgent(llm=self.mock_llm)

        system_msg = agent.system_message
        assert isinstance(system_msg, str)
        assert "áudio" in system_msg.lower() or "audio" in system_msg.lower()
        assert "voz" in system_msg.lower() or "speech" in system_msg.lower()
        assert "TTS" in system_msg
        assert "STT" in system_msg
        assert "AudioProcessorTool" in system_msg
        assert "ReAct" in system_msg

    def test_prepare_agent_state(self):
        """Test AudioAgent state preparation."""
        agent = AudioAgent(llm=self.mock_llm)

        state = create_initial_state("test_session")
        state["audio"]["current_mode"] = "listening"
        state["audio"]["voice_settings"] = {"rate": 1.0, "volume": 0.8}
        state["audio"]["speech_type"] = "natural"
        state["audio"]["language"] = "pt-BR"
        state["conversation"]["user_preferences"] = {
            "voice": "female",
            "volume": 0.9,
            "speech_rate": 1.2,
        }

        prepared_state = agent._prepare_agent_state(state)

        # Check base state preparation
        assert "conversation_history" in prepared_state
        assert "session_id" in prepared_state

        # Check audio-specific additions
        assert "audio_context" in prepared_state

        audio_context = prepared_state["audio_context"]
        assert "current_mode" in audio_context
        assert "voice_settings" in audio_context
        assert "speech_type" in audio_context
        assert "language" in audio_context
        assert "preferred_voice" in audio_context
        assert "volume_level" in audio_context
        assert "speech_rate" in audio_context

        assert audio_context["current_mode"] == "listening"
        assert audio_context["voice_settings"]["rate"] == 1.0
        assert audio_context["speech_type"] == "natural"
        assert audio_context["language"] == "pt-BR"
        assert audio_context["preferred_voice"] == "female"
        assert audio_context["volume_level"] == 0.9
        assert audio_context["speech_rate"] == 1.2


class TestConversationAgent:
    """Test cases for the ConversationAgent."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = MockLanguageModel("Olá! Como posso ajudá-lo?")

    def test_agent_initialization(self):
        """Test ConversationAgent initialization."""
        agent = ConversationAgent(llm=self.mock_llm)

        assert agent.name == "conversation_agent"
        assert agent.llm == self.mock_llm
        assert len(agent.tools) == 1  # MemoryTool only
        assert agent.config.name == "conversation_agent"
        assert agent.config.max_iterations == 8
        assert agent.config.conversation_memory is True
        assert agent.config.max_context_length == 15000

        # Verify tool is correct type
        assert isinstance(agent.tools[0], MemoryTool)
        assert agent.tools[0].name == "memory_manager"

    def test_system_message(self):
        """Test ConversationAgent system message."""
        agent = ConversationAgent(llm=self.mock_llm)

        system_msg = agent.system_message
        assert isinstance(system_msg, str)
        assert "Gianna" in system_msg
        assert "conversa" in system_msg.lower() or "conversation" in system_msg.lower()
        assert "MemoryTool" in system_msg
        assert "CommandAgent" in system_msg  # Should mention when to forward
        assert "AudioAgent" in system_msg
        assert "ReAct" in system_msg

    def test_prepare_agent_state(self):
        """Test ConversationAgent state preparation."""
        agent = ConversationAgent(llm=self.mock_llm)

        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Oi, como você está?"},
            {"role": "assistant", "content": "Olá! Estou bem, obrigada por perguntar."},
            {"role": "user", "content": "Pode executar um comando shell para mim?"},
            {
                "role": "assistant",
                "content": "Claro, qual comando você gostaria de executar?",
            },
        ]
        state["conversation"]["user_preferences"] = {"name": "João", "tone": "casual"}
        state["conversation"]["context_summary"] = "User asking about shell commands"

        prepared_state = agent._prepare_agent_state(state)

        # Check base state preparation
        assert "conversation_history" in prepared_state
        assert "session_id" in prepared_state

        # Check conversation-specific additions
        assert "conversation_context" in prepared_state

        conv_context = prepared_state["conversation_context"]
        assert "session_duration" in conv_context
        assert "user_name" in conv_context
        assert "conversation_tone" in conv_context
        assert "context_summary" in conv_context
        assert "recent_topics" in conv_context
        assert "user_mood" in conv_context

        assert conv_context["session_duration"] == 4  # 4 messages
        assert conv_context["user_name"] == "João"
        assert conv_context["conversation_tone"] == "casual"
        assert conv_context["context_summary"] == "User asking about shell commands"
        assert (
            "comandos" in conv_context["recent_topics"]
        )  # Should detect shell/command topic

    def test_extract_recent_topics(self):
        """Test topic extraction from conversation."""
        agent = ConversationAgent(llm=self.mock_llm)

        messages = [
            {"role": "user", "content": "Execute o comando ls para mim"},
            {"role": "assistant", "content": "Executando comando..."},
            {"role": "user", "content": "Agora toque uma música relaxante"},
            {"role": "assistant", "content": "Tocando música..."},
            {"role": "user", "content": "Salve esse arquivo na pasta documentos"},
        ]

        topics = agent._extract_recent_topics(messages)

        assert isinstance(topics, list)
        assert "comandos" in topics  # Command-related
        assert "áudio" in topics  # Audio-related
        assert "arquivos" in topics  # File-related

    def test_assess_user_mood(self):
        """Test user mood assessment."""
        agent = ConversationAgent(llm=self.mock_llm)

        # Test positive mood
        positive_messages = [
            {"role": "user", "content": "Obrigado, isso foi perfeito!"},
            {"role": "user", "content": "Ótimo trabalho, muito legal!"},
        ]
        mood = agent._assess_user_mood(positive_messages)
        assert mood == "positive"

        # Test frustrated mood
        negative_messages = [
            {"role": "user", "content": "Isso não funciona, tem algum problema"},
            {"role": "user", "content": "Está muito difícil de usar"},
        ]
        mood = agent._assess_user_mood(negative_messages)
        assert mood == "frustrated"

        # Test neutral mood
        neutral_messages = [
            {"role": "user", "content": "Como você está hoje?"},
            {"role": "user", "content": "Qual é a temperatura?"},
        ]
        mood = agent._assess_user_mood(neutral_messages)
        assert mood == "neutral"

        # Test empty messages
        mood = agent._assess_user_mood([])
        assert mood == "neutral"


class TestMemoryAgent:
    """Test cases for the MemoryAgent."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = MockLanguageModel("Memory operation completed")

    def test_agent_initialization(self):
        """Test MemoryAgent initialization."""
        agent = MemoryAgent(llm=self.mock_llm)

        assert agent.name == "memory_agent"
        assert agent.llm == self.mock_llm
        assert len(agent.tools) == 1  # MemoryTool only
        assert agent.config.name == "memory_agent"
        assert agent.config.max_iterations == 5
        assert agent.config.persistent_memory is True
        assert agent.config.max_context_length == 20000

        # Verify tool is correct type
        assert isinstance(agent.tools[0], MemoryTool)
        assert agent.tools[0].name == "memory_manager"

    def test_system_message(self):
        """Test MemoryAgent system message."""
        agent = MemoryAgent(llm=self.mock_llm)

        system_msg = agent.system_message
        assert isinstance(system_msg, str)
        assert "memória" in system_msg.lower() or "memory" in system_msg.lower()
        assert "contexto" in system_msg.lower() or "context" in system_msg.lower()
        assert "MemoryTool" in system_msg
        assert (
            "preferências" in system_msg.lower() or "preferences" in system_msg.lower()
        )
        assert "ReAct" in system_msg

    def test_prepare_agent_state(self):
        """Test MemoryAgent state preparation."""
        agent = MemoryAgent(llm=self.mock_llm)

        state = create_initial_state("test_session")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]
        state["conversation"]["user_preferences"] = {
            "language": "pt",
            "voice": "female",
        }
        state["conversation"]["context_summary"] = "User preferences discussion"
        state["commands"]["execution_history"] = [
            {"command": "ls", "success": True},
            {"command": "pwd", "success": True},
        ]

        prepared_state = agent._prepare_agent_state(state)

        # Check base state preparation
        assert "conversation_history" in prepared_state
        assert "session_id" in prepared_state

        # Check memory-specific additions
        assert "memory_context" in prepared_state

        memory_context = prepared_state["memory_context"]
        assert "session_age" in memory_context
        assert "user_preferences" in memory_context
        assert "context_summary" in memory_context
        assert "memory_capacity" in memory_context
        assert "persistent_enabled" in memory_context
        assert "key_information" in memory_context
        assert "memory_priorities" in memory_context

        assert memory_context["session_age"] == 2  # 2 messages
        assert memory_context["user_preferences"]["language"] == "pt"
        assert memory_context["context_summary"] == "User preferences discussion"
        assert memory_context["memory_capacity"] == 20000
        assert memory_context["persistent_enabled"] is True
        assert isinstance(memory_context["key_information"], list)
        assert "user_preferences" in memory_context["memory_priorities"]

    def test_extract_key_information(self):
        """Test key information extraction."""
        agent = MemoryAgent(llm=self.mock_llm)

        state = create_initial_state("test_session")
        state["conversation"]["user_preferences"] = {
            "name": "Maria",
            "language": "pt",
            "preferred_voice": "female",
        }
        state["commands"]["execution_history"] = [
            {"command": "git status", "success": True},
            {"command": "npm install", "success": True},
            {"command": "invalid_cmd", "success": False},
        ]

        key_info = agent._extract_key_information(state)

        assert isinstance(key_info, list)
        assert len(key_info) > 0

        # Should extract user preferences
        pref_info = [info for info in key_info if "Preference" in info]
        assert len(pref_info) == 3  # name, language, preferred_voice

        # Should note successful operations
        success_info = [info for info in key_info if "successful operations" in info]
        assert len(success_info) == 1


class TestReActAgentIntegration:
    """Integration tests for ReAct agents."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_llm = MockLanguageModel("Integration test response")

    def test_all_agents_initialization(self):
        """Test that all agent types can be initialized."""
        agents = [
            CommandAgent(self.mock_llm),
            AudioAgent(self.mock_llm),
            ConversationAgent(self.mock_llm),
            MemoryAgent(self.mock_llm),
        ]

        expected_names = [
            "command_agent",
            "audio_agent",
            "conversation_agent",
            "memory_agent",
        ]

        for agent, expected_name in zip(agents, expected_names):
            assert agent.name == expected_name
            assert agent.llm == self.mock_llm
            assert hasattr(agent, "tools")
            assert len(agent.tools) > 0
            assert hasattr(agent, "config")
            assert isinstance(agent.config, AgentConfig)

    def test_agent_tool_specialization(self):
        """Test that agents have appropriate tool specialization."""
        command_agent = CommandAgent(self.mock_llm)
        audio_agent = AudioAgent(self.mock_llm)
        conversation_agent = ConversationAgent(self.mock_llm)
        memory_agent = MemoryAgent(self.mock_llm)

        # Command agent should have shell and filesystem tools
        command_tools = [tool.name for tool in command_agent.tools]
        assert "shell_executor" in command_tools
        assert "filesystem_manager" in command_tools

        # Audio agent should have audio-related tools
        audio_tools = [tool.name for tool in audio_agent.tools]
        assert "audio_processor" in audio_tools
        assert "text_to_speech" in audio_tools
        assert "speech_to_text" in audio_tools

        # Conversation agent should have memory tool
        conv_tools = [tool.name for tool in conversation_agent.tools]
        assert "memory_manager" in conv_tools
        assert len(conv_tools) == 1  # Only memory tool

        # Memory agent should have memory tool
        memory_tools = [tool.name for tool in memory_agent.tools]
        assert "memory_manager" in memory_tools
        assert len(memory_tools) == 1  # Only memory tool

    def test_state_preparation_consistency(self):
        """Test that all agents prepare state consistently."""
        agents = [
            CommandAgent(self.mock_llm),
            AudioAgent(self.mock_llm),
            ConversationAgent(self.mock_llm),
            MemoryAgent(self.mock_llm),
        ]

        state = create_initial_state("integration_test")
        state["conversation"]["messages"] = [
            {"role": "user", "content": "Test message for integration"}
        ]

        for agent in agents:
            prepared_state = agent._prepare_agent_state(state)

            # All agents should include these base elements
            assert "conversation_history" in prepared_state
            assert "session_id" in prepared_state
            assert "user_preferences" in prepared_state
            assert "audio_mode" in prepared_state
            assert "language" in prepared_state
            assert "command_history" in prepared_state
            assert "metadata" in prepared_state

            # Verify data consistency
            assert prepared_state["session_id"] == "integration_test"
            assert len(prepared_state["conversation_history"]) == 1
            assert (
                prepared_state["conversation_history"][0]["content"]
                == "Test message for integration"
            )

    def test_agent_configuration_inheritance(self):
        """Test agent configuration inheritance and customization."""
        # Test default configurations
        agents = [
            CommandAgent(self.mock_llm),
            AudioAgent(self.mock_llm),
            ConversationAgent(self.mock_llm),
            MemoryAgent(self.mock_llm),
        ]

        for agent in agents:
            assert isinstance(agent.config, AgentConfig)
            assert agent.config.name == agent.name
            assert isinstance(agent.config.description, str)
            assert len(agent.config.description) > 0
            assert agent.config.max_iterations > 0

        # Test custom configuration
        custom_config = AgentConfig(
            name="custom_agent",
            description="Custom test agent",
            max_iterations=25,
            custom_attribute="custom_value",
        )

        custom_command_agent = CommandAgent(self.mock_llm, config=custom_config)
        assert custom_command_agent.config == custom_config
        assert custom_command_agent.config.max_iterations == 25
        assert hasattr(custom_command_agent.config, "custom_attribute")
        assert custom_command_agent.config.custom_attribute == "custom_value"

    def test_agent_system_message_differentiation(self):
        """Test that agents have differentiated system messages."""
        agents = [
            CommandAgent(self.mock_llm),
            AudioAgent(self.mock_llm),
            ConversationAgent(self.mock_llm),
            MemoryAgent(self.mock_llm),
        ]

        system_messages = [agent.system_message for agent in agents]

        # All system messages should be unique
        assert len(set(system_messages)) == len(system_messages)

        # Verify domain-specific content
        command_msg = agents[0].system_message
        assert "comando" in command_msg.lower() or "shell" in command_msg.lower()

        audio_msg = agents[1].system_message
        assert "áudio" in audio_msg.lower() or "voz" in audio_msg.lower()

        conv_msg = agents[2].system_message
        assert "conversa" in conv_msg.lower() or "Gianna" in conv_msg

        memory_msg = agents[3].system_message
        assert "memória" in memory_msg.lower() or "contexto" in memory_msg.lower()

    @patch("gianna.agents.react_agents.LANGGRAPH_AVAILABLE", True)
    @patch("gianna.agents.react_agents.create_react_agent")
    def test_langgraph_integration(self, mock_create_react_agent):
        """Test LangGraph integration when available."""
        mock_agent = Mock()
        mock_create_react_agent.return_value = mock_agent

        agent = CommandAgent(self.mock_llm)

        # Verify LangGraph agent was created
        mock_create_react_agent.assert_called_once()
        assert agent.agent == mock_agent

        # Verify parameters passed to create_react_agent
        args, kwargs = mock_create_react_agent.call_args
        assert args[0] == self.mock_llm  # LLM passed
        assert len(args[1]) == 2  # Tools passed (ShellExecutorTool, FileSystemTool)
        assert "state_modifier" in kwargs  # State modifier function passed

    def test_error_handling_consistency(self):
        """Test consistent error handling across all agents."""
        agents = [
            CommandAgent(self.mock_llm),
            AudioAgent(self.mock_llm),
            ConversationAgent(self.mock_llm),
            MemoryAgent(self.mock_llm),
        ]

        # Test with invalid state
        invalid_state = {"invalid": "state"}

        for agent in agents:
            try:
                # This should handle the error gracefully or raise a meaningful exception
                agent._prepare_agent_state(invalid_state)
            except (KeyError, AttributeError, TypeError) as e:
                # Expected - invalid state should cause errors
                assert isinstance(e, (KeyError, AttributeError, TypeError))
            except Exception as e:
                # Unexpected exception type
                pytest.fail(
                    f"Unexpected exception type {type(e)} from {agent.name}: {e}"
                )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
