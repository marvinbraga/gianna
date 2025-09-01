"""
ReAct Agents System for Gianna AI Assistant - FASE 2

This module provides specialized ReAct agents that integrate with LangGraph
for advanced reasoning and acting capabilities. Each agent is specialized
for specific domains while maintaining compatibility with the existing
Gianna state management system.

The agents use the ReAct (Reasoning and Acting) pattern via LangGraph's
create_react_agent function, providing structured thought processes and
tool usage for complex tasks.

Available Agents:
- CommandAgent: Shell commands and system operations
- AudioAgent: Audio processing (TTS/STT/playbook)
- ConversationAgent: Natural dialogue management
- MemoryAgent: Context and memory management

Agent Factory:
- AgentFactory: Factory pattern for easy agent creation and management
"""

from .base_agent import AgentConfig, BaseReActAgent
from .react_agents import (
    AudioAgent,
    CommandAgent,
    ConversationAgent,
    GiannaReActAgent,
    MemoryAgent,
)


# Agent factory for easy creation
class AgentFactory:
    """
    Factory class for creating and managing ReAct agents.

    Provides centralized agent creation with consistent configuration
    and integration with the existing LLM provider system.
    """

    @staticmethod
    def create_agent(agent_type: str, llm, config: AgentConfig = None):
        """
        Create an agent of the specified type.

        Args:
            agent_type: Type of agent to create ('command', 'audio', 'conversation', 'memory')
            llm: Language model instance (BaseLanguageModel)
            config: Optional agent configuration

        Returns:
            GiannaReActAgent: Specialized agent instance

        Raises:
            ValueError: If agent_type is not supported
        """
        agent_classes = {
            "command": CommandAgent,
            "audio": AudioAgent,
            "conversation": ConversationAgent,
            "memory": MemoryAgent,
        }

        if agent_type not in agent_classes:
            raise ValueError(
                f"Unsupported agent type: {agent_type}. "
                f"Available types: {list(agent_classes.keys())}"
            )

        agent_class = agent_classes[agent_type]

        if config:
            return agent_class(llm, config)
        else:
            return agent_class(llm)

    @staticmethod
    def get_available_agent_types():
        """
        Get list of available agent types.

        Returns:
            List[str]: List of supported agent types
        """
        return ["command", "audio", "conversation", "memory"]


__all__ = [
    "BaseReActAgent",
    "AgentConfig",
    "GiannaReActAgent",
    "CommandAgent",
    "AudioAgent",
    "ConversationAgent",
    "MemoryAgent",
    "AgentFactory",
]
