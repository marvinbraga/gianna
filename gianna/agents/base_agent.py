"""
Base Agent Classes for Gianna ReAct Agents System

This module provides the foundational classes and interfaces for all
ReAct agents in the Gianna system. It defines the common structure,
configuration, and lifecycle management for specialized agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from loguru import logger

from ..core.state import GiannaState


class AgentStatus(Enum):
    """Agent execution status enumeration."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentConfig:
    """
    Configuration class for ReAct agents.

    Contains common configuration parameters that can be shared
    across different agent types while allowing for specialization.
    """

    # Agent identification
    name: str = ""
    description: str = ""
    version: str = "1.0"

    # Execution settings
    max_iterations: int = 10
    timeout_seconds: int = 300
    verbose: bool = True

    # LangGraph settings
    checkpointer: Optional[Any] = None
    interrupt_before: List[str] = field(default_factory=list)
    interrupt_after: List[str] = field(default_factory=list)

    # Memory and context
    max_context_length: int = 10000
    conversation_memory: bool = True
    persistent_memory: bool = False

    # Error handling
    retry_attempts: int = 3
    error_recovery: bool = True
    fallback_enabled: bool = True

    # Safety and validation
    validate_inputs: bool = True
    sanitize_outputs: bool = True
    safety_checks: bool = True

    # Custom parameters for agent specialization
    custom_params: Dict[str, Any] = field(default_factory=dict)


class BaseReActAgent(ABC):
    """
    Abstract base class for all Gianna ReAct agents.

    Provides the common interface and lifecycle management that all
    specialized agents must implement. Uses the ReAct pattern for
    structured reasoning and action execution.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        config: Optional[AgentConfig] = None,
    ):
        """
        Initialize the base ReAct agent.

        Args:
            name: Agent name/identifier
            llm: Language model instance
            tools: List of tools available to the agent
            config: Optional agent configuration
        """
        self.name = name
        self.llm = llm
        self.tools = tools
        self.config = config or AgentConfig(name=name)

        # Agent state
        self.status = AgentStatus.IDLE
        self.current_iteration = 0
        self.execution_history: List[Dict[str, Any]] = []
        self.errors: List[Exception] = []

        # Initialize the ReAct agent (to be implemented by subclasses)
        self.agent = None

        logger.info(f"Initialized {self.__class__.__name__}: {self.name}")

    @property
    @abstractmethod
    def system_message(self) -> str:
        """
        Get the specialized system message for this agent type.

        Returns:
            str: System message defining agent behavior and capabilities
        """
        pass

    @abstractmethod
    def _prepare_agent_state(self, state: GiannaState) -> Dict[str, Any]:
        """
        Prepare the state for the ReAct agent execution.

        Agents can customize how the GiannaState is transformed into
        the format expected by their LangGraph ReAct agent.

        Args:
            state: Current Gianna system state

        Returns:
            Dict[str, Any]: State prepared for agent execution
        """
        pass

    @abstractmethod
    def _process_agent_output(self, output: Any, state: GiannaState) -> Dict[str, Any]:
        """
        Process the output from the ReAct agent back into Gianna format.

        Args:
            output: Output from the LangGraph ReAct agent
            state: Current Gianna system state

        Returns:
            Dict[str, Any]: Processed output in Gianna format
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data before processing.

        Args:
            input_data: Input to validate

        Returns:
            bool: True if input is valid

        Raises:
            ValueError: If validation fails and strict mode is enabled
        """
        if not self.config.validate_inputs:
            return True

        try:
            # Basic validation - can be overridden by subclasses
            if input_data is None:
                raise ValueError("Input cannot be None")

            return True

        except Exception as e:
            logger.error(f"Input validation failed for {self.name}: {e}")
            if self.config.safety_checks:
                raise
            return False

    def sanitize_output(self, output: Any) -> Any:
        """
        Sanitize agent output for safety and consistency.

        Args:
            output: Raw agent output

        Returns:
            Any: Sanitized output
        """
        if not self.config.sanitize_outputs:
            return output

        # Basic sanitization - can be overridden by subclasses
        if isinstance(output, dict):
            # Remove potentially sensitive keys
            sensitive_keys = ["password", "token", "key", "secret"]
            return {
                k: v
                for k, v in output.items()
                if not any(sensitive in k.lower() for sensitive in sensitive_keys)
            }

        return output

    def execute(self, input_data: Any, state: GiannaState, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent with given input and state.

        This is the main entry point for agent execution. It handles
        the complete lifecycle including validation, execution, and
        error handling.

        Args:
            input_data: Input for the agent to process
            state: Current Gianna system state
            **kwargs: Additional execution parameters

        Returns:
            Dict[str, Any]: Agent execution results
        """
        self.status = AgentStatus.THINKING
        start_time = logger.info(f"Starting execution for {self.name}")

        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Input validation failed")

            # Prepare state for agent
            agent_state = self._prepare_agent_state(state)

            # Execute the ReAct agent
            self.status = AgentStatus.ACTING
            raw_output = self._execute_agent(input_data, agent_state, **kwargs)

            # Process output
            processed_output = self._process_agent_output(raw_output, state)

            # Sanitize output
            safe_output = self.sanitize_output(processed_output)

            self.status = AgentStatus.COMPLETED

            # Record execution
            self._record_execution(input_data, safe_output, success=True)

            logger.info(f"Completed execution for {self.name}")
            return safe_output

        except Exception as e:
            self.status = AgentStatus.ERROR
            self.errors.append(e)

            logger.error(f"Execution failed for {self.name}: {e}")

            # Record failed execution
            self._record_execution(input_data, None, success=False, error=str(e))

            # Attempt recovery if enabled
            if self.config.error_recovery:
                return self._attempt_recovery(input_data, state, e, **kwargs)

            raise

    def _execute_agent(
        self, input_data: Any, agent_state: Dict[str, Any], **kwargs
    ) -> Any:
        """
        Execute the underlying ReAct agent.

        Args:
            input_data: Input for the agent
            agent_state: Prepared state for execution
            **kwargs: Additional parameters

        Returns:
            Any: Raw agent output

        Raises:
            NotImplementedError: If agent is not properly initialized
        """
        if self.agent is None:
            raise NotImplementedError("ReAct agent not initialized in subclass")

        # Create input for LangGraph agent
        graph_input = {
            "messages": [{"role": "user", "content": str(input_data)}],
            **agent_state,
        }

        # Execute the agent
        result = self.agent.invoke(graph_input)

        self.current_iteration += 1
        return result

    def _attempt_recovery(
        self, input_data: Any, state: GiannaState, error: Exception, **kwargs
    ) -> Dict[str, Any]:
        """
        Attempt to recover from execution error.

        Args:
            input_data: Original input data
            state: Current system state
            error: The exception that occurred
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Recovery attempt result
        """
        logger.warning(f"Attempting recovery for {self.name} after error: {error}")

        if self.config.fallback_enabled:
            return {
                "success": False,
                "error": str(error),
                "fallback_used": True,
                "message": f"Agent {self.name} encountered an error but provided fallback response",
            }

        raise error

    def _record_execution(
        self, input_data: Any, output: Any, success: bool, error: str = None
    ):
        """
        Record execution details for monitoring and debugging.

        Args:
            input_data: Input that was processed
            output: Output that was generated (or None if failed)
            success: Whether execution was successful
            error: Error message if execution failed
        """
        execution_record = {
            "timestamp": logger.info(f"Recording execution for {self.name}"),
            "agent": self.name,
            "input": str(input_data)[:200],  # Truncate for safety
            "success": success,
            "iteration": self.current_iteration,
            "status": self.status.value,
        }

        if success:
            execution_record["output"] = str(output)[:200] if output else None
        else:
            execution_record["error"] = error

        self.execution_history.append(execution_record)

        # Keep history bounded
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-50:]

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status and statistics.

        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "current_iteration": self.current_iteration,
            "total_executions": len(self.execution_history),
            "error_count": len(self.errors),
            "last_execution": (
                self.execution_history[-1] if self.execution_history else None
            ),
            "config": {
                "max_iterations": self.config.max_iterations,
                "timeout_seconds": self.config.timeout_seconds,
                "error_recovery": self.config.error_recovery,
            },
        }

    def reset(self):
        """Reset agent state for fresh execution."""
        self.status = AgentStatus.IDLE
        self.current_iteration = 0
        self.errors.clear()
        logger.info(f"Reset agent {self.name}")

    def cleanup(self):
        """Cleanup agent resources."""
        self.reset()
        logger.info(f"Cleaned up agent {self.name}")
