"""
Multi-Agent Orchestrator for Gianna AI Assistant

This module implements the central orchestration system that manages multiple
specialized agents, coordinates their execution, handles conflicts, and ensures
optimal resource utilization.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from loguru import logger

from ..agents.react_agents import (
    AudioAgent,
    CommandAgent,
    ConversationAgent,
    GiannaReActAgent,
    MemoryAgent,
)
from ..core.state import GiannaState
from .router import AgentRouter, AgentType


class ExecutionMode(Enum):
    """Execution modes for agent coordination."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class AgentStatus(Enum):
    """Status of agents in the orchestrator."""

    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AgentInfo:
    """Information about a registered agent."""

    agent: GiannaReActAgent
    status: AgentStatus = AgentStatus.AVAILABLE
    last_used: Optional[datetime] = None
    total_executions: int = 0
    success_rate: float = 1.0
    avg_execution_time: float = 0.0
    current_task: Optional[str] = None
    error_count: int = 0
    max_concurrent_tasks: int = 1
    current_tasks: Set[str] = field(default_factory=set)


@dataclass
class ExecutionRequest:
    """Represents a request for agent execution."""

    request_id: str
    state: GiannaState
    requested_agent: Optional[AgentType] = None
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    priority: int = 1
    timeout: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Result of agent execution."""

    request_id: str
    agent_type: AgentType
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    """
    Central orchestrator for multi-agent coordination and execution.

    Manages agent lifecycle, routing, execution, conflict resolution,
    and performance optimization across all specialized agents.
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize the Agent Orchestrator.

        Args:
            max_workers: Maximum number of concurrent execution threads
        """
        self.agents: Dict[AgentType, AgentInfo] = {}
        self.router = AgentRouter()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.execution_queue = []
        self.execution_history = []
        self.performance_metrics = {}
        self._lock = threading.RLock()
        self._shutdown = False

        logger.info(f"AgentOrchestrator initialized with {max_workers} max workers")

    def register_agent(self, agent: GiannaReActAgent) -> None:
        """
        Register an agent with the orchestrator.

        Args:
            agent: Agent instance to register
        """
        try:
            agent_type = AgentType(agent.name)
        except ValueError:
            logger.error(f"Unknown agent type: {agent.name}")
            return

        with self._lock:
            self.agents[agent_type] = AgentInfo(
                agent=agent,
                status=AgentStatus.AVAILABLE,
                last_used=None,
                total_executions=0,
            )

        logger.info(f"Registered agent: {agent_type.value}")

    def unregister_agent(self, agent_type: AgentType) -> bool:
        """
        Unregister an agent from the orchestrator.

        Args:
            agent_type: Type of agent to unregister

        Returns:
            bool: True if successfully unregistered
        """
        with self._lock:
            if agent_type not in self.agents:
                logger.warning(f"Cannot unregister unknown agent: {agent_type.value}")
                return False

            agent_info = self.agents[agent_type]
            if agent_info.status == AgentStatus.BUSY:
                logger.warning(f"Cannot unregister busy agent: {agent_type.value}")
                return False

            del self.agents[agent_type]
            logger.info(f"Unregistered agent: {agent_type.value}")
            return True

    def get_registered_agents(self) -> List[AgentType]:
        """
        Get list of all registered agent types.

        Returns:
            List[AgentType]: List of registered agent types
        """
        with self._lock:
            return list(self.agents.keys())

    def get_agent_status(self, agent_type: AgentType) -> Optional[AgentStatus]:
        """
        Get the current status of a specific agent.

        Args:
            agent_type: Type of agent to check

        Returns:
            Optional[AgentStatus]: Agent status or None if not registered
        """
        with self._lock:
            return (
                self.agents.get(agent_type, {}).status
                if agent_type in self.agents
                else None
            )

    def route_and_execute(
        self,
        state: GiannaState,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        requested_agent: Optional[AgentType] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Route a request to the appropriate agent and execute it.

        Args:
            state: Current system state
            execution_mode: How to execute the request
            requested_agent: Specific agent to use (overrides routing)
            timeout: Maximum execution time in seconds

        Returns:
            ExecutionResult: Result of the execution
        """
        request_id = f"req_{int(time.time() * 1000)}"

        # Create execution request
        request = ExecutionRequest(
            request_id=request_id,
            state=state,
            requested_agent=requested_agent,
            execution_mode=execution_mode,
            timeout=timeout,
        )

        try:
            # Route to appropriate agent if not specified
            if requested_agent is None:
                agent_type, confidence = self.router.route_request(state)
                logger.info(
                    f"Routed request {request_id} to {agent_type.value} (confidence: {confidence:.2f})"
                )
            else:
                agent_type = requested_agent
                confidence = 1.0
                logger.info(
                    f"Direct routing to {agent_type.value} for request {request_id}"
                )

            # Execute the request
            result = self._execute_request(request, agent_type, confidence)

            # Update performance metrics
            self._update_performance_metrics(agent_type, result)

            return result

        except Exception as e:
            logger.error(f"Failed to execute request {request_id}: {e}")
            return ExecutionResult(
                request_id=request_id,
                agent_type=(
                    agent_type if "agent_type" in locals() else AgentType.CONVERSATION
                ),
                success=False,
                result=None,
                execution_time=0.0,
                error=str(e),
            )

    def coordinate_agents(
        self,
        agents: List[AgentType],
        state: GiannaState,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    ) -> List[ExecutionResult]:
        """
        Coordinate execution across multiple agents.

        Args:
            agents: List of agents to coordinate
            state: Current system state
            execution_mode: How to execute (sequential or parallel)

        Returns:
            List[ExecutionResult]: Results from all agents
        """
        if execution_mode == ExecutionMode.SEQUENTIAL:
            return self._execute_sequential(agents, state)
        elif execution_mode == ExecutionMode.PARALLEL:
            return self._execute_parallel(agents, state)
        else:  # HYBRID
            return self._execute_hybrid(agents, state)

    def _execute_request(
        self, request: ExecutionRequest, agent_type: AgentType, confidence: float
    ) -> ExecutionResult:
        """
        Execute a single request on the specified agent.

        Args:
            request: Execution request
            agent_type: Target agent type
            confidence: Routing confidence score

        Returns:
            ExecutionResult: Execution result
        """
        start_time = time.time()

        with self._lock:
            if agent_type not in self.agents:
                raise ValueError(f"Agent {agent_type.value} not registered")

            agent_info = self.agents[agent_type]

            if agent_info.status != AgentStatus.AVAILABLE:
                # Try to find alternative or wait
                alternative = self._find_alternative_agent(agent_type)
                if alternative:
                    agent_type = alternative
                    agent_info = self.agents[agent_type]
                else:
                    raise RuntimeError(f"Agent {agent_type.value} is not available")

            # Mark agent as busy
            agent_info.status = AgentStatus.BUSY
            agent_info.current_task = request.request_id

        try:
            # Execute the agent
            agent = agent_info.agent

            # Prepare input from state
            last_message = (
                request.state["conversation"].messages[-1]
                if request.state["conversation"].messages
                else ""
            )
            user_input = (
                last_message.get("content", "")
                if isinstance(last_message, dict)
                else str(last_message)
            )

            # Execute with timeout
            result = self._execute_with_timeout(
                agent.execute,
                args=(user_input, request.state),
                timeout=request.timeout or 30,
            )

            execution_time = time.time() - start_time

            # Mark as successful
            with self._lock:
                agent_info.status = AgentStatus.AVAILABLE
                agent_info.current_task = None
                agent_info.last_used = datetime.now()
                agent_info.total_executions += 1

                # Update success rate and execution time
                total_time = (
                    agent_info.avg_execution_time * (agent_info.total_executions - 1)
                    + execution_time
                )
                agent_info.avg_execution_time = total_time / agent_info.total_executions

            return ExecutionResult(
                request_id=request.request_id,
                agent_type=agent_type,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata={"confidence": confidence},
            )

        except Exception as e:
            execution_time = time.time() - start_time

            with self._lock:
                agent_info.status = (
                    AgentStatus.ERROR
                    if "timeout" not in str(e).lower()
                    else AgentStatus.AVAILABLE
                )
                agent_info.current_task = None
                agent_info.error_count += 1

                # Update success rate
                if agent_info.total_executions > 0:
                    agent_info.success_rate = (
                        agent_info.total_executions - agent_info.error_count
                    ) / agent_info.total_executions

            logger.error(f"Agent {agent_type.value} execution failed: {e}")

            return ExecutionResult(
                request_id=request.request_id,
                agent_type=agent_type,
                success=False,
                result=None,
                execution_time=execution_time,
                error=str(e),
            )

    def _execute_sequential(
        self, agents: List[AgentType], state: GiannaState
    ) -> List[ExecutionResult]:
        """
        Execute agents sequentially, passing state between them.

        Args:
            agents: List of agents to execute
            state: Initial system state

        Returns:
            List[ExecutionResult]: Results from all agents
        """
        results = []
        current_state = state.copy()  # Work with a copy

        for agent_type in agents:
            request = ExecutionRequest(
                request_id=f"seq_{int(time.time() * 1000)}_{len(results)}",
                state=current_state,
            )

            result = self._execute_request(request, agent_type, 1.0)
            results.append(result)

            # Update state for next agent if successful
            if result.success and result.result:
                # Update conversation with result
                if isinstance(result.result, dict) and "content" in result.result:
                    current_state["conversation"].messages.append(
                        {
                            "role": "assistant",
                            "content": result.result["content"],
                            "agent": agent_type.value,
                        }
                    )

            # Stop on error unless it's the conversation agent
            if not result.success and agent_type != AgentType.CONVERSATION:
                logger.warning(
                    f"Sequential execution stopped due to error in {agent_type.value}"
                )
                break

        return results

    def _execute_parallel(
        self, agents: List[AgentType], state: GiannaState
    ) -> List[ExecutionResult]:
        """
        Execute agents in parallel with the same state.

        Args:
            agents: List of agents to execute
            state: System state for all agents

        Returns:
            List[ExecutionResult]: Results from all agents
        """
        futures = []
        request_base_id = f"par_{int(time.time() * 1000)}"

        for i, agent_type in enumerate(agents):
            request = ExecutionRequest(request_id=f"{request_base_id}_{i}", state=state)

            future = self.executor.submit(
                self._execute_request, request, agent_type, 1.0
            )
            futures.append(future)

        results = []
        for future in as_completed(futures, timeout=60):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel execution error: {e}")
                results.append(
                    ExecutionResult(
                        request_id="unknown",
                        agent_type=AgentType.CONVERSATION,  # Default
                        success=False,
                        result=None,
                        execution_time=0.0,
                        error=str(e),
                    )
                )

        # Sort results by request ID to maintain order
        results.sort(key=lambda r: r.request_id)
        return results

    def _execute_hybrid(
        self, agents: List[AgentType], state: GiannaState
    ) -> List[ExecutionResult]:
        """
        Execute agents in hybrid mode (smart sequential/parallel mix).

        Args:
            agents: List of agents to execute
            state: System state

        Returns:
            List[ExecutionResult]: Results from all agents
        """
        # Group agents by dependency requirements
        # Command and Memory agents should run before Conversation
        # Audio can run in parallel with others

        priority_groups = {
            "high": [AgentType.COMMAND, AgentType.MEMORY],
            "medium": [AgentType.AUDIO],
            "low": [AgentType.CONVERSATION],
        }

        results = []
        current_state = state.copy()

        for priority, group_types in priority_groups.items():
            group_agents = [a for a in agents if a in group_types]
            if not group_agents:
                continue

            if len(group_agents) == 1:
                # Single agent - execute directly
                group_results = self._execute_sequential(group_agents, current_state)
            else:
                # Multiple agents - execute in parallel
                group_results = self._execute_parallel(group_agents, current_state)

            results.extend(group_results)

            # Update state with successful results
            for result in group_results:
                if result.success and result.result:
                    if isinstance(result.result, dict) and "content" in result.result:
                        current_state["conversation"].messages.append(
                            {
                                "role": "assistant",
                                "content": result.result["content"],
                                "agent": result.agent_type.value,
                            }
                        )

        return results

    def _execute_with_timeout(self, func, args=(), timeout=30):
        """
        Execute a function with timeout.

        Args:
            func: Function to execute
            args: Function arguments
            timeout: Timeout in seconds

        Returns:
            Function result

        Raises:
            TimeoutError: If execution exceeds timeout
        """
        future = self.executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(f"Execution timed out after {timeout} seconds")

    def _find_alternative_agent(
        self, preferred_agent: AgentType
    ) -> Optional[AgentType]:
        """
        Find an alternative agent when the preferred one is unavailable.

        Args:
            preferred_agent: Preferred agent that's unavailable

        Returns:
            Optional[AgentType]: Alternative agent or None
        """
        with self._lock:
            # Look for available agents with similar capabilities
            alternatives = {
                AgentType.COMMAND: [AgentType.CONVERSATION],
                AgentType.AUDIO: [AgentType.CONVERSATION],
                AgentType.MEMORY: [AgentType.CONVERSATION],
                AgentType.CONVERSATION: [],  # No alternatives for conversation agent
            }

            for alternative in alternatives.get(preferred_agent, []):
                if (
                    alternative in self.agents
                    and self.agents[alternative].status == AgentStatus.AVAILABLE
                ):
                    logger.info(
                        f"Using alternative agent {alternative.value} for {preferred_agent.value}"
                    )
                    return alternative

            return None

    def _update_performance_metrics(
        self, agent_type: AgentType, result: ExecutionResult
    ) -> None:
        """
        Update performance metrics for monitoring and optimization.

        Args:
            agent_type: Agent type
            result: Execution result
        """
        if agent_type not in self.performance_metrics:
            self.performance_metrics[agent_type] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0,
                "error_rate": 0.0,
            }

        metrics = self.performance_metrics[agent_type]
        metrics["total_requests"] += 1
        metrics["total_execution_time"] += result.execution_time

        if result.success:
            metrics["successful_requests"] += 1

        metrics["average_execution_time"] = (
            metrics["total_execution_time"] / metrics["total_requests"]
        )
        metrics["error_rate"] = 1.0 - (
            metrics["successful_requests"] / metrics["total_requests"]
        )

    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all agents.

        Returns:
            Dict[str, Dict[str, float]]: Performance metrics by agent
        """
        return {
            agent_type.value: metrics.copy()
            for agent_type, metrics in self.performance_metrics.items()
        }

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.

        Returns:
            Dict[str, Any]: System status information
        """
        with self._lock:
            agent_status = {}
            for agent_type, agent_info in self.agents.items():
                agent_status[agent_type.value] = {
                    "status": agent_info.status.value,
                    "total_executions": agent_info.total_executions,
                    "success_rate": agent_info.success_rate,
                    "average_execution_time": agent_info.avg_execution_time,
                    "error_count": agent_info.error_count,
                    "last_used": (
                        agent_info.last_used.isoformat()
                        if agent_info.last_used
                        else None
                    ),
                    "current_task": agent_info.current_task,
                }

        return {
            "orchestrator_status": "running" if not self._shutdown else "shutdown",
            "total_agents": len(self.agents),
            "available_agents": sum(
                1
                for info in self.agents.values()
                if info.status == AgentStatus.AVAILABLE
            ),
            "busy_agents": sum(
                1 for info in self.agents.values() if info.status == AgentStatus.BUSY
            ),
            "error_agents": sum(
                1 for info in self.agents.values() if info.status == AgentStatus.ERROR
            ),
            "agents": agent_status,
            "routing_stats": self.router.get_routing_stats(),
            "performance_metrics": self.get_performance_metrics(),
        }

    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all registered agents.

        Returns:
            Dict[str, bool]: Health status for each agent
        """
        health_status = {}

        with self._lock:
            for agent_type, agent_info in self.agents.items():
                try:
                    # Simple health check - try to access agent properties
                    _ = agent_info.agent.name
                    _ = agent_info.agent.llm
                    health_status[agent_type.value] = (
                        agent_info.status != AgentStatus.ERROR
                    )
                except Exception as e:
                    logger.error(f"Health check failed for {agent_type.value}: {e}")
                    health_status[agent_type.value] = False
                    agent_info.status = AgentStatus.ERROR

        return health_status

    def reset_agent_errors(self, agent_type: Optional[AgentType] = None) -> None:
        """
        Reset error status for agents.

        Args:
            agent_type: Specific agent to reset, or None for all agents
        """
        with self._lock:
            agents_to_reset = [agent_type] if agent_type else list(self.agents.keys())

            for agent in agents_to_reset:
                if agent in self.agents:
                    agent_info = self.agents[agent]
                    if agent_info.status == AgentStatus.ERROR:
                        agent_info.status = AgentStatus.AVAILABLE
                        agent_info.error_count = 0
                        logger.info(f"Reset error status for {agent.value}")

    def shutdown(self) -> None:
        """Shutdown the orchestrator and cleanup resources."""
        logger.info("Shutting down AgentOrchestrator")
        self._shutdown = True

        # Wait for running tasks to complete
        with self._lock:
            busy_agents = [
                agent_type
                for agent_type, info in self.agents.items()
                if info.status == AgentStatus.BUSY
            ]

        if busy_agents:
            logger.info(f"Waiting for {len(busy_agents)} busy agents to complete")
            # Give agents time to complete current tasks
            time.sleep(2)

        # Shutdown executor
        self.executor.shutdown(wait=True)
        logger.info("AgentOrchestrator shutdown complete")
