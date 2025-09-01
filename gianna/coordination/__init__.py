"""
Multi-Agent Coordination System for Gianna AI Assistant

This module implements the multi-agent coordination system that enables intelligent
routing, orchestration, and collaboration between specialized agents.

The coordination system provides:
- Intelligent routing based on Portuguese keyword analysis
- Agent registration and discovery
- Sequential and parallel execution coordination
- Conflict resolution and error recovery
- Agent lifecycle management
- Load balancing and performance optimization

Main Components:
- AgentOrchestrator: Central coordination and orchestration
- AgentRouter: Intelligent routing and decision making
- AgentManager: Lifecycle and resource management
"""

from .orchestrator import AgentOrchestrator, ExecutionMode
from .router import AgentRouter, AgentType

__all__ = ["AgentOrchestrator", "AgentRouter", "ExecutionMode", "AgentType"]
