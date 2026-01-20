"""
Intelligent Agent Router for Gianna Multi-Agent System

This module implements intelligent routing logic that analyzes Portuguese text
to determine the most appropriate agent for handling specific requests.

The routing rules are loaded from a YAML configuration file, making it easy
to customize and extend without modifying code.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

from loguru import logger

from ..core.state import GiannaState
from .router_config import (
    AgentTypeEnum,
    RoutingConfigLoader,
    RoutingRuleConfig,
    RoutingRulesConfig,
    load_routing_config,
)


class AgentType(Enum):
    """Enumeration of available agent types."""

    COMMAND = "command_agent"
    AUDIO = "audio_agent"
    CONVERSATION = "conversation_agent"
    MEMORY = "memory_agent"

    @classmethod
    def from_config_enum(cls, config_enum: AgentTypeEnum) -> "AgentType":
        """Convert from configuration enum to AgentType."""
        mapping = {
            AgentTypeEnum.COMMAND: cls.COMMAND,
            AgentTypeEnum.AUDIO: cls.AUDIO,
            AgentTypeEnum.MEMORY: cls.MEMORY,
            AgentTypeEnum.CONVERSATION: cls.CONVERSATION,
        }
        return mapping[config_enum]


@dataclass
class RoutingRule:
    """Represents a routing rule with keywords and compiled patterns."""

    agent_type: AgentType
    keywords: List[str]
    compiled_patterns: List[Pattern] = field(default_factory=list)
    priority: int = 1
    confidence_threshold: float = 0.6
    description: Optional[str] = None

    @classmethod
    def from_config(cls, config: RoutingRuleConfig) -> "RoutingRule":
        """Create a RoutingRule from a configuration object."""
        # Compile regex patterns for performance
        compiled_patterns = []
        for pattern in config.patterns:
            try:
                compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(f"Skipping invalid pattern '{pattern}': {e}")

        return cls(
            agent_type=AgentType.from_config_enum(AgentTypeEnum(config.agent_type)),
            keywords=config.keywords,
            compiled_patterns=compiled_patterns,
            priority=config.priority,
            confidence_threshold=config.confidence_threshold,
            description=config.description,
        )


class AgentRouter:
    """
    Intelligent router that determines the best agent for handling requests.

    Uses Portuguese keyword analysis, pattern matching, and contextual
    information to make routing decisions with confidence scoring.

    Routing rules are loaded from a YAML configuration file for easy
    customization without code changes.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the router with routing rules from configuration.

        Args:
            config_path: Optional path to routing rules YAML file.
                        Defaults to config/routing_rules.yaml
        """
        self.config_path = config_path
        self._config: Optional[RoutingRulesConfig] = None
        self.routing_rules = self._load_routing_rules()
        self.routing_history: List[Dict] = []
        self._default_agent = AgentType.CONVERSATION
        self._default_confidence = 0.5

    def _load_routing_rules(self) -> List[RoutingRule]:
        """
        Load routing rules from YAML configuration file.

        Returns:
            List[RoutingRule]: List of compiled routing rules
        """
        try:
            self._config = load_routing_config(
                config_path=self.config_path, use_default_on_error=True
            )

            # Convert config rules to RoutingRule objects
            rules = [RoutingRule.from_config(rule_config) for rule_config in self._config.rules]

            # Set defaults from config
            self._default_agent = AgentType.from_config_enum(
                AgentTypeEnum(self._config.default_agent)
            )
            self._default_confidence = self._config.default_confidence

            logger.info(f"Loaded {len(rules)} routing rules from configuration")
            return rules

        except Exception as e:
            logger.error(f"Failed to load routing rules: {e}. Using hardcoded defaults.")
            return self._get_fallback_rules()

    def _get_fallback_rules(self) -> List[RoutingRule]:
        """
        Get minimal fallback rules when configuration loading fails.

        Returns:
            List[RoutingRule]: Minimal set of routing rules
        """
        return [
            RoutingRule(
                agent_type=AgentType.COMMAND,
                keywords=["comando", "executar", "rodar", "run", "shell"],
                compiled_patterns=[re.compile(r"\b(execute|run|rodar)\s+", re.IGNORECASE)],
                priority=3,
                confidence_threshold=0.15,
            ),
            RoutingRule(
                agent_type=AgentType.AUDIO,
                keywords=["falar", "áudio", "voz", "ouvir", "tocar"],
                compiled_patterns=[re.compile(r"\b(falar|tocar)\s+", re.IGNORECASE)],
                priority=3,
                confidence_threshold=0.15,
            ),
            RoutingRule(
                agent_type=AgentType.MEMORY,
                keywords=["lembrar", "memória", "histórico", "salvar"],
                compiled_patterns=[re.compile(r"\blembrar\s+", re.IGNORECASE)],
                priority=2,
                confidence_threshold=0.1,
            ),
            RoutingRule(
                agent_type=AgentType.CONVERSATION,
                keywords=["conversar", "ajuda", "explicar", "pergunta"],
                compiled_patterns=[re.compile(r"\bme\s+(explique|ajude)", re.IGNORECASE)],
                priority=1,
                confidence_threshold=0.05,
            ),
        ]

    def reload_rules(self) -> None:
        """Reload routing rules from configuration file."""
        self.routing_rules = self._load_routing_rules()
        logger.info("Routing rules reloaded")

    def route_request(self, state: GiannaState) -> Tuple[AgentType, float]:
        """
        Route a request to the most appropriate agent.

        Args:
            state: Current Gianna system state

        Returns:
            Tuple[AgentType, float]: Selected agent type and confidence score
        """
        if not state["conversation"].messages:
            return self._default_agent, self._default_confidence

        last_message = state["conversation"].messages[-1]
        if last_message.get("role") != "user":
            return self._default_agent, self._default_confidence

        content = last_message.get("content", "").lower().strip()
        if not content:
            return self._default_agent, self._default_confidence

        # Calculate scores for each agent type
        agent_scores = self._calculate_routing_scores(content, state)

        # Select the best agent based on scores and context
        selected_agent, confidence = self._select_best_agent(agent_scores, state)

        # Log routing decision
        self._log_routing_decision(content, selected_agent, confidence, agent_scores)

        # Update routing history
        self.routing_history.append(
            {
                "message": content[:100],  # Truncate for privacy
                "selected_agent": selected_agent.value,
                "confidence": confidence,
                "all_scores": {k.value: v for k, v in agent_scores.items()},
            }
        )

        return selected_agent, confidence

    def _calculate_routing_scores(
        self, content: str, state: GiannaState
    ) -> Dict[AgentType, float]:
        """
        Calculate routing scores for all agent types.

        Args:
            content: Message content to analyze
            state: Current system state for context

        Returns:
            Dict[AgentType, float]: Scores for each agent type
        """
        scores = {agent_type: 0.0 for agent_type in AgentType}

        for rule in self.routing_rules:
            score = self._evaluate_rule(rule, content, state)
            scores[rule.agent_type] += score * rule.priority

        # Apply contextual adjustments
        scores = self._apply_contextual_adjustments(scores, content, state)

        # Normalize scores
        max_possible_score = max(
            (rule.priority for rule in self.routing_rules), default=1
        ) * 3
        for agent_type in scores:
            scores[agent_type] = min(1.0, scores[agent_type] / max_possible_score)

        return scores

    def _evaluate_rule(
        self, rule: RoutingRule, content: str, state: GiannaState
    ) -> float:
        """
        Evaluate a single routing rule against the content.

        Args:
            rule: Routing rule to evaluate
            content: Message content
            state: Current system state

        Returns:
            float: Rule evaluation score
        """
        score = 0.0

        # Keyword matching
        keyword_matches = sum(1 for keyword in rule.keywords if keyword in content)
        keyword_score = min(1.0, keyword_matches / max(1, len(rule.keywords) * 0.1))

        # Pattern matching (using pre-compiled patterns)
        pattern_matches = sum(
            1 for pattern in rule.compiled_patterns if pattern.search(content)
        )
        pattern_score = min(
            1.0, pattern_matches / max(1, len(rule.compiled_patterns))
        ) if rule.compiled_patterns else 0.0

        # Combine scores
        if rule.compiled_patterns:
            score = (keyword_score * 0.6) + (pattern_score * 0.4)
        else:
            score = keyword_score

        return score if score >= rule.confidence_threshold else 0.0

    def _apply_contextual_adjustments(
        self, scores: Dict[AgentType, float], content: str, state: GiannaState
    ) -> Dict[AgentType, float]:
        """
        Apply contextual adjustments based on conversation history and state.

        Args:
            scores: Current routing scores
            content: Message content
            state: Current system state

        Returns:
            Dict[AgentType, float]: Adjusted scores
        """
        adjusted_scores = scores.copy()

        # Recent agent usage - slight preference for continuity
        if self.routing_history:
            last_agent = AgentType(self.routing_history[-1]["selected_agent"])
            if adjusted_scores[last_agent] > 0.3:  # Only if already reasonably relevant
                adjusted_scores[last_agent] *= 1.1

        # Audio mode context
        if state["audio"].current_mode in ["listening", "speaking"]:
            adjusted_scores[AgentType.AUDIO] *= 1.2

        # Command history context
        recent_commands = state["commands"].execution_history[-3:]
        if recent_commands and any(
            "error" in str(cmd).lower() for cmd in recent_commands
        ):
            # If recent commands had errors, boost command agent for follow-ups
            if any(
                word in content
                for word in ["erro", "problema", "não", "funciona", "fix"]
            ):
                adjusted_scores[AgentType.COMMAND] *= 1.3

        # Memory-related follow-up detection
        user_prefs = state["conversation"].user_preferences
        if user_prefs and any(
            word in content
            for word in ["mudança", "alterar", "configurar", "preferência"]
        ):
            adjusted_scores[AgentType.MEMORY] *= 1.2

        return adjusted_scores

    def _select_best_agent(
        self, scores: Dict[AgentType, float], state: GiannaState
    ) -> Tuple[AgentType, float]:
        """
        Select the best agent based on scores and system constraints.

        Args:
            scores: Routing scores for each agent
            state: Current system state

        Returns:
            Tuple[AgentType, float]: Selected agent and confidence score
        """
        # Find the highest scoring agent
        best_agent = max(scores, key=scores.get)
        best_score = scores[best_agent]

        # Minimum confidence check
        if best_score < 0.05:
            # Fall back to default agent for low confidence
            return self._default_agent, self._default_confidence

        # If multiple agents have similar high scores, apply tie-breaking rules
        high_scores = {k: v for k, v in scores.items() if v > best_score * 0.9}

        if len(high_scores) > 1:
            # Tie-breaking priority: Command > Audio > Memory > Conversation
            priority_order = [
                AgentType.COMMAND,
                AgentType.AUDIO,
                AgentType.MEMORY,
                AgentType.CONVERSATION,
            ]

            for agent_type in priority_order:
                if agent_type in high_scores:
                    return agent_type, high_scores[agent_type]

        return best_agent, best_score

    def _log_routing_decision(
        self,
        content: str,
        selected_agent: AgentType,
        confidence: float,
        all_scores: Dict[AgentType, float],
    ) -> None:
        """
        Log the routing decision for debugging and monitoring.

        Args:
            content: Message content (truncated)
            selected_agent: Selected agent
            confidence: Confidence score
            all_scores: All calculated scores
        """
        content_preview = content[:50] + "..." if len(content) > 50 else content

        logger.info(
            f"Routing decision: '{content_preview}' -> {selected_agent.value} "
            f"(confidence: {confidence:.2f})"
        )

        # Log detailed scores in debug mode
        score_details = ", ".join(
            [f"{agent.value}: {score:.2f}" for agent, score in all_scores.items()]
        )
        logger.debug(f"All routing scores: {score_details}")

    def get_routing_stats(self) -> Dict[str, any]:
        """
        Get routing statistics and performance metrics.

        Returns:
            Dict[str, any]: Routing statistics
        """
        if not self.routing_history:
            return {"total_routes": 0, "agent_distribution": {}, "avg_confidence": 0.0}

        total_routes = len(self.routing_history)
        agent_counts = {}
        total_confidence = 0.0

        for route in self.routing_history:
            agent = route["selected_agent"]
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            total_confidence += route["confidence"]

        agent_distribution = {
            agent: count / total_routes for agent, count in agent_counts.items()
        }

        return {
            "total_routes": total_routes,
            "agent_distribution": agent_distribution,
            "avg_confidence": total_confidence / total_routes,
            "recent_routes": self.routing_history[-10:],  # Last 10 for analysis
        }

    def get_config_info(self) -> Dict[str, any]:
        """
        Get information about the loaded configuration.

        Returns:
            Dict[str, any]: Configuration information
        """
        if self._config:
            return {
                "version": self._config.version,
                "description": self._config.description,
                "rules_count": len(self._config.rules),
                "default_agent": self._default_agent.value,
                "config_path": str(self.config_path) if self.config_path else "default",
            }
        return {"status": "using_fallback_rules", "rules_count": len(self.routing_rules)}

    def clear_routing_history(self) -> None:
        """Clear routing history (useful for testing or privacy)."""
        self.routing_history.clear()
        logger.info("Routing history cleared")
