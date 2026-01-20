"""
Router Configuration Validator for Gianna Multi-Agent System

This module provides Pydantic models for validating routing rules
configuration loaded from YAML files.
"""

import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator


class AgentTypeEnum(str, Enum):
    """Enumeration of available agent types for routing."""

    COMMAND = "COMMAND"
    AUDIO = "AUDIO"
    MEMORY = "MEMORY"
    CONVERSATION = "CONVERSATION"


class RoutingRuleConfig(BaseModel):
    """Configuration model for a single routing rule."""

    agent_type: AgentTypeEnum = Field(
        ..., description="The type of agent this rule routes to"
    )
    priority: int = Field(
        default=1, ge=1, le=10, description="Priority level for this rule (1-10)"
    )
    confidence_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to trigger this rule",
    )
    description: Optional[str] = Field(
        default=None, description="Human-readable description of this rule"
    )
    keywords: List[str] = Field(
        ..., min_length=1, description="List of keywords that trigger this rule"
    )
    patterns: List[str] = Field(
        default_factory=list, description="Regex patterns that trigger this rule"
    )

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        """Validate that keywords list is not empty and contains valid strings."""
        if not v:
            raise ValueError("Keywords list cannot be empty")
        # Normalize keywords to lowercase
        return [keyword.lower().strip() for keyword in v if keyword.strip()]

    @field_validator("patterns")
    @classmethod
    def validate_patterns(cls, v: List[str]) -> List[str]:
        """Validate that all patterns are valid regex expressions."""
        validated_patterns = []
        for pattern in v:
            if not pattern.strip():
                continue
            try:
                re.compile(pattern)
                validated_patterns.append(pattern)
            except re.error as e:
                raise ValueError(f'Invalid regex pattern "{pattern}": {e}')
        return validated_patterns

    class Config:
        use_enum_values = True


class RoutingRulesConfig(BaseModel):
    """Root configuration model for routing rules."""

    version: str = Field(..., description="Configuration version")
    description: Optional[str] = Field(
        default=None, description="Description of this configuration"
    )
    default_agent: AgentTypeEnum = Field(
        default=AgentTypeEnum.CONVERSATION,
        description="Default agent when no rule matches",
    )
    default_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default confidence for the default agent",
    )
    rules: List[RoutingRuleConfig] = Field(
        ..., min_length=1, description="List of routing rules"
    )

    @model_validator(mode="after")
    def validate_agent_coverage(self) -> "RoutingRulesConfig":
        """Ensure all agent types have at least one rule."""
        agent_types_with_rules = {rule.agent_type for rule in self.rules}
        missing_agents = set(AgentTypeEnum) - agent_types_with_rules

        if missing_agents:
            logger.warning(
                f"Some agent types have no routing rules: {missing_agents}. "
                f"They will only be selected via the default agent fallback."
            )

        return self

    class Config:
        use_enum_values = True


class RoutingConfigLoader:
    """Utility class for loading and validating routing configuration."""

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "routing_rules.yaml"

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the YAML configuration file.
                        Defaults to config/routing_rules.yaml
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Optional[RoutingRulesConfig] = None

    def load(self) -> RoutingRulesConfig:
        """
        Load and validate the routing configuration.

        Returns:
            RoutingRulesConfig: Validated configuration object

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the configuration is invalid
        """
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Routing configuration file not found: {self.config_path}"
            )

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            self._config = RoutingRulesConfig(**raw_config)
            logger.info(
                f"Loaded routing configuration from {self.config_path}: "
                f"{len(self._config.rules)} rules"
            )
            return self._config

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in routing config: {e}")
        except Exception as e:
            raise ValueError(f"Error loading routing config: {e}")

    def reload(self) -> RoutingRulesConfig:
        """Force reload the configuration from disk."""
        self._config = None
        return self.load()

    @staticmethod
    def get_default_config() -> RoutingRulesConfig:
        """
        Get a minimal default configuration for fallback.

        Returns:
            RoutingRulesConfig: Minimal default configuration
        """
        return RoutingRulesConfig(
            version="1.0-default",
            description="Minimal default routing configuration",
            default_agent=AgentTypeEnum.CONVERSATION,
            default_confidence=0.5,
            rules=[
                RoutingRuleConfig(
                    agent_type=AgentTypeEnum.COMMAND,
                    priority=3,
                    confidence_threshold=0.15,
                    keywords=["comando", "executar", "rodar", "run", "shell", "bash"],
                    patterns=[r"\b(execute|run|rodar)\s+"],
                ),
                RoutingRuleConfig(
                    agent_type=AgentTypeEnum.AUDIO,
                    priority=3,
                    confidence_threshold=0.15,
                    keywords=["falar", "áudio", "voz", "ouvir", "tocar", "gravar"],
                    patterns=[r"\b(falar|tocar|gravar)\s+"],
                ),
                RoutingRuleConfig(
                    agent_type=AgentTypeEnum.MEMORY,
                    priority=2,
                    confidence_threshold=0.1,
                    keywords=["lembrar", "memória", "histórico", "salvar", "buscar"],
                    patterns=[r"\blembrar\s+(de|que|sobre)"],
                ),
                RoutingRuleConfig(
                    agent_type=AgentTypeEnum.CONVERSATION,
                    priority=1,
                    confidence_threshold=0.05,
                    keywords=["conversar", "ajuda", "explicar", "pergunta", "obrigado"],
                    patterns=[r"\bme\s+(explique|conte|ajude)"],
                ),
            ],
        )


def load_routing_config(
    config_path: Optional[Path] = None, use_default_on_error: bool = True
) -> RoutingRulesConfig:
    """
    Convenience function to load routing configuration.

    Args:
        config_path: Optional path to configuration file
        use_default_on_error: If True, return default config on errors

    Returns:
        RoutingRulesConfig: Loaded or default configuration
    """
    loader = RoutingConfigLoader(config_path)

    try:
        return loader.load()
    except (FileNotFoundError, ValueError) as e:
        if use_default_on_error:
            logger.warning(f"Failed to load routing config: {e}. Using defaults.")
            return RoutingConfigLoader.get_default_config()
        raise
