"""
Tests for Router Configuration and YAML-based routing rules.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from gianna.coordination.router_config import (
    AgentTypeEnum,
    RoutingConfigLoader,
    RoutingRuleConfig,
    RoutingRulesConfig,
    load_routing_config,
)


class TestRoutingRuleConfig:
    """Tests for RoutingRuleConfig model."""

    def test_valid_rule_config(self):
        """Test creating a valid rule configuration."""
        rule = RoutingRuleConfig(
            agent_type=AgentTypeEnum.COMMAND,
            priority=3,
            confidence_threshold=0.15,
            keywords=["executar", "rodar", "comando"],
            patterns=[r"\bexecute\s+"],
        )

        assert rule.agent_type == AgentTypeEnum.COMMAND
        assert rule.priority == 3
        assert len(rule.keywords) == 3
        assert len(rule.patterns) == 1

    def test_keywords_normalized_to_lowercase(self):
        """Test that keywords are normalized to lowercase."""
        rule = RoutingRuleConfig(
            agent_type=AgentTypeEnum.AUDIO,
            keywords=["FALAR", "AUDIO", "ToCAR"],
        )

        assert all(k.islower() for k in rule.keywords)

    def test_empty_keywords_validation(self):
        """Test that empty keywords list raises error."""
        with pytest.raises(ValueError):
            RoutingRuleConfig(
                agent_type=AgentTypeEnum.COMMAND,
                keywords=[],
            )

    def test_invalid_regex_pattern(self):
        """Test that invalid regex patterns raise error."""
        with pytest.raises(ValueError):
            RoutingRuleConfig(
                agent_type=AgentTypeEnum.COMMAND,
                keywords=["test"],
                patterns=["[invalid(regex"],  # Invalid regex
            )

    def test_priority_bounds(self):
        """Test priority validation bounds."""
        # Valid priority
        rule = RoutingRuleConfig(
            agent_type=AgentTypeEnum.COMMAND,
            keywords=["test"],
            priority=5,
        )
        assert rule.priority == 5

        # Invalid priority (too low)
        with pytest.raises(ValueError):
            RoutingRuleConfig(
                agent_type=AgentTypeEnum.COMMAND,
                keywords=["test"],
                priority=0,
            )

        # Invalid priority (too high)
        with pytest.raises(ValueError):
            RoutingRuleConfig(
                agent_type=AgentTypeEnum.COMMAND,
                keywords=["test"],
                priority=11,
            )

    def test_confidence_threshold_bounds(self):
        """Test confidence threshold validation."""
        # Valid threshold
        rule = RoutingRuleConfig(
            agent_type=AgentTypeEnum.COMMAND,
            keywords=["test"],
            confidence_threshold=0.5,
        )
        assert rule.confidence_threshold == 0.5

        # Invalid threshold (too high)
        with pytest.raises(ValueError):
            RoutingRuleConfig(
                agent_type=AgentTypeEnum.COMMAND,
                keywords=["test"],
                confidence_threshold=1.5,
            )


class TestRoutingRulesConfig:
    """Tests for RoutingRulesConfig model."""

    def test_valid_config(self):
        """Test creating a valid routing configuration."""
        config = RoutingRulesConfig(
            version="1.0",
            description="Test config",
            rules=[
                RoutingRuleConfig(
                    agent_type=AgentTypeEnum.COMMAND,
                    keywords=["execute"],
                ),
                RoutingRuleConfig(
                    agent_type=AgentTypeEnum.AUDIO,
                    keywords=["play"],
                ),
            ],
        )

        assert config.version == "1.0"
        assert len(config.rules) == 2

    def test_empty_rules_validation(self):
        """Test that empty rules list raises error."""
        with pytest.raises(ValueError):
            RoutingRulesConfig(
                version="1.0",
                rules=[],
            )

    def test_default_agent_setting(self):
        """Test default agent configuration."""
        config = RoutingRulesConfig(
            version="1.0",
            default_agent=AgentTypeEnum.MEMORY,
            rules=[
                RoutingRuleConfig(
                    agent_type=AgentTypeEnum.COMMAND,
                    keywords=["test"],
                ),
            ],
        )

        assert config.default_agent == AgentTypeEnum.MEMORY


class TestRoutingConfigLoader:
    """Tests for RoutingConfigLoader class."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML configuration."""
        yaml_content = """
version: "1.0"
description: "Test routing rules"
default_agent: "CONVERSATION"
rules:
  - agent_type: "COMMAND"
    priority: 3
    keywords:
      - "executar"
      - "rodar"
    patterns:
      - "\\\\bexecute\\\\s+"
  - agent_type: "AUDIO"
    priority: 2
    keywords:
      - "falar"
      - "tocar"
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            loader = RoutingConfigLoader(Path(f.name))
            config = loader.load()

            assert config.version == "1.0"
            assert len(config.rules) == 2
            assert config.rules[0].agent_type == AgentTypeEnum.COMMAND

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        loader = RoutingConfigLoader(Path("/nonexistent/path.yaml"))

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            loader = RoutingConfigLoader(Path(f.name))

            with pytest.raises(ValueError):
                loader.load()

    def test_reload(self):
        """Test reloading configuration."""
        yaml_content1 = """
version: "1.0"
rules:
  - agent_type: "COMMAND"
    keywords:
      - "test1"
"""
        yaml_content2 = """
version: "2.0"
rules:
  - agent_type: "AUDIO"
    keywords:
      - "test2"
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content1)
            f.flush()

            loader = RoutingConfigLoader(Path(f.name))
            config1 = loader.load()
            assert config1.version == "1.0"

            # Update file
            with open(f.name, "w") as f2:
                f2.write(yaml_content2)

            config2 = loader.reload()
            assert config2.version == "2.0"

    def test_get_default_config(self):
        """Test getting default configuration."""
        default = RoutingConfigLoader.get_default_config()

        assert default.version == "1.0-default"
        assert len(default.rules) >= 1


class TestLoadRoutingConfig:
    """Tests for load_routing_config convenience function."""

    def test_load_with_fallback(self):
        """Test loading with fallback on error."""
        # Load nonexistent file with fallback
        config = load_routing_config(
            Path("/nonexistent/path.yaml"),
            use_default_on_error=True,
        )

        # Should return default config
        assert config is not None
        assert config.version == "1.0-default"

    def test_load_without_fallback(self):
        """Test loading without fallback raises error."""
        with pytest.raises(FileNotFoundError):
            load_routing_config(
                Path("/nonexistent/path.yaml"),
                use_default_on_error=False,
            )
