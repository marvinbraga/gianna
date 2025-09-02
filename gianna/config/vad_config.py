"""
VAD Configuration Management for Gianna.

Provides a high-level interface for managing Voice Activity Detection
configurations with validation, environment management, and runtime
configuration access.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .loader import ConfigLoader
from .validator import ConfigValidator, ValidationError


class VADConfigManager:
    """
    High-level VAD configuration management.

    Provides easy access to VAD configurations with validation,
    environment management, and runtime configuration.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize VAD configuration manager.

        Args:
            config_dir: Directory containing configuration files.
                       Defaults to project config directory.
        """
        self.loader = ConfigLoader(config_dir)
        self.validator = ConfigValidator(config_dir)
        self._current_config: Optional[Dict[str, Any]] = None
        self._current_environment: Optional[str] = None

    def load_environment(
        self, environment: str, validate: bool = True
    ) -> Dict[str, Any]:
        """
        Load VAD configuration for specified environment.

        Args:
            environment: Environment name (development, production, etc.)
            validate: Whether to validate configuration after loading

        Returns:
            Loaded and optionally validated configuration

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValidationError: If validation fails
        """
        # Load configuration
        config = self.loader.load_vad_config(environment)

        # Validate if requested
        if validate:
            validation_result = self.validator.validate_vad_config(config)
            if not validation_result["valid"]:
                raise ValidationError(
                    f"Configuration validation failed for {environment}: "
                    f"{validation_result['errors']}"
                )

        # Store as current configuration
        self._current_config = config
        self._current_environment = environment

        return config

    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """
        Get currently loaded configuration.

        Returns:
            Current configuration dictionary or None if none loaded
        """
        return self._current_config

    def get_current_environment(self) -> Optional[str]:
        """
        Get currently loaded environment name.

        Returns:
            Current environment name or None if none loaded
        """
        return self._current_environment

    def list_environments(self) -> List[str]:
        """
        List available VAD configuration environments.

        Returns:
            List of available environment names
        """
        return self.loader.list_vad_configs()

    def validate_environment(self, environment: str) -> Dict[str, Any]:
        """
        Validate VAD configuration for specified environment.

        Args:
            environment: Environment name to validate

        Returns:
            Validation results dictionary

        Raises:
            FileNotFoundError: If configuration file doesn't exist
        """
        config = self.loader.load_vad_config(environment)
        return self.validator.validate_vad_config(config)

    def get_algorithm_config(
        self, algorithm_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get configuration for specific VAD algorithm.

        Args:
            algorithm_name: Name of algorithm. If None, uses default algorithm.

        Returns:
            Algorithm configuration dictionary

        Raises:
            ValueError: If no configuration is loaded or algorithm not found
        """
        if self._current_config is None:
            raise ValueError("No configuration loaded. Call load_environment() first.")

        algorithms = self._current_config.get("algorithms", {})

        if algorithm_name is None:
            # Use default algorithm
            algorithm_name = self._current_config.get("vad", {}).get(
                "default_algorithm"
            )
            if algorithm_name is None:
                raise ValueError("No default algorithm specified in configuration")

        if algorithm_name not in algorithms:
            available = list(algorithms.keys())
            raise ValueError(
                f"Algorithm '{algorithm_name}' not found. Available: {available}"
            )

        return algorithms[algorithm_name]

    def get_vad_settings(self) -> Dict[str, Any]:
        """
        Get VAD general settings.

        Returns:
            VAD settings dictionary

        Raises:
            ValueError: If no configuration is loaded
        """
        if self._current_config is None:
            raise ValueError("No configuration loaded. Call load_environment() first.")

        return self._current_config.get("vad", {})

    def get_audio_settings(self) -> Dict[str, Any]:
        """
        Get audio processing settings.

        Returns:
            Audio settings dictionary

        Raises:
            ValueError: If no configuration is loaded
        """
        if self._current_config is None:
            raise ValueError("No configuration loaded. Call load_environment() first.")

        return self._current_config.get("audio", {})

    def get_performance_settings(self) -> Dict[str, Any]:
        """
        Get performance-related settings.

        Returns:
            Performance settings dictionary

        Raises:
            ValueError: If no configuration is loaded
        """
        if self._current_config is None:
            raise ValueError("No configuration loaded. Call load_environment() first.")

        return self._current_config.get("vad", {}).get("performance", {})

    def is_real_time_enabled(self) -> bool:
        """
        Check if real-time processing is enabled.

        Returns:
            True if real-time mode is enabled, False otherwise

        Raises:
            ValueError: If no configuration is loaded
        """
        if self._current_config is None:
            raise ValueError("No configuration loaded. Call load_environment() first.")

        return self._current_config.get("real_time", {}).get("enabled", False)

    def get_monitoring_settings(self) -> Dict[str, Any]:
        """
        Get monitoring and alerting settings.

        Returns:
            Monitoring settings dictionary

        Raises:
            ValueError: If no configuration is loaded
        """
        if self._current_config is None:
            raise ValueError("No configuration loaded. Call load_environment() first.")

        return self._current_config.get("monitoring", {})

    def get_testing_settings(self) -> Dict[str, Any]:
        """
        Get testing and validation settings.

        Returns:
            Testing settings dictionary

        Raises:
            ValueError: If no configuration is loaded
        """
        if self._current_config is None:
            raise ValueError("No configuration loaded. Call load_environment() first.")

        return self._current_config.get("testing", {})

    def auto_select_environment(self) -> str:
        """
        Automatically select appropriate environment based on context.

        Returns:
            Selected environment name
        """
        # Check environment variables
        env_var = os.environ.get("GIANNA_VAD_ENV")
        if env_var:
            return env_var

        # Check if we're in a development context
        if os.environ.get("GIANNA_DEV", "").lower() in ("1", "true", "yes"):
            return "development"

        # Check for production indicators
        if any(os.environ.get(var) for var in ["PRODUCTION", "PROD", "GIANNA_PROD"]):
            return "production"

        # Default to development
        return "development"

    def create_runtime_config(
        self,
        environment: Optional[str] = None,
        algorithm_override: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a runtime configuration with optional overrides.

        Args:
            environment: Environment to load (auto-selected if None)
            algorithm_override: Override default algorithm
            **kwargs: Additional configuration overrides

        Returns:
            Runtime configuration dictionary
        """
        if environment is None:
            environment = self.auto_select_environment()

        # Load base configuration
        config = self.load_environment(environment, validate=True)

        # Apply overrides
        if algorithm_override:
            config.setdefault("vad", {})["default_algorithm"] = algorithm_override

        # Apply additional overrides
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like "vad.log_level"
                keys = key.split(".")
                current = config
                for k in keys[:-1]:
                    current = current.setdefault(k, {})
                current[keys[-1]] = value
            else:
                config[key] = value

        return config

    def export_config(
        self, output_path: Path, environment: Optional[str] = None, format: str = "yaml"
    ) -> None:
        """
        Export current or specified configuration to file.

        Args:
            output_path: Path to write configuration file
            environment: Environment to export (current if None)
            format: Export format ("yaml" or "json")

        Raises:
            ValueError: If invalid format specified or no config to export
        """
        if format not in ["yaml", "json"]:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'.")

        # Get configuration to export
        if environment:
            config = self.loader.load_vad_config(environment)
        elif self._current_config:
            config = self._current_config
        else:
            raise ValueError("No configuration to export. Load an environment first.")

        # Write to file
        if format == "yaml":
            import yaml

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        else:  # json
            import json

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration.

        Returns:
            Configuration summary dictionary

        Raises:
            ValueError: If no configuration is loaded
        """
        if self._current_config is None:
            raise ValueError("No configuration loaded. Call load_environment() first.")

        vad_settings = self.get_vad_settings()
        algorithms = self._current_config.get("algorithms", {})

        summary = {
            "environment": self._current_environment,
            "default_algorithm": vad_settings.get("default_algorithm"),
            "log_level": vad_settings.get("log_level"),
            "available_algorithms": list(algorithms.keys()),
            "real_time_enabled": self.is_real_time_enabled(),
            "performance": {
                "max_concurrent_streams": self.get_performance_settings().get(
                    "max_concurrent_streams"
                ),
                "memory_limit_mb": self.get_performance_settings().get(
                    "memory_limit_mb"
                ),
            },
            "audio": {
                "sample_rate": self.get_audio_settings().get("default_sample_rate"),
                "channels": self.get_audio_settings().get("default_channels"),
                "supported_formats": self.get_audio_settings().get("supported_formats"),
            },
        }

        return summary
