"""
Test configuration management for VAD testing.

This module provides utilities for loading and managing test configurations
across different testing scenarios.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from gianna.audio.vad import VADConfig

logger = logging.getLogger(__name__)

# Path to configuration files
CONFIG_DIR = Path(__file__).parent
VAD_CONFIGS_FILE = CONFIG_DIR / "vad_test_configs.yaml"


class TestConfigManager:
    """Manager for VAD test configurations."""

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize test configuration manager.

        Args:
            config_file: Path to configuration file. If None, uses default.
        """
        self.config_file = config_file or VAD_CONFIGS_FILE
        self._configs = None
        self._load_configs()

    def _load_configs(self) -> None:
        """Load configurations from YAML file."""
        try:
            with open(self.config_file, "r") as f:
                self._configs = yaml.safe_load(f)
            logger.info(f"Loaded test configurations from {self.config_file}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_file}")
            self._configs = {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            self._configs = {}

    def get_config(self, category: str, name: str) -> Dict[str, Any]:
        """
        Get configuration by category and name.

        Args:
            category: Configuration category (e.g., 'basic_configs')
            name: Configuration name within category

        Returns:
            Dict containing configuration parameters

        Raises:
            KeyError: If category or name not found
        """
        if self._configs is None:
            self._load_configs()

        try:
            return self._configs[category][name].copy()
        except KeyError:
            available_categories = list(self._configs.keys())
            if category in self._configs:
                available_names = list(self._configs[category].keys())
                raise KeyError(
                    f"Configuration '{name}' not found in category '{category}'. Available: {available_names}"
                )
            else:
                raise KeyError(
                    f"Category '{category}' not found. Available categories: {available_categories}"
                )

    def get_vad_config(self, category: str, name: str) -> VADConfig:
        """
        Get VAD configuration object.

        Args:
            category: Configuration category
            name: Configuration name

        Returns:
            VADConfig object
        """
        config_dict = self.get_config(category, name)

        # Extract VAD-specific parameters
        vad_params = {
            k: v
            for k, v in config_dict.items()
            if k not in ["algorithm", "description"]
        }

        return VADConfig(**vad_params)

    def list_categories(self) -> List[str]:
        """
        List all available configuration categories.

        Returns:
            List of category names
        """
        if self._configs is None:
            self._load_configs()
        return list(self._configs.keys())

    def list_configs(self, category: str) -> List[str]:
        """
        List all configurations in a category.

        Args:
            category: Configuration category

        Returns:
            List of configuration names
        """
        if self._configs is None:
            self._load_configs()

        if category not in self._configs:
            return []

        return list(self._configs[category].keys())

    def get_all_configs_in_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all configurations in a category.

        Args:
            category: Configuration category

        Returns:
            Dict mapping config names to config dictionaries
        """
        if self._configs is None:
            self._load_configs()

        if category not in self._configs:
            return {}

        return {name: config.copy() for name, config in self._configs[category].items()}

    def create_vad_from_config(self, category: str, name: str):
        """
        Create VAD instance from configuration.

        Args:
            category: Configuration category
            name: Configuration name

        Returns:
            Configured VAD instance
        """
        from gianna.audio.vad import create_vad

        config_dict = self.get_config(category, name)
        algorithm = config_dict.pop("algorithm", "energy")
        description = config_dict.pop("description", "")

        logger.info(
            f"Creating VAD with config '{name}' from '{category}': {description}"
        )

        return create_vad(algorithm, **config_dict)


# Global configuration manager instance
config_manager = TestConfigManager()


# Convenience functions
def get_config(category: str, name: str) -> Dict[str, Any]:
    """Get configuration by category and name."""
    return config_manager.get_config(category, name)


def get_vad_config(category: str, name: str) -> VADConfig:
    """Get VAD configuration object."""
    return config_manager.get_vad_config(category, name)


def create_vad_from_config(category: str, name: str):
    """Create VAD instance from configuration."""
    return config_manager.create_vad_from_config(category, name)


def list_categories() -> List[str]:
    """List all available configuration categories."""
    return config_manager.list_categories()


def list_configs(category: str) -> List[str]:
    """List all configurations in a category."""
    return config_manager.list_configs(category)


# Configuration presets for common testing scenarios
class ConfigPresets:
    """Common configuration presets for testing."""

    @staticmethod
    def get_basic_energy_config() -> VADConfig:
        """Get basic Energy VAD configuration."""
        return get_vad_config("basic_configs", "energy_default")

    @staticmethod
    def get_performance_config(profile: str = "low_latency") -> VADConfig:
        """Get performance-optimized configuration."""
        return get_vad_config("performance_configs", profile)

    @staticmethod
    def get_environment_config(environment: str = "office_environment") -> VADConfig:
        """Get environment-specific configuration."""
        return get_vad_config("environment_configs", environment)

    @staticmethod
    def get_streaming_config(quality: str = "real_time_basic") -> VADConfig:
        """Get streaming configuration."""
        return get_vad_config("streaming_configs", quality)

    @staticmethod
    def get_algorithm_configs() -> Dict[str, VADConfig]:
        """Get configurations for all available algorithms."""
        configs = {}

        algorithm_configs = config_manager.get_all_configs_in_category(
            "algorithm_configs"
        )
        for name, config_dict in algorithm_configs.items():
            try:
                algorithm = config_dict.get("algorithm", "energy")
                vad_params = {
                    k: v
                    for k, v in config_dict.items()
                    if k not in ["algorithm", "description"]
                }
                configs[name] = VADConfig(**vad_params)
            except Exception as e:
                logger.warning(f"Could not create config for {name}: {e}")

        return configs


# Test data configuration
class TestDataConfig:
    """Configuration for test data generation."""

    @staticmethod
    def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
        """Get test dataset configuration."""
        return get_config("test_datasets", dataset_name)

    @staticmethod
    def get_all_dataset_configs() -> Dict[str, Dict[str, Any]]:
        """Get all test dataset configurations."""
        return config_manager.get_all_configs_in_category("test_datasets")


# Benchmark configuration utilities
class BenchmarkConfig:
    """Benchmark-specific configuration utilities."""

    @staticmethod
    def get_benchmark_configs() -> Dict[str, Dict[str, Any]]:
        """Get all benchmark configurations."""
        return config_manager.get_all_configs_in_category("benchmark_configs")

    @staticmethod
    def get_speed_benchmark_config() -> Dict[str, Any]:
        """Get speed benchmark configuration."""
        return get_config("benchmark_configs", "benchmark_speed")

    @staticmethod
    def get_accuracy_benchmark_config() -> Dict[str, Any]:
        """Get accuracy benchmark configuration."""
        return get_config("benchmark_configs", "benchmark_accuracy")


# Validation functions
def validate_config(config_dict: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.

    Args:
        config_dict: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["algorithm"]

    # Check required fields
    for field in required_fields:
        if field not in config_dict:
            logger.error(f"Missing required field: {field}")
            return False

    # Validate algorithm
    from gianna.audio.vad import get_available_algorithms

    available_algorithms = get_available_algorithms()

    algorithm = config_dict.get("algorithm")
    if algorithm not in available_algorithms:
        logger.error(
            f"Algorithm '{algorithm}' not available. Available: {available_algorithms}"
        )
        return False

    # Validate VAD config parameters
    try:
        vad_params = {
            k: v
            for k, v in config_dict.items()
            if k not in ["algorithm", "description"]
        }
        VADConfig(**vad_params)
    except Exception as e:
        logger.error(f"Invalid VAD configuration: {e}")
        return False

    return True


def load_custom_config(file_path: str) -> TestConfigManager:
    """
    Load custom configuration from file.

    Args:
        file_path: Path to custom configuration file

    Returns:
        TestConfigManager instance with custom configurations
    """
    return TestConfigManager(Path(file_path))


# Environment-based configuration selection
def get_config_for_environment() -> Dict[str, Any]:
    """
    Get configuration based on environment variables.

    Environment variables:
    - VAD_TEST_ALGORITHM: Algorithm to use
    - VAD_TEST_PROFILE: Performance profile (low_latency, balanced, high_quality)
    - VAD_TEST_ENVIRONMENT: Environment type (office, noisy, quiet, car)

    Returns:
        Configuration dictionary
    """
    algorithm = os.environ.get("VAD_TEST_ALGORITHM", "energy")
    profile = os.environ.get("VAD_TEST_PROFILE", "balanced")
    environment = os.environ.get("VAD_TEST_ENVIRONMENT", "office")

    # Try to get algorithm-specific config first
    try:
        if algorithm == "energy":
            return get_config("basic_configs", f"energy_default")
        else:
            return get_config("algorithm_configs", f"{algorithm}_default")
    except KeyError:
        pass

    # Fallback to environment config
    try:
        return get_config("environment_configs", f"{environment}_environment")
    except KeyError:
        pass

    # Final fallback to basic config
    return get_config("basic_configs", "energy_default")


if __name__ == "__main__":
    # Test configuration loading
    print("Available categories:")
    for category in list_categories():
        print(f"  - {category}")
        configs = list_configs(category)
        for config_name in configs[:3]:  # Show first 3 configs
            print(f"    - {config_name}")

    # Test basic config loading
    basic_config = get_vad_config("basic_configs", "energy_default")
    print(f"\nBasic config threshold: {basic_config.threshold}")

    # Test VAD creation
    try:
        vad = create_vad_from_config("basic_configs", "energy_default")
        print(f"Created VAD: {vad.__class__.__name__}")
    except Exception as e:
        print(f"Error creating VAD: {e}")
