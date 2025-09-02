"""
Configuration validation utilities for Gianna.

Provides schema validation for configuration files, with special
support for VAD configurations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None


class ValidationError(Exception):
    """Configuration validation error."""

    pass


class ConfigValidator:
    """Validate configuration files against schemas."""

    def __init__(self, schema_dir: Optional[Path] = None):
        """
        Initialize configuration validator.

        Args:
            schema_dir: Directory containing schema files.
                       Defaults to config/vad for VAD schemas
        """
        if schema_dir is None:
            # Default to project config directory
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            schema_dir = project_root / "config" / "vad"

        self.schema_dir = Path(schema_dir)

    def validate_vad_config(
        self, config: Dict[str, Any], schema_name: str = "schema"
    ) -> Dict[str, Any]:
        """
        Validate VAD configuration against schema.

        Args:
            config: Configuration dictionary to validate
            schema_name: Name of schema file (without .yaml extension)

        Returns:
            Validation results dictionary

        Raises:
            ValidationError: If validation fails
            FileNotFoundError: If schema file doesn't exist
        """
        schema_file = self.schema_dir / f"{schema_name}.yaml"

        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        # Load schema
        if yaml is None:
            raise ImportError(
                "PyYAML is required for validation. " "Install with: pip install pyyaml"
            )

        with open(schema_file, "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)

        # Perform validation
        errors = []
        warnings = []

        # Basic structural validation
        self._validate_structure(config, schema, errors, warnings)

        # VAD-specific validation
        self._validate_vad_specific(config, errors, warnings)

        # Compile results
        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "schema_version": schema.get("schema_version", "unknown"),
        }

        if errors:
            raise ValidationError(f"Validation failed: {errors}")

        return result

    def _validate_structure(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """
        Validate configuration structure against schema.

        Args:
            config: Configuration to validate
            schema: Schema definition
            errors: List to collect errors
            warnings: List to collect warnings
        """
        schema_props = schema.get("properties", {})
        required_fields = schema.get("required", [])

        # Check required fields
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate VAD section if present
        if "vad" in config:
            self._validate_vad_section(
                config["vad"], schema_props.get("vad", {}), errors, warnings
            )

        # Validate algorithms section if present
        if "algorithms" in config:
            self._validate_algorithms_section(
                config["algorithms"],
                schema_props.get("algorithms", {}),
                errors,
                warnings,
            )

        # Validate audio section if present
        if "audio" in config:
            self._validate_audio_section(
                config["audio"], schema_props.get("audio", {}), errors, warnings
            )

    def _validate_vad_section(
        self,
        vad_config: Dict[str, Any],
        vad_schema: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate VAD configuration section."""
        vad_props = vad_schema.get("properties", {})
        vad_required = vad_schema.get("required", [])

        # Check required VAD fields
        for field in vad_required:
            if field not in vad_config:
                errors.append(f"Missing required VAD field: {field}")

        # Validate log level
        if "log_level" in vad_config:
            log_level = vad_config["log_level"]
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if log_level not in valid_levels:
                errors.append(
                    f"Invalid log_level: {log_level}. Must be one of: {valid_levels}"
                )

        # Validate default algorithm
        if "default_algorithm" in vad_config:
            algorithm = vad_config["default_algorithm"]
            valid_algorithms = [
                "webrtc",
                "energy_based",
                "spectral_centroid",
                "zero_crossing_rate",
                "ml_vad",
                "ensemble",
            ]
            if algorithm not in valid_algorithms:
                errors.append(
                    f"Invalid default_algorithm: {algorithm}. Must be one of: {valid_algorithms}"
                )

        # Validate performance settings
        if "performance" in vad_config:
            perf = vad_config["performance"]

            if "max_concurrent_streams" in perf:
                streams = perf["max_concurrent_streams"]
                if not isinstance(streams, int) or streams < 1 or streams > 100:
                    errors.append(
                        "max_concurrent_streams must be an integer between 1 and 100"
                    )

            if "memory_limit_mb" in perf:
                memory = perf["memory_limit_mb"]
                if not isinstance(memory, int) or memory < 64 or memory > 16384:
                    errors.append(
                        "memory_limit_mb must be an integer between 64 and 16384"
                    )

    def _validate_algorithms_section(
        self,
        algorithms: Dict[str, Any],
        algo_schema: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate algorithms configuration section."""
        # Validate WebRTC settings
        if "webrtc" in algorithms:
            webrtc = algorithms["webrtc"]

            if "aggressiveness" in webrtc:
                aggr = webrtc["aggressiveness"]
                if not isinstance(aggr, int) or aggr < 0 or aggr > 3:
                    errors.append(
                        "webrtc.aggressiveness must be an integer between 0 and 3"
                    )

            if "frame_duration_ms" in webrtc:
                frame_dur = webrtc["frame_duration_ms"]
                if frame_dur not in [10, 20, 30]:
                    errors.append("webrtc.frame_duration_ms must be 10, 20, or 30")

            if "sample_rate" in webrtc:
                sr = webrtc["sample_rate"]
                valid_rates = [8000, 16000, 32000, 48000]
                if sr not in valid_rates:
                    errors.append(f"webrtc.sample_rate must be one of: {valid_rates}")

        # Validate energy-based settings
        if "energy_based" in algorithms:
            energy = algorithms["energy_based"]

            if "threshold" in energy:
                threshold = energy["threshold"]
                if (
                    not isinstance(threshold, (int, float))
                    or threshold < 0
                    or threshold > 1
                ):
                    errors.append(
                        "energy_based.threshold must be a number between 0.0 and 1.0"
                    )

        # Validate ensemble settings
        if "ensemble" in algorithms:
            ensemble = algorithms["ensemble"]

            if "algorithms" in ensemble:
                algos = ensemble["algorithms"]
                valid_algos = [
                    "webrtc",
                    "energy_based",
                    "spectral_centroid",
                    "zero_crossing_rate",
                    "ml_vad",
                ]

                if not isinstance(algos, list) or len(algos) < 2:
                    errors.append(
                        "ensemble.algorithms must be a list with at least 2 algorithms"
                    )
                else:
                    for algo in algos:
                        if algo not in valid_algos:
                            errors.append(
                                f"Invalid ensemble algorithm: {algo}. Must be one of: {valid_algos}"
                            )

            if "voting_strategy" in ensemble:
                strategy = ensemble["voting_strategy"]
                valid_strategies = ["majority", "weighted_average", "unanimous"]
                if strategy not in valid_strategies:
                    errors.append(
                        f"Invalid voting_strategy: {strategy}. Must be one of: {valid_strategies}"
                    )

    def _validate_audio_section(
        self,
        audio: Dict[str, Any],
        audio_schema: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate audio configuration section."""
        if "supported_formats" in audio:
            formats = audio["supported_formats"]
            valid_formats = ["wav", "mp3", "flac", "m4a", "ogg", "aac"]

            if not isinstance(formats, list):
                errors.append("audio.supported_formats must be a list")
            else:
                for fmt in formats:
                    if fmt not in valid_formats:
                        warnings.append(f"Unsupported audio format: {fmt}")

        if "default_sample_rate" in audio:
            sr = audio["default_sample_rate"]
            common_rates = [8000, 11025, 16000, 22050, 44100, 48000]
            if sr not in common_rates:
                warnings.append(
                    f"Uncommon sample rate: {sr}. Common rates: {common_rates}"
                )

        if "default_channels" in audio:
            channels = audio["default_channels"]
            if not isinstance(channels, int) or channels < 1 or channels > 8:
                errors.append(
                    "audio.default_channels must be an integer between 1 and 8"
                )

    def _validate_vad_specific(
        self, config: Dict[str, Any], errors: List[str], warnings: List[str]
    ) -> None:
        """Perform VAD-specific validation checks."""
        vad_config = config.get("vad", {})
        algorithms = config.get("algorithms", {})

        # Check if default algorithm is configured
        default_algo = vad_config.get("default_algorithm")
        if default_algo and default_algo not in algorithms:
            warnings.append(
                f"Default algorithm '{default_algo}' is not configured in algorithms section"
            )

        # Check real-time configuration consistency
        if config.get("real_time", {}).get("enabled", False):
            # Real-time mode should prefer fast algorithms
            if default_algo in ["ml_vad", "ensemble"]:
                warnings.append(
                    f"Algorithm '{default_algo}' may be too slow for real-time processing. "
                    "Consider 'webrtc' or 'energy_based' for real-time mode."
                )

            # Check latency settings
            rt_config = config.get("real_time", {})
            latency = rt_config.get("latency", {})
            max_latency = latency.get("max_total_latency_ms", 0)
            target_latency = latency.get("target_latency_ms", 0)

            if max_latency > 0 and target_latency > 0 and target_latency > max_latency:
                errors.append(
                    "target_latency_ms cannot be greater than max_total_latency_ms"
                )

        # Check production configuration
        if vad_config.get("log_level") == "DEBUG" and any(
            key in config for key in ["production", "monitoring"]
        ):
            warnings.append("DEBUG log level detected in production-like configuration")

        # Check memory limits vs concurrent streams
        performance = vad_config.get("performance", {})
        memory_limit = performance.get("memory_limit_mb", 0)
        max_streams = performance.get("max_concurrent_streams", 1)

        if memory_limit > 0 and max_streams > 1:
            memory_per_stream = memory_limit / max_streams
            if memory_per_stream < 32:
                warnings.append(
                    f"Low memory per stream ({memory_per_stream:.1f}MB). "
                    "Consider increasing memory_limit_mb or reducing max_concurrent_streams."
                )
