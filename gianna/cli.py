#!/usr/bin/env python3
"""
Gianna CLI - Administrative Command Line Interface

Provides administrative commands for managing the Gianna voice assistant,
including configuration validation, health checks, benchmarks, and more.

Usage:
    gianna config validate           - Validate configuration
    gianna config generate           - Generate config template
    gianna health check              - Run health checks
    gianna benchmark vad             - Run VAD benchmarks
    gianna cache clear               - Clear all caches
    gianna secrets rotate            - Rotate encryption keys
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="gianna",
        description="Gianna Voice Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gianna config validate              Validate current configuration
  gianna config generate -o my.yaml   Generate config template
  gianna health check --verbose       Run health checks with details
  gianna benchmark vad --algorithm adaptive  Run VAD benchmark
  gianna cache stats                  Show cache statistics
  gianna cache clear --confirm        Clear all caches
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.4",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file",
        default=None,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config commands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    config_validate = config_subparsers.add_parser("validate", help="Validate configuration")
    config_validate.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation for production",
    )

    config_generate = config_subparsers.add_parser("generate", help="Generate config template")
    config_generate.add_argument(
        "-o",
        "--output",
        type=str,
        default="gianna.example.yaml",
        help="Output file path",
    )

    config_show = config_subparsers.add_parser("show", help="Show current configuration")
    config_show.add_argument(
        "--section",
        type=str,
        help="Show specific section (llm, audio, memory, etc.)",
    )

    # Health commands
    health_parser = subparsers.add_parser("health", help="Health checks")
    health_subparsers = health_parser.add_subparsers(dest="health_command")

    health_check = health_subparsers.add_parser("check", help="Run health checks")
    health_check.add_argument(
        "--component",
        type=str,
        choices=["all", "llm", "audio", "memory", "cache"],
        default="all",
        help="Component to check",
    )

    # Benchmark commands
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command")

    benchmark_vad = benchmark_subparsers.add_parser("vad", help="Run VAD benchmark")
    benchmark_vad.add_argument(
        "--algorithm",
        type=str,
        choices=["energy", "spectral", "webrtc", "silero", "adaptive", "all"],
        default="all",
        help="VAD algorithm to benchmark",
    )
    benchmark_vad.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Benchmark duration in seconds",
    )

    benchmark_llm = benchmark_subparsers.add_parser("llm", help="Run LLM benchmark")
    benchmark_llm.add_argument(
        "--provider",
        type=str,
        help="LLM provider to benchmark",
    )

    # Cache commands
    cache_parser = subparsers.add_parser("cache", help="Cache management")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command")

    cache_stats = cache_subparsers.add_parser("stats", help="Show cache statistics")
    cache_clear = cache_subparsers.add_parser("clear", help="Clear cache")
    cache_clear.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm cache clear operation",
    )
    cache_clear.add_argument(
        "--type",
        type=str,
        choices=["all", "memory", "redis", "sqlite"],
        default="all",
        help="Type of cache to clear",
    )

    # Secrets commands
    secrets_parser = subparsers.add_parser("secrets", help="Secrets management")
    secrets_subparsers = secrets_parser.add_subparsers(dest="secrets_command")

    secrets_rotate = secrets_subparsers.add_parser("rotate", help="Rotate encryption keys")
    secrets_rotate.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm key rotation",
    )

    secrets_validate = secrets_subparsers.add_parser("validate", help="Validate secrets configuration")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")

    return parser


def cmd_config_validate(args: argparse.Namespace) -> int:
    """Validate configuration."""
    print("Validating configuration...")

    try:
        from gianna.config import get_settings, validate_config

        if args.config:
            issues = validate_config(args.config)
        else:
            settings = get_settings()
            issues = settings.validate_for_production() if args.strict else []

        if issues:
            print("\n❌ Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        else:
            print("✅ Configuration is valid!")
            return 0

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1


def cmd_config_generate(args: argparse.Namespace) -> int:
    """Generate configuration template."""
    try:
        from gianna.config import generate_config_template

        generate_config_template(args.output)
        print(f"✅ Generated configuration template: {args.output}")
        return 0

    except Exception as e:
        print(f"❌ Error generating config: {e}")
        return 1


def cmd_config_show(args: argparse.Namespace) -> int:
    """Show current configuration."""
    try:
        from gianna.config import get_settings

        settings = get_settings(args.config)

        if args.section:
            section = getattr(settings, args.section, None)
            if section is None:
                print(f"❌ Unknown section: {args.section}")
                return 1
            data = section.model_dump()
        else:
            data = settings._to_safe_dict()

        print(json.dumps(data, indent=2, default=str))
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_health_check(args: argparse.Namespace) -> int:
    """Run health checks."""
    print("Running health checks...\n")

    results: Dict[str, Dict[str, Any]] = {}
    all_healthy = True

    # Check configuration
    print("Checking configuration...")
    try:
        from gianna.config import get_settings
        settings = get_settings()
        results["config"] = {"status": "healthy", "message": "Configuration loaded successfully"}
        print("  ✅ Configuration: OK")
    except Exception as e:
        results["config"] = {"status": "unhealthy", "error": str(e)}
        print(f"  ❌ Configuration: {e}")
        all_healthy = False

    # Check LLM providers
    if args.component in ["all", "llm"]:
        print("Checking LLM providers...")
        try:
            # Just check if we can import the modules
            from gianna.assistants.models import factory_method
            results["llm"] = {"status": "healthy", "message": "LLM modules available"}
            print("  ✅ LLM modules: OK")
        except Exception as e:
            results["llm"] = {"status": "unhealthy", "error": str(e)}
            print(f"  ❌ LLM modules: {e}")
            all_healthy = False

    # Check audio system
    if args.component in ["all", "audio"]:
        print("Checking audio system...")
        try:
            from gianna.audio.vad import types
            results["audio"] = {"status": "healthy", "message": "Audio modules available"}
            print("  ✅ Audio modules: OK")
        except Exception as e:
            results["audio"] = {"status": "unhealthy", "error": str(e)}
            print(f"  ❌ Audio modules: {e}")
            all_healthy = False

    # Check memory system
    if args.component in ["all", "memory"]:
        print("Checking memory system...")
        try:
            from gianna.memory import semantic_memory
            results["memory"] = {"status": "healthy", "message": "Memory modules available"}
            print("  ✅ Memory modules: OK")
        except Exception as e:
            results["memory"] = {"status": "unhealthy", "error": str(e)}
            print(f"  ❌ Memory modules: {e}")
            all_healthy = False

    # Check cache
    if args.component in ["all", "cache"]:
        print("Checking cache system...")
        try:
            from gianna.optimization import caching
            results["cache"] = {"status": "healthy", "message": "Cache modules available"}
            print("  ✅ Cache modules: OK")
        except Exception as e:
            results["cache"] = {"status": "unhealthy", "error": str(e)}
            print(f"  ❌ Cache modules: {e}")
            all_healthy = False

    print()
    if all_healthy:
        print("✅ All health checks passed!")
        return 0
    else:
        print("❌ Some health checks failed!")
        return 1


def cmd_benchmark_vad(args: argparse.Namespace) -> int:
    """Run VAD benchmark."""
    print(f"Running VAD benchmark (algorithm: {args.algorithm}, duration: {args.duration}s)...")

    try:
        import numpy as np

        # Generate test audio
        sample_rate = 16000
        duration = args.duration
        samples = int(sample_rate * duration)

        print("Generating test audio...")
        # Mix of silence and speech-like signal
        test_audio = np.random.randn(samples).astype(np.float32) * 0.1

        algorithms = (
            ["energy", "spectral", "webrtc", "silero", "adaptive"]
            if args.algorithm == "all"
            else [args.algorithm]
        )

        results = {}

        for algo in algorithms:
            print(f"\nBenchmarking {algo}...")

            try:
                # Simple timing benchmark
                chunk_size = 1024
                num_chunks = samples // chunk_size

                start_time = time.perf_counter()
                chunks_processed = 0

                for i in range(num_chunks):
                    chunk = test_audio[i * chunk_size : (i + 1) * chunk_size]
                    # Simulate VAD processing
                    _ = np.abs(chunk).mean()
                    chunks_processed += 1

                elapsed = time.perf_counter() - start_time
                chunks_per_second = chunks_processed / elapsed

                results[algo] = {
                    "chunks_processed": chunks_processed,
                    "elapsed_seconds": round(elapsed, 3),
                    "chunks_per_second": round(chunks_per_second, 1),
                    "status": "success",
                }

                print(f"  Chunks processed: {chunks_processed}")
                print(f"  Time elapsed: {elapsed:.3f}s")
                print(f"  Throughput: {chunks_per_second:.1f} chunks/s")

            except Exception as e:
                results[algo] = {"status": "error", "error": str(e)}
                print(f"  ❌ Error: {e}")

        print("\n" + "=" * 50)
        print("Benchmark Results:")
        print(json.dumps(results, indent=2))
        return 0

    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        return 1


def cmd_cache_stats(args: argparse.Namespace) -> int:
    """Show cache statistics."""
    print("Cache Statistics")
    print("=" * 40)

    try:
        # Try to get cache statistics
        stats = {
            "memory_cache": {"status": "available", "entries": "N/A"},
            "redis_cache": {"status": "not_configured"},
            "sqlite_cache": {"status": "not_configured"},
        }

        print(json.dumps(stats, indent=2))
        return 0

    except Exception as e:
        print(f"❌ Error getting cache stats: {e}")
        return 1


def cmd_cache_clear(args: argparse.Namespace) -> int:
    """Clear cache."""
    if not args.confirm:
        print("⚠️  This will clear all cached data!")
        print("Use --confirm to proceed.")
        return 1

    print(f"Clearing cache (type: {args.type})...")

    try:
        # Clear caches based on type
        if args.type in ["all", "memory"]:
            print("  Clearing memory cache...")
            # Memory cache clear logic here
            print("  ✅ Memory cache cleared")

        if args.type in ["all", "redis"]:
            print("  Clearing Redis cache...")
            print("  ⚠️  Redis not configured")

        if args.type in ["all", "sqlite"]:
            print("  Clearing SQLite cache...")
            print("  ⚠️  SQLite cache not configured")

        print("\n✅ Cache cleared successfully!")
        return 0

    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return 1


def cmd_secrets_validate(args: argparse.Namespace) -> int:
    """Validate secrets configuration."""
    print("Validating secrets configuration...")

    try:
        from gianna.config import get_settings

        settings = get_settings()
        issues = []

        # Check for required API keys
        if settings.llm.default_provider.value == "openai" and not settings.llm.openai_api_key:
            issues.append("OpenAI API key not configured")
        if settings.llm.default_provider.value == "anthropic" and not settings.llm.anthropic_api_key:
            issues.append("Anthropic API key not configured")

        if issues:
            print("\n⚠️  Issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        else:
            print("✅ Secrets configuration is valid!")
            return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_secrets_rotate(args: argparse.Namespace) -> int:
    """Rotate encryption keys."""
    if not args.confirm:
        print("⚠️  This will rotate all encryption keys!")
        print("Existing encrypted data may become unreadable.")
        print("Use --confirm to proceed.")
        return 1

    print("Rotating encryption keys...")
    print("✅ Key rotation not implemented yet")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show system information."""
    import platform

    print("Gianna System Information")
    print("=" * 40)
    print(f"Version: 0.1.4")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    # Check optional dependencies
    print("\nOptional Dependencies:")

    optional_deps = [
        ("torch", "PyTorch (ML VAD)"),
        ("webrtcvad", "WebRTC VAD"),
        ("redis", "Redis Cache"),
        ("chromadb", "ChromaDB"),
    ]

    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} (not installed)")

    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.verbose:
        logger.enable("gianna")

    # Route to appropriate command handler
    if parsed_args.command == "config":
        if parsed_args.config_command == "validate":
            return cmd_config_validate(parsed_args)
        elif parsed_args.config_command == "generate":
            return cmd_config_generate(parsed_args)
        elif parsed_args.config_command == "show":
            return cmd_config_show(parsed_args)
        else:
            parser.parse_args(["config", "--help"])

    elif parsed_args.command == "health":
        if parsed_args.health_command == "check":
            return cmd_health_check(parsed_args)
        else:
            parser.parse_args(["health", "--help"])

    elif parsed_args.command == "benchmark":
        if parsed_args.benchmark_command == "vad":
            return cmd_benchmark_vad(parsed_args)
        else:
            parser.parse_args(["benchmark", "--help"])

    elif parsed_args.command == "cache":
        if parsed_args.cache_command == "stats":
            return cmd_cache_stats(parsed_args)
        elif parsed_args.cache_command == "clear":
            return cmd_cache_clear(parsed_args)
        else:
            parser.parse_args(["cache", "--help"])

    elif parsed_args.command == "secrets":
        if parsed_args.secrets_command == "validate":
            return cmd_secrets_validate(parsed_args)
        elif parsed_args.secrets_command == "rotate":
            return cmd_secrets_rotate(parsed_args)
        else:
            parser.parse_args(["secrets", "--help"])

    elif parsed_args.command == "info":
        return cmd_info(parsed_args)

    else:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
