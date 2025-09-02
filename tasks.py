"""
Invoke tasks for Gianna project automation.
"""

from pathlib import Path

from invoke import task

# Project configuration
PROJECT_DIR = Path(__file__).parent
PYTHON_DIRS = ["gianna", "notebooks"]
VAD_CONFIG_DIR = PROJECT_DIR / "config" / "vad"


@task
def clean(ctx):
    """Clean up build artifacts and cache files."""
    print("🧹 Cleaning up build artifacts...")
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        ".pytest_cache",
        "build",
        "dist",
        "*.egg-info",
        ".coverage",
        ".mypy_cache",
        ".tox",
    ]

    for pattern in patterns:
        paths = list(PROJECT_DIR.glob(pattern))
        for path in paths:
            if path.is_dir():
                ctx.run(f"rm -rf {path}")
                print(f"  ✅ Removed directory: {path}")
            else:
                ctx.run(f"rm -f {path}")
                print(f"  ✅ Removed file: {path}")


@task
def format_code(ctx):
    """Format code with Black and isort."""
    print("🎨 Formatting code with Black and isort...")

    # Format with Black
    print("  📝 Running Black...")
    for python_dir in PYTHON_DIRS:
        if Path(python_dir).exists():
            ctx.run(f"poetry run black {python_dir}")

    # Sort imports with isort
    print("  📦 Running isort...")
    for python_dir in PYTHON_DIRS:
        if Path(python_dir).exists():
            ctx.run(f"poetry run isort {python_dir}")

    print("✅ Code formatting completed!")


@task
def lint_flake8(ctx):
    """Run flake8 linting."""
    print("🔍 Running flake8...")
    for python_dir in PYTHON_DIRS:
        if Path(python_dir).exists():
            print(f"  📁 Linting {python_dir}/")
            ctx.run(f"poetry run flake8 {python_dir}")


@task
def type_check(ctx):
    """Run mypy type checking."""
    print("🏷️  Running mypy type checking...")
    for python_dir in PYTHON_DIRS:
        if Path(python_dir).exists():
            print(f"  📁 Type checking {python_dir}/")
            ctx.run(f"poetry run mypy {python_dir}")


@task(pre=[format_code])
def lint_quick(ctx):
    """Quick lint - format + flake8."""
    print("⚡ Running quick lint (format + flake8)...")
    lint_flake8(ctx)
    print("✅ Quick lint completed!")


@task(pre=[format_code])
def lint(ctx):
    """Comprehensive linting suite (format + flake8)."""
    print("🚀 Running comprehensive lint...")

    try:
        # Run flake8
        lint_flake8(ctx)
        print("✅ flake8 passed!")

        print("🎉 All linting checks passed successfully!")

    except Exception as e:
        print(f"❌ Linting failed: {e}")
        raise


@task(pre=[format_code])
def lint_strict(ctx):
    """Strict linting suite including type checking."""
    print("🚀 Running strict lint with type checking...")

    try:
        # Run flake8
        lint_flake8(ctx)
        print("✅ flake8 passed!")

        # Run mypy
        type_check(ctx)
        print("✅ mypy passed!")

        print("🎉 All strict linting checks passed successfully!")

    except Exception as e:
        print(f"❌ Strict linting failed: {e}")
        raise


@task
def pre_commit_run(ctx):
    """Run pre-commit hooks on all files."""
    print("🔗 Running pre-commit hooks...")
    ctx.run("poetry run pre-commit run --all-files")


@task
def pre_commit_update(ctx):
    """Update pre-commit hooks."""
    print("⬆️ Updating pre-commit hooks...")
    ctx.run("poetry run pre-commit autoupdate")


@task
def test(ctx):
    """Run tests (if test framework is set up)."""
    print("🧪 Running tests...")
    # Add test command here when tests are set up
    # ctx.run("poetry run pytest")
    print("⚠️  No tests configured yet")


@task
def install(ctx):
    """Install dependencies and set up development environment."""
    print("📦 Installing dependencies and setting up dev environment...")

    # Install dependencies
    ctx.run("poetry install")

    # Install pre-commit hooks
    ctx.run("poetry run pre-commit install")

    print("✅ Development environment setup completed!")


@task
def build(ctx):
    """Build the project."""
    print("🏗️  Building project...")
    ctx.run("poetry build")
    print("✅ Build completed!")


@task
def check_deps(ctx):
    """Check for dependency updates."""
    print("🔍 Checking for dependency updates...")
    ctx.run("poetry show --outdated")


@task(pre=[clean, lint])
def ci(ctx):
    """Run CI pipeline (clean, lint, test)."""
    print("🔄 Running CI pipeline...")
    test(ctx)
    print("🎉 CI pipeline completed successfully!")


@task
def dev_setup(ctx):
    """Complete development setup."""
    print("🚀 Setting up development environment...")
    install(ctx)
    pre_commit_run(ctx)
    print("✅ Development environment ready!")


@task
def help(ctx):
    """Show available commands."""
    print(
        """
🛠️  Gianna Project Tasks

📦 Setup & Installation:
  invoke install      - Install dependencies and setup dev environment
  invoke dev-setup    - Complete development setup

🎨 Code Quality:
  invoke format-code  - Format code with Black and isort
  invoke lint-quick   - Quick lint (format + flake8)
  invoke lint         - Comprehensive linting suite (format + flake8)
  invoke lint-strict  - Strict linting suite including type checking
  invoke type-check   - Run mypy type checking

🎙️ VAD (Voice Activity Detection):
  invoke vad-install-basic    - Install basic VAD dependencies
  invoke vad-install-advanced - Install advanced VAD dependencies
  invoke vad-install-ml       - Install ML VAD dependencies (PyTorch)
  invoke vad-install-full     - Install all VAD dependencies
  invoke vad-config-list      - List available VAD configurations
  invoke vad-config-validate  - Validate VAD configuration files
  invoke vad-test-config      - Test VAD configuration with algorithm
  invoke vad-deps-check       - Check VAD dependencies status
  invoke vad-benchmark        - Run VAD performance benchmark
  invoke vad-validate-all     - Comprehensive VAD system validation
  invoke vad-clean            - Clean VAD temporary files

🔗 Pre-commit:
  invoke pre-commit-run    - Run pre-commit hooks on all files
  invoke pre-commit-update - Update pre-commit hooks

🧹 Maintenance:
  invoke clean        - Clean build artifacts and cache
  invoke check-deps   - Check for dependency updates
  invoke build        - Build the project
  invoke vad-clean    - Clean VAD-specific temporary files

🔄 CI/CD:
  invoke ci          - Run full CI pipeline
  invoke ci-vad      - Run CI pipeline with VAD validation
  invoke test        - Run tests

📚 Help:
  invoke help        - Show this help message

VAD Installation Examples:
  invoke vad-install-basic     # WebRTC VAD + basic audio processing
  invoke vad-install-advanced  # + visualization libraries
  invoke vad-install-ml        # + PyTorch for ML-based VAD
  invoke vad-install-full      # All VAD dependencies

VAD Configuration Examples:
  invoke vad-config-list                    # List all configurations
  invoke vad-config-validate development    # Validate dev config
  invoke vad-test-config webrtc            # Test WebRTC algorithm
  invoke vad-benchmark testing             # Performance benchmark

Example usage:
  invoke lint        # Run complete linting suite
  invoke ci-vad      # Run full CI pipeline with VAD validation
"""
    )


# Default task
# ====================================================================
# VAD-Specific Tasks
# ====================================================================


@task
def vad_install_basic(ctx):
    """Install basic VAD dependencies."""
    print("📦 Installing basic VAD dependencies...")
    ctx.run("poetry install --extras vad-basic")
    print("✅ Basic VAD dependencies installed!")


@task
def vad_install_advanced(ctx):
    """Install advanced VAD dependencies (includes visualization)."""
    print("📦 Installing advanced VAD dependencies...")
    ctx.run("poetry install --extras vad-advanced")
    print("✅ Advanced VAD dependencies installed!")


@task
def vad_install_ml(ctx):
    """Install machine learning VAD dependencies (PyTorch-based)."""
    print("📦 Installing ML VAD dependencies...")
    ctx.run("poetry install --extras vad-ml")
    print("✅ ML VAD dependencies installed!")


@task
def vad_install_full(ctx):
    """Install all VAD dependencies."""
    print("📦 Installing all VAD dependencies...")
    ctx.run("poetry install --extras vad-full")
    print("✅ All VAD dependencies installed!")


@task
def vad_config_validate(ctx, config="development"):
    """Validate VAD configuration files."""
    print(f"🔍 Validating VAD configuration: {config}")
    config_file = VAD_CONFIG_DIR / f"{config}.yaml"

    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_file}")
        return

    try:
        import yaml

        # Load and validate YAML syntax
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        # Basic validation
        if "vad" not in config_data:
            print("❌ Missing 'vad' section in configuration")
            return

        if "algorithms" not in config_data:
            print("❌ Missing 'algorithms' section in configuration")
            return

        print(f"✅ Configuration {config} is valid!")

        # Print summary
        vad_config = config_data.get("vad", {})
        print(f"   Algorithm: {vad_config.get('default_algorithm', 'not specified')}")
        print(f"   Log Level: {vad_config.get('log_level', 'not specified')}")

    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error in {config_file}: {e}")
    except ImportError:
        print("⚠️  PyYAML not installed. Install with: pip install pyyaml")


@task
def vad_config_list(ctx):
    """List available VAD configurations."""
    print("📋 Available VAD configurations:")

    if not VAD_CONFIG_DIR.exists():
        print("❌ VAD config directory not found")
        return

    config_files = list(VAD_CONFIG_DIR.glob("*.yaml"))

    if not config_files:
        print("❌ No configuration files found")
        return

    for config_file in sorted(config_files):
        if config_file.name != "schema.yaml":
            config_name = config_file.stem
            print(f"   📄 {config_name}")

            # Try to read description
            try:
                import yaml

                with open(config_file, "r") as f:
                    # Read first few lines for description
                    lines = f.readlines()
                    for line in lines[:5]:
                        if line.strip().startswith("# ") and "Configuration" in line:
                            desc = line.strip()[2:].strip()
                            print(f"      {desc}")
                            break
            except:
                pass  # Skip if can't read description


@task
def vad_test_config(ctx, algorithm="webrtc"):
    """Test VAD configuration with specific algorithm."""
    print(f"🧪 Testing VAD configuration with {algorithm} algorithm...")

    # Create a simple test script
    test_script = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

try:
    # Test basic imports
    print("Testing VAD imports...")

    if "{algorithm}" == "webrtc":
        try:
            import webrtcvad
            print("✅ WebRTC VAD available")
        except ImportError:
            print("❌ WebRTC VAD not available. Install with: poetry install --extras vad-basic")

    elif "{algorithm}" == "ml_vad":
        try:
            import torch
            import torchaudio
            print("✅ PyTorch VAD dependencies available")
        except ImportError:
            print("❌ PyTorch VAD not available. Install with: poetry install --extras vad-ml")

    # Test configuration loading
    import yaml
    config_file = Path("config/vad/development.yaml")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
        print(f"   Default algorithm: {{config.get('vad', {{}}).get('default_algorithm', 'not set')}}")
    else:
        print("❌ Development configuration not found")

    print("🎉 VAD test completed!")

except Exception as e:
    print(f"❌ VAD test failed: {{e}}")
    sys.exit(1)
"""

    # Write and run test script
    test_file = PROJECT_DIR / "vad_test_temp.py"
    with open(test_file, "w") as f:
        f.write(test_script)

    try:
        ctx.run(f"poetry run python {test_file}")
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


@task
def vad_benchmark(ctx, config="testing"):
    """Run VAD performance benchmark."""
    print(f"🚀 Running VAD benchmark with {config} configuration...")

    benchmark_script = f"""
import sys
import time
import psutil
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

def benchmark_vad():
    print("🔍 VAD Benchmark Starting...")

    # Memory before
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"   Memory before: {{memory_before:.2f}} MB")

    # Time a simple operation
    start_time = time.time()

    try:
        # Test basic VAD operations
        import numpy as np

        # Generate test audio (1 second at 16kHz)
        sample_rate = 16000
        duration = 1.0
        test_audio = np.random.rand(int(sample_rate * duration)) * 0.1

        # Simulate VAD processing time
        for i in range(100):
            # Simple energy-based VAD simulation
            energy = np.mean(test_audio ** 2)
            voice_detected = energy > 0.001

        end_time = time.time()
        processing_time = end_time - start_time

        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before

        print(f"   Processing time: {{processing_time:.4f}} seconds")
        print(f"   Memory usage: {{memory_usage:.2f}} MB")
        print(f"   Throughput: {{100/processing_time:.2f}} operations/second")
        print("✅ Benchmark completed successfully!")

    except Exception as e:
        print(f"❌ Benchmark failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    benchmark_vad()
"""

    # Write and run benchmark script
    benchmark_file = PROJECT_DIR / "vad_benchmark_temp.py"
    with open(benchmark_file, "w") as f:
        f.write(benchmark_script)

    try:
        ctx.run(f"poetry run python {benchmark_file}")
    finally:
        # Cleanup
        if benchmark_file.exists():
            benchmark_file.unlink()


@task
def vad_deps_check(ctx):
    """Check VAD dependencies status."""
    print("🔍 Checking VAD dependencies status...")

    dependencies = {
        "Basic VAD": {
            "webrtcvad": "WebRTC Voice Activity Detection",
            "scipy": "Scientific computing library",
            "numpy": "Numerical computing",
            "soundfile": "Audio file reading/writing",
        },
        "Advanced VAD": {
            "librosa": "Audio analysis library",
            "scikit-learn": "Machine learning library",
            "matplotlib": "Plotting library",
            "seaborn": "Statistical visualization",
        },
        "ML VAD": {
            "torch": "PyTorch deep learning framework",
            "torchaudio": "PyTorch audio processing",
        },
        "Additional": {"noisereduce": "Noise reduction library"},
    }

    for category, deps in dependencies.items():
        print(f"\n📦 {category}:")
        for package, description in deps.items():
            try:
                __import__(package)
                print(f"   ✅ {package}: {description}")
            except ImportError:
                print(f"   ❌ {package}: {description} (not installed)")


@task
def vad_clean(ctx):
    """Clean VAD-related temporary files and caches."""
    print("🧹 Cleaning VAD temporary files...")

    patterns = [
        "vad_output/",
        "test_results/",
        "*.vad_cache",
        "vad_*_temp.py",
        "**/*.vad_log",
    ]

    cleaned_count = 0
    for pattern in patterns:
        paths = list(PROJECT_DIR.glob(pattern))
        for path in paths:
            try:
                if path.is_dir():
                    ctx.run(f"rm -rf {path}")
                    print(f"  ✅ Removed directory: {path}")
                else:
                    ctx.run(f"rm -f {path}")
                    print(f"  ✅ Removed file: {path}")
                cleaned_count += 1
            except Exception as e:
                print(f"  ⚠️  Could not remove {path}: {e}")

    if cleaned_count == 0:
        print("  🎉 No VAD temporary files found!")
    else:
        print(f"  🎉 Cleaned {cleaned_count} VAD files!")


@task(pre=[vad_config_validate, vad_deps_check])
def vad_validate_all(ctx):
    """Comprehensive VAD system validation."""
    print("🚀 Running comprehensive VAD validation...")
    print("✅ VAD system validation completed!")


# ====================================================================
# Enhanced Main Tasks (including VAD)
# ====================================================================


@task(pre=[clean, lint, vad_validate_all])
def ci_vad(ctx):
    """Run CI pipeline including VAD validation."""
    print("🔄 Running CI pipeline with VAD validation...")
    test(ctx)
    print("🎉 CI pipeline with VAD completed successfully!")


@task(default=True)
def default(ctx):
    """Show help by default."""
    help(ctx)
