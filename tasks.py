"""
Invoke tasks for Gianna project automation.
"""

from pathlib import Path

from invoke import task

# Project configuration
PROJECT_DIR = Path(__file__).parent
PYTHON_DIRS = ["gianna", "notebooks"]


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

🔗 Pre-commit:
  invoke pre-commit-run    - Run pre-commit hooks on all files
  invoke pre-commit-update - Update pre-commit hooks

🧹 Maintenance:
  invoke clean        - Clean build artifacts and cache
  invoke check-deps   - Check for dependency updates
  invoke build        - Build the project

🔄 CI/CD:
  invoke ci          - Run full CI pipeline
  invoke test        - Run tests

📚 Help:
  invoke help        - Show this help message

Example usage:
  invoke lint        # Run complete linting suite
  invoke ci          # Run full CI pipeline
"""
    )


# Default task
@task(default=True)
def default(ctx):
    """Show help by default."""
    help(ctx)
