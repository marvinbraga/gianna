#!/usr/bin/env python3
"""
Gianna Test Runner - Comprehensive Testing Suite

This script provides a centralized test runner for all Gianna testing categories
with flexible filtering, reporting, and coverage options.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --unit             # Run only unit tests
    python tests/run_tests.py --integration      # Run only integration tests
    python tests/run_tests.py --performance      # Run only performance tests
    python tests/run_tests.py --phase fase1      # Run tests for specific phase
    python tests/run_tests.py --coverage         # Run with coverage reporting
    python tests/run_tests.py --slow             # Include slow tests
    python tests/run_tests.py --report html      # Generate HTML report

Categories:
- Unit Tests: Individual component testing
- Integration Tests: Multi-component workflows
- Performance Tests: Load, stress, and benchmark testing
- End-to-End Tests: Complete workflow validation

Phases:
- FASE 1: Core state management, LangGraph chains
- FASE 2: ReAct agents, tools, orchestration
- FASE 3: VAD, streaming pipeline, voice workflows
- FASE 4: Semantic memory, learning, optimization
- FASE 5: End-to-end integration, production readiness
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


class GiannaTestRunner:
    """Centralized test runner for Gianna testing framework."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"

        # Test categories and their paths
        self.test_categories = {
            "unit": "tests/unit/",
            "integration": "tests/integration/",
            "performance": "tests/performance/",
            "workflows": "tests/test_workflows.py",
            "end_to_end": "tests/test_end_to_end.py",
        }

        # Phase mappings
        self.phases = {
            "fase1": ["test_core.py", "test_models.py"],
            "fase2": [
                "test_react_agents.py",
                "test_tools_integration.py",
                "test_coordination.py",
            ],
            "fase3": ["test_audio.py", "test_voice_workflows.py"],
            "fase4": ["test_memory_integration.py", "test_learning_system.py"],
            "fase5": ["test_workflows.py", "test_end_to_end.py", "test_performance.py"],
        }

    def build_pytest_command(self, args) -> List[str]:
        """Build pytest command based on arguments."""
        cmd = ["python", "-m", "pytest"]

        # Add test paths based on categories
        test_paths = []

        if args.unit:
            test_paths.append(self.test_categories["unit"])
        if args.integration:
            test_paths.append(self.test_categories["integration"])
        if args.performance:
            test_paths.append(self.test_categories["performance"])
        if args.workflows:
            test_paths.append(self.test_categories["workflows"])
        if args.end_to_end:
            test_paths.append(self.test_categories["end_to_end"])

        # If no specific category, run all tests
        if not test_paths:
            test_paths = ["tests/"]

        cmd.extend(test_paths)

        # Add markers for filtering
        markers = []
        if args.phase:
            markers.append(args.phase)
        if not args.slow:
            markers.append("not slow")
        if args.mock_only:
            markers.append("mock_only")
        if args.external_api:
            markers.append("external_api")

        if markers:
            cmd.extend(["-m", " and ".join(markers)])

        # Verbosity
        if args.verbose >= 2:
            cmd.append("-vv")
        elif args.verbose >= 1:
            cmd.append("-v")

        # Coverage options
        if args.coverage:
            cmd.extend(
                [
                    "--cov=gianna",
                    f"--cov-report=html:htmlcov",
                    f"--cov-report=term-missing",
                    f"--cov-report=xml:coverage.xml",
                ]
            )

            if args.coverage_min:
                cmd.extend([f"--cov-fail-under={args.coverage_min}"])

        # Performance options
        if args.benchmark:
            cmd.append("--benchmark-only")

        # Output options
        if args.report == "junit":
            cmd.extend(["--junit-xml=test_results.xml"])
        elif args.report == "json":
            cmd.extend(["--json-report", "--json-report-file=test_results.json"])

        # Parallel execution
        if args.parallel and args.parallel > 1:
            cmd.extend(["-n", str(args.parallel)])

        # Additional pytest options
        if args.maxfail:
            cmd.extend(["--maxfail", str(args.maxfail)])

        if args.timeout:
            cmd.extend(["--timeout", str(args.timeout)])

        # Show durations for slowest tests
        if args.durations:
            cmd.extend(["--durations", str(args.durations)])

        return cmd

    def run_tests(self, args) -> int:
        """Execute the test suite."""
        print("=" * 80)
        print("GIANNA TESTING FRAMEWORK")
        print("=" * 80)

        # Display test configuration
        self.print_test_config(args)

        # Build and execute pytest command
        cmd = self.build_pytest_command(args)

        print(f"\nExecuting: {' '.join(cmd)}")
        print("-" * 80)

        # Set environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)
        env["TESTING"] = "true"

        # Execute tests
        start_time = time.perf_counter()

        try:
            result = subprocess.run(cmd, cwd=self.project_root, env=env, check=False)

            end_time = time.perf_counter()
            duration = end_time - start_time

            # Print summary
            self.print_test_summary(result.returncode, duration, args)

            return result.returncode

        except KeyboardInterrupt:
            print("\n\n❌ Tests interrupted by user")
            return 130
        except Exception as e:
            print(f"\n\n❌ Error running tests: {e}")
            return 1

    def print_test_config(self, args):
        """Print test configuration summary."""
        print(f"Project Root: {self.project_root}")
        print(f"Test Directory: {self.test_dir}")

        # Test categories
        categories = []
        if args.unit:
            categories.append("Unit")
        if args.integration:
            categories.append("Integration")
        if args.performance:
            categories.append("Performance")
        if args.workflows:
            categories.append("Workflows")
        if args.end_to_end:
            categories.append("End-to-End")

        if not categories:
            categories = ["All"]

        print(f"Categories: {', '.join(categories)}")

        if args.phase:
            print(f"Phase Filter: {args.phase}")

        print(f"Include Slow Tests: {args.slow}")
        print(f"Coverage Enabled: {args.coverage}")

        if args.coverage and args.coverage_min:
            print(f"Coverage Minimum: {args.coverage_min}%")

        if args.parallel:
            print(f"Parallel Workers: {args.parallel}")

    def print_test_summary(self, return_code: int, duration: float, args):
        """Print test execution summary."""
        print("\n" + "=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)

        # Status
        if return_code == 0:
            print("✅ STATUS: ALL TESTS PASSED")
        else:
            print("❌ STATUS: SOME TESTS FAILED")

        print(f"⏱️  DURATION: {duration:.2f} seconds")

        # Coverage information
        if args.coverage:
            print("\n📊 COVERAGE REPORTS:")
            print(f"  • HTML Report: {self.project_root}/htmlcov/index.html")
            print(f"  • XML Report: {self.project_root}/coverage.xml")
            print("  • Terminal Report: See above")

        # Additional reports
        if args.report == "junit":
            print(f"\n📋 JUNIT REPORT: {self.project_root}/test_results.xml")
        elif args.report == "json":
            print(f"\n📋 JSON REPORT: {self.project_root}/test_results.json")

        # Next steps
        print("\n🔧 NEXT STEPS:")
        if return_code == 0:
            print("  • All tests passed! Ready for deployment.")
            if args.coverage:
                print("  • Review coverage report for any gaps.")
        else:
            print("  • Check failed tests and fix issues.")
            print("  • Run specific test categories to isolate problems.")
            print("  • Use --verbose for more detailed output.")

        print("\n💡 USEFUL COMMANDS:")
        print("  • Run only failing tests: pytest --lf")
        print("  • Run with more detail: python tests/run_tests.py -vv")
        print("  • Run specific phase: python tests/run_tests.py --phase fase1")
        print("  • Performance tests: python tests/run_tests.py --performance")

    def list_available_tests(self):
        """List all available tests."""
        print("📋 AVAILABLE TESTS")
        print("=" * 50)

        for category, path in self.test_categories.items():
            full_path = self.project_root / path
            print(f"\n{category.upper()} TESTS:")

            if full_path.is_file():
                print(f"  • {path}")
            elif full_path.is_dir():
                test_files = list(full_path.glob("**/test_*.py"))
                for test_file in sorted(test_files):
                    rel_path = test_file.relative_to(self.project_root)
                    print(f"  • {rel_path}")

        print(f"\nPHASES:")
        for phase, files in self.phases.items():
            print(f"  • {phase}: {', '.join(files)}")

    def check_dependencies(self):
        """Check if all testing dependencies are available."""
        required_packages = [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "pytest-asyncio",
            "psutil",
            "numpy",
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)

        if missing:
            print("❌ MISSING DEPENDENCIES:")
            for package in missing:
                print(f"  • {package}")
            print("\nInstall missing dependencies with:")
            print(f"  pip install {' '.join(missing)}")
            return False

        print("✅ All testing dependencies available")
        return True


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for test runner."""
    parser = argparse.ArgumentParser(
        description="Gianna Testing Framework Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py                    # Run all tests
  python tests/run_tests.py --unit             # Unit tests only
  python tests/run_tests.py --integration      # Integration tests only
  python tests/run_tests.py --performance      # Performance tests only
  python tests/run_tests.py --coverage         # Run with coverage
  python tests/run_tests.py --phase fase1      # Run FASE 1 tests
  python tests/run_tests.py --slow             # Include slow tests
  python tests/run_tests.py -v                 # Verbose output
        """,
    )

    # Test categories
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    parser.add_argument("--workflows", action="store_true", help="Run workflow tests")
    parser.add_argument(
        "--end-to-end", action="store_true", help="Run end-to-end tests"
    )

    # Phase filtering
    parser.add_argument(
        "--phase",
        choices=["fase1", "fase2", "fase3", "fase4", "fase5"],
        help="Run tests for specific phase",
    )

    # Test filtering
    parser.add_argument(
        "--slow", action="store_true", help="Include slow tests (default: exclude)"
    )
    parser.add_argument(
        "--mock-only", action="store_true", help="Run only tests using mocks"
    )
    parser.add_argument(
        "--external-api",
        action="store_true",
        help="Include tests requiring external APIs",
    )

    # Coverage options
    parser.add_argument(
        "--coverage", action="store_true", help="Enable coverage reporting"
    )
    parser.add_argument(
        "--coverage-min",
        type=int,
        default=80,
        help="Minimum coverage percentage (default: 80)",
    )

    # Output options
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv)",
    )
    parser.add_argument(
        "--report",
        choices=["html", "junit", "json"],
        help="Generate additional reports",
    )

    # Execution options
    parser.add_argument(
        "--parallel", type=int, help="Run tests in parallel (number of workers)"
    )
    parser.add_argument(
        "--maxfail", type=int, default=10, help="Stop after N failures (default: 10)"
    )
    parser.add_argument("--timeout", type=int, help="Test timeout in seconds")
    parser.add_argument(
        "--durations", type=int, default=10, help="Show N slowest tests (default: 10)"
    )

    # Performance options
    parser.add_argument(
        "--benchmark", action="store_true", help="Run only benchmark tests"
    )

    # Utility options
    parser.add_argument("--list", action="store_true", help="List available tests")
    parser.add_argument(
        "--check-deps", action="store_true", help="Check testing dependencies"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    runner = GiannaTestRunner()

    # Handle utility commands
    if args.list:
        runner.list_available_tests()
        return 0

    if args.check_deps:
        return 0 if runner.check_dependencies() else 1

    # Run tests
    return runner.run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
