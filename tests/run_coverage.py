#!/usr/bin/env python3
"""
Coverage Analysis Runner for Gianna Testing Framework

Specialized script for comprehensive coverage analysis with detailed reporting,
coverage gaps identification, and improvement recommendations.

Features:
- Comprehensive coverage measurement
- Coverage gap analysis
- Component-wise coverage breakdown
- Coverage trend analysis
- HTML and XML report generation
- Coverage improvement recommendations

Usage:
    python tests/run_coverage.py                     # Full coverage analysis
    python tests/run_coverage.py --component core    # Component-specific coverage
    python tests/run_coverage.py --html              # Generate HTML report
    python tests/run_coverage.py --gaps              # Identify coverage gaps
    python tests/run_coverage.py --trend             # Coverage trend analysis
"""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


class CoverageAnalyzer:
    """Comprehensive coverage analysis for Gianna."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.coverage_dir = self.project_root / "htmlcov"
        self.coverage_data_file = self.project_root / ".coverage"
        self.coverage_xml = self.project_root / "coverage.xml"
        self.coverage_history = self.project_root / "coverage_history.json"

        # Component mappings
        self.components = {
            "core": ["gianna/core/"],
            "models": ["gianna/assistants/models/"],
            "audio": ["gianna/assistants/audio/", "gianna/audio/"],
            "agents": ["gianna/agents/"],
            "tools": ["gianna/tools/"],
            "memory": ["gianna/memory/"],
            "learning": ["gianna/learning/"],
            "coordination": ["gianna/coordination/"],
            "workflows": ["gianna/workflows/"],
            "optimization": ["gianna/optimization/"],
        }

        # Coverage thresholds
        self.thresholds = {"excellent": 95, "good": 85, "acceptable": 75, "poor": 60}

    def run_coverage_analysis(self, args) -> Dict[str, Any]:
        """Run comprehensive coverage analysis."""
        print("🔍 GIANNA COVERAGE ANALYSIS")
        print("=" * 60)

        # Run tests with coverage
        if not self._run_tests_with_coverage(args):
            print("❌ Failed to run tests with coverage")
            return {}

        # Analyze coverage data
        coverage_data = self._analyze_coverage_data(args)

        # Generate reports
        if args.html:
            self._generate_html_report()

        if args.gaps:
            gaps = self._identify_coverage_gaps(coverage_data)
            self._print_coverage_gaps(gaps)

        if args.trend:
            self._analyze_coverage_trend(coverage_data)

        # Print summary
        self._print_coverage_summary(coverage_data)

        # Save coverage history
        self._save_coverage_history(coverage_data)

        return coverage_data

    def _run_tests_with_coverage(self, args) -> bool:
        """Run tests with coverage collection."""
        cmd = [
            "python",
            "-m",
            "pytest",
            "--cov=gianna",
            "--cov-report=xml",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-branch",  # Include branch coverage
        ]

        # Add component-specific paths
        if args.component:
            if args.component in self.components:
                test_paths = []
                for comp_path in self.components[args.component]:
                    test_paths.extend(
                        [f"tests/unit/test_{args.component}.py", f"tests/integration/"]
                    )
                cmd.extend(test_paths)
            else:
                cmd.append(f"tests/unit/test_{args.component}.py")
        else:
            cmd.append("tests/")

        # Exclude slow tests unless specifically requested
        if not args.slow:
            cmd.extend(["-m", "not slow"])

        print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                print("⚠️  Some tests failed, but coverage data may still be valid")
                print(result.stderr)

            return self.coverage_xml.exists()

        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False

    def _analyze_coverage_data(self, args) -> Dict[str, Any]:
        """Analyze coverage data from XML report."""
        if not self.coverage_xml.exists():
            return {}

        try:
            tree = ET.parse(self.coverage_xml)
            root = tree.getroot()

            # Overall coverage
            overall = {
                "line_rate": float(root.get("line-rate", 0)) * 100,
                "branch_rate": float(root.get("branch-rate", 0)) * 100,
                "lines_covered": int(root.get("lines-covered", 0)),
                "lines_valid": int(root.get("lines-valid", 0)),
                "branches_covered": int(root.get("branches-covered", 0)),
                "branches_valid": int(root.get("branches-valid", 0)),
            }

            # Component-wise coverage
            components = {}
            packages = root.find("packages")

            if packages is not None:
                for package in packages.findall("package"):
                    package_name = package.get("name", "")

                    # Map to component
                    component = self._map_package_to_component(package_name)

                    if component not in components:
                        components[component] = {
                            "line_rate": 0,
                            "branch_rate": 0,
                            "lines_covered": 0,
                            "lines_valid": 0,
                            "branches_covered": 0,
                            "branches_valid": 0,
                            "files": [],
                        }

                    # Aggregate package data
                    pkg_data = components[component]
                    pkg_data["lines_covered"] += int(package.get("lines-covered", 0))
                    pkg_data["lines_valid"] += int(package.get("lines-valid", 0))
                    pkg_data["branches_covered"] += int(
                        package.get("branches-covered", 0)
                    )
                    pkg_data["branches_valid"] += int(package.get("branches-valid", 0))

                    # File-level data
                    classes = package.find("classes")
                    if classes is not None:
                        for cls in classes.findall("class"):
                            filename = cls.get("filename", "")
                            line_rate = float(cls.get("line-rate", 0)) * 100
                            branch_rate = float(cls.get("branch-rate", 0)) * 100

                            pkg_data["files"].append(
                                {
                                    "filename": filename,
                                    "line_rate": line_rate,
                                    "branch_rate": branch_rate,
                                }
                            )

            # Calculate component rates
            for component, data in components.items():
                if data["lines_valid"] > 0:
                    data["line_rate"] = (
                        data["lines_covered"] / data["lines_valid"]
                    ) * 100
                if data["branches_valid"] > 0:
                    data["branch_rate"] = (
                        data["branches_covered"] / data["branches_valid"]
                    ) * 100

            return {
                "timestamp": datetime.now().isoformat(),
                "overall": overall,
                "components": components,
                "thresholds": self.thresholds,
            }

        except Exception as e:
            print(f"❌ Error analyzing coverage data: {e}")
            return {}

    def _map_package_to_component(self, package_name: str) -> str:
        """Map package name to component."""
        for component, paths in self.components.items():
            for path in paths:
                if package_name.startswith(path.replace("/", ".")):
                    return component
        return "other"

    def _identify_coverage_gaps(
        self, coverage_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify coverage gaps and improvement opportunities."""
        gaps = []

        if "components" not in coverage_data:
            return gaps

        for component, data in coverage_data["components"].items():
            line_rate = data.get("line_rate", 0)
            branch_rate = data.get("branch_rate", 0)

            # Overall component gaps
            if line_rate < self.thresholds["acceptable"]:
                gaps.append(
                    {
                        "type": "component_lines",
                        "component": component,
                        "current": line_rate,
                        "threshold": self.thresholds["acceptable"],
                        "severity": self._get_coverage_severity(line_rate),
                        "recommendation": f"Add more unit tests for {component} module",
                    }
                )

            if branch_rate < self.thresholds["acceptable"]:
                gaps.append(
                    {
                        "type": "component_branches",
                        "component": component,
                        "current": branch_rate,
                        "threshold": self.thresholds["acceptable"],
                        "severity": self._get_coverage_severity(branch_rate),
                        "recommendation": f"Add tests for conditional branches in {component}",
                    }
                )

            # File-level gaps
            for file_data in data.get("files", []):
                filename = file_data["filename"]
                file_line_rate = file_data["line_rate"]

                if file_line_rate < self.thresholds["acceptable"]:
                    gaps.append(
                        {
                            "type": "file_lines",
                            "component": component,
                            "filename": filename,
                            "current": file_line_rate,
                            "threshold": self.thresholds["acceptable"],
                            "severity": self._get_coverage_severity(file_line_rate),
                            "recommendation": f"Add tests for {filename}",
                        }
                    )

        # Sort by severity and impact
        gaps.sort(
            key=lambda x: (
                ["critical", "high", "medium", "low"].index(x["severity"]),
                -x["current"],  # Lower coverage first
            )
        )

        return gaps

    def _get_coverage_severity(self, rate: float) -> str:
        """Determine coverage gap severity."""
        if rate < self.thresholds["poor"]:
            return "critical"
        elif rate < self.thresholds["acceptable"]:
            return "high"
        elif rate < self.thresholds["good"]:
            return "medium"
        else:
            return "low"

    def _analyze_coverage_trend(self, current_data: Dict[str, Any]):
        """Analyze coverage trends over time."""
        if not self.coverage_history.exists():
            print("📊 No historical coverage data available yet")
            return

        try:
            with open(self.coverage_history, "r") as f:
                history = json.load(f)

            if len(history) < 2:
                print("📊 Insufficient historical data for trend analysis")
                return

            # Compare with previous run
            previous = history[-1]
            current_overall = current_data.get("overall", {})
            previous_overall = previous.get("overall", {})

            line_trend = current_overall.get("line_rate", 0) - previous_overall.get(
                "line_rate", 0
            )
            branch_trend = current_overall.get("branch_rate", 0) - previous_overall.get(
                "branch_rate", 0
            )

            print("\n📊 COVERAGE TREND ANALYSIS")
            print("-" * 30)

            print(
                f"Line Coverage: {current_overall.get('line_rate', 0):.1f}% "
                + f"({'+' if line_trend >= 0 else ''}{line_trend:.1f}%)"
            )
            print(
                f"Branch Coverage: {current_overall.get('branch_rate', 0):.1f}% "
                + f"({'+' if branch_trend >= 0 else ''}{branch_trend:.1f}%)"
            )

            # Component trends
            print("\nComponent Trends:")
            current_components = current_data.get("components", {})
            previous_components = previous.get("components", {})

            for component in current_components:
                if component in previous_components:
                    curr_rate = current_components[component].get("line_rate", 0)
                    prev_rate = previous_components[component].get("line_rate", 0)
                    trend = curr_rate - prev_rate

                    trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                    print(
                        f"  {trend_icon} {component}: {curr_rate:.1f}% "
                        + f"({'+' if trend >= 0 else ''}{trend:.1f}%)"
                    )

        except Exception as e:
            print(f"❌ Error analyzing coverage trend: {e}")

    def _print_coverage_gaps(self, gaps: List[Dict[str, Any]]):
        """Print coverage gaps analysis."""
        if not gaps:
            print("\n✅ No significant coverage gaps identified!")
            return

        print(f"\n🔍 COVERAGE GAPS ANALYSIS ({len(gaps)} issues found)")
        print("=" * 60)

        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for gap in gaps[:20]:  # Show top 20 gaps
            severity = gap["severity"]
            severity_counts[severity] += 1

            severity_icon = {
                "critical": "🚨",
                "high": "⚠️ ",
                "medium": "💛",
                "low": "ℹ️ ",
            }[severity]

            print(
                f"\n{severity_icon} {severity.upper()}: {gap['type'].replace('_', ' ').title()}"
            )

            if "filename" in gap:
                print(f"   File: {gap['filename']}")
            else:
                print(f"   Component: {gap['component']}")

            print(f"   Current: {gap['current']:.1f}% | Target: {gap['threshold']}%")
            print(f"   💡 {gap['recommendation']}")

        if len(gaps) > 20:
            print(f"\n... and {len(gaps) - 20} more issues")

        print(f"\n📊 SEVERITY BREAKDOWN:")
        for severity, count in severity_counts.items():
            if count > 0:
                print(f"   {severity.title()}: {count} issues")

    def _print_coverage_summary(self, coverage_data: Dict[str, Any]):
        """Print comprehensive coverage summary."""
        if not coverage_data:
            return

        overall = coverage_data.get("overall", {})
        components = coverage_data.get("components", {})

        print("\n📊 COVERAGE SUMMARY")
        print("=" * 50)

        # Overall metrics
        line_rate = overall.get("line_rate", 0)
        branch_rate = overall.get("branch_rate", 0)

        line_status = self._get_coverage_status(line_rate)
        branch_status = self._get_coverage_status(branch_rate)

        print(f"📈 Overall Line Coverage: {line_rate:.1f}% {line_status}")
        print(f"🌿 Overall Branch Coverage: {branch_rate:.1f}% {branch_status}")
        print(f"📏 Total Lines: {overall.get('lines_valid', 0):,}")
        print(f"✅ Lines Covered: {overall.get('lines_covered', 0):,}")

        # Component breakdown
        print(f"\n🧩 COMPONENT BREAKDOWN:")
        print("-" * 40)

        for component, data in sorted(components.items()):
            comp_line_rate = data.get("line_rate", 0)
            comp_status = self._get_coverage_status(comp_line_rate)
            file_count = len(data.get("files", []))

            print(
                f"{component:12}: {comp_line_rate:5.1f}% {comp_status} ({file_count} files)"
            )

        # Coverage grade
        overall_grade = self._calculate_coverage_grade(line_rate, branch_rate)
        print(f"\n🏆 OVERALL GRADE: {overall_grade}")

        # Recommendations
        self._print_coverage_recommendations(coverage_data)

    def _get_coverage_status(self, rate: float) -> str:
        """Get coverage status emoji."""
        if rate >= self.thresholds["excellent"]:
            return "🟢"
        elif rate >= self.thresholds["good"]:
            return "🟡"
        elif rate >= self.thresholds["acceptable"]:
            return "🟠"
        else:
            return "🔴"

    def _calculate_coverage_grade(self, line_rate: float, branch_rate: float) -> str:
        """Calculate overall coverage grade."""
        combined = (line_rate + branch_rate) / 2

        if combined >= self.thresholds["excellent"]:
            return "A+ (Excellent)"
        elif combined >= self.thresholds["good"]:
            return "A (Good)"
        elif combined >= self.thresholds["acceptable"]:
            return "B (Acceptable)"
        elif combined >= self.thresholds["poor"]:
            return "C (Needs Improvement)"
        else:
            return "D (Poor)"

    def _print_coverage_recommendations(self, coverage_data: Dict[str, Any]):
        """Print coverage improvement recommendations."""
        overall = coverage_data.get("overall", {})
        line_rate = overall.get("line_rate", 0)
        branch_rate = overall.get("branch_rate", 0)

        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 25)

        if line_rate < self.thresholds["good"]:
            print("• Add more unit tests to increase line coverage")
            print("• Focus on untested functions and methods")
            print("• Use coverage report to identify specific gaps")

        if branch_rate < self.thresholds["good"]:
            print("• Add tests for conditional branches (if/else)")
            print("• Test exception handling paths")
            print("• Add tests for different input scenarios")

        # Component-specific recommendations
        components = coverage_data.get("components", {})
        low_coverage_components = [
            comp
            for comp, data in components.items()
            if data.get("line_rate", 0) < self.thresholds["acceptable"]
        ]

        if low_coverage_components:
            print(
                f"• Priority components for improvement: {', '.join(low_coverage_components)}"
            )

        print("• Consider adding integration tests for workflow coverage")
        print("• Review and add tests for error handling scenarios")

    def _generate_html_report(self):
        """Generate HTML coverage report."""
        if self.coverage_dir.exists():
            report_path = self.coverage_dir / "index.html"
            print(f"\n📊 HTML Coverage Report: file://{report_path.absolute()}")
        else:
            print("⚠️  HTML coverage report not generated")

    def _save_coverage_history(self, coverage_data: Dict[str, Any]):
        """Save coverage data to history file."""
        try:
            history = []
            if self.coverage_history.exists():
                with open(self.coverage_history, "r") as f:
                    history = json.load(f)

            history.append(coverage_data)

            # Keep only last 50 entries
            if len(history) > 50:
                history = history[-50:]

            with open(self.coverage_history, "w") as f:
                json.dump(history, f, indent=2)

            print(f"💾 Coverage data saved to history ({len(history)} entries)")

        except Exception as e:
            print(f"❌ Error saving coverage history: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for coverage runner."""
    parser = argparse.ArgumentParser(
        description="Gianna Coverage Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--component",
        choices=[
            "core",
            "models",
            "audio",
            "agents",
            "tools",
            "memory",
            "learning",
            "coordination",
            "workflows",
            "optimization",
        ],
        help="Analyze specific component coverage",
    )

    parser.add_argument(
        "--html", action="store_true", help="Generate HTML coverage report"
    )

    parser.add_argument(
        "--gaps", action="store_true", help="Identify and analyze coverage gaps"
    )

    parser.add_argument(
        "--trend", action="store_true", help="Analyze coverage trends over time"
    )

    parser.add_argument(
        "--slow", action="store_true", help="Include slow tests in coverage"
    )

    parser.add_argument(
        "--threshold", type=int, default=80, help="Coverage threshold percentage"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    analyzer = CoverageAnalyzer()
    coverage_data = analyzer.run_coverage_analysis(args)

    # Return appropriate exit code
    if coverage_data:
        overall_coverage = coverage_data.get("overall", {}).get("line_rate", 0)
        return 0 if overall_coverage >= args.threshold else 1
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
