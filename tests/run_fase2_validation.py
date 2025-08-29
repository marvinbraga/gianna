#!/usr/bin/env python3
"""
FASE 2 Complete Validation Suite Runner

This script runs all FASE 2 tests and generates comprehensive validation reports
for the complete implementation including tools, agents, coordination, and
performance benchmarks.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class FASE2ValidationRunner:
    """Comprehensive validation runner for FASE 2 implementation."""

    def __init__(self):
        self.test_modules = [
            {
                "name": "Tool Integration Layer",
                "module": "test_tools_integration.py",
                "description": "Tests all tools for functionality, safety, and integration",
                "weight": 25,  # Percentage weight in overall score
            },
            {
                "name": "ReAct Agents System",
                "module": "test_react_agents.py",
                "description": "Tests agent creation, execution, and lifecycle management",
                "weight": 25,
            },
            {
                "name": "Multi-Agent Coordination",
                "module": "test_coordination.py",
                "description": "Tests routing, orchestration, and agent coordination",
                "weight": 20,
            },
            {
                "name": "End-to-End Integration",
                "module": "test_end_to_end.py",
                "description": "Tests complete system workflows and integration",
                "weight": 20,
            },
            {
                "name": "Performance & Benchmarks",
                "module": "test_performance.py",
                "description": "Performance validation and FASE 2 criteria verification",
                "weight": 10,
            },
        ]

        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_test_module(self, module_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test module and collect results."""
        module_name = module_info["name"]
        module_file = module_info["module"]

        print(f"\n🗨  Running {module_name}...")
        print(f"   Description: {module_info['description']}")
        print(f"   Module: {module_file}")

        # Construct pytest command
        test_file_path = os.path.join(os.path.dirname(__file__), module_file)

        if not os.path.exists(test_file_path):
            print(f"   ⚠️  Warning: Test file not found: {test_file_path}")
            return {
                "status": "skipped",
                "reason": "Test file not found",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "execution_time": 0,
                "coverage": 0,
            }

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_file_path,
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file=/tmp/pytest_report_{module_file}.json",
            "--durations=10",
        ]

        start_time = time.time()

        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per module
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Parse pytest output
            stdout_lines = result.stdout.split("\n")
            stderr_lines = result.stderr.split("\n")

            # Extract test statistics from pytest output
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            tests_skipped = 0

            for line in stdout_lines:
                if "passed" in line and "failed" in line:
                    # Line like: "5 failed, 10 passed in 2.3s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "failed" and i > 0:
                            tests_failed = int(parts[i - 1])
                        elif part == "passed" and i > 0:
                            tests_passed = int(parts[i - 1])
                        elif part == "skipped" and i > 0:
                            tests_skipped = int(parts[i - 1])
                elif "passed in" in line and "failed" not in line:
                    # Line like: "15 passed in 3.2s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            tests_passed = int(parts[i - 1])

            tests_run = tests_passed + tests_failed + tests_skipped

            # Try to load JSON report for more detailed information
            json_report_file = f"/tmp/pytest_report_{module_file}.json"
            detailed_results = None

            if os.path.exists(json_report_file):
                try:
                    with open(json_report_file, "r") as f:
                        detailed_results = json.load(f)
                except:
                    pass

            # Determine overall status
            if result.returncode == 0:
                status = "passed"
            elif tests_run == 0:
                status = "no_tests"
            else:
                status = "failed"

            module_result = {
                "status": status,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tests_skipped": tests_skipped,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout_sample": "\n".join(stdout_lines[-10:]) if stdout_lines else "",
                "stderr_sample": "\n".join(stderr_lines[-5:]) if stderr_lines else "",
                "detailed_results": detailed_results,
            }

            # Print summary
            if status == "passed":
                print(
                    f"   ✅ {module_name}: {tests_passed} passed, {tests_failed} failed ({execution_time:.1f}s)"
                )
            elif status == "failed":
                print(
                    f"   ❌ {module_name}: {tests_passed} passed, {tests_failed} failed ({execution_time:.1f}s)"
                )
            else:
                print(f"   ⚠️  {module_name}: {status} ({execution_time:.1f}s)")

            return module_result

        except subprocess.TimeoutExpired:
            print(f"   ⏰ {module_name}: Timeout after 5 minutes")
            return {
                "status": "timeout",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "execution_time": 300,
                "error": "Test execution timeout",
            }

        except Exception as e:
            print(f"   ❌ {module_name}: Execution error: {e}")
            return {
                "status": "error",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "execution_time": 0,
                "error": str(e),
            }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test modules and collect comprehensive results."""
        print("🚀 Starting FASE 2 Complete Validation Suite")
        print(f"   Timestamp: {datetime.now().isoformat()}")
        print(f"   Test Modules: {len(self.test_modules)}")
        print("=" * 70)

        self.start_time = time.time()

        # Run each test module
        for module_info in self.test_modules:
            module_result = self.run_test_module(module_info)
            self.results[module_info["name"]] = {
                **module_result,
                "weight": module_info["weight"],
                "description": module_info["description"],
            }

        self.end_time = time.time()
        total_time = self.end_time - self.start_time

        # Calculate overall statistics
        overall_stats = self.calculate_overall_statistics()

        # Generate summary report
        summary_report = self.generate_summary_report(overall_stats, total_time)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "module_results": self.results,
            "overall_statistics": overall_stats,
            "summary_report": summary_report,
            "fase2_validation": self.validate_fase2_criteria(),
        }

    def calculate_overall_statistics(self) -> Dict[str, Any]:
        """Calculate overall test statistics across all modules."""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        modules_passed = 0
        modules_failed = 0
        modules_error = 0
        weighted_success_score = 0

        for module_name, result in self.results.items():
            total_tests += result.get("tests_run", 0)
            total_passed += result.get("tests_passed", 0)
            total_failed += result.get("tests_failed", 0)
            total_skipped += result.get("tests_skipped", 0)

            module_weight = (
                result.get("weight", 0) / 100
            )  # Convert percentage to decimal

            if result["status"] == "passed":
                modules_passed += 1
                weighted_success_score += module_weight
            elif result["status"] in ["failed", "timeout"]:
                modules_failed += 1
                # Partial credit based on passed tests
                if result.get("tests_run", 0) > 0:
                    partial_score = result.get("tests_passed", 0) / result.get(
                        "tests_run", 1
                    )
                    weighted_success_score += module_weight * partial_score
            else:
                modules_error += 1

        success_rate = total_passed / max(total_tests, 1)

        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_skipped": total_skipped,
            "success_rate": success_rate,
            "modules_passed": modules_passed,
            "modules_failed": modules_failed,
            "modules_error": modules_error,
            "weighted_success_score": weighted_success_score,
        }

    def validate_fase2_criteria(self) -> Dict[str, Any]:
        """Validate against specific FASE 2 success criteria."""
        criteria_validation = {
            "react_agents_responding": False,
            "tools_integrated_functioning": False,
            "multi_agent_coordination_operational": False,
            "overall_system_functional": False,
        }

        # ✅ Agentes ReAct respondendo a comandos
        react_agents_result = self.results.get("ReAct Agents System", {})
        if (
            react_agents_result.get("status") == "passed"
            and react_agents_result.get("tests_passed", 0) > 0
        ):
            criteria_validation["react_agents_responding"] = True

        # ✅ Ferramentas integradas funcionando
        tools_result = self.results.get("Tool Integration Layer", {})
        if (
            tools_result.get("status") == "passed"
            and tools_result.get("tests_passed", 0) > 0
        ):
            criteria_validation["tools_integrated_functioning"] = True

        # ✅ Coordenação multi-agente operacional
        coordination_result = self.results.get("Multi-Agent Coordination", {})
        e2e_result = self.results.get("End-to-End Integration", {})
        if (
            coordination_result.get("status") == "passed"
            or e2e_result.get("status") == "passed"
        ) and (
            coordination_result.get("tests_passed", 0)
            + e2e_result.get("tests_passed", 0)
        ) > 0:
            criteria_validation["multi_agent_coordination_operational"] = True

        # Overall system functionality
        overall_stats = self.calculate_overall_statistics()
        if (
            overall_stats["success_rate"] >= 0.7
            and overall_stats["modules_passed"] >= len(self.test_modules) * 0.6
        ):
            criteria_validation["overall_system_functional"] = True

        # Calculate criteria success rate
        criteria_passed = sum(criteria_validation.values())
        criteria_total = len(criteria_validation)
        criteria_success_rate = criteria_passed / criteria_total

        return {
            "criteria": criteria_validation,
            "criteria_passed": criteria_passed,
            "criteria_total": criteria_total,
            "criteria_success_rate": criteria_success_rate,
            "fase2_validated": criteria_success_rate
            >= 0.75,  # Require 75% criteria success
        }

    def generate_summary_report(
        self, overall_stats: Dict[str, Any], total_time: float
    ) -> str:
        """Generate a comprehensive summary report."""
        report_lines = []

        # Header
        report_lines.extend(
            [
                "",
                "📈 FASE 2 VALIDATION SUMMARY REPORT",
                "=" * 50,
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Total Execution Time: {total_time:.1f} seconds",
                "",
            ]
        )

        # Overall Statistics
        report_lines.extend(
            [
                "📊 OVERALL STATISTICS",
                f"  Total Tests: {overall_stats['total_tests']}",
                f"  Passed: {overall_stats['total_passed']} ({overall_stats['success_rate']:.1%})",
                f"  Failed: {overall_stats['total_failed']}",
                f"  Skipped: {overall_stats['total_skipped']}",
                f"  Weighted Success Score: {overall_stats['weighted_success_score']:.1%}",
                "",
            ]
        )

        # Module Results
        report_lines.extend(
            [
                "🗏 MODULE RESULTS",
            ]
        )

        for module_name, result in self.results.items():
            status_icon = {
                "passed": "✅",
                "failed": "❌",
                "timeout": "⏰",
                "error": "⚠️",
                "skipped": "⏭️",
                "no_tests": "🚫",
            }.get(result["status"], "❓")

            report_lines.extend(
                [
                    f"  {status_icon} {module_name} (Weight: {result['weight']}%)",
                    f"     Status: {result['status'].upper()}",
                    f"     Tests: {result.get('tests_passed', 0)}/{result.get('tests_run', 0)} passed",
                    f"     Time: {result.get('execution_time', 0):.1f}s",
                    f"     Description: {result['description']}",
                ]
            )

            if result["status"] in ["failed", "error", "timeout"]:
                error_info = result.get(
                    "error", "See detailed logs for more information"
                )
                report_lines.append(f"     Error: {error_info}")

            report_lines.append("")

        # FASE 2 Criteria Validation
        fase2_validation = self.validate_fase2_criteria()
        report_lines.extend(
            [
                "🎯 FASE 2 SUCCESS CRITERIA VALIDATION",
            ]
        )

        for criterion, validated in fase2_validation["criteria"].items():
            icon = "✅" if validated else "❌"
            criterion_name = criterion.replace("_", " ").title()
            report_lines.append(f"  {icon} {criterion_name}")

        report_lines.extend(
            [
                "",
                f"Criteria Success Rate: {fase2_validation['criteria_success_rate']:.1%} ({fase2_validation['criteria_passed']}/{fase2_validation['criteria_total']})",
                f"FASE 2 Validation: {'PASSED' if fase2_validation['fase2_validated'] else 'FAILED'}",
                "",
            ]
        )

        # Recommendations
        report_lines.extend(
            [
                "💡 RECOMMENDATIONS",
            ]
        )

        if overall_stats["success_rate"] >= 0.9:
            report_lines.append("  • Excellent test coverage and system stability.")
        elif overall_stats["success_rate"] >= 0.7:
            report_lines.append(
                "  • Good overall performance with some areas for improvement."
            )
        else:
            report_lines.append(
                "  • Significant issues detected requiring immediate attention."
            )

        if overall_stats["total_failed"] > 0:
            report_lines.append(
                "  • Review failed tests and address underlying issues."
            )

        if not fase2_validation["fase2_validated"]:
            report_lines.append(
                "  • FASE 2 criteria not fully met - review implementation."
            )

        # Performance Insights
        avg_time_per_test = total_time / max(overall_stats["total_tests"], 1)
        if avg_time_per_test > 1.0:
            report_lines.append(
                f"  • Performance optimization needed (avg {avg_time_per_test:.2f}s per test)."
            )

        report_lines.extend(
            ["", "=" * 50, f"Report generated at {datetime.now().isoformat()}", ""]
        )

        return "\n".join(report_lines)

    def save_detailed_report(
        self, results: Dict[str, Any], filename: str = None
    ) -> str:
        """Save detailed results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fase2_validation_report_{timestamp}.json"

        filepath = os.path.join(project_root, filename)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return filepath


def main():
    """Main function to run FASE 2 validation suite."""
    print("🎆 FASE 2 Complete Validation Suite")
    print(
        "Comprehensive testing of Tool Integration, ReAct Agents, Coordination, and Performance"
    )
    print()

    # Initialize and run validation
    runner = FASE2ValidationRunner()
    results = runner.run_all_tests()

    # Print summary report
    print(results["summary_report"])

    # Save detailed report
    report_file = runner.save_detailed_report(results)
    print(f"📋 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    fase2_validated = results["fase2_validation"]["fase2_validated"]
    overall_success_rate = results["overall_statistics"]["success_rate"]

    if fase2_validated and overall_success_rate >= 0.8:
        print(
            f"\n🎉 FASE 2 VALIDATION SUCCESSFUL! ({overall_success_rate:.1%} success rate)"
        )
        sys.exit(0)
    elif fase2_validated:
        print(
            f"\n⚠️  FASE 2 CRITERIA MET with some test failures ({overall_success_rate:.1%} success rate)"
        )
        sys.exit(1)
    else:
        print(
            f"\n❌ FASE 2 VALIDATION FAILED ({overall_success_rate:.1%} success rate)"
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
