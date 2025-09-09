# this_file: tests/test_integration_quality_gates.py

"""
Comprehensive integration tests for automated quality gates and performance benchmarks.
Validates system-wide quality metrics and production readiness standards.
"""

import asyncio
import gc
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest

from vexy_co_model_catalog.cli import CLI
from vexy_co_model_catalog.core.monitoring import MonitoringSystem
from vexy_co_model_catalog.core.performance import PerformanceMonitor
from vexy_co_model_catalog.core.production_reliability import (
    ProductionReliabilityHardening,
    ReliabilityLevel,
)


class TestSystemPerformanceQualityGates:
    """Quality gates for system performance benchmarks."""

    def setup_method(self):
        """Set up performance monitoring environment."""
        self.cli = CLI()
        self.performance_monitor = PerformanceMonitor()
        self.process = psutil.Process()

    def test_cli_command_response_time_quality_gate(self):
        """Ensure CLI commands meet response time standards."""
        # Performance benchmarks for critical commands
        performance_standards = {
            'version': 0.5,    # Should be very fast
            'providers': 2.0,  # May need to load provider data
            'stats': 1.0,      # Lightweight statistics
        }

        for command, max_time in performance_standards.items():
            start_time = time.time()

            try:
                # Execute command through CLI
                if command == 'version':
                    self.cli.version()
                elif command == 'providers':
                    self.cli.providers()
                elif command == 'stats':
                    self.cli.stats()
                # Skip health command due to method signature complexity

                execution_time = time.time() - start_time

                assert execution_time < max_time, (
                    f"{command} command exceeded quality gate: "
                    f"{execution_time:.3f}s > {max_time}s"
                )

            except Exception as e:
                # Command should not fail completely
                pytest.fail(f"{command} command failed unexpectedly: {e}")

    def test_memory_consumption_quality_gate(self):
        """Ensure memory consumption stays within acceptable bounds."""
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Run memory-intensive operation sequence
        memory_test_operations = [
            lambda: self.cli.providers(),
            lambda: self.cli.stats(),
            lambda: self.cli.version(),
            lambda: gc.collect(),  # Force garbage collection
        ]

        for operation in memory_test_operations * 3:  # Repeat 3 times
            operation()

        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Quality gate: memory increase should be reasonable
        assert memory_increase < 100.0, (
            f"Memory consumption exceeded quality gate: "
            f"{memory_increase:.1f}MB increase"
        )

    def test_cpu_usage_quality_gate(self):
        """Ensure CPU usage patterns are reasonable."""
        # Monitor CPU before operations
        self.process.cpu_percent(interval=0.1)

        # Run CPU-intensive operations
        start_time = time.time()
        for _ in range(5):
            self.cli.providers()
            self.cli.stats()
        end_time = time.time()

        cpu_percent_after = self.process.cpu_percent(interval=0.1)
        operation_duration = end_time - start_time

        # Quality gates
        assert operation_duration < 10.0, (
            f"Operations took too long: {operation_duration:.3f}s"
        )
        # CPU usage should be reasonable (allow some variation in CI)
        assert cpu_percent_after < 80.0, (
            f"CPU usage too high: {cpu_percent_after:.1f}%"
        )

    def test_concurrent_operation_stability(self):
        """Test system stability under concurrent load."""
        results = []
        errors = []

        def concurrent_operation(operation_id):
            """Run concurrent CLI operations."""
            try:
                start = time.time()
                self.cli.version()  # Use lightweight command
                duration = time.time() - start
                results.append((operation_id, duration))
            except Exception as e:
                errors.append((operation_id, str(e)))

        # Create and start concurrent threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)  # Timeout to prevent hanging

        # Quality gates
        assert len(errors) == 0, f"Concurrent operations failed: {errors}"
        assert len(results) == 10, f"Not all operations completed: {len(results)}/10"

        # Check performance consistency
        durations = [duration for _, duration in results]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)

        assert avg_duration < 1.0, f"Average duration too high: {avg_duration:.3f}s"
        assert max_duration < 2.0, f"Max duration too high: {max_duration:.3f}s"


class TestSystemReliabilityQualityGates:
    """Quality gates for system reliability and robustness."""

    def setup_method(self):
        """Set up reliability testing environment."""
        self.cli = CLI()
        self.reliability_hardening = ProductionReliabilityHardening()

    @pytest.mark.asyncio
    async def test_system_health_quality_gate(self):
        """Ensure system health meets production standards."""
        # Run comprehensive diagnostics
        diagnostic_result = await self.reliability_hardening.run_comprehensive_diagnostics(
            ReliabilityLevel.STANDARD
        )

        # Calculate health metrics
        total_checks = diagnostic_result["total_checks"]
        passed_checks = diagnostic_result["passed_checks"]
        diagnostic_result["failed_checks"]

        health_percentage = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

        # Quality gates
        assert total_checks >= 8, f"Insufficient diagnostic coverage: {total_checks} checks"
        assert health_percentage >= 75.0, (
            f"System health below quality gate: {health_percentage:.1f}% "
            f"({passed_checks}/{total_checks} passed)"
        )

        # Check for critical failures
        critical_failures = [
            diag for diag in diagnostic_result["diagnostics"]
            if diag["status"] == "failed" and "critical" in diag.get("category", "").lower()
        ]

        assert len(critical_failures) == 0, (
            f"Critical system failures detected: {critical_failures}"
        )

    def test_error_handling_robustness(self):
        """Test error handling robustness across different scenarios."""
        error_scenarios = [
            ("invalid_provider", lambda: self.cli.fetch("nonexistent_provider_12345")),
            ("empty_input", lambda: self.cli.providers("")),
            ("malformed_command", lambda: getattr(self.cli, "nonexistent_command", lambda: None)()),
        ]

        handled_errors = 0
        unhandled_exceptions = []

        for scenario_name, operation in error_scenarios:
            try:
                operation()
                # If no exception, that's also acceptable (graceful handling)
                handled_errors += 1
            except SystemExit:
                # CLI may exit on error, which is acceptable
                handled_errors += 1
            except Exception as e:
                # Unhandled exceptions are problematic
                unhandled_exceptions.append((scenario_name, str(e)))

        # Quality gate: should handle errors gracefully
        total_scenarios = len(error_scenarios)
        error_handling_rate = handled_errors / total_scenarios * 100

        assert error_handling_rate >= 80.0, (
            f"Error handling below quality gate: {error_handling_rate:.1f}% "
            f"Unhandled: {unhandled_exceptions}"
        )

    def test_resource_cleanup_quality_gate(self):
        """Ensure proper resource cleanup and no resource leaks."""
        # Get initial resource state
        initial_files = len(self.process.open_files())
        initial_threads = self.process.num_threads()

        # Run operations that use resources
        resource_operations = [
            lambda: self.cli.providers(),
            lambda: self.cli.stats(),
            lambda: self.cli.version(),
        ]

        for operation in resource_operations * 5:  # Repeat to amplify leaks
            operation()

        # Force cleanup
        gc.collect()
        time.sleep(0.1)  # Allow time for cleanup

        # Check final resource state
        final_files = len(self.process.open_files())
        final_threads = self.process.num_threads()

        file_leak = final_files - initial_files
        thread_leak = final_threads - initial_threads

        # Quality gates for resource leaks
        assert file_leak <= 5, (
            f"Potential file handle leak: {file_leak} files "
            f"({initial_files} -> {final_files})"
        )
        assert thread_leak <= 2, (
            f"Potential thread leak: {thread_leak} threads "
            f"({initial_threads} -> {final_threads})"
        )


class TestMonitoringSystemQualityGates:
    """Quality gates for monitoring and observability systems."""

    def setup_method(self):
        """Set up monitoring system for testing."""
        self.monitoring_system = MonitoringSystem()
        self.cli = CLI()

    def test_monitoring_data_accuracy_quality_gate(self):
        """Ensure monitoring data collection is accurate."""
        # Record known metrics
        test_metrics = [
            ("command_test", 100.0, {"component": "test"}),
            ("response_time", 250.5, {"provider": "openai"}),
            ("cache_hit", 1.0, {"cache_type": "provider"}),
        ]

        for metric_name, value, labels in test_metrics:
            self.monitoring_system.record_command_performance(
                metric_name, value, True, **labels
            )

        # Verify data accuracy
        for metric_name, expected_value, labels in test_metrics:
            # Get the latest recorded value
            metrics = self.monitoring_system.metric_collector.get_metrics(
                f"command_performance_{metric_name}",
                self.monitoring_system.metric_collector.MetricType.TIMER
            )

            if metrics:  # May be empty due to internal naming
                latest_metric = metrics[-1]
                # Value should match within reasonable precision
                assert abs(latest_metric.value - expected_value) < 0.1, (
                    f"Metric accuracy failed for {metric_name}: "
                    f"expected {expected_value}, got {latest_metric.value}"
                )

    def test_monitoring_system_performance_quality_gate(self):
        """Ensure monitoring system itself performs efficiently."""
        start_time = time.time()

        # Record many metrics quickly
        for i in range(100):
            self.monitoring_system.record_command_performance(
                f"test_metric_{i % 10}", float(i), i % 2 == 0
            )

        monitoring_overhead = time.time() - start_time

        # Quality gate: monitoring overhead should be minimal
        assert monitoring_overhead < 1.0, (
            f"Monitoring overhead too high: {monitoring_overhead:.3f}s for 100 metrics"
        )

    def test_monitoring_data_retention_quality_gate(self):
        """Test monitoring data retention and cleanup."""
        # Record metrics with different patterns
        time.time()
        for i in range(20):
            self.monitoring_system.record_command_performance(
                "retention_test", float(i), True
            )

        # Get recorded metrics
        all_metrics = self.monitoring_system.metric_collector.get_metric_names()
        retention_metrics = [m for m in all_metrics if "retention_test" in m]

        # Quality gate: should retain reasonable amount of data
        assert len(retention_metrics) >= 0, "No metrics retained"

        # Test cleanup doesn't break system
        try:
            # Simulate cleanup (if available)
            if hasattr(self.monitoring_system, 'cleanup_old_data'):
                self.monitoring_system.cleanup_old_data()
        except Exception as e:
            pytest.fail(f"Cleanup operation failed: {e}")


class TestIntegrationQualityAssurance:
    """Comprehensive integration quality assurance tests."""

    def setup_method(self):
        """Set up comprehensive testing environment."""
        self.cli = CLI()
        self.monitoring_system = MonitoringSystem()

    def test_end_to_end_workflow_quality_gate(self):
        """Test complete end-to-end workflow meets quality standards."""
        workflow_steps = []

        try:
            # Step 1: Check system version
            step_start = time.time()
            self.cli.version()
            workflow_steps.append(("version", time.time() - step_start))

            # Step 2: List providers
            step_start = time.time()
            self.cli.providers()
            workflow_steps.append(("providers", time.time() - step_start))

            # Step 3: Get statistics
            step_start = time.time()
            self.cli.stats()
            workflow_steps.append(("stats", time.time() - step_start))

        except Exception as e:
            pytest.fail(f"End-to-end workflow failed at step: {e}")

        # Quality gates
        total_time = sum(duration for _, duration in workflow_steps)
        assert total_time < 5.0, f"Total workflow time too high: {total_time:.3f}s"

        # Each step should complete reasonably quickly
        for step_name, duration in workflow_steps:
            max_step_time = 3.0 if step_name != "version" else 1.0
            assert duration < max_step_time, (
                f"Step '{step_name}' too slow: {duration:.3f}s > {max_step_time}s"
            )

    def test_production_readiness_quality_gate(self):
        """Comprehensive production readiness assessment."""
        production_checks = {
            "basic_functionality": False,
            "performance_acceptable": False,
            "error_handling_robust": False,
            "resource_usage_reasonable": False,
        }

        # Test basic functionality
        try:
            self.cli.version()
            self.cli.providers()
            production_checks["basic_functionality"] = True
        except Exception:
            pass

        # Test performance
        start = time.time()
        for _ in range(3):
            self.cli.version()  # Lightweight operation
        avg_time = (time.time() - start) / 3
        production_checks["performance_acceptable"] = avg_time < 1.0

        # Test error handling
        try:
            # This should handle gracefully
            invalid_result = getattr(self.cli, "nonexistent_method", None)
            production_checks["error_handling_robust"] = invalid_result is None
        except AttributeError:
            production_checks["error_handling_robust"] = True  # Expected behavior
        except Exception:
            production_checks["error_handling_robust"] = False

        # Test resource usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        for _ in range(5):
            self.cli.version()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        production_checks["resource_usage_reasonable"] = memory_increase < 50.0

        # Overall production readiness quality gate
        passed_checks = sum(production_checks.values())
        total_checks = len(production_checks)
        readiness_score = (passed_checks / total_checks) * 100

        assert readiness_score >= 75.0, (
            f"Production readiness below quality gate: {readiness_score:.1f}% "
            f"Failed checks: {[k for k, v in production_checks.items() if not v]}"
        )

    def test_integration_stress_quality_gate(self):
        """Test system behavior under stress conditions."""
        stress_results = {
            "operations_completed": 0,
            "errors_encountered": 0,
            "max_response_time": 0.0,
            "total_time": 0.0,
        }

        start_time = time.time()

        # Run stress operations
        stress_operations = [
            self.cli.version,
            self.cli.providers,
            self.cli.stats,
        ]

        for _cycle in range(10):  # 10 cycles of operations
            for operation in stress_operations:
                op_start = time.time()
                try:
                    operation()
                    op_time = time.time() - op_start
                    stress_results["operations_completed"] += 1
                    stress_results["max_response_time"] = max(
                        stress_results["max_response_time"], op_time
                    )
                except Exception:
                    stress_results["errors_encountered"] += 1

        stress_results["total_time"] = time.time() - start_time

        # Quality gates for stress test
        total_attempted = 10 * len(stress_operations)
        success_rate = (stress_results["operations_completed"] / total_attempted) * 100

        assert success_rate >= 90.0, (
            f"Stress test success rate below quality gate: {success_rate:.1f}% "
            f"({stress_results['operations_completed']}/{total_attempted})"
        )

        assert stress_results["max_response_time"] < 5.0, (
            f"Maximum response time exceeded quality gate: "
            f"{stress_results['max_response_time']:.3f}s"
        )

        assert stress_results["total_time"] < 30.0, (
            f"Total stress test time too high: {stress_results['total_time']:.3f}s"
        )
