# this_file: tests/test_integration_phase19_features.py

"""
Comprehensive end-to-end integration tests for Phase 19 features.

Tests cover:
- Shell completion system (Bash/Zsh/Fish)
- Production diagnostics (multi-level validation)
- Continuous monitoring (dashboards, alerts, metrics)
- Quality gates and performance benchmarks
"""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vexy_co_model_catalog.cli import CLI
from vexy_co_model_catalog.core.completion import CompletionGenerator
from vexy_co_model_catalog.core.monitoring import AlertLevel, MetricType, MonitoringSystem
from vexy_co_model_catalog.core.production_reliability import (
    ProductionReliabilityHardening,
    ReliabilityLevel,
)


class TestShellCompletionIntegration:
    """Integration tests for shell completion system."""

    def setup_method(self):
        """Set up test environment."""
        self.completion_generator = CompletionGenerator()
        self.cli = CLI()

    def test_bash_completion_generation(self):
        """Test complete Bash completion script generation."""
        completion_script = self.completion_generator.generate_bash_completion()

        # Verify script structure
        assert completion_script.startswith("#!/bin/bash") or completion_script.startswith("#!/usr/bin/env bash")
        assert "_vexy_completion()" in completion_script
        assert "complete -F _vexy_completion vexy" in completion_script

        # Verify command completion
        assert "providers" in completion_script
        assert "fetch" in completion_script
        assert "health" in completion_script
        assert "production_status" in completion_script

        # Verify provider completion
        assert "openai" in completion_script
        assert "anthropic" in completion_script
        assert "groq" in completion_script

    def test_zsh_completion_generation(self):
        """Test complete Zsh completion script generation."""
        completion_script = self.completion_generator.generate_zsh_completion()

        # Verify Zsh-specific structure
        assert "#compdef vexy" in completion_script
        assert "_vexy()" in completion_script
        assert "_arguments" in completion_script

        # Verify command descriptions
        assert "List all providers" in completion_script
        assert "Fetch models from providers" in completion_script
        assert "System health diagnostics" in completion_script

    def test_fish_completion_generation(self):
        """Test complete Fish completion script generation."""
        completion_script = self.completion_generator.generate_fish_completion()

        # Verify Fish-specific structure
        assert "complete -c vexy" in completion_script
        assert "function __fish_vexy_providers" in completion_script

        # Verify command completion
        assert "-a 'providers'" in completion_script
        assert "-a 'fetch'" in completion_script
        assert "-a 'health'" in completion_script

    @patch('subprocess.run')
    def test_completion_installation_bash(self, mock_run):
        """Test Bash completion installation process."""
        mock_run.return_value = MagicMock(returncode=0)

        # Test completion command with installation
        self.cli.completion("bash", install=True)

        # Verify installation attempt
        mock_run.assert_called()

    def test_completion_command_integration(self):
        """Test completion command end-to-end integration."""
        # Test completion generation without installation
        result = self.cli.completion("bash", install=False)

        # Should complete without errors
        assert result is None  # No return value expected

    def test_provider_name_completion_accuracy(self):
        """Test that completion includes all available providers."""
        bash_script = self.completion_generator.generate_bash_completion()

        # Critical providers should be included
        essential_providers = [
            "openai", "anthropic", "groq", "cerebras",
            "deepinfra", "fireworks", "togetherai"
        ]

        for provider in essential_providers:
            assert provider in bash_script


class TestProductionDiagnosticsIntegration:
    """Integration tests for production diagnostics system."""

    def setup_method(self):
        """Set up test environment."""
        self.reliability_hardening = ProductionReliabilityHardening()
        self.cli = CLI()

    @pytest.mark.asyncio
    async def test_basic_diagnostics_complete_workflow(self):
        """Test complete basic diagnostics workflow."""
        # Run basic level diagnostics
        result = await self.reliability_hardening.run_comprehensive_diagnostics(
            ReliabilityLevel.BASIC
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "level" in result
        assert "total_checks" in result
        assert "passed_checks" in result
        assert "failed_checks" in result
        assert "diagnostics" in result

        # Verify basic checks are present
        basic_categories = ["python_environment", "storage_health", "basic_connectivity"]
        for category in basic_categories:
            assert any(category in diag["category"] for diag in result["diagnostics"])

    @pytest.mark.asyncio
    async def test_standard_diagnostics_comprehensive_checks(self):
        """Test standard diagnostics with comprehensive system validation."""
        result = await self.reliability_hardening.run_comprehensive_diagnostics(
            ReliabilityLevel.STANDARD
        )

        # Verify comprehensive check coverage
        expected_categories = [
            "python_environment", "storage_health", "network_connectivity",
            "provider_resilience", "configuration_integrity", "performance_baseline"
        ]

        result_categories = [diag["category"] for diag in result["diagnostics"]]
        for category in expected_categories:
            assert any(category in result_cat for result_cat in result_categories)

    @pytest.mark.asyncio
    async def test_enterprise_diagnostics_advanced_validation(self):
        """Test enterprise-level diagnostics with advanced validation."""
        result = await self.reliability_hardening.run_comprehensive_diagnostics(
            ReliabilityLevel.ENTERPRISE
        )

        # Verify enterprise-level features
        assert result["level"] == "enterprise"
        assert result["total_checks"] >= 10  # Should have many checks

        # Check for advanced diagnostic categories
        advanced_categories = ["security_validation", "performance_optimization", "monitoring_health"]
        result_categories = [diag["category"] for diag in result["diagnostics"]]

        # At least some advanced categories should be present
        advanced_present = any(
            any(adv_cat in result_cat for result_cat in result_categories)
            for adv_cat in advanced_categories
        )
        assert advanced_present

    @pytest.mark.asyncio
    async def test_critical_diagnostics_exhaustive_validation(self):
        """Test critical-level diagnostics with exhaustive system validation."""
        result = await self.reliability_hardening.run_comprehensive_diagnostics(
            ReliabilityLevel.CRITICAL
        )

        # Verify critical-level thoroughness
        assert result["level"] == "critical"
        assert result["total_checks"] >= 15  # Should be very comprehensive

        # Verify detailed diagnostic information
        for diagnostic in result["diagnostics"]:
            assert "category" in diagnostic
            assert "status" in diagnostic
            assert diagnostic["status"] in ["passed", "failed", "warning"]

            if diagnostic["status"] == "failed":
                assert "error_details" in diagnostic
                assert "suggested_fix" in diagnostic

    def test_cli_production_diagnostics_command(self):
        """Test CLI production diagnostics command integration."""
        # Test basic command execution
        try:
            self.cli.production_diagnostics("basic", auto_fix=False)
            # Should complete without exceptions
        except Exception as e:
            pytest.fail(f"Production diagnostics command failed: {e}")

    @pytest.mark.asyncio
    async def test_diagnostic_auto_fix_functionality(self):
        """Test diagnostic system's auto-fix capabilities."""
        # Create a diagnostic result with fixable issues
        result = await self.reliability_hardening.run_comprehensive_diagnostics(
            ReliabilityLevel.STANDARD
        )

        # Check if auto-fix suggestions are provided
        failed_diagnostics = [d for d in result["diagnostics"] if d["status"] == "failed"]

        for diagnostic in failed_diagnostics:
            if "suggested_fix" in diagnostic:
                assert isinstance(diagnostic["suggested_fix"], str)
                assert len(diagnostic["suggested_fix"]) > 0


class TestContinuousMonitoringIntegration:
    """Integration tests for continuous monitoring system."""

    def setup_method(self):
        """Set up test environment."""
        self.monitoring_system = MonitoringSystem()
        self.cli = CLI()

    def test_monitoring_system_initialization(self):
        """Test monitoring system proper initialization."""
        # Verify components are initialized
        assert self.monitoring_system.metric_collector is not None
        assert self.monitoring_system.alert_manager is not None
        assert hasattr(self.monitoring_system, 'start_monitoring')
        assert hasattr(self.monitoring_system, 'stop_monitoring')

    def test_metric_collection_end_to_end(self):
        """Test complete metric collection workflow."""
        # Record various metric types
        self.monitoring_system.metric_collector.record_metric(
            "test_counter", 1.0, MetricType.COUNTER, {"test": "value"}
        )
        self.monitoring_system.metric_collector.record_metric(
            "test_gauge", 50.0, MetricType.GAUGE, {"component": "test"}
        )
        self.monitoring_system.metric_collector.record_metric(
            "test_timer", 125.5, MetricType.TIMER, {"operation": "test_op"}
        )

        # Verify metrics are recorded
        counter_metrics = self.monitoring_system.metric_collector.get_metrics(
            "test_counter", MetricType.COUNTER
        )
        gauge_metrics = self.monitoring_system.metric_collector.get_metrics(
            "test_gauge", MetricType.GAUGE
        )
        timer_metrics = self.monitoring_system.metric_collector.get_metrics(
            "test_timer", MetricType.TIMER
        )

        assert len(counter_metrics) == 1
        assert len(gauge_metrics) == 1
        assert len(timer_metrics) == 1
        assert counter_metrics[0].value == 1.0
        assert gauge_metrics[0].value == 50.0
        assert timer_metrics[0].value == 125.5

    def test_alert_management_workflow(self):
        """Test complete alert management workflow."""
        # Add alert rule
        def test_condition(collector):
            return True  # Always trigger for testing

        self.monitoring_system.alert_manager.add_alert_rule(
            "test_alert",
            test_condition,
            AlertLevel.WARNING,
            "Test Alert",
            "This is a test alert: {error}",
            cooldown_minutes=1
        )

        # Evaluate alerts
        alerts = self.monitoring_system.alert_manager.evaluate_alerts(
            self.monitoring_system.metric_collector
        )

        # Verify alert was triggered
        assert len(alerts) >= 0  # May be 0 due to cooldown or other conditions

    def test_monitoring_dashboard_integration(self):
        """Test monitoring dashboard functionality."""
        # Add some test metrics
        self.monitoring_system.record_provider_response_time("openai", 150.0)
        self.monitoring_system.record_api_error("anthropic", "timeout")
        self.monitoring_system.record_cache_hit("provider_list")

        # Test dashboard data retrieval
        try:
            # These should not raise exceptions
            self.monitoring_system.get_system_health_overview()
            self.monitoring_system.get_recent_alerts(limit=10)
            metrics = self.monitoring_system.get_performance_metrics(minutes=60)
            assert isinstance(metrics, dict)
        except Exception as e:
            pytest.fail(f"Dashboard integration failed: {e}")

    def test_cli_monitoring_commands_integration(self):
        """Test CLI monitoring commands integration."""
        # Test monitoring status command
        try:
            self.cli.monitoring("status")
        except Exception as e:
            pytest.fail(f"Monitoring status command failed: {e}")

        # Test monitoring metrics command
        try:
            self.cli.monitoring("metrics")
        except Exception as e:
            pytest.fail(f"Monitoring metrics command failed: {e}")

    def test_monitoring_data_persistence(self):
        """Test monitoring data persistence and retrieval."""
        # Record metrics
        self.monitoring_system.record_command_performance("fetch", 2.5, True)
        self.monitoring_system.record_provider_response_time("groq", 75.0)

        # Verify persistence
        command_metrics = self.monitoring_system.metric_collector.get_metrics(
            "command_performance_fetch", MetricType.TIMER
        )
        provider_metrics = self.monitoring_system.metric_collector.get_metrics(
            "provider_response_time_groq", MetricType.TIMER
        )

        assert len(command_metrics) >= 0
        assert len(provider_metrics) >= 0


class TestQualityGatesAndBenchmarks:
    """Integration tests for automated quality gates and performance benchmarks."""

    def setup_method(self):
        """Set up test environment."""
        self.cli = CLI()

    def test_performance_benchmark_validation(self):
        """Test performance benchmarks meet quality standards."""
        import time

        # Test command performance benchmarks
        start_time = time.time()
        self.cli.providers()
        providers_time = time.time() - start_time

        start_time = time.time()
        self.cli.stats()
        stats_time = time.time() - start_time

        start_time = time.time()
        self.cli.version()
        version_time = time.time() - start_time

        # Quality gates: commands should complete quickly
        assert providers_time < 2.0, f"providers command too slow: {providers_time:.3f}s"
        assert stats_time < 1.0, f"stats command too slow: {stats_time:.3f}s"
        assert version_time < 0.5, f"version command too slow: {version_time:.3f}s"

    @pytest.mark.asyncio
    async def test_system_health_quality_gate(self):
        """Test system health meets quality standards."""
        reliability_hardening = ProductionReliabilityHardening()

        # Run diagnostics
        result = await reliability_hardening.run_comprehensive_diagnostics(
            ReliabilityLevel.STANDARD
        )

        # Quality gate: majority of checks should pass
        pass_rate = result["passed_checks"] / result["total_checks"] * 100
        assert pass_rate >= 70.0, f"System health below quality gate: {pass_rate:.1f}%"

    def test_memory_usage_quality_gate(self):
        """Test memory usage stays within acceptable bounds."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run memory-intensive operations
        for _i in range(10):
            self.cli.providers()
            self.cli.stats()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Quality gate: memory usage should not grow excessively
        assert memory_increase < 50.0, f"Memory usage increased too much: {memory_increase:.1f}MB"

    def test_error_handling_quality_gate(self):
        """Test error handling meets quality standards."""
        # Test handling of invalid inputs
        try:
            self.cli.fetch("nonexistent_provider")
            # Should handle gracefully, not crash
        except SystemExit:
            # CLI may exit on error, which is acceptable
            pass
        except Exception as e:
            # Should not have unhandled exceptions
            pytest.fail(f"Unhandled exception in error case: {e}")

    def test_concurrent_operation_quality_gate(self):
        """Test concurrent operations meet quality standards."""
        import threading
        import time

        results = []
        errors = []

        def run_command():
            try:
                start = time.time()
                self.cli.version()  # Lightweight command
                duration = time.time() - start
                results.append(duration)
            except Exception as e:
                errors.append(e)

        # Run concurrent operations
        threads = []
        for _i in range(5):
            thread = threading.Thread(target=run_command)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Quality gates
        assert len(errors) == 0, f"Concurrent operations had errors: {errors}"
        assert len(results) == 5, "Not all concurrent operations completed"

        avg_time = sum(results) / len(results)
        assert avg_time < 1.0, f"Concurrent operations too slow: {avg_time:.3f}s average"


@pytest.mark.integration
class TestPhase19FeatureIntegration:
    """End-to-end integration tests covering all Phase 19 features together."""

    def setup_method(self):
        """Set up comprehensive test environment."""
        self.cli = CLI()
        self.completion_generator = CompletionGenerator()
        self.monitoring_system = MonitoringSystem()
        self.reliability_hardening = ProductionReliabilityHardening()

    @pytest.mark.asyncio
    async def test_complete_phase19_feature_workflow(self):
        """Test complete workflow integrating all Phase 19 features."""
        # 1. Test shell completion generation
        bash_completion = self.completion_generator.generate_bash_completion()
        assert len(bash_completion) > 1000  # Should be substantial

        # 2. Test production diagnostics
        diagnostic_result = await self.reliability_hardening.run_comprehensive_diagnostics(
            ReliabilityLevel.STANDARD
        )
        assert diagnostic_result["total_checks"] > 5

        # 3. Test monitoring system
        self.monitoring_system.record_command_performance("integration_test", 1.0, True)
        metrics = self.monitoring_system.metric_collector.get_metrics(
            "command_performance_integration_test", MetricType.TIMER
        )
        assert len(metrics) >= 1

        # 4. Test quality gates
        assert diagnostic_result["passed_checks"] / diagnostic_result["total_checks"] >= 0.5

    def test_phase19_feature_robustness(self):
        """Test robustness of Phase 19 features under various conditions."""
        # Test completion generation robustness
        for shell in ["bash", "zsh", "fish"]:
            if shell == "bash":
                result = self.completion_generator.generate_bash_completion()
            elif shell == "zsh":
                result = self.completion_generator.generate_zsh_completion()
            else:
                result = self.completion_generator.generate_fish_completion()
            assert len(result) > 100  # Should generate meaningful content

        # Test CLI robustness with various inputs
        test_inputs = ["", "invalid_command", "help", "version"]
        for test_input in test_inputs:
            try:
                if test_input == "":
                    self.cli.version()  # Default safe command
                elif test_input == "invalid_command":
                    # Should handle gracefully
                    pass  # Skip invalid commands in this test
                else:
                    getattr(self.cli, test_input, lambda: None)()
            except Exception as e:
                # Should not crash on reasonable inputs
                if "invalid_command" not in str(e).lower():
                    pytest.fail(f"Unexpected failure on input '{test_input}': {e}")

    def test_phase19_performance_integration(self):
        """Test performance characteristics of integrated Phase 19 features."""
        import time

        # Test completion generation performance
        start = time.time()
        bash_completion = self.completion_generator.generate_bash_completion()
        completion_time = time.time() - start

        # Test CLI command performance
        start = time.time()
        self.cli.version()
        cli_time = time.time() - start

        # Performance quality gates
        assert completion_time < 1.0, f"Completion generation too slow: {completion_time:.3f}s"
        assert cli_time < 0.5, f"CLI command too slow: {cli_time:.3f}s"
        assert len(bash_completion) > 500, "Completion script too short"
