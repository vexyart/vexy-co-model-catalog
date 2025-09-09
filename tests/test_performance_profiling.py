# this_file: tests/test_performance_profiling.py

"""
Test suite for enhanced performance profiling, memory optimization, and resource monitoring.
"""

import asyncio
import gc
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from vexy_co_model_catalog.core.performance import (
    PerformanceMonitor,
    enable_detailed_profiling,
    enable_memory_profiling,
    get_performance_monitor,
    get_system_resource_usage,
    performance_monitor,
    profile_memory,
)


class TestPerformanceMonitor(unittest.TestCase):
    """Test enhanced performance monitoring capabilities."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.monitor = PerformanceMonitor(self.temp_dir / "performance")

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_monitoring(self):
        """Test basic performance monitoring functionality."""
        self.monitor.start_monitoring("test_command")
        time.sleep(0.1)  # Simulate work
        metric = self.monitor.stop_monitoring(success=True)

        assert metric is not None
        assert metric.command == "test_command"
        assert metric.success
        assert metric.duration_seconds > 0
        assert metric.memory_mb_start > 0

    def test_enhanced_resource_tracking(self):
        """Test enhanced resource tracking with disk I/O and network monitoring."""
        self.monitor.start_monitoring("resource_test", {"test": True})

        # Simulate some work that might cause resource usage
        time.sleep(0.1)
        list(range(1000))  # Memory allocation

        metric = self.monitor.stop_monitoring(success=True)

        # Check enhanced metrics
        assert metric.duration_seconds > 0
        assert metric.memory_peak_mb >= metric.memory_mb_start
        assert metric.thread_count_start is not None
        assert metric.file_handles_start is not None
        assert isinstance(metric.gc_collections, dict)

    def test_memory_leak_detection(self):
        """Test memory leak detection capabilities."""
        # Enable memory profiling for this test
        os.environ["VEXY_MEMORY_PROFILING"] = "true"
        monitor = PerformanceMonitor(self.temp_dir / "performance")

        monitor.start_monitoring("memory_test")

        # Simulate potential memory leak by creating large objects
        large_data = []
        for _i in range(100):
            large_data.append([0] * 1000)  # Create large lists

        time.sleep(0.1)
        metric = monitor.stop_monitoring(success=True)

        # Check memory growth tracking
        assert metric.memory_usage_change > 0
        assert metric.memory_peak_mb >= metric.memory_mb_start

        # Clean up
        os.environ.pop("VEXY_MEMORY_PROFILING", None)

    def test_performance_statistics(self):
        """Test performance statistics generation."""
        # Generate some test metrics
        for i in range(5):
            self.monitor.start_monitoring(f"test_command_{i}")
            time.sleep(0.01)
            self.monitor.stop_monitoring(success=i % 2 == 0)

        stats = self.monitor.get_statistics()

        assert stats["total_commands"] == 5
        assert stats["successful_commands"] == 3  # 0, 2, 4
        assert stats["failed_commands"] == 2     # 1, 3
        assert stats["total_execution_time"] > 0
        assert "commands_by_type" in stats
        assert "resource_usage" in stats

    def test_bottleneck_identification(self):
        """Test bottleneck identification functionality."""
        # Create a mix of fast and slow commands
        for i in range(3):
            self.monitor.start_monitoring(f"fast_command_{i}")
            time.sleep(0.01)  # Fast
            self.monitor.stop_monitoring(success=True)

        for i in range(2):
            self.monitor.start_monitoring(f"slow_command_{i}")
            time.sleep(0.5)   # Slow (above 0.3s threshold)
            self.monitor.stop_monitoring(success=True)

        bottlenecks = self.monitor.identify_bottlenecks(threshold_seconds=0.3)

        assert bottlenecks["total_commands_analyzed"] == 5
        assert bottlenecks["slow_commands"]["count"] == 2
        assert len(bottlenecks["recommendations"]) > 0

    def test_memory_optimization(self):
        """Test memory optimization functionality."""
        # Create some garbage to collect
        temp_data = []
        for i in range(1000):
            temp_data.append(f"test_string_{i}")
        del temp_data

        optimization_result = self.monitor.optimize_memory_usage()

        assert "initial_memory_mb" in optimization_result
        assert "final_memory_mb" in optimization_result
        assert "memory_freed_mb" in optimization_result
        assert "objects_collected" in optimization_result
        assert "recommendations" in optimization_result

    def test_performance_report_generation(self):
        """Test comprehensive performance report generation."""
        # Generate some metrics first
        self.monitor.start_monitoring("test_report")
        time.sleep(0.1)
        self.monitor.stop_monitoring(success=True)

        report = self.monitor.get_performance_report()

        assert "report_generated" in report
        assert "system_info" in report
        assert "performance_statistics" in report
        assert "optimization_summary" in report
        assert "overall_health_score" in report["optimization_summary"]

    def test_health_score_calculation(self):
        """Test system health score calculation."""
        # Test with perfect performance
        self.monitor.start_monitoring("perfect_command")
        time.sleep(0.01)  # Fast
        self.monitor.stop_monitoring(success=True)

        health_score = self.monitor._calculate_health_score()
        assert health_score > 90  # Should be high

        # Test with failures and slow performance
        self.monitor.start_monitoring("slow_command")
        time.sleep(0.1)
        self.monitor.stop_monitoring(success=False, error_message="Test failure")

        health_score = self.monitor._calculate_health_score()
        assert health_score < 100  # Should be lower

    def test_metrics_persistence(self):
        """Test saving and loading performance metrics."""
        # Generate test metrics
        self.monitor.start_monitoring("persist_test")
        time.sleep(0.01)
        self.monitor.stop_monitoring(success=True)

        # Save metrics
        filepath = self.monitor.save_metrics()
        assert filepath.exists()

        # Verify file contains valid JSON
        import json
        with open(filepath) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "statistics" in data
        assert "metrics" in data


class TestPerformanceDecorators(unittest.TestCase):
    """Test performance monitoring decorators."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('vexy_co_model_catalog.core.performance.get_performance_monitor')
    def test_performance_monitor_decorator(self, mock_get_monitor):
        """Test performance monitoring decorator."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor

        @performance_monitor(command_name="test_function")
        def test_function(x):
            time.sleep(0.01)
            return x * 2

        result = test_function(5)

        assert result == 10
        mock_monitor.start_monitoring.assert_called_once()
        mock_monitor.stop_monitoring.assert_called_once()

    @patch('vexy_co_model_catalog.core.performance.get_performance_monitor')
    def test_performance_monitor_decorator_with_exception(self, mock_get_monitor):
        """Test performance monitoring decorator handles exceptions."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor

        @performance_monitor(command_name="failing_function")
        def failing_function():
            msg = "Test error"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        mock_monitor.start_monitoring.assert_called_once()
        mock_monitor.stop_monitoring.assert_called_once_with(
            success=False,
            error_message="ValueError: Test error"
        )

    def test_memory_profiling_decorator(self):
        """Test memory profiling decorator."""
        # Enable memory profiling
        os.environ["VEXY_MEMORY_PROFILING"] = "true"

        @profile_memory
        def memory_intensive_function():
            # Create some data that uses memory
            data = list(range(10000))
            return len(data)

        try:
            result = memory_intensive_function()
            assert result == 10000
        finally:
            os.environ.pop("VEXY_MEMORY_PROFILING", None)


class TestSystemResourceMonitoring(unittest.TestCase):
    """Test system resource monitoring functionality."""

    def test_system_resource_usage(self):
        """Test system resource usage snapshot."""
        resource_usage = get_system_resource_usage()

        if "error" not in resource_usage:
            assert "timestamp" in resource_usage
            assert "cpu_percent" in resource_usage
            assert "memory_rss_mb" in resource_usage
            assert "thread_count" in resource_usage
            assert "system_cpu_percent" in resource_usage
            assert "system_memory_percent" in resource_usage

    def test_profiling_enablement(self):
        """Test enabling profiling features."""
        # Test enabling memory profiling
        enable_memory_profiling()
        assert os.environ.get("VEXY_MEMORY_PROFILING") == "true"

        # Test enabling detailed profiling
        enable_detailed_profiling()
        assert os.environ.get("VEXY_DETAILED_PROFILING") == "true"

        # Clean up
        os.environ.pop("VEXY_MEMORY_PROFILING", None)
        os.environ.pop("VEXY_DETAILED_PROFILING", None)


class TestPerformanceIntegration(unittest.TestCase):
    """Integration tests for performance monitoring system."""

    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up integration test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_global_monitor_integration(self):
        """Test global performance monitor integration."""
        # Get global monitor
        monitor = get_performance_monitor(self.temp_dir / "performance")
        assert isinstance(monitor, PerformanceMonitor)

        # Test that subsequent calls return the same instance
        monitor2 = get_performance_monitor()
        assert monitor is monitor2

    def test_concurrent_monitoring(self):
        """Test performance monitoring under concurrent operations."""
        monitor = PerformanceMonitor(self.temp_dir / "performance")
        results = []

        def worker_thread(worker_id):
            monitor.start_monitoring(f"worker_{worker_id}")
            time.sleep(0.05)  # Simulate work
            metric = monitor.stop_monitoring(success=True)
            results.append(metric)

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(results) == 3
        for result in results:
            assert result.success
            assert result.duration_seconds > 0

    async def test_async_monitoring_compatibility(self):
        """Test performance monitoring with async operations."""
        monitor = PerformanceMonitor(self.temp_dir / "performance")

        async def async_operation():
            monitor.start_monitoring("async_test")
            await asyncio.sleep(0.05)  # Async work simulation
            return monitor.stop_monitoring(success=True)

        metric = await async_operation()
        assert metric.success
        assert metric.duration_seconds > 0.04


@pytest.mark.integration
class TestPerformanceProfilingCLI(unittest.TestCase):
    """Integration tests for CLI performance profiling commands."""

    def setUp(self):
        """Set up CLI test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up CLI test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_performance_stats(self):
        """Test CLI performance stats command."""
        # This would be tested with actual CLI integration
        # For now, we test the underlying functionality
        monitor = PerformanceMonitor(self.temp_dir / "performance")

        # Generate test data
        monitor.start_monitoring("cli_test")
        time.sleep(0.01)
        monitor.stop_monitoring(success=True)

        stats = monitor.get_statistics()
        assert stats["total_commands"] > 0

    def test_cli_bottleneck_analysis(self):
        """Test CLI bottleneck analysis command."""
        monitor = PerformanceMonitor(self.temp_dir / "performance")

        # Create test bottleneck
        monitor.start_monitoring("slow_operation")
        time.sleep(0.5)  # Slow operation
        monitor.stop_monitoring(success=True)

        bottlenecks = monitor.identify_bottlenecks(threshold_seconds=0.3)
        assert bottlenecks["slow_commands"]["count"] == 1

    def test_cli_memory_optimization(self):
        """Test CLI memory optimization command."""
        monitor = PerformanceMonitor(self.temp_dir / "performance")

        # Create some garbage
        temp_data = list(range(1000))
        del temp_data

        optimization_result = monitor.optimize_memory_usage()
        assert "memory_freed_mb" in optimization_result


if __name__ == "__main__":
    unittest.main()
