# this_file: tests/test_monitoring.py

"""
Test suite for sophisticated monitoring capabilities with metrics dashboards and operational intelligence.
"""

import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from vexy_co_model_catalog.core.monitoring import (
    AlertLevel,
    AlertManager,
    Dashboard,
    MetricCollector,
    MetricPoint,
    MetricType,
    MonitoringSystem,
    get_monitoring_system,
    initialize_monitoring,
    shutdown_monitoring,
)


class TestMetricCollector(unittest.TestCase):
    """Test metric collection and storage functionality."""

    def setUp(self):
        """Set up test environment."""
        self.collector = MetricCollector(retention_hours=1)

    def test_record_metric(self):
        """Test recording metrics of different types."""
        # Record counter metric
        self.collector.record_metric("requests", 1, MetricType.COUNTER)

        # Record gauge metric
        self.collector.record_metric("memory_usage", 512.5, MetricType.GAUGE)

        # Record timer metric
        self.collector.record_metric("response_time", 0.125, MetricType.TIMER)

        # Verify metrics are stored
        assert len(self.collector.get_metric_names()) == 3
        assert "requests" in self.collector.get_metric_names()
        assert "memory_usage" in self.collector.get_metric_names()
        assert "response_time" in self.collector.get_metric_names()

    def test_get_latest_value(self):
        """Test getting the most recent metric value."""
        self.collector.record_metric("test_metric", 10, MetricType.GAUGE)
        self.collector.record_metric("test_metric", 20, MetricType.GAUGE)
        self.collector.record_metric("test_metric", 30, MetricType.GAUGE)

        latest = self.collector.get_latest_value("test_metric", MetricType.GAUGE)
        assert latest == 30

    def test_get_metrics_with_time_range(self):
        """Test getting metrics within a specific time range."""
        now = datetime.now(timezone.utc)

        # Record metrics at different times (simulated)
        self.collector.record_metric("test_metric", 1, MetricType.COUNTER)
        time.sleep(0.01)  # Small delay
        self.collector.record_metric("test_metric", 2, MetricType.COUNTER)

        # Get all metrics
        all_metrics = self.collector.get_metrics("test_metric", MetricType.COUNTER)
        assert len(all_metrics) == 2

        # Test time range filtering
        start_time = now + timedelta(minutes=1)  # Future time, should return empty
        filtered_metrics = self.collector.get_metrics(
            "test_metric", MetricType.COUNTER, start_time=start_time
        )
        assert len(filtered_metrics) == 0

    def test_metric_statistics(self):
        """Test metric statistical calculations."""
        # Record some test data
        values = [10, 20, 30, 40, 50]
        for value in values:
            self.collector.record_metric("test_stats", value, MetricType.GAUGE)
            time.sleep(0.001)  # Small delay to ensure different timestamps

        stats = self.collector.get_metric_statistics("test_stats", MetricType.GAUGE, minutes=1)

        assert stats["count"] == 5
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["mean"] == 30
        assert stats["median"] == 30

    def test_metric_cleanup(self):
        """Test automatic cleanup of old metrics."""
        collector = MetricCollector(retention_hours=0.001)  # Very short retention

        collector.record_metric("test_cleanup", 1, MetricType.COUNTER)
        assert len(collector.get_metrics("test_cleanup", MetricType.COUNTER)) == 1

        time.sleep(0.01)  # Wait for retention period to pass

        # Record new metric, which should trigger cleanup
        collector.record_metric("test_cleanup", 2, MetricType.COUNTER)

        # Should have cleaned up the old metric
        metrics = collector.get_metrics("test_cleanup", MetricType.COUNTER)
        assert len(metrics) == 1
        assert metrics[0].value == 2

    def test_metric_labels_and_metadata(self):
        """Test metric labels and metadata functionality."""
        labels = {"service": "api", "endpoint": "/users"}
        metadata = {"server": "web-01", "region": "us-east"}

        self.collector.record_metric(
            "request_count",
            1,
            MetricType.COUNTER,
            labels=labels,
            metadata=metadata
        )

        metrics = self.collector.get_metrics("request_count", MetricType.COUNTER)
        assert len(metrics) == 1

        metric = metrics[0]
        assert metric.labels == labels
        assert metric.metadata == metadata


class TestAlertManager(unittest.TestCase):
    """Test alert management functionality."""

    def setUp(self):
        """Set up test environment."""
        self.alert_manager = AlertManager()
        self.metric_collector = MetricCollector()

    def test_create_alert_rule(self):
        """Test creating and evaluating alert rules."""
        # Add an alert rule
        self.alert_manager.add_alert_rule(
            name="high_memory",
            condition=lambda collector: collector.get_latest_value("memory", MetricType.GAUGE) or 0 > 100,
            level=AlertLevel.WARNING,
            title="High Memory Usage",
            message_template="Memory usage is too high",
            cooldown_minutes=5
        )

        assert len(self.alert_manager.alert_rules) == 1

    def test_evaluate_alerts(self):
        """Test alert evaluation and triggering."""
        # Add alert rule that should trigger
        self.alert_manager.add_alert_rule(
            name="test_alert",
            condition=lambda collector: True,  # Always trigger
            level=AlertLevel.CRITICAL,
            title="Test Alert",
            message_template="This is a test alert",
            cooldown_minutes=1
        )

        # Evaluate alerts
        new_alerts = self.alert_manager.evaluate_alerts(self.metric_collector)

        assert len(new_alerts) == 1
        assert new_alerts[0].level == AlertLevel.CRITICAL
        assert new_alerts[0].title == "Test Alert"
        assert not new_alerts[0].resolved

    def test_alert_cooldown(self):
        """Test alert cooldown functionality."""
        # Add alert rule with short cooldown
        self.alert_manager.add_alert_rule(
            name="cooldown_test",
            condition=lambda collector: True,  # Always trigger
            level=AlertLevel.INFO,
            title="Cooldown Test",
            message_template="Testing cooldown",
            cooldown_minutes=0.01  # Very short cooldown (0.6 seconds)
        )

        # First evaluation should trigger alert
        alerts1 = self.alert_manager.evaluate_alerts(self.metric_collector)
        assert len(alerts1) == 1

        # Second evaluation immediately should not trigger (cooldown)
        alerts2 = self.alert_manager.evaluate_alerts(self.metric_collector)
        assert len(alerts2) == 0

        # Wait for cooldown and try again
        time.sleep(0.7)
        alerts3 = self.alert_manager.evaluate_alerts(self.metric_collector)
        assert len(alerts3) == 1

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        # Create alert manually
        self.alert_manager._create_alert(
            level=AlertLevel.WARNING,
            title="Test Alert 1",
            message="Active alert",
            source="test"
        )

        active_alerts = self.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1

    def test_resolve_alert(self):
        """Test resolving alerts."""
        alert = self.alert_manager._create_alert(
            level=AlertLevel.INFO,
            title="Test Alert",
            message="To be resolved",
            source="test"
        )

        assert not alert.resolved

        # Resolve the alert
        success = self.alert_manager.resolve_alert(alert.id)
        assert success
        assert alert.resolved
        assert alert.resolved_at is not None

    def test_alert_summary(self):
        """Test alert summary generation."""
        # Create alerts of different levels
        self.alert_manager._create_alert(AlertLevel.CRITICAL, "Critical 1", "Message", "test")
        self.alert_manager._create_alert(AlertLevel.WARNING, "Warning 1", "Message", "test")
        self.alert_manager._create_alert(AlertLevel.INFO, "Info 1", "Message", "test")

        # Resolve one alert
        alerts = list(self.alert_manager.alerts.values())
        self.alert_manager.resolve_alert(alerts[0].id)

        summary = self.alert_manager.get_alert_summary()

        assert summary["total_alerts"] == 3
        assert summary["active_alerts"] == 2
        assert summary["resolved_alerts"] == 1
        assert summary["by_level"]["critical"] == 0  # The critical one was resolved


class TestDashboard(unittest.TestCase):
    """Test monitoring dashboard functionality."""

    def setUp(self):
        """Set up test environment."""
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.dashboard = Dashboard(self.metric_collector, self.alert_manager)

    def test_get_dashboard_data(self):
        """Test dashboard data generation."""
        # Add some test data
        self.metric_collector.record_metric("cpu_usage", 45.5, MetricType.GAUGE)
        self.metric_collector.record_metric("memory_usage", 512, MetricType.GAUGE)

        dashboard_data = self.dashboard.get_dashboard_data()

        assert "timestamp" in dashboard_data
        assert "system_health" in dashboard_data
        assert "metrics" in dashboard_data
        assert "alerts" in dashboard_data
        assert "trends" in dashboard_data
        assert "uptime_info" in dashboard_data

    def test_system_health_assessment(self):
        """Test system health status assessment."""
        # Test healthy status (no alerts)
        health = self.dashboard._get_system_health_overview()
        assert health["status"] == "healthy"

        # Add critical alert
        self.alert_manager._create_alert(
            AlertLevel.CRITICAL, "Critical Issue", "Test", "test"
        )

        # Should now show critical status
        health = self.dashboard._get_system_health_overview()
        assert health["status"] == "critical"

    def test_text_dashboard_generation(self):
        """Test text dashboard generation for CLI display."""
        # Add some test metrics and alerts
        self.metric_collector.record_metric("command_duration", 1.5, MetricType.TIMER)
        self.alert_manager._create_alert(
            AlertLevel.WARNING, "Test Warning", "Test message", "test"
        )

        text_dashboard = self.dashboard.generate_text_dashboard()

        assert isinstance(text_dashboard, str)
        assert "SYSTEM MONITORING DASHBOARD" in text_dashboard
        assert "SYSTEM HEALTH" in text_dashboard
        assert "ACTIVE ALERTS" in text_dashboard

    def test_performance_trends(self):
        """Test performance trend analysis."""
        # Record metrics to simulate trends
        datetime.now(timezone.utc)

        # Simulate increasing memory usage trend
        for i in range(10):
            self.metric_collector.record_metric("memory_usage", 100 + i * 10, MetricType.GAUGE)
            time.sleep(0.001)

        trends = self.dashboard._get_performance_trends()

        # Should detect trends (though implementation is simplified in this test)
        assert isinstance(trends, dict)


class TestMonitoringSystem(unittest.TestCase):
    """Test complete monitoring system integration."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.monitoring_system = MonitoringSystem(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        self.monitoring_system.stop_monitoring()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test monitoring system initialization."""
        assert self.monitoring_system.metric_collector is not None
        assert self.monitoring_system.alert_manager is not None
        assert self.monitoring_system.dashboard is not None
        assert self.monitoring_system.storage_dir.exists()

    def test_start_stop_monitoring(self):
        """Test starting and stopping background monitoring."""
        assert self.monitoring_system._monitoring_thread is None

        # Start monitoring
        self.monitoring_system.start_monitoring()
        assert self.monitoring_system._monitoring_thread is not None
        assert self.monitoring_system._monitoring_thread.is_alive()

        # Stop monitoring
        self.monitoring_system.stop_monitoring()
        time.sleep(0.1)  # Give thread time to stop
        assert not self.monitoring_system._monitoring_thread.is_alive()

    def test_default_alert_rules(self):
        """Test that default alert rules are configured."""
        rule_names = [rule["name"] for rule in self.monitoring_system.alert_manager.alert_rules]

        expected_rules = ["high_memory_usage", "high_error_rate", "slow_response_time"]
        for rule_name in expected_rules:
            assert rule_name in rule_names

    def test_record_command_metric(self):
        """Test recording command execution metrics."""
        self.monitoring_system.record_command_metric("test_command", 1.5, True)

        # Check that metrics were recorded
        duration = self.monitoring_system.metric_collector.get_latest_value(
            "command_duration", MetricType.TIMER
        )
        assert duration == 1.5

        count = self.monitoring_system.metric_collector.get_latest_value(
            "command_count", MetricType.COUNTER
        )
        assert count == 1

    def test_export_metrics(self):
        """Test exporting metrics to JSON file."""
        # Record some test metrics
        self.monitoring_system.record_command_metric("export_test", 0.5, True)

        # Export metrics
        filepath = self.monitoring_system.export_metrics("test_export.json")

        assert filepath.exists()

        # Verify export content
        with open(filepath) as f:
            export_data = json.load(f)

        assert "timestamp" in export_data
        assert "metrics" in export_data
        assert "alerts" in export_data

    @patch('vexy_co_model_catalog.core.monitoring.psutil')
    def test_system_metrics_collection(self, mock_psutil):
        """Test automatic system metrics collection."""
        # Mock psutil for testing
        mock_psutil.virtual_memory.return_value.percent = 75.0
        mock_psutil.cpu_percent.return_value = 45.0
        mock_psutil.disk_usage.return_value.total = 1000000
        mock_psutil.disk_usage.return_value.used = 500000

        # Collect metrics
        self.monitoring_system._collect_system_metrics()

        # Verify metrics were collected
        memory_metric = self.monitoring_system.metric_collector.get_latest_value(
            "system_memory_percent", MetricType.GAUGE
        )
        assert memory_metric == 75.0

    def test_dashboard_integration(self):
        """Test dashboard integration with monitoring system."""
        dashboard_data = self.monitoring_system.get_dashboard_data()

        assert "timestamp" in dashboard_data
        assert "system_health" in dashboard_data

        # Test text dashboard
        text_dashboard = self.monitoring_system.get_text_dashboard()
        assert isinstance(text_dashboard, str)
        assert "SYSTEM MONITORING DASHBOARD" in text_dashboard


class TestMonitoringIntegration(unittest.TestCase):
    """Integration tests for monitoring system."""

    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up integration test environment."""
        shutdown_monitoring()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_global_monitoring_system(self):
        """Test global monitoring system management."""
        # Get global monitoring system
        system1 = get_monitoring_system(self.temp_dir)
        assert isinstance(system1, MonitoringSystem)

        # Should return same instance
        system2 = get_monitoring_system()
        assert system1 is system2

    def test_initialize_and_shutdown(self):
        """Test system initialization and shutdown."""
        # Initialize monitoring
        system = initialize_monitoring()
        assert isinstance(system, MonitoringSystem)

        # Should have started background monitoring
        assert system._monitoring_thread.is_alive()

        # Shutdown monitoring
        shutdown_monitoring()
        time.sleep(0.1)  # Give thread time to stop

        # Thread should be stopped
        if system._monitoring_thread:
            assert not system._monitoring_thread.is_alive()

    def test_concurrent_metric_collection(self):
        """Test metric collection under concurrent load."""
        system = get_monitoring_system(self.temp_dir)
        results = []

        def worker_thread(worker_id):
            for i in range(10):
                system.record_command_metric(f"worker_{worker_id}_cmd_{i}", 0.1, True)
                time.sleep(0.001)
            results.append(worker_id)

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify all workers completed
        assert len(results) == 3

        # Verify metrics were collected
        metric_names = system.metric_collector.get_metric_names()
        assert "command_duration" in metric_names
        assert "command_count" in metric_names

    def test_alert_evaluation_under_load(self):
        """Test alert evaluation with high metric load."""
        system = get_monitoring_system(self.temp_dir)

        # Add alert rule that should trigger
        system.alert_manager.add_alert_rule(
            name="load_test_alert",
            condition=lambda collector: len(collector.get_metric_names()) > 2,
            level=AlertLevel.INFO,
            title="Load Test Alert",
            message_template="Metrics under load",
            cooldown_minutes=0.01
        )

        # Generate metrics rapidly
        for i in range(50):
            system.record_command_metric(f"load_test_{i}", 0.01, True)

        # Evaluate alerts
        new_alerts = system.alert_manager.evaluate_alerts(system.metric_collector)

        # Should have generated at least one alert
        assert len(new_alerts) > 0


@pytest.mark.integration
class TestMonitoringCLI(unittest.TestCase):
    """Integration tests for monitoring CLI commands."""

    def setUp(self):
        """Set up CLI test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up CLI test environment."""
        shutdown_monitoring()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dashboard_generation(self):
        """Test CLI dashboard generation."""
        system = get_monitoring_system(self.temp_dir)

        # Add test data
        system.record_command_metric("cli_test", 1.0, True)
        system.alert_manager._create_alert(
            AlertLevel.WARNING, "CLI Test Alert", "Test message", "cli_test"
        )

        # Generate dashboard
        dashboard_text = system.get_text_dashboard()

        assert "SYSTEM MONITORING DASHBOARD" in dashboard_text
        assert "SYSTEM HEALTH" in dashboard_text
        assert "ACTIVE ALERTS" in dashboard_text

    def test_metrics_export_functionality(self):
        """Test metrics export functionality."""
        system = get_monitoring_system(self.temp_dir)

        # Record test metrics
        for i in range(5):
            system.record_command_metric(f"export_test_{i}", 0.5 + i * 0.1, i % 2 == 0)

        # Export metrics
        export_path = system.export_metrics("cli_test_export.json")

        # Verify export
        assert export_path.exists()

        with open(export_path) as f:
            data = json.load(f)

        assert "metrics" in data
        assert "command_duration" in data["metrics"]

    def test_alert_management_cli(self):
        """Test alert management through CLI interface."""
        system = get_monitoring_system(self.temp_dir)

        # Create test alerts
        system.alert_manager._create_alert(
            AlertLevel.CRITICAL, "Critical Test", "Critical message", "test"
        )
        system.alert_manager._create_alert(
            AlertLevel.WARNING, "Warning Test", "Warning message", "test"
        )

        # Get alert summary
        summary = system.alert_manager.get_alert_summary()

        assert summary["total_alerts"] == 2
        assert summary["active_alerts"] == 2
        assert summary["by_level"]["critical"] == 1
        assert summary["by_level"]["warning"] == 1


if __name__ == "__main__":
    unittest.main()
