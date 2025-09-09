# this_file: src/vexy_co_model_catalog/core/monitoring.py

"""
Sophisticated monitoring capabilities with metrics dashboards and operational intelligence.
Provides real-time monitoring, alert systems, metric collection, and dashboard generation.
"""

from __future__ import annotations

import json
import statistics
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable


class AlertLevel(Enum):
    """Alert severity levels for monitoring system."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"           # Ever-increasing value
    GAUGE = "gauge"              # Point-in-time value
    HISTOGRAM = "histogram"      # Distribution of values
    TIMER = "timer"              # Duration measurements
    RATE = "rate"                # Events per time unit


# Monitoring thresholds and constants
ALERT_THRESHOLD_WARNING_COUNT = 5       # Number of active alerts to trigger warning status
MEMORY_USAGE_WARNING_MB = 500           # Memory usage threshold in MB for warning
ERROR_RATE_CRITICAL_THRESHOLD = 0.1     # Error rate threshold (10%) for critical alert
RESPONSE_TIME_WARNING_SECONDS = 5.0     # Response time threshold in seconds for warning


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels,
            "metadata": self.metadata,
        }


@dataclass
class Alert:
    """Monitoring alert with metadata."""
    id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source: str
    resolved: bool = False
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now(timezone.utc)


class MetricCollector:
    """Collects and stores metrics with time-series data."""

    def __init__(self, retention_hours: int = 24) -> None:
        """Initialize metric collector.

        Args:
            retention_hours: How long to keep metrics in memory
        """
        self.retention_hours = retention_hours
        self.metrics: dict[str, dict[MetricType, deque]] = defaultdict(lambda: defaultdict(deque))
        self._lock = threading.RLock()

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a metric data point."""
        point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )

        with self._lock:
            self.metrics[name][metric_type].append(point)
            self._cleanup_old_metrics(name, metric_type)

    def _cleanup_old_metrics(self, name: str, metric_type: MetricType) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        metric_queue = self.metrics[name][metric_type]

        # Remove old entries from the front of deque
        while metric_queue and metric_queue[0].timestamp < cutoff_time:
            metric_queue.popleft()

    def get_metrics(
        self,
        name: str,
        metric_type: MetricType,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> list[MetricPoint]:
        """Get metrics for a specific name and type within time range."""
        with self._lock:
            if name not in self.metrics or metric_type not in self.metrics[name]:
                return []

            points = list(self.metrics[name][metric_type])

            # Filter by time range
            if start_time or end_time:
                filtered_points = []
                for point in points:
                    if start_time and point.timestamp < start_time:
                        continue
                    if end_time and point.timestamp > end_time:
                        continue
                    filtered_points.append(point)
                return filtered_points

            return points

    def get_metric_names(self) -> list[str]:
        """Get all metric names currently being collected."""
        with self._lock:
            return list(self.metrics.keys())

    def get_latest_value(self, name: str, metric_type: MetricType) -> float | None:
        """Get the most recent value for a metric."""
        points = self.get_metrics(name, metric_type)
        return points[-1].value if points else None

    def get_metric_statistics(
        self,
        name: str,
        metric_type: MetricType,
        minutes: int = 60
    ) -> dict[str, float]:
        """Get statistical summary for a metric over the last N minutes."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=minutes)

        points = self.get_metrics(name, metric_type, start_time, end_time)
        if not points:
            return {}

        values = [p.value for p in points]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99),
        }

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile
        f = int(k)
        c = k - f
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c


class AlertManager:
    """Manages monitoring alerts and notifications."""

    def __init__(self) -> None:
        """Initialize alert manager."""
        self.alerts: dict[str, Alert] = {}
        self.alert_rules: list[dict] = []
        self._lock = threading.RLock()
        self._alert_counter = 0

    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[MetricCollector], bool],
        level: AlertLevel,
        title: str,
        message_template: str,
        cooldown_minutes: int = 15
    ) -> None:
        """Add an alert rule that will be evaluated periodically."""
        rule = {
            "name": name,
            "condition": condition,
            "level": level,
            "title": title,
            "message_template": message_template,
            "cooldown_minutes": cooldown_minutes,
            "last_triggered": None,
        }
        self.alert_rules.append(rule)

    def evaluate_alerts(self, metric_collector: MetricCollector) -> list[Alert]:
        """Evaluate all alert rules and generate new alerts if needed."""
        new_alerts = []
        current_time = datetime.now(timezone.utc)

        for rule in self.alert_rules:
            # Check cooldown period
            if rule["last_triggered"]:
                cooldown_end = rule["last_triggered"] + timedelta(minutes=rule["cooldown_minutes"])
                if current_time < cooldown_end:
                    continue

            # Evaluate condition
            try:
                if rule["condition"](metric_collector):
                    alert = self._create_alert(
                        level=rule["level"],
                        title=rule["title"],
                        message=rule["message_template"],
                        source=rule["name"]
                    )
                    new_alerts.append(alert)
                    rule["last_triggered"] = current_time
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule['name']}: {e}")

        return new_alerts

    def _create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str,
        metadata: dict | None = None
    ) -> Alert:
        """Create a new alert."""
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter:06d}"

            alert = Alert(
                id=alert_id,
                level=level,
                title=title,
                message=message,
                timestamp=datetime.now(timezone.utc),
                source=source,
                metadata=metadata or {}
            )

            self.alerts[alert_id] = alert
            return alert

    def get_active_alerts(self) -> list[Alert]:
        """Get all unresolved alerts."""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]

    def get_alerts_by_level(self, level: AlertLevel) -> list[Alert]:
        """Get all alerts of a specific level."""
        with self._lock:
            return [alert for alert in self.alerts.values() if alert.level == level]

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID."""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolve()
                return True
            return False

    def get_alert_summary(self) -> dict[str, Any]:
        """Get summary of alerts by level and status."""
        with self._lock:
            active_alerts = self.get_active_alerts()

            summary = {
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "resolved_alerts": len(self.alerts) - len(active_alerts),
                "by_level": {level.value: 0 for level in AlertLevel},
                "recent_alerts": [],
            }

            # Count by level
            for alert in active_alerts:
                summary["by_level"][alert.level.value] += 1

            # Get recent alerts (last 10)
            recent = sorted(self.alerts.values(), key=lambda a: a.timestamp, reverse=True)[:10]
            summary["recent_alerts"] = [alert.to_dict() for alert in recent]

            return summary


class Dashboard:
    """Monitoring dashboard with real-time metrics visualization."""

    def __init__(self, metric_collector: MetricCollector, alert_manager: AlertManager) -> None:
        """Initialize dashboard."""
        self.metric_collector = metric_collector
        self.alert_manager = alert_manager

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get complete dashboard data for visualization."""
        current_time = datetime.now(timezone.utc)

        # System health overview
        system_health = self._get_system_health_overview()

        # Metric summaries
        metric_summaries = self._get_metric_summaries()

        # Alert summary
        alert_summary = self.alert_manager.get_alert_summary()

        # Performance trends
        performance_trends = self._get_performance_trends()

        return {
            "timestamp": current_time.isoformat(),
            "system_health": system_health,
            "metrics": metric_summaries,
            "alerts": alert_summary,
            "trends": performance_trends,
            "uptime_info": self._get_uptime_info(),
        }

    def _get_system_health_overview(self) -> dict[str, Any]:
        """Get high-level system health indicators."""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]

        # Determine overall health status
        if critical_alerts:
            health_status = "critical"
        elif len(active_alerts) > ALERT_THRESHOLD_WARNING_COUNT:
            health_status = "warning"
        elif active_alerts:
            health_status = "degraded"
        else:
            health_status = "healthy"

        return {
            "status": health_status,
            "active_alerts": len(active_alerts),
            "critical_issues": len(critical_alerts),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

    def _get_metric_summaries(self) -> dict[str, Any]:
        """Get summaries for all collected metrics."""
        summaries = {}

        for metric_name in self.metric_collector.get_metric_names():
            for metric_type in MetricType:
                stats = self.metric_collector.get_metric_statistics(metric_name, metric_type, minutes=30)
                if stats:
                    key = f"{metric_name}_{metric_type.value}"
                    summaries[key] = {
                        "current_value": self.metric_collector.get_latest_value(metric_name, metric_type),
                        "statistics": stats,
                        "metric_type": metric_type.value,
                    }

        return summaries

    def _get_performance_trends(self) -> dict[str, Any]:
        """Get performance trend analysis."""
        trends = {}

        # Analyze key performance metrics
        key_metrics = [
            ("command_duration", MetricType.TIMER),
            ("memory_usage", MetricType.GAUGE),
            ("error_rate", MetricType.RATE),
        ]

        for metric_name, metric_type in key_metrics:
            recent_stats = self.metric_collector.get_metric_statistics(metric_name, metric_type, minutes=30)
            older_stats = self.metric_collector.get_metric_statistics(metric_name, metric_type, minutes=60)

            if recent_stats and older_stats:
                trend = "stable"
                change_pct = 0.0

                if recent_stats["mean"] > older_stats["mean"] * 1.1:
                    trend = "increasing"
                    change_pct = ((recent_stats["mean"] - older_stats["mean"]) / older_stats["mean"]) * 100
                elif recent_stats["mean"] < older_stats["mean"] * 0.9:
                    trend = "decreasing"
                    change_pct = ((older_stats["mean"] - recent_stats["mean"]) / older_stats["mean"]) * 100

                trends[metric_name] = {
                    "trend": trend,
                    "change_percent": change_pct,
                    "current_value": recent_stats["mean"],
                    "previous_value": older_stats["mean"],
                }

        return trends

    def _get_uptime_info(self) -> dict[str, Any]:
        """Get system uptime and availability information."""
        # This would integrate with the actual system start time
        # For now, return basic placeholder information
        return {
            "uptime_hours": 24.0,  # Would be calculated from actual start time
            "availability_percent": 99.9,
            "last_restart": (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
        }

    def generate_text_dashboard(self) -> str:
        """Generate text-based dashboard for CLI display."""
        dashboard_data = self.get_dashboard_data()

        lines = [
            "🖥️  SYSTEM MONITORING DASHBOARD",
            "=" * 50,
            f"⏰ Last Updated: {dashboard_data['timestamp'][:19]}",
            "",
            "📊 SYSTEM HEALTH",
            f"   Status: {dashboard_data['system_health']['status'].upper()}",
            f"   Active Alerts: {dashboard_data['system_health']['active_alerts']}",
            f"   Critical Issues: {dashboard_data['system_health']['critical_issues']}",
            "",
            "📈 PERFORMANCE TRENDS",
        ]

        for metric_name, trend_data in dashboard_data["trends"].items():
            trend_icon = {"increasing": "📈", "decreasing": "📉", "stable": "➖"}.get(trend_data["trend"], "➖")
            lines.append(f"   {trend_icon} {metric_name}: {trend_data['trend']} ({trend_data['change_percent']:+.1f}%)")

        if dashboard_data["alerts"]["active_alerts"] > 0:
            lines.extend([
                "",
                "🚨 ACTIVE ALERTS",
            ])

            for alert_data in dashboard_data["alerts"]["recent_alerts"][:5]:
                if not alert_data["resolved"]:
                    level_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵", "emergency": "🚨"}.get(alert_data["level"], "⚪")
                    lines.append(f"   {level_icon} {alert_data['title']}")

        lines.extend([
            "",
            f"⏱️  Uptime: {dashboard_data['uptime_info']['uptime_hours']:.1f} hours",
            f"📊 Availability: {dashboard_data['uptime_info']['availability_percent']:.2f}%",
            "",
            "Use 'vexy monitor alerts' for detailed alert information",
            "Use 'vexy monitor metrics' for detailed metric information",
        ])

        return "\n".join(lines)


class MonitoringSystem:
    """Main monitoring system that coordinates all monitoring components."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        """Initialize monitoring system."""
        self.storage_dir = storage_dir or Path.cwd() / "monitoring"
        self.storage_dir.mkdir(exist_ok=True)

        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.dashboard = Dashboard(self.metric_collector, self.alert_manager)

        self._monitoring_thread: threading.Thread | None = None
        self._stop_monitoring = threading.Event()

        self._setup_default_alerts()

    def _setup_default_alerts(self) -> None:
        """Set up default monitoring alerts."""
        # High memory usage alert
        self.alert_manager.add_alert_rule(
            name="high_memory_usage",
            condition=lambda collector: (
                collector.get_latest_value("memory_usage", MetricType.GAUGE) or 0
            ) > MEMORY_USAGE_WARNING_MB,
            level=AlertLevel.WARNING,
            title="High Memory Usage",
            message_template="Memory usage is above 500MB",
            cooldown_minutes=10
        )

        # High error rate alert
        self.alert_manager.add_alert_rule(
            name="high_error_rate",
            condition=lambda collector: (
                collector.get_latest_value("error_rate", MetricType.RATE) or 0
            ) > ERROR_RATE_CRITICAL_THRESHOLD,
            level=AlertLevel.CRITICAL,
            title="High Error Rate",
            message_template="Error rate is above 10%",
            cooldown_minutes=5
        )

        # Slow response time alert
        self.alert_manager.add_alert_rule(
            name="slow_response_time",
            condition=lambda collector: (
                collector.get_latest_value("response_time", MetricType.TIMER) or 0
            ) > RESPONSE_TIME_WARNING_SECONDS,
            level=AlertLevel.WARNING,
            title="Slow Response Time",
            message_template="Response time is above 5 seconds",
            cooldown_minutes=15
        )

    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring already started")
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Monitoring system started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Monitoring system stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs in background thread."""
        while not self._stop_monitoring.wait(30.0):  # Check every 30 seconds
            try:
                # Evaluate alert rules
                new_alerts = self.alert_manager.evaluate_alerts(self.metric_collector)

                # Log new alerts
                for alert in new_alerts:
                    level_name = alert.level.value.upper()
                    logger.warning(f"ALERT [{level_name}] {alert.title}: {alert.message}")

                # Collect system metrics
                self._collect_system_metrics()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def _collect_system_metrics(self) -> None:
        """Collect basic system metrics."""
        try:
            import psutil

            # Memory usage
            memory = psutil.virtual_memory()
            self.metric_collector.record_metric(
                "system_memory_percent",
                memory.percent,
                MetricType.GAUGE
            )

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metric_collector.record_metric(
                "system_cpu_percent",
                cpu_percent,
                MetricType.GAUGE
            )

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metric_collector.record_metric(
                "system_disk_percent",
                disk_percent,
                MetricType.GAUGE
            )

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def record_command_metric(self, command: str, duration: float, success: bool) -> None:
        """Record metrics for a command execution."""
        # Record command duration
        self.metric_collector.record_metric(
            "command_duration",
            duration,
            MetricType.TIMER,
            labels={"command": command, "success": str(success)}
        )

        # Record success/failure counter
        self.metric_collector.record_metric(
            "command_count",
            1,
            MetricType.COUNTER,
            labels={"command": command, "success": str(success)}
        )

        # Update error rate if command failed
        if not success:
            self.metric_collector.record_metric(
                "error_rate",
                1.0,
                MetricType.RATE,
                labels={"command": command}
            )

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard.get_dashboard_data()

    def get_text_dashboard(self) -> str:
        """Get text-based dashboard."""
        return self.dashboard.generate_text_dashboard()

    def export_metrics(self, filename: str | None = None) -> Path:
        """Export all metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json"

        filepath = self.storage_dir / filename

        export_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
            "alerts": self.alert_manager.get_alert_summary(),
        }

        # Export all metrics
        for metric_name in self.metric_collector.get_metric_names():
            export_data["metrics"][metric_name] = {}
            for metric_type in MetricType:
                points = self.metric_collector.get_metrics(metric_name, metric_type)
                if points:
                    export_data["metrics"][metric_name][metric_type.value] = [
                        point.to_dict() for point in points
                    ]

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Metrics exported to {filepath}")
        return filepath


# Global monitoring system instance
_global_monitoring_system: MonitoringSystem | None = None


def get_monitoring_system(storage_dir: Path | None = None) -> MonitoringSystem:
    """Get or create the global monitoring system instance."""
    global _global_monitoring_system
    if _global_monitoring_system is None:
        _global_monitoring_system = MonitoringSystem(storage_dir)
    return _global_monitoring_system


def initialize_monitoring() -> MonitoringSystem:
    """Initialize and start the monitoring system."""
    monitoring = get_monitoring_system()
    monitoring.start_monitoring()
    logger.info("Monitoring system initialized and started")
    return monitoring


def shutdown_monitoring() -> None:
    """Shutdown the monitoring system."""
    global _global_monitoring_system
    if _global_monitoring_system:
        _global_monitoring_system.stop_monitoring()
        _global_monitoring_system = None
    logger.info("Monitoring system shutdown")
