"""
this_file: src/vexy_co_model_catalog/core/health_monitor.py

Comprehensive system health monitoring and self-diagnostics.
Provides automated health checks, issue detection, and actionable recommendations.
"""

from __future__ import annotations

import asyncio
import os
import platform
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import psutil
from loguru import logger

from vexy_co_model_catalog.core.failure_tracker import FailureTracker
from vexy_co_model_catalog.core.rate_limiter import get_rate_limiter
from vexy_co_model_catalog.core.storage import StorageManager


class HealthStatus(Enum):
    """Health status levels for different components."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components to monitor."""

    SYSTEM = "system"
    STORAGE = "storage"
    NETWORK = "network"
    PROVIDER = "provider"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"


@dataclass
class HealthIssue:
    """Represents a detected health issue."""

    component: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    details: str | None = None
    recommendations: list[str] = field(default_factory=list)
    metric_values: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "component": self.component,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "recommendations": self.recommendations,
            "metric_values": self.metric_values,
            "timestamp": self.timestamp,
        }


@dataclass
class SystemMetrics:
    """System performance and resource metrics."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    available_memory_mb: float = 0.0
    disk_free_gb: float = 0.0
    python_version: str = ""
    platform_info: str = ""
    uptime_seconds: float = 0.0

    # Network metrics
    network_connectivity: bool = False
    dns_resolution: bool = False
    avg_response_time_ms: float = 0.0

    # Application metrics
    total_providers: int = 0
    healthy_providers: int = 0
    rate_limited_providers: int = 0
    failed_requests: int = 0
    successful_requests: int = 0


class SystemHealthMonitor:
    """Comprehensive system health monitoring and diagnostics."""

    def __init__(self, storage_root: Path) -> None:
        """Initialize health monitor with storage path."""
        self.storage_root = Path(storage_root)
        self.issues: list[HealthIssue] = []
        self.metrics = SystemMetrics()
        self.last_check_time = 0.0

        # Health check configuration
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 85.0,
            "memory_critical": 95.0,
            "disk_warning": 90.0,
            "disk_critical": 98.0,
            "response_time_warning": 5000.0,  # 5 seconds
            "response_time_critical": 15000.0,  # 15 seconds
            "min_free_disk_gb": 1.0,
            "min_free_memory_mb": 100.0,
        }

    async def run_comprehensive_health_check(self) -> dict[str, Any]:
        """Run complete health check and return detailed report."""
        start_time = time.time()
        self.issues.clear()

        logger.info("Starting comprehensive system health check...")

        # Run all health checks
        await asyncio.gather(
            self._check_system_resources(),
            self._check_storage_health(),
            self._check_network_connectivity(),
            self._check_configuration_health(),
            self._check_dependencies(),
            self._check_provider_health(),
            return_exceptions=True,
        )

        # Calculate overall health status
        overall_status = self._calculate_overall_health()

        # Generate health report
        execution_time = (time.time() - start_time) * 1000
        self.last_check_time = time.time()

        report = {
            "overall_status": overall_status.value,
            "check_timestamp": self.last_check_time,
            "execution_time_ms": execution_time,
            "total_issues": len(self.issues),
            "critical_issues": len([i for i in self.issues if i.status == HealthStatus.CRITICAL]),
            "warning_issues": len([i for i in self.issues if i.status == HealthStatus.WARNING]),
            "system_metrics": self.metrics.__dict__,
            "issues_by_component": self._group_issues_by_component(),
            "recommendations": self._generate_prioritized_recommendations(),
            "quick_fixes": self._generate_quick_fixes(),
        }

        logger.info(f"Health check completed in {execution_time:.1f}ms - Status: {overall_status.value}")
        return report

    async def _check_system_resources(self) -> None:
        """Check system CPU, memory, and disk usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.cpu_percent = cpu_percent

            if cpu_percent > self.thresholds["cpu_critical"]:
                self.issues.append(
                    HealthIssue(
                        component="cpu",
                        component_type=ComponentType.SYSTEM,
                        status=HealthStatus.CRITICAL,
                        message=f"Critical CPU usage: {cpu_percent:.1f}%",
                        details="High CPU usage may impact performance and cause request timeouts",
                        recommendations=[
                            "Reduce concurrent provider requests (--max-concurrency flag)",
                            "Close other resource-intensive applications",
                            "Consider running during off-peak hours",
                        ],
                        metric_values={"cpu_percent": cpu_percent},
                    )
                )
            elif cpu_percent > self.thresholds["cpu_warning"]:
                self.issues.append(
                    HealthIssue(
                        component="cpu",
                        component_type=ComponentType.SYSTEM,
                        status=HealthStatus.WARNING,
                        message=f"High CPU usage: {cpu_percent:.1f}%",
                        recommendations=["Monitor system performance", "Consider reducing concurrency"],
                        metric_values={"cpu_percent": cpu_percent},
                    )
                )

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.memory_percent = memory.percent
            self.metrics.available_memory_mb = memory.available / 1024 / 1024

            if memory.percent > self.thresholds["memory_critical"]:
                self.issues.append(
                    HealthIssue(
                        component="memory",
                        component_type=ComponentType.SYSTEM,
                        status=HealthStatus.CRITICAL,
                        message=f"Critical memory usage: {memory.percent:.1f}%",
                        details=f"Only {self.metrics.available_memory_mb:.0f}MB available",
                        recommendations=[
                            "Restart the application to free memory",
                            "Reduce batch size for large operations",
                            "Close other applications to free memory",
                        ],
                        metric_values={
                            "memory_percent": memory.percent,
                            "available_mb": self.metrics.available_memory_mb,
                        },
                    )
                )
            elif memory.percent > self.thresholds["memory_warning"]:
                self.issues.append(
                    HealthIssue(
                        component="memory",
                        component_type=ComponentType.SYSTEM,
                        status=HealthStatus.WARNING,
                        message=f"High memory usage: {memory.percent:.1f}%",
                        recommendations=["Monitor memory usage", "Consider restarting if usage increases"],
                        metric_values={"memory_percent": memory.percent},
                    )
                )

            # Disk usage
            disk = psutil.disk_usage(str(self.storage_root))
            self.metrics.disk_usage_percent = (disk.used / disk.total) * 100
            self.metrics.disk_free_gb = disk.free / 1024 / 1024 / 1024

            if self.metrics.disk_usage_percent > self.thresholds["disk_critical"]:
                self.issues.append(
                    HealthIssue(
                        component="disk",
                        component_type=ComponentType.STORAGE,
                        status=HealthStatus.CRITICAL,
                        message=f"Critical disk usage: {self.metrics.disk_usage_percent:.1f}%",
                        details=f"Only {self.metrics.disk_free_gb:.1f}GB free",
                        recommendations=[
                            "Clean up old model catalog files",
                            "Use 'vexy clean --temp' to remove temporary files",
                            "Consider moving data to external storage",
                        ],
                        metric_values={
                            "disk_percent": self.metrics.disk_usage_percent,
                            "free_gb": self.metrics.disk_free_gb,
                        },
                    )
                )
            elif self.metrics.disk_free_gb < self.thresholds["min_free_disk_gb"]:
                self.issues.append(
                    HealthIssue(
                        component="disk",
                        component_type=ComponentType.STORAGE,
                        status=HealthStatus.WARNING,
                        message=f"Low disk space: {self.metrics.disk_free_gb:.1f}GB free",
                        recommendations=["Clean up old files", "Monitor disk usage"],
                        metric_values={"free_gb": self.metrics.disk_free_gb},
                    )
                )

            # System info
            self.metrics.python_version = f"{platform.python_version()}"
            self.metrics.platform_info = f"{platform.system()} {platform.release()}"

        except Exception as e:
            self.issues.append(
                HealthIssue(
                    component="system_resources",
                    component_type=ComponentType.SYSTEM,
                    status=HealthStatus.WARNING,
                    message=f"Failed to check system resources: {e}",
                    recommendations=["Check system permissions", "Ensure psutil is installed"],
                )
            )

    async def _check_storage_health(self) -> None:
        """Check storage directory structure and permissions."""
        try:
            # Check if storage directories exist and are writable
            required_dirs = [
                self.storage_root / "config",
                self.storage_root / "models",
                self.storage_root / "analytics",
            ]

            for dir_path in required_dirs:
                if not dir_path.exists():
                    self.issues.append(
                        HealthIssue(
                            component="storage",
                            component_type=ComponentType.STORAGE,
                            status=HealthStatus.WARNING,
                            message=f"Missing directory: {dir_path}",
                            recommendations=["Directory will be created automatically when needed"],
                            metric_values={"path": str(dir_path)},
                        )
                    )
                elif not os.access(dir_path, os.W_OK):
                    self.issues.append(
                        HealthIssue(
                            component="storage",
                            component_type=ComponentType.STORAGE,
                            status=HealthStatus.CRITICAL,
                            message=f"No write permission: {dir_path}",
                            recommendations=[
                                f"Fix permissions: chmod 755 {dir_path}",
                                "Check file ownership",
                                "Run as user with appropriate permissions",
                            ],
                            metric_values={"path": str(dir_path)},
                        )
                    )

            # Check for backup directories
            backup_dir = self.storage_root / "config" / "backups"
            if backup_dir.exists():
                backup_files = list(backup_dir.glob("*"))
                if len(backup_files) > 50:
                    self.issues.append(
                        HealthIssue(
                            component="backups",
                            component_type=ComponentType.STORAGE,
                            status=HealthStatus.WARNING,
                            message=f"Many backup files: {len(backup_files)} files",
                            recommendations=[
                                "Clean up old backup files periodically",
                                "Consider automated backup cleanup",
                            ],
                            metric_values={"backup_count": len(backup_files)},
                        )
                    )

        except Exception as e:
            self.issues.append(
                HealthIssue(
                    component="storage",
                    component_type=ComponentType.STORAGE,
                    status=HealthStatus.WARNING,
                    message=f"Storage health check failed: {e}",
                    recommendations=["Check storage permissions and accessibility"],
                )
            )

    async def _check_network_connectivity(self) -> None:
        """Check network connectivity and DNS resolution."""
        try:
            # Test basic connectivity
            test_urls = ["https://api.openai.com", "https://api.anthropic.com", "https://httpbin.org/status/200"]

            successful_connections = 0
            total_response_time = 0.0

            async with httpx.AsyncClient(timeout=10.0) as client:
                for url in test_urls:
                    try:
                        start_time = time.time()
                        response = await client.get(url)
                        response_time = (time.time() - start_time) * 1000
                        total_response_time += response_time

                        if response.status_code in [200, 401, 403]:  # 401/403 means endpoint exists
                            successful_connections += 1

                    except Exception as e:
                        logger.debug(f"Connection test failed for {url}: {e}")

            self.metrics.network_connectivity = successful_connections > 0
            self.metrics.avg_response_time_ms = total_response_time / len(test_urls) if test_urls else 0

            if not self.metrics.network_connectivity:
                self.issues.append(
                    HealthIssue(
                        component="network",
                        component_type=ComponentType.NETWORK,
                        status=HealthStatus.CRITICAL,
                        message="No network connectivity detected",
                        details="Cannot reach any test endpoints",
                        recommendations=[
                            "Check internet connection",
                            "Verify firewall settings",
                            "Check proxy configuration if applicable",
                        ],
                        metric_values={"successful_connections": successful_connections},
                    )
                )
            elif self.metrics.avg_response_time_ms > self.thresholds["response_time_critical"]:
                self.issues.append(
                    HealthIssue(
                        component="network",
                        component_type=ComponentType.NETWORK,
                        status=HealthStatus.CRITICAL,
                        message=f"Very slow network: {self.metrics.avg_response_time_ms:.0f}ms average",
                        recommendations=[
                            "Check network connection quality",
                            "Consider running during better network conditions",
                            "Increase timeout values if needed",
                        ],
                        metric_values={"avg_response_time_ms": self.metrics.avg_response_time_ms},
                    )
                )
            elif self.metrics.avg_response_time_ms > self.thresholds["response_time_warning"]:
                self.issues.append(
                    HealthIssue(
                        component="network",
                        component_type=ComponentType.NETWORK,
                        status=HealthStatus.WARNING,
                        message=f"Slow network: {self.metrics.avg_response_time_ms:.0f}ms average",
                        recommendations=["Monitor network performance"],
                        metric_values={"avg_response_time_ms": self.metrics.avg_response_time_ms},
                    )
                )

        except Exception as e:
            self.issues.append(
                HealthIssue(
                    component="network",
                    component_type=ComponentType.NETWORK,
                    status=HealthStatus.WARNING,
                    message=f"Network check failed: {e}",
                    recommendations=["Check network configuration"],
                )
            )

    async def _check_configuration_health(self) -> None:
        """Check configuration files and environment variables."""
        try:
            # Check for common environment variables
            important_env_vars = [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GROQ_API_KEY",
                "TOGETHER_API_KEY",
                "OPENROUTER_API_KEY",
            ]

            missing_keys = []
            present_keys = []

            for var in important_env_vars:
                if os.environ.get(var):
                    present_keys.append(var)
                else:
                    missing_keys.append(var)

            if len(missing_keys) == len(important_env_vars):
                self.issues.append(
                    HealthIssue(
                        component="api_keys",
                        component_type=ComponentType.CONFIGURATION,
                        status=HealthStatus.WARNING,
                        message="No API keys found in environment",
                        details="Most providers require API keys for access",
                        recommendations=[
                            "Set up API keys for providers you want to use",
                            "Example: export OPENAI_API_KEY='sk-***[YOUR_KEY_HERE]'",
                            "Check provider documentation for key setup",
                        ],
                        metric_values={
                            "missing_count": len(missing_keys),
                            "present_count": len(present_keys),
                            "total_checked": len(important_env_vars),
                        },
                    )
                )
            elif missing_keys:
                self.issues.append(
                    HealthIssue(
                        component="api_keys",
                        component_type=ComponentType.CONFIGURATION,
                        status=HealthStatus.WARNING,
                        message=f"Some API keys missing: {len(missing_keys)} of {len(important_env_vars)}",
                        details=f"Missing {len(missing_keys)} API key(s) for full provider access",
                        recommendations=[
                            "Set up missing API keys for full provider access",
                            "Use 'providers' command to see required environment variables",
                        ],
                        metric_values={
                            "missing_count": len(missing_keys),
                            "present_count": len(present_keys),
                            "total_checked": len(important_env_vars),
                        },
                    )
                )

        except Exception as e:
            self.issues.append(
                HealthIssue(
                    component="configuration",
                    component_type=ComponentType.CONFIGURATION,
                    status=HealthStatus.WARNING,
                    message=f"Configuration check failed: {e}",
                    recommendations=["Check configuration accessibility"],
                )
            )

    async def _check_dependencies(self) -> None:
        """Check required dependencies and versions."""
        try:
            # Check critical dependencies
            critical_deps = ["httpx", "rich", "loguru", "fire", "psutil"]
            missing_deps = []

            for dep in critical_deps:
                try:
                    __import__(dep)
                except ImportError:
                    missing_deps.append(dep)

            if missing_deps:
                self.issues.append(
                    HealthIssue(
                        component="dependencies",
                        component_type=ComponentType.SYSTEM,
                        status=HealthStatus.CRITICAL,
                        message=f"Missing dependencies: {', '.join(missing_deps)}",
                        recommendations=[
                            "Install missing dependencies: pip install " + " ".join(missing_deps),
                            "Check virtual environment activation",
                            "Verify package installation",
                        ],
                        metric_values={"missing_deps": missing_deps},
                    )
                )

        except Exception as e:
            self.issues.append(
                HealthIssue(
                    component="dependencies",
                    component_type=ComponentType.SYSTEM,
                    status=HealthStatus.WARNING,
                    message=f"Dependency check failed: {e}",
                    recommendations=["Verify package installation"],
                )
            )

    async def _check_provider_health(self) -> None:
        """Check provider-specific health indicators."""
        try:
            # Get rate limiter stats
            rate_limiter = get_rate_limiter()
            rate_stats = rate_limiter.get_all_stats()

            # Get failure tracker stats
            storage = StorageManager(self.storage_root)
            failure_tracker = FailureTracker(storage)
            failure_summary = failure_tracker.get_failure_summary()

            self.metrics.rate_limited_providers = len([s for s in rate_stats.values() if s.get("current_delay", 0) > 0])

            self.metrics.total_providers = failure_summary.get("total_providers", 0)
            self.metrics.healthy_providers = self.metrics.total_providers - failure_summary.get("currently_failed", 0)

            # Check for systemic issues
            if failure_summary.get("currently_failed", 0) > 5:
                self.issues.append(
                    HealthIssue(
                        component="providers",
                        component_type=ComponentType.PROVIDER,
                        status=HealthStatus.WARNING,
                        message=f"Many providers failing: {failure_summary['currently_failed']} of {self.metrics.total_providers}",
                        recommendations=[
                            "Check API keys and network connectivity",
                            "Review provider status pages",
                            "Consider running validation: vexy validate",
                        ],
                        metric_values=failure_summary,
                    )
                )

            if self.metrics.rate_limited_providers > 3:
                self.issues.append(
                    HealthIssue(
                        component="rate_limits",
                        component_type=ComponentType.PERFORMANCE,
                        status=HealthStatus.WARNING,
                        message=f"Many providers rate limited: {self.metrics.rate_limited_providers}",
                        recommendations=[
                            "Reduce request frequency",
                            "Check rate limiting status: vexy rate-limits",
                            "Consider staggering requests across time",
                        ],
                        metric_values={"rate_limited_count": self.metrics.rate_limited_providers},
                    )
                )

        except Exception as e:
            self.issues.append(
                HealthIssue(
                    component="provider_health",
                    component_type=ComponentType.PROVIDER,
                    status=HealthStatus.WARNING,
                    message=f"Provider health check failed: {e}",
                    recommendations=["Check provider monitoring systems"],
                )
            )

    def _calculate_overall_health(self) -> HealthStatus:
        """Calculate overall health status from all issues."""
        if any(issue.status == HealthStatus.CRITICAL for issue in self.issues):
            return HealthStatus.CRITICAL
        if any(issue.status == HealthStatus.WARNING for issue in self.issues):
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY

    def _group_issues_by_component(self) -> dict[str, list[dict[str, Any]]]:
        """Group issues by component type for organized reporting."""
        grouped = {}
        for issue in self.issues:
            component_type = issue.component_type.value
            if component_type not in grouped:
                grouped[component_type] = []
            grouped[component_type].append(issue.to_dict())
        return grouped

    def _generate_prioritized_recommendations(self) -> list[str]:
        """Generate prioritized list of recommendations."""
        recommendations = []

        # Critical issues first
        for issue in self.issues:
            if issue.status == HealthStatus.CRITICAL:
                recommendations.extend(issue.recommendations)

        # Then warnings
        for issue in self.issues:
            if issue.status == HealthStatus.WARNING:
                recommendations.extend(issue.recommendations)

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations[:10]  # Top 10 recommendations

    def _generate_quick_fixes(self) -> list[str]:
        """Generate list of quick fixes for common issues."""
        quick_fixes = []

        for issue in self.issues:
            if issue.component == "disk" and issue.status == HealthStatus.CRITICAL:
                quick_fixes.append("vexy clean --temp --configs")
            elif issue.component == "network" and not self.metrics.network_connectivity:
                quick_fixes.append("Check internet connection and firewall settings")
            elif issue.component == "api_keys":
                quick_fixes.append("Set up API keys: export OPENAI_API_KEY='sk-***[YOUR_KEY_HERE]'")
            elif issue.component == "dependencies":
                quick_fixes.append("pip install -r requirements.txt")

        return quick_fixes


# Global health monitor instance
_global_health_monitor: SystemHealthMonitor | None = None


def get_health_monitor(storage_root: Path | None = None) -> SystemHealthMonitor:
    """Get the global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        if storage_root is None:
            storage_root = Path.cwd()
        _global_health_monitor = SystemHealthMonitor(storage_root)
    return _global_health_monitor
