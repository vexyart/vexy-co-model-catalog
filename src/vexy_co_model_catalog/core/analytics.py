"""
this_file: src/vexy_co_model_catalog/core/analytics.py

CLI command usage analytics and performance metrics logging.
Provides privacy-conscious tracking of command usage patterns and performance data.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

try:
    from vexy_co_model_catalog._version import __version__
except ImportError:
    __version__ = "unknown"

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class CommandMetrics:
    """Metrics for a single command execution."""

    command_name: str
    start_time: float
    end_time: float | None = None
    success: bool = False
    error_type: str | None = None
    error_message: str | None = None

    # Command parameters (anonymized)
    parameter_count: int = 0
    has_force_flag: bool = False
    has_config_flags: bool = False
    provider_count: int = 0

    # System metrics
    memory_usage_mb: float | None = None

    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    @property
    def execution_time_seconds(self) -> float:
        """Get execution time in seconds."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


@dataclass
class AnalyticsSession:
    """Analytics data for a complete CLI session."""

    session_id: str
    timestamp: str
    commands: list[CommandMetrics] = field(default_factory=list)
    total_duration_ms: float = 0.0
    python_version: str = ""
    platform: str = ""
    package_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "total_duration_ms": self.total_duration_ms,
            "python_version": self.python_version,
            "platform": self.platform,
            "package_version": self.package_version,
            "commands": [
                {
                    "command_name": cmd.command_name,
                    "execution_time_ms": cmd.execution_time_ms,
                    "success": cmd.success,
                    "error_type": cmd.error_type,
                    "error_message": cmd.error_message,
                    "parameter_count": cmd.parameter_count,
                    "has_force_flag": cmd.has_force_flag,
                    "has_config_flags": cmd.has_config_flags,
                    "provider_count": cmd.provider_count,
                    "memory_usage_mb": cmd.memory_usage_mb,
                }
                for cmd in self.commands
            ],
        }


class AnalyticsCollector:
    """Collects and manages CLI usage analytics and performance metrics."""

    def __init__(self, storage_dir: Path, enabled: bool = True) -> None:
        """
        Initialize analytics collector.

        Args:
            storage_dir: Directory to store analytics data
            enabled: Whether to collect analytics (privacy setting)
        """
        self.storage_dir = Path(storage_dir)
        self.enabled = enabled
        self.current_session: AnalyticsSession | None = None
        self.session_start_time = time.time()

        if self.enabled:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._initialize_session()

    def _initialize_session(self) -> None:
        """Initialize a new analytics session."""
        package_version = __version__

        session_id = str(uuid.uuid4())[:8]  # Short session ID
        timestamp = datetime.now(timezone.utc).isoformat()

        self.current_session = AnalyticsSession(
            session_id=session_id,
            timestamp=timestamp,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=platform.system(),
            package_version=package_version,
        )

        logger.debug(f"Started analytics session: {session_id}")

    @contextmanager
    def track_command(self, command_name: str, **kwargs: Any) -> Generator[CommandMetrics | None, None, None]:
        """
        Context manager to track command execution metrics.

        Usage:
            with analytics.track_command("fetch", provider_count=5):
                # Execute command
                pass
        """
        if not self.enabled or self.current_session is None:
            yield None
            return

        # Create metrics object
        metrics = CommandMetrics(
            command_name=command_name,
            start_time=time.time(),
            parameter_count=kwargs.get("parameter_count", 0),
            has_force_flag=kwargs.get("has_force_flag", False),
            has_config_flags=kwargs.get("has_config_flags", False),
            provider_count=kwargs.get("provider_count", 0),
        )

        # Get initial memory usage
        try:
            import psutil

            process = psutil.Process()
            metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass  # psutil not available
        except Exception as e:
            logger.debug(f"Failed to get memory usage: {e}")

        try:
            yield metrics
            # Command succeeded
            metrics.success = True

        except Exception as e:
            # Command failed
            metrics.success = False
            metrics.error_type = type(e).__name__
            metrics.error_message = str(e)[:200]  # Limit error message length
            raise

        finally:
            # Always record end time and add to session
            metrics.end_time = time.time()
            self.current_session.commands.append(metrics)

            logger.debug(
                f"Command {command_name} completed in {metrics.execution_time_ms:.1f}ms (success: {metrics.success})"
            )

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable analytics collection."""
        self.enabled = enabled
        if enabled and self.current_session is None:
            self._initialize_session()
        logger.info(f"Analytics collection {'enabled' if enabled else 'disabled'}")

    def save_session(self) -> Path | None:
        """Save the current session to disk."""
        if not self.enabled or self.current_session is None:
            return None

        # Calculate total session duration
        self.current_session.total_duration_ms = (time.time() - self.session_start_time) * 1000

        # Save to file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"analytics_{timestamp}_{self.current_session.session_id}.json"
        file_path = self.storage_dir / filename

        try:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(self.current_session.to_dict(), f, indent=2)

            logger.debug(f"Saved analytics session to: {file_path}")
            return file_path

        except Exception as e:
            logger.warning(f"Failed to save analytics session: {e}")
            return None

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for all stored analytics."""
        if not self.enabled:
            return {"analytics_enabled": False}

        analytics_files = list(self.storage_dir.glob("analytics_*.json"))
        if not analytics_files:
            return {"analytics_enabled": True, "total_sessions": 0}

        total_sessions = len(analytics_files)
        total_commands = 0
        command_counts = {}
        success_rate_sum = 0.0

        try:
            for file_path in analytics_files:
                with file_path.open("r", encoding="utf-8") as f:
                    session_data = json.load(f)

                commands = session_data.get("commands", [])
                total_commands += len(commands)

                session_success_count = 0
                for cmd in commands:
                    cmd_name = cmd.get("command_name", "unknown")
                    command_counts[cmd_name] = command_counts.get(cmd_name, 0) + 1

                    if cmd.get("success", False):
                        session_success_count += 1

                if commands:
                    success_rate_sum += session_success_count / len(commands)

            avg_success_rate = (success_rate_sum / total_sessions) if total_sessions > 0 else 0.0
            most_used_command = max(command_counts.items(), key=lambda x: x[1]) if command_counts else ("none", 0)

            return {
                "analytics_enabled": True,
                "total_sessions": total_sessions,
                "total_commands": total_commands,
                "average_success_rate": avg_success_rate,
                "most_used_command": most_used_command[0],
                "command_usage": command_counts,
            }

        except Exception as e:
            logger.warning(f"Failed to calculate analytics summary: {e}")
            return {"analytics_enabled": True, "total_sessions": total_sessions, "error": str(e)}

    def cleanup_old_analytics(self, max_files: int = 100) -> None:
        """Clean up old analytics files to prevent disk space issues."""
        if not self.enabled:
            return

        analytics_files = list(self.storage_dir.glob("analytics_*.json"))
        if len(analytics_files) <= max_files:
            return

        # Sort by modification time and remove oldest files
        analytics_files.sort(key=lambda p: p.stat().st_mtime)
        files_to_remove = analytics_files[:-max_files]

        removed_count = 0
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove old analytics file {file_path}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old analytics files")


# Global analytics collector instance
_global_analytics: AnalyticsCollector | None = None


def get_analytics() -> AnalyticsCollector | None:
    """Get the global analytics collector instance."""
    return _global_analytics


def initialize_analytics(storage_dir: Path, enabled: bool = True) -> AnalyticsCollector:
    """Initialize the global analytics collector."""
    global _global_analytics
    _global_analytics = AnalyticsCollector(storage_dir, enabled)
    return _global_analytics


def finalize_analytics() -> None:
    """Finalize analytics collection and save session."""
    global _global_analytics
    if _global_analytics:
        _global_analytics.save_session()
        _global_analytics = None
