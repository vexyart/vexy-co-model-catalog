# this_file: src/vexy_co_model_catalog/core/performance.py

"""
Performance monitoring and timing utilities for CLI commands and operations.
Provides detailed metrics collection, timing decorators, and resource usage tracking.
"""

from __future__ import annotations

import functools
import gc
import json
import os
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class PerformanceMetric:
    """Single performance measurement with detailed system metrics."""

    command: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    cpu_percent_start: float
    cpu_percent_end: float
    memory_mb_start: float
    memory_mb_end: float
    memory_peak_mb: float
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Enhanced metrics
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    thread_count_start: int = 0
    thread_count_end: int = 0
    file_handles_start: int = 0
    file_handles_end: int = 0
    gc_collections: dict[str, int] = field(default_factory=dict)
    memory_leaks_detected: bool = False
    memory_growth_rate_mb_per_sec: float = 0.0

    @property
    def cpu_usage_change(self) -> float:
        """Change in CPU usage during command execution."""
        return self.cpu_percent_end - self.cpu_percent_start

    @property
    def memory_usage_change(self) -> float:
        """Change in memory usage during command execution (MB)."""
        return self.memory_mb_end - self.memory_mb_start

    @property
    def disk_io_total_mb(self) -> float:
        """Total disk I/O during command execution (MB)."""
        return self.disk_io_read_mb + self.disk_io_write_mb

    @property
    def network_total_bytes(self) -> int:
        """Total network bytes transferred during command execution."""
        return self.network_bytes_sent + self.network_bytes_recv

    @property
    def thread_count_change(self) -> int:
        """Change in thread count during command execution."""
        return self.thread_count_end - self.thread_count_start

    @property
    def file_handles_change(self) -> int:
        """Change in file handles during command execution."""
        return self.file_handles_end - self.file_handles_start

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "command": self.command,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "cpu_percent_start": self.cpu_percent_start,
            "cpu_percent_end": self.cpu_percent_end,
            "cpu_usage_change": self.cpu_usage_change,
            "memory_mb_start": self.memory_mb_start,
            "memory_mb_end": self.memory_mb_end,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_usage_change": self.memory_usage_change,
            "memory_growth_rate_mb_per_sec": self.memory_growth_rate_mb_per_sec,
            "disk_io_read_mb": self.disk_io_read_mb,
            "disk_io_write_mb": self.disk_io_write_mb,
            "disk_io_total_mb": self.disk_io_total_mb,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "network_total_bytes": self.network_total_bytes,
            "thread_count_start": self.thread_count_start,
            "thread_count_end": self.thread_count_end,
            "thread_count_change": self.thread_count_change,
            "file_handles_start": self.file_handles_start,
            "file_handles_end": self.file_handles_end,
            "file_handles_change": self.file_handles_change,
            "gc_collections": self.gc_collections,
            "memory_leaks_detected": self.memory_leaks_detected,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class PerformanceMonitor:
    """Advanced performance monitor with memory profiling and resource tracking."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        """Initialize performance monitor with enhanced capabilities."""
        self.storage_dir = storage_dir or Path.cwd() / "performance"
        self.storage_dir.mkdir(exist_ok=True)

        self.metrics: list[PerformanceMetric] = []
        self._monitoring_active = False
        self._monitor_thread: threading.Thread | None = None
        self._current_metric: PerformanceMetric | None = None
        self._peak_memory = 0.0

        # Enhanced monitoring state
        self._memory_profiling_enabled = os.getenv("VEXY_MEMORY_PROFILING", "false").lower() == "true"
        self._detailed_profiling_enabled = os.getenv("VEXY_DETAILED_PROFILING", "false").lower() == "true"
        self._initial_memory_snapshot = None
        self._gc_stats_start = None
        self._io_counters_start = None
        self._network_counters_start = None

        # Process handle for monitoring
        try:
            self.process = psutil.Process()
        except psutil.NoSuchProcess:
            logger.warning("Could not initialize process monitoring")
            self.process = None

        # Initialize memory profiling if enabled
        if self._memory_profiling_enabled:
            self._init_memory_profiling()

    def _get_current_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        if self.process:
            try:
                return self.process.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return 0.0

    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if self.process:
            try:
                memory_info = self.process.memory_info()
                return memory_info.rss / 1024 / 1024  # Convert bytes to MB
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return 0.0

    def _init_memory_profiling(self) -> None:
        """Initialize memory profiling with tracemalloc."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.debug("Memory profiling initialized with tracemalloc")

    def _get_resource_counters(self) -> dict[str, Any]:
        """Get comprehensive resource usage counters."""
        counters = {
            "disk_io": {"read_bytes": 0, "write_bytes": 0},
            "network": {"bytes_sent": 0, "bytes_recv": 0},
            "threads": 0,
            "file_handles": 0,
        }

        if self.process:
            try:
                # Disk I/O counters
                io_counters = self.process.io_counters()
                counters["disk_io"] = {
                    "read_bytes": io_counters.read_bytes,
                    "write_bytes": io_counters.write_bytes,
                }

                # Thread count
                counters["threads"] = self.process.num_threads()

                # File handles (approximate)
                try:
                    counters["file_handles"] = self.process.num_fds()  # Unix-like systems
                except AttributeError:
                    counters["file_handles"] = len(self.process.open_files())  # Fallback

            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                pass

        # Network I/O (system-wide, best we can do)
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                counters["network"] = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                }
        except (psutil.AccessDenied, AttributeError):
            pass

        return counters

    def _detect_memory_leaks(self) -> bool:
        """Detect potential memory leaks using tracemalloc."""
        if not self._memory_profiling_enabled or not tracemalloc.is_tracing():
            return False

        try:
            current, peak = tracemalloc.get_traced_memory()

            # Simple heuristic: if peak is significantly higher than current,
            # and we've allocated more than 10MB, consider it a potential leak
            if peak > current * 1.5 and current > 10 * 1024 * 1024:  # 10MB threshold
                return True

        except Exception as e:
            logger.debug(f"Error detecting memory leaks: {e}")

        return False

    def _get_garbage_collection_stats(self) -> dict[str, int]:
        """Get garbage collection statistics."""
        return {
            f"generation_{i}": gc.get_count()[i] if i < len(gc.get_count()) else 0
            for i in range(3)  # Python has 3 GC generations
        }

    def _monitor_peak_memory(self) -> None:
        """Monitor peak memory usage in a background thread."""
        while self._monitoring_active:
            current_memory = self._get_current_memory_mb()
            self._peak_memory = max(self._peak_memory, current_memory)
            time.sleep(0.1)  # Check every 100ms

    def start_monitoring(self, command: str, metadata: dict[str, Any] | None = None) -> None:
        """Start monitoring a command execution with enhanced resource tracking."""
        if self._monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        start_time = datetime.now(timezone.utc)
        cpu_start = self._get_current_cpu_percent()
        memory_start = self._get_current_memory_mb()

        # Collect initial resource counters
        resource_counters = self._get_resource_counters()
        self._io_counters_start = resource_counters["disk_io"]
        self._network_counters_start = resource_counters["network"]

        # Initialize garbage collection tracking
        self._gc_stats_start = self._get_garbage_collection_stats()

        # Take memory snapshot if profiling enabled
        if self._memory_profiling_enabled and tracemalloc.is_tracing():
            self._initial_memory_snapshot = tracemalloc.take_snapshot()

        self._current_metric = PerformanceMetric(
            command=command,
            start_time=start_time,
            end_time=start_time,  # Will be updated on stop
            duration_seconds=0.0,
            cpu_percent_start=cpu_start,
            cpu_percent_end=cpu_start,  # Will be updated on stop
            memory_mb_start=memory_start,
            memory_mb_end=memory_start,  # Will be updated on stop
            memory_peak_mb=memory_start,
            thread_count_start=resource_counters["threads"],
            file_handles_start=resource_counters["file_handles"],
            metadata=metadata or {},
        )

        self._peak_memory = memory_start
        self._monitoring_active = True

        # Start background thread to monitor peak memory
        if self.process:
            self._monitor_thread = threading.Thread(target=self._monitor_peak_memory, daemon=True)
            self._monitor_thread.start()

        logger.debug(f"Started enhanced performance monitoring for command '{command}'")

    def stop_monitoring(self, success: bool = True, error_message: str | None = None) -> PerformanceMetric:
        """Stop monitoring and record comprehensive final metrics."""
        if not self._monitoring_active or not self._current_metric:
            logger.warning("No active performance monitoring to stop")
            return None

        end_time = datetime.now(timezone.utc)
        cpu_end = self._get_current_cpu_percent()
        memory_end = self._get_current_memory_mb()

        # Collect final resource counters
        resource_counters = self._get_resource_counters()

        # Calculate resource usage deltas
        disk_read_delta = resource_counters["disk_io"]["read_bytes"] - self._io_counters_start["read_bytes"]
        disk_write_delta = resource_counters["disk_io"]["write_bytes"] - self._io_counters_start["write_bytes"]
        network_sent_delta = resource_counters["network"]["bytes_sent"] - self._network_counters_start["bytes_sent"]
        network_recv_delta = resource_counters["network"]["bytes_recv"] - self._network_counters_start["bytes_recv"]

        # Calculate garbage collection activity
        gc_stats_end = self._get_garbage_collection_stats()
        gc_collections = {
            gen: gc_stats_end[gen] - self._gc_stats_start.get(gen, 0)
            for gen in gc_stats_end
        }

        # Detect memory leaks
        memory_leaks = self._detect_memory_leaks()

        # Calculate memory growth rate
        duration = (end_time - self._current_metric.start_time).total_seconds()
        memory_growth_rate = (memory_end - self._current_metric.memory_mb_start) / duration if duration > 0 else 0.0

        # Stop background monitoring
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)

        # Update final metrics with enhanced data
        self._current_metric.end_time = end_time
        self._current_metric.duration_seconds = duration
        self._current_metric.cpu_percent_end = cpu_end
        self._current_metric.memory_mb_end = memory_end
        self._current_metric.memory_peak_mb = max(self._peak_memory, memory_end)
        self._current_metric.memory_growth_rate_mb_per_sec = memory_growth_rate
        self._current_metric.disk_io_read_mb = disk_read_delta / (1024 * 1024)  # Convert to MB
        self._current_metric.disk_io_write_mb = disk_write_delta / (1024 * 1024)
        self._current_metric.network_bytes_sent = network_sent_delta
        self._current_metric.network_bytes_recv = network_recv_delta
        self._current_metric.thread_count_end = resource_counters["threads"]
        self._current_metric.file_handles_end = resource_counters["file_handles"]
        self._current_metric.gc_collections = gc_collections
        self._current_metric.memory_leaks_detected = memory_leaks
        self._current_metric.success = success
        self._current_metric.error_message = error_message

        # Store the metric
        self.metrics.append(self._current_metric)
        completed_metric = self._current_metric
        self._current_metric = None

        # Reset profiling state
        self._initial_memory_snapshot = None
        self._gc_stats_start = None
        self._io_counters_start = None
        self._network_counters_start = None

        logger.debug(f"Completed enhanced performance monitoring: {completed_metric.duration_seconds:.3f}s")
        if self._detailed_profiling_enabled:
            self._log_detailed_metrics(completed_metric)

        return completed_metric

    def _log_detailed_metrics(self, metric: PerformanceMetric) -> None:
        """Log detailed performance metrics for analysis."""
        logger.info(f"Performance Profile - {metric.command}:")
        logger.info(f"  Duration: {metric.duration_seconds:.3f}s")
        logger.info(f"  Memory: {metric.memory_mb_start:.1f} → {metric.memory_mb_end:.1f} MB (peak: {metric.memory_peak_mb:.1f} MB)")
        logger.info(f"  Memory Growth Rate: {metric.memory_growth_rate_mb_per_sec:.2f} MB/s")
        logger.info(f"  CPU: {metric.cpu_percent_start:.1f}% → {metric.cpu_percent_end:.1f}%")
        logger.info(f"  Disk I/O: Read {metric.disk_io_read_mb:.2f} MB, Write {metric.disk_io_write_mb:.2f} MB")
        sent_bytes = f"{metric.network_bytes_sent:,}"
        recv_bytes = f"{metric.network_bytes_recv:,}"
        logger.info(f"  Network: Sent {sent_bytes} bytes, Received {recv_bytes} bytes")
        logger.info(f"  Threads: {metric.thread_count_start} → {metric.thread_count_end}")
        logger.info(f"  File Handles: {metric.file_handles_start} → {metric.file_handles_end}")
        if metric.memory_leaks_detected:
            logger.warning("  ⚠️  Potential memory leak detected")
        if any(metric.gc_collections.values()):
            logger.info(f"  GC Collections: {metric.gc_collections}")

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics of all recorded metrics."""
        if not self.metrics:
            return {
                "total_commands": 0,
                "successful_commands": 0,
                "failed_commands": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0,
                "fastest_command": None,
                "slowest_command": None,
            }

        successful = [m for m in self.metrics if m.success]
        failed = [m for m in self.metrics if not m.success]
        durations = [m.duration_seconds for m in self.metrics]

        fastest = min(self.metrics, key=lambda m: m.duration_seconds)
        slowest = max(self.metrics, key=lambda m: m.duration_seconds)

        return {
            "total_commands": len(self.metrics),
            "successful_commands": len(successful),
            "failed_commands": len(failed),
            "total_execution_time": sum(durations),
            "average_execution_time": sum(durations) / len(durations),
            "min_execution_time": min(durations),
            "max_execution_time": max(durations),
            "fastest_command": {
                "command": fastest.command,
                "duration": fastest.duration_seconds,
                "timestamp": fastest.start_time.isoformat(),
            },
            "slowest_command": {
                "command": slowest.command,
                "duration": slowest.duration_seconds,
                "timestamp": slowest.start_time.isoformat(),
            },
            "commands_by_type": self._get_command_statistics(),
            "resource_usage": self._get_resource_statistics(),
        }

    def _get_command_statistics(self) -> dict[str, dict[str, Any]]:
        """Get statistics grouped by command type."""
        command_stats = {}

        for metric in self.metrics:
            command = metric.command
            if command not in command_stats:
                command_stats[command] = {
                    "count": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "min_time": float("inf"),
                    "max_time": 0.0,
                }

            stats = command_stats[command]
            stats["count"] += 1
            stats["total_time"] += metric.duration_seconds
            stats["min_time"] = min(stats["min_time"], metric.duration_seconds)
            stats["max_time"] = max(stats["max_time"], metric.duration_seconds)

            if metric.success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1

            stats["avg_time"] = stats["total_time"] / stats["count"]

        return command_stats

    def _get_resource_statistics(self) -> dict[str, Any]:
        """Get comprehensive resource usage statistics."""
        if not self.metrics:
            return {}

        cpu_changes = [m.cpu_usage_change for m in self.metrics if m.cpu_usage_change is not None]
        memory_changes = [m.memory_usage_change for m in self.metrics if m.memory_usage_change is not None]
        peak_memories = [m.memory_peak_mb for m in self.metrics if m.memory_peak_mb is not None]
        disk_reads = [m.disk_io_read_mb for m in self.metrics if m.disk_io_read_mb is not None]
        disk_writes = [m.disk_io_write_mb for m in self.metrics if m.disk_io_write_mb is not None]
        network_sent = [m.network_bytes_sent for m in self.metrics if m.network_bytes_sent is not None]
        network_recv = [m.network_bytes_recv for m in self.metrics if m.network_bytes_recv is not None]
        growth_rates = [
            m.memory_growth_rate_mb_per_sec for m in self.metrics 
            if m.memory_growth_rate_mb_per_sec is not None
        ]

        # Count memory leaks
        memory_leak_count = sum(1 for m in self.metrics if m.memory_leaks_detected)

        return {
            "cpu_usage": {
                "avg_change": sum(cpu_changes) / len(cpu_changes) if cpu_changes else 0.0,
                "max_change": max(cpu_changes) if cpu_changes else 0.0,
                "min_change": min(cpu_changes) if cpu_changes else 0.0,
            },
            "memory_usage": {
                "avg_change_mb": sum(memory_changes) / len(memory_changes) if memory_changes else 0.0,
                "max_change_mb": max(memory_changes) if memory_changes else 0.0,
                "min_change_mb": min(memory_changes) if memory_changes else 0.0,
                "max_peak_mb": max(peak_memories) if peak_memories else 0.0,
                "avg_growth_rate_mb_per_sec": sum(growth_rates) / len(growth_rates) if growth_rates else 0.0,
                "max_growth_rate_mb_per_sec": max(growth_rates) if growth_rates else 0.0,
                "potential_leaks_detected": memory_leak_count,
            },
            "disk_io": {
                "total_read_mb": sum(disk_reads) if disk_reads else 0.0,
                "total_write_mb": sum(disk_writes) if disk_writes else 0.0,
                "avg_read_mb": sum(disk_reads) / len(disk_reads) if disk_reads else 0.0,
                "avg_write_mb": sum(disk_writes) / len(disk_writes) if disk_writes else 0.0,
                "max_read_mb": max(disk_reads) if disk_reads else 0.0,
                "max_write_mb": max(disk_writes) if disk_writes else 0.0,
            },
            "network_io": {
                "total_sent_bytes": sum(network_sent) if network_sent else 0,
                "total_recv_bytes": sum(network_recv) if network_recv else 0,
                "avg_sent_bytes": sum(network_sent) / len(network_sent) if network_sent else 0,
                "avg_recv_bytes": sum(network_recv) / len(network_recv) if network_recv else 0,
                "max_sent_bytes": max(network_sent) if network_sent else 0,
                "max_recv_bytes": max(network_recv) if network_recv else 0,
            },
            "system_health": {
                "commands_with_memory_leaks": memory_leak_count,
                "memory_leak_percentage": (memory_leak_count / len(self.metrics) * 100) if self.metrics else 0,
            },
        }

    def save_metrics(self, filename: str | None = None) -> Path:
        """Save metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"

        filepath = self.storage_dir / filename

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": self.get_statistics(),
            "metrics": [m.to_dict() for m in self.metrics],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Performance metrics saved to {filepath}")
        return filepath

    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        self.metrics.clear()
        logger.debug("Performance metrics cleared")

    def get_memory_profile(self, top_n: int = 10) -> dict[str, Any]:
        """Get detailed memory profiling information."""
        if not self._memory_profiling_enabled or not tracemalloc.is_tracing():
            return {"error": "Memory profiling not enabled. Set VEXY_MEMORY_PROFILING=true"}

        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            profile_data = {
                "current_memory_mb": self._get_current_memory_mb(),
                "traced_memory_mb": tracemalloc.get_traced_memory()[0] / 1024 / 1024,
                "peak_traced_memory_mb": tracemalloc.get_traced_memory()[1] / 1024 / 1024,
                "top_allocations": [],
            }

            for index, stat in enumerate(top_stats[:top_n]):
                profile_data["top_allocations"].append({
                    "rank": index + 1,
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                    "filename": stat.traceback.format()[-1] if stat.traceback else "unknown",
                })

            return profile_data

        except Exception as e:
            return {"error": f"Memory profiling failed: {e}"}

    def identify_bottlenecks(self, threshold_seconds: float = 1.0) -> dict[str, Any]:
        """Identify performance bottlenecks based on execution times and resource usage."""
        if not self.metrics:
            return {"error": "No performance metrics available"}

        slow_commands = [m for m in self.metrics if m.duration_seconds > threshold_seconds]
        high_memory_commands = [m for m in self.metrics if m.memory_usage_change > 50]  # > 50MB
        high_cpu_commands = [m for m in self.metrics if abs(m.cpu_usage_change) > 20]  # > 20% change
        high_io_commands = [m for m in self.metrics if m.disk_io_total_mb > 10]  # > 10MB I/O

        return {
            "analysis_threshold_seconds": threshold_seconds,
            "total_commands_analyzed": len(self.metrics),
            "slow_commands": {
                "count": len(slow_commands),
                "examples": [{
                    "command": cmd.command,
                    "duration_seconds": cmd.duration_seconds,
                    "memory_change_mb": cmd.memory_usage_change,
                    "timestamp": cmd.start_time.isoformat(),
                } for cmd in sorted(slow_commands, key=lambda x: x.duration_seconds, reverse=True)[:5]]
            },
            "high_memory_usage": {
                "count": len(high_memory_commands),
                "examples": [{
                    "command": cmd.command,
                    "memory_change_mb": cmd.memory_usage_change,
                    "peak_memory_mb": cmd.memory_peak_mb,
                    "timestamp": cmd.start_time.isoformat(),
                } for cmd in sorted(high_memory_commands, key=lambda x: x.memory_usage_change, reverse=True)[:5]]
            },
            "high_cpu_usage": {
                "count": len(high_cpu_commands),
                "examples": [{
                    "command": cmd.command,
                    "cpu_change_percent": cmd.cpu_usage_change,
                    "duration_seconds": cmd.duration_seconds,
                    "timestamp": cmd.start_time.isoformat(),
                } for cmd in sorted(high_cpu_commands, key=lambda x: abs(x.cpu_usage_change), reverse=True)[:5]]
            },
            "high_io_usage": {
                "count": len(high_io_commands),
                "examples": [{
                    "command": cmd.command,
                    "total_io_mb": cmd.disk_io_total_mb,
                    "read_mb": cmd.disk_io_read_mb,
                    "write_mb": cmd.disk_io_write_mb,
                    "timestamp": cmd.start_time.isoformat(),
                } for cmd in sorted(high_io_commands, key=lambda x: x.disk_io_total_mb, reverse=True)[:5]]
            },
            "recommendations": self._get_optimization_recommendations(slow_commands, high_memory_commands, high_cpu_commands, high_io_commands)
        }

    def _get_optimization_recommendations(self, slow_cmds, memory_cmds, cpu_cmds, io_cmds) -> list[str]:
        """Generate optimization recommendations based on bottleneck analysis."""
        recommendations = []

        if slow_cmds:
            recommendations.append(f"{len(slow_cmds)} commands are running slowly. Consider optimizing algorithms or adding caching.")

        if memory_cmds:
            recommendations.append(f"{len(memory_cmds)} commands use excessive memory. Review data structures and consider streaming.")

        if cpu_cmds:
            recommendations.append(f"{len(cpu_cmds)} commands show high CPU usage. Consider async processing or optimization.")

        if io_cmds:
            recommendations.append(f"{len(io_cmds)} commands perform heavy I/O. Consider batching or caching strategies.")

        # Check for memory leaks
        leak_count = sum(1 for m in self.metrics if m.memory_leaks_detected)
        if leak_count > 0:
            recommendations.append(f"{leak_count} commands show potential memory leaks. Review object lifecycle management.")

        if not recommendations:
            recommendations.append("No significant performance issues detected. System is running efficiently.")

        return recommendations

    def optimize_memory_usage(self) -> dict[str, Any]:
        """Perform memory optimization operations."""
        initial_memory = self._get_current_memory_mb()

        # Force garbage collection
        collected = gc.collect()

        # Clear internal caches if available
        if hasattr(sys, 'intern'):
            # Python string interning cleanup (limited effect)
            pass

        final_memory = self._get_current_memory_mb()
        memory_freed = initial_memory - final_memory

        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_freed_mb": memory_freed,
            "objects_collected": collected,
            "gc_stats": self._get_garbage_collection_stats(),
            "recommendations": [
                "Regular garbage collection can help free unused objects",
                "Consider implementing object pooling for frequently created objects",
                "Use generators and iterators to reduce memory footprint",
                "Profile memory usage with tracemalloc for detailed analysis",
            ]
        }

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        stats = self.get_statistics()
        bottlenecks = self.identify_bottlenecks()
        memory_profile = self.get_memory_profile()

        return {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "system_info": {
                "python_version": sys.version,
                "memory_profiling_enabled": self._memory_profiling_enabled,
                "detailed_profiling_enabled": self._detailed_profiling_enabled,
            },
            "performance_statistics": stats,
            "bottleneck_analysis": bottlenecks,
            "memory_profile": memory_profile,
            "optimization_summary": {
                "total_execution_time": stats.get("total_execution_time", 0),
                "average_memory_usage_mb": stats.get("resource_usage", {}).get("memory_usage", {}).get("avg_change_mb", 0),
                "commands_with_issues": len([m for m in self.metrics if not m.success or m.memory_leaks_detected]),
                "overall_health_score": self._calculate_health_score(),
            }
        }

    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        if not self.metrics:
            return 100.0

        total_score = 100.0

        # Deduct for failures
        failure_rate = len([m for m in self.metrics if not m.success]) / len(self.metrics)
        total_score -= failure_rate * 30

        # Deduct for memory leaks
        leak_rate = len([m for m in self.metrics if m.memory_leaks_detected]) / len(self.metrics)
        total_score -= leak_rate * 20

        # Deduct for slow commands (> 5 seconds)
        slow_rate = len([m for m in self.metrics if m.duration_seconds > 5.0]) / len(self.metrics)
        total_score -= slow_rate * 15

        # Deduct for high memory usage (> 100MB change)
        high_mem_rate = len([m for m in self.metrics if m.memory_usage_change > 100]) / len(self.metrics)
        total_score -= high_mem_rate * 10

        return max(0.0, total_score)


def performance_monitor(command_name: str | None = None, metadata: dict[str, Any] | None = None) -> Callable:
    """Enhanced decorator to monitor performance with detailed profiling."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get monitor instance (assume it's available globally or passed)
            monitor = get_performance_monitor()

            # Use function name if no command name provided
            cmd_name = command_name or f"{func.__module__}.{func.__name__}"

            # Add enhanced function metadata
            func_metadata = {
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_count": len(kwargs),
                "thread_id": threading.current_thread().ident,
                "process_id": os.getpid(),
                **(metadata or {})
            }

            monitor.start_monitoring(cmd_name, func_metadata)

            try:
                result = func(*args, **kwargs)
                monitor.stop_monitoring(success=True)
                return result
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e!s}"
                monitor.stop_monitoring(success=False, error_message=error_msg)
                raise

        return wrapper

    return decorator


def profile_memory(func: Callable) -> Callable:
    """Decorator specifically for memory profiling of functions."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        tracemalloc.take_snapshot()
        memory_before, _ = tracemalloc.get_traced_memory()

        try:
            result = func(*args, **kwargs)

            memory_after, peak = tracemalloc.get_traced_memory()
            memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
            peak_used = peak / 1024 / 1024  # MB

            if memory_used > 1.0:  # Log if > 1MB used
                logger.info(f"Memory Profile {func.__name__}: {memory_used:.1f}MB used, {peak_used:.1f}MB peak")

            return result

        except Exception:
            memory_after, peak = tracemalloc.get_traced_memory()
            memory_used = (memory_after - memory_before) / 1024 / 1024
            logger.error(f"Memory Profile {func.__name__} (ERROR): {memory_used:.1f}MB used before exception")
            raise

    return wrapper


# Global monitor instance and configuration
_global_monitor: PerformanceMonitor | None = None
_profiling_enabled = os.getenv("VEXY_PERFORMANCE_ENABLED", "false").lower() == "true"


def get_performance_monitor(storage_dir: Path | None = None) -> PerformanceMonitor:
    """Get or create the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(storage_dir)
    return _global_monitor


def initialize_performance_monitoring(storage_dir: Path | None = None) -> PerformanceMonitor:
    """Initialize performance monitoring system."""
    global _global_monitor
    _global_monitor = PerformanceMonitor(storage_dir)
    logger.debug("Performance monitoring system initialized")
    return _global_monitor


def shutdown_performance_monitoring() -> None:
    """Shutdown performance monitoring system with cleanup."""
    global _global_monitor
    if _global_monitor and _global_monitor._monitoring_active:
        _global_monitor.stop_monitoring(success=False, error_message="Shutdown")

    # Stop memory profiling if enabled
    if tracemalloc.is_tracing():
        tracemalloc.stop()

    _global_monitor = None
    logger.debug("Performance monitoring system shutdown")


def enable_memory_profiling() -> None:
    """Enable memory profiling for the current session."""
    os.environ["VEXY_MEMORY_PROFILING"] = "true"
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    logger.info("Memory profiling enabled")


def enable_detailed_profiling() -> None:
    """Enable detailed performance profiling for the current session."""
    os.environ["VEXY_DETAILED_PROFILING"] = "true"
    logger.info("Detailed performance profiling enabled")


def get_system_resource_usage() -> dict[str, Any]:
    """Get current system resource usage snapshot."""
    try:
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": cpu_percent,
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024,
            "thread_count": process.num_threads(),
            "file_descriptors": process.num_fds() if hasattr(process, 'num_fds') else len(process.open_files()),
            "system_cpu_percent": psutil.cpu_percent(),
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_disk_usage_percent": psutil.disk_usage('/').percent,
        }
    except Exception as e:
        return {"error": f"Failed to get system resource usage: {e}"}
