#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["pytest", "rich", "loguru", "httpx", "psutil"]
# ///
# this_file: scripts/performance_benchmark.py

"""
Performance benchmarking suite for regression prevention.

Ensures sub-100ms cache performance and other critical optimizations
are maintained over time with automated quality gates.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
from loguru import logger
from rich.console import Console

# Add project-specific imports that can be safely moved to module level
from vexy_co_model_catalog.core.caching import (
    FAST_CACHE_CONFIG,
    get_api_response_cache,
    get_model_cache,
    get_validation_cache,
)
from rich.table import Table

# Performance thresholds - these are our quality gates
CACHE_ACCESS_MAX_MS = 100           # Sub-100ms cache access requirement
CACHE_WRITE_MAX_MS = 50            # Cache write operations should be fast
PROVIDER_FETCH_MAX_S = 5.0         # Provider fetching should complete reasonably
CLI_STARTUP_MAX_MS = 2000          # CLI startup should be responsive
MEMORY_GROWTH_MAX_MB = 50          # Memory growth during operations
MIN_CACHE_HIT_RATE = 75            # Minimum acceptable cache hit rate %

# Test iteration counts for reliable measurements
CACHE_OPERATIONS_COUNT = 1000      # Number of cache operations to test
PROVIDER_TEST_COUNT = 5            # Number of provider tests
MEMORY_SAMPLE_COUNT = 100          # Memory sampling frequency


@dataclass
class BenchmarkResult:
    """Individual benchmark test result."""
    name: str
    duration_ms: float
    memory_mb: float
    success: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Complete performance benchmark report."""
    cache_performance: list[BenchmarkResult] = field(default_factory=list)
    cli_performance: list[BenchmarkResult] = field(default_factory=list)
    memory_performance: list[BenchmarkResult] = field(default_factory=list)
    quality_gates_passed: int = 0
    quality_gates_failed: int = 0
    overall_status: str = "UNKNOWN"


class PerformanceBenchmark:
    """Performance regression prevention benchmarking suite."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize performance benchmark suite."""
        self.console = console or Console()
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

    def run_cache_benchmarks(self) -> list[BenchmarkResult]:
        """Benchmark cache performance to ensure sub-100ms response times."""
        self.console.print("🚀 [cyan]Running cache performance benchmarks...[/cyan]")
        results = []

        # Cache modules already imported at module level
        try:
            # Test cache access performance
            pass
        except ImportError as e:
            results.append(BenchmarkResult(
                name="cache_import",
                duration_ms=0,
                memory_mb=0,
                success=False,
                details={"error": str(e)}
            ))
            return results

        # Test cache creation performance
        start_time = time.perf_counter()
        model_cache = get_model_cache()
        get_validation_cache()
        get_api_response_cache()
        creation_time = (time.perf_counter() - start_time) * 1000

        results.append(BenchmarkResult(
            name="cache_creation",
            duration_ms=creation_time,
            memory_mb=self._get_current_memory(),
            success=creation_time < CACHE_ACCESS_MAX_MS,
            details={"caches_created": 3}
        ))

        # Test cache write performance
        test_data = {"provider": "test", "models": ["model1", "model2"] * 50}  # Realistic payload size
        write_times = []

        for i in range(CACHE_OPERATIONS_COUNT // 10):  # Sample subset for performance
            start_time = time.perf_counter()
            model_cache.put(f"benchmark_key_{i}", test_data, ttl_seconds=300)
            write_time = (time.perf_counter() - start_time) * 1000
            write_times.append(write_time)

        avg_write_time = statistics.mean(write_times)
        max_write_time = max(write_times)

        results.append(BenchmarkResult(
            name="cache_write_performance",
            duration_ms=avg_write_time,
            memory_mb=self._get_current_memory(),
            success=avg_write_time < CACHE_WRITE_MAX_MS and max_write_time < CACHE_ACCESS_MAX_MS,
            details={
                "avg_write_ms": avg_write_time,
                "max_write_ms": max_write_time,
                "operations": len(write_times)
            }
        ))

        # Test cache read performance with hit rate tracking
        read_times = []
        hits = 0

        for i in range(CACHE_OPERATIONS_COUNT // 10):
            start_time = time.perf_counter()
            result = model_cache.get(f"benchmark_key_{i}")
            read_time = (time.perf_counter() - start_time) * 1000
            read_times.append(read_time)
            if result is not None:
                hits += 1

        avg_read_time = statistics.mean(read_times)
        max_read_time = max(read_times)
        hit_rate = (hits / len(read_times)) * 100

        results.append(BenchmarkResult(
            name="cache_read_performance",
            duration_ms=avg_read_time,
            memory_mb=self._get_current_memory(),
            success=avg_read_time < CACHE_ACCESS_MAX_MS and hit_rate >= MIN_CACHE_HIT_RATE,
            details={
                "avg_read_ms": avg_read_time,
                "max_read_ms": max_read_time,
                "hit_rate_percent": hit_rate,
                "operations": len(read_times)
            }
        ))

        return results

    def run_cli_benchmarks(self) -> list[BenchmarkResult]:
        """Benchmark CLI performance for startup and common operations."""
        self.console.print("⚡ [cyan]Running CLI performance benchmarks...[/cyan]")
        results = []

        # Test CLI import performance (simulates startup)
        start_time = time.perf_counter()
        try:
            from vexy_co_model_catalog.cli import CLI
            import_time = (time.perf_counter() - start_time) * 1000

            results.append(BenchmarkResult(
                name="cli_import_time",
                duration_ms=import_time,
                memory_mb=self._get_current_memory(),
                success=import_time < CLI_STARTUP_MAX_MS,
                details={"import_modules": "CLI and dependencies"}
            ))

            # Test CLI instantiation performance
            start_time = time.perf_counter()
            CLI()
            instantiation_time = (time.perf_counter() - start_time) * 1000

            results.append(BenchmarkResult(
                name="cli_instantiation",
                duration_ms=instantiation_time,
                memory_mb=self._get_current_memory(),
                success=instantiation_time < CLI_STARTUP_MAX_MS // 4,  # Should be very fast
                details={"cli_ready": True}
            ))

        except ImportError as e:
            results.append(BenchmarkResult(
                name="cli_import_failure",
                duration_ms=0,
                memory_mb=0,
                success=False,
                details={"error": str(e)}
            ))

        return results

    def run_memory_benchmarks(self) -> list[BenchmarkResult]:
        """Benchmark memory usage patterns and growth."""
        self.console.print("🧠 [cyan]Running memory performance benchmarks...[/cyan]")
        results = []

        initial_memory = self._get_current_memory()

        # Simulate typical usage pattern
        try:
            cache = get_model_cache()
            test_data = {"provider": "test_provider", "models": [f"model_{i}" for i in range(100)]}

            # Perform memory-intensive operations
            for i in range(MEMORY_SAMPLE_COUNT):
                cache.put(f"memory_test_{i}", test_data, ttl_seconds=300)
                if i % 10 == 0:  # Sample memory periodically
                    self._get_current_memory()

            final_memory = self._get_current_memory()
            memory_growth = final_memory - initial_memory

            results.append(BenchmarkResult(
                name="memory_growth_test",
                duration_ms=0,  # Not time-based
                memory_mb=memory_growth,
                success=memory_growth < MEMORY_GROWTH_MAX_MB,
                details={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "growth_mb": memory_growth,
                    "operations": MEMORY_SAMPLE_COUNT
                }
            ))

        except Exception as e:
            results.append(BenchmarkResult(
                name="memory_test_failure",
                duration_ms=0,
                memory_mb=0,
                success=False,
                details={"error": str(e)}
            ))

        return results

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def analyze_results(self, report: PerformanceReport) -> None:
        """Analyze benchmark results and apply quality gates."""
        all_results = report.cache_performance + report.cli_performance + report.memory_performance

        passed = sum(1 for result in all_results if result.success)
        failed = sum(1 for result in all_results if not result.success)

        report.quality_gates_passed = passed
        report.quality_gates_failed = failed

        # Overall status determination
        if failed == 0:
            report.overall_status = "EXCELLENT"
        elif failed <= 2:
            report.overall_status = "GOOD"
        elif failed <= 4:
            report.overall_status = "WARNING"
        else:
            report.overall_status = "CRITICAL"

    def display_results(self, report: PerformanceReport) -> None:
        """Display formatted benchmark results."""
        # Summary status
        status_colors = {
            "EXCELLENT": "green",
            "GOOD": "cyan",
            "WARNING": "yellow",
            "CRITICAL": "red"
        }
        status_color = status_colors.get(report.overall_status, "white")

        self.console.print(f"\n[bold {status_color}]🎯 Performance Status: {report.overall_status}[/bold {status_color}]")
        self.console.print(f"[green]✅ Quality Gates Passed: {report.quality_gates_passed}[/green]")
        if report.quality_gates_failed > 0:
            self.console.print(f"[red]❌ Quality Gates Failed: {report.quality_gates_failed}[/red]")

        # Detailed results table
        table = Table(title="Performance Benchmark Results")
        table.add_column("Test Category", style="cyan")
        table.add_column("Test Name", style="white")
        table.add_column("Duration (ms)", justify="right")
        table.add_column("Memory (MB)", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        categories = [
            ("Cache Performance", report.cache_performance),
            ("CLI Performance", report.cli_performance),
            ("Memory Performance", report.memory_performance)
        ]

        for category_name, results in categories:
            for i, result in enumerate(results):
                status_icon = "✅" if result.success else "❌"
                duration_str = f"{result.duration_ms:.2f}" if result.duration_ms > 0 else "N/A"
                memory_str = f"{result.memory_mb:.1f}" if result.memory_mb > 0 else "N/A"
                details_str = ", ".join(f"{k}: {v}" for k, v in list(result.details.items())[:2])

                table.add_row(
                    category_name if i == 0 else "",
                    result.name,
                    duration_str,
                    memory_str,
                    status_icon,
                    details_str[:60] + "..." if len(details_str) > 60 else details_str
                )

        self.console.print(table)

        # Recommendations
        if report.quality_gates_failed > 0:
            self.console.print("\n[bold yellow]🔧 Performance Recommendations:[/bold yellow]")
            for category_name, results in categories:
                failed_results = [r for r in results if not r.success]
                for result in failed_results:
                    self.console.print(f"  • {result.name}: Review {result.details}")

    async def run_full_benchmark(self) -> PerformanceReport:
        """Run complete performance benchmark suite."""
        self.console.print("[bold cyan]🚀 Starting Performance Regression Prevention Benchmark Suite[/bold cyan]\n")

        report = PerformanceReport()

        # Run benchmark categories
        report.cache_performance = self.run_cache_benchmarks()
        report.cli_performance = self.run_cli_benchmarks()
        report.memory_performance = self.run_memory_benchmarks()

        # Analyze and display results
        self.analyze_results(report)
        self.display_results(report)

        return report


def main() -> None:
    """Main benchmark execution function."""
    console = Console()
    benchmark = PerformanceBenchmark(console)

    try:
        report = asyncio.run(benchmark.run_full_benchmark())

        # Exit with error code if critical performance issues detected
        if report.overall_status == "CRITICAL":
            console.print("\n[bold red]💥 CRITICAL: Performance regression detected![/bold red]")
            console.print("[red]Some optimizations have regressed beyond acceptable thresholds.[/red]")
            sys.exit(1)
        elif report.overall_status == "WARNING":
            console.print("\n[bold yellow]⚠️  WARNING: Performance degradation detected[/bold yellow]")
            console.print("[yellow]Monitor these metrics closely and consider optimization.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]❌ Benchmark execution failed: {e}[/bold red]")
        logger.exception("Benchmark failure")
        sys.exit(1)


if __name__ == "__main__":
    main()
