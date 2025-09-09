# this_file: tests/test_performance_benchmark.py

"""
Test suite for performance benchmark validation.
Ensures the performance regression prevention system works correctly.
"""

import asyncio
import stat
import sys
from pathlib import Path

import pytest
from rich.console import Console

# Add scripts directory to path for performance benchmark imports
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from performance_benchmark import (
    BenchmarkResult,
    PerformanceBenchmark, 
    PerformanceReport,
    CACHE_ACCESS_MAX_MS,
    CACHE_WRITE_MAX_MS,
    CLI_STARTUP_MAX_MS,
    MEMORY_GROWTH_MAX_MB,
    MIN_CACHE_HIT_RATE
)


def test_performance_benchmark_imports():
    """Test that performance benchmark script can be imported."""
    # Should be able to import the benchmark module - imports are now at top level
    try:
        # Test that the classes are accessible
        assert BenchmarkResult is not None
        assert PerformanceBenchmark is not None
        assert PerformanceReport is not None
        assert True  # Import successful
    except NameError:
        pytest.fail("Failed to import performance benchmark components")

def test_benchmark_result_creation():
    """Test that BenchmarkResult objects can be created properly."""

    result = BenchmarkResult(
        name="test_cache_access",
        duration_ms=85.0,
        memory_mb=12.5,
        success=True,
        details={"operations": 100, "hit_rate": 95.0}
    )

    assert result.name == "test_cache_access"
    assert result.duration_ms == 85.0
    assert result.memory_mb == 12.5
    assert result.success is True
    assert result.details["operations"] == 100

def test_performance_report_creation():
    """Test that PerformanceReport objects work correctly."""

    report = PerformanceReport()

    # Add sample results
    cache_result = BenchmarkResult("cache_test", 95.0, 10.0, True)
    report.cache_performance.append(cache_result)

    assert len(report.cache_performance) == 1
    assert report.cache_performance[0].name == "cache_test"
    assert report.overall_status == "UNKNOWN"  # Before analysis

def test_benchmark_thresholds():
    """Test that performance thresholds are properly defined."""

    # Verify thresholds are reasonable
    assert CACHE_ACCESS_MAX_MS == 100  # Sub-100ms requirement
    assert CACHE_WRITE_MAX_MS == 50    # Write operations should be fast
    assert CLI_STARTUP_MAX_MS == 2000  # 2 second startup is reasonable
    assert MEMORY_GROWTH_MAX_MB == 50  # 50MB growth limit
    assert MIN_CACHE_HIT_RATE == 75    # 75% minimum hit rate

@pytest.mark.slow
def test_benchmark_execution():
    """Test that benchmark can be executed (slow test)."""

    console = Console()
    benchmark = PerformanceBenchmark(console)

    # Test that benchmark methods can be called without errors
    try:
        cli_results = benchmark.run_cli_benchmarks()
        assert isinstance(cli_results, list)
        assert len(cli_results) > 0

        memory_results = benchmark.run_memory_benchmarks()
        assert isinstance(memory_results, list)
        assert len(memory_results) > 0

    except Exception as e:
        pytest.fail(f"Benchmark execution failed: {e}")

def test_performance_gates_script_exists():
    """Test that the performance gates script exists and is executable."""
    script_path = Path(__file__).parent.parent / "scripts" / "run_performance_gates.sh"

    assert script_path.exists(), "Performance gates script not found"
    assert script_path.is_file(), "Performance gates script is not a file"

    # Check if script has executable permissions (on Unix systems)
    if hasattr(stat, 'S_IEXEC'):
        file_stat = script_path.stat()
        assert file_stat.st_mode & stat.S_IEXEC, "Script is not executable"

def test_github_workflow_exists():
    """Test that GitHub Actions workflow exists."""
    workflow_path = Path(__file__).parent.parent / ".github" / "workflows" / "performance-regression-prevention.yml"

    assert workflow_path.exists(), "GitHub Actions workflow not found"
    assert workflow_path.is_file(), "Workflow path is not a file"

    # Basic validation that it's a YAML file
    content = workflow_path.read_text()
    assert "name: Performance Regression Prevention" in content
    assert "performance-benchmark:" in content
    assert "CACHE_ACCESS_MAX_MS" in content
