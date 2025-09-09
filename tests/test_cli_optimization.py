"""
this_file: tests/test_cli_optimization.py

Comprehensive unit tests for CLI optimization components.
"""

import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from vexy_co_model_catalog.core.cli_optimization import (
    CacheEntry,
    CLIPerformanceOptimizer,
    CLIResponseCache,
    LazyImportManager,
    cached_command,
    get_cached_help_text,
    get_cached_provider_list,
    get_cli_optimizer,
    get_cli_performance_report,
    invalidate_cli_cache,
    lazy_import,
    optimize_cli_startup,
    persist_cli_cache,
)


class TestCacheEntry(unittest.TestCase):
    """Test CacheEntry functionality."""

    def test_cache_entry_creation(self):
        """Test cache entry creation with proper defaults."""
        data = {"test": "data"}
        entry = CacheEntry(data=data, created_at=datetime.now(timezone.utc), ttl_seconds=300)

        assert entry.data == data
        assert entry.ttl_seconds == 300
        assert entry.access_count == 0
        assert isinstance(entry.last_accessed, datetime)

    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic."""
        # Non-expiring entry
        entry_no_expire = CacheEntry(data="test", created_at=datetime.now(timezone.utc), ttl_seconds=0)
        assert not entry_no_expire.is_expired()

        # Expired entry
        entry_expired = CacheEntry(
            data="test", created_at=datetime.now(timezone.utc) - timedelta(seconds=400), ttl_seconds=300
        )
        assert entry_expired.is_expired()

        # Valid entry
        entry_valid = CacheEntry(
            data="test", created_at=datetime.now(timezone.utc) - timedelta(seconds=100), ttl_seconds=300
        )
        assert not entry_valid.is_expired()

    def test_cache_entry_touch(self):
        """Test cache entry access tracking."""
        entry = CacheEntry(data="test", created_at=datetime.now(timezone.utc), ttl_seconds=300)
        initial_count = entry.access_count
        initial_time = entry.last_accessed

        time.sleep(0.01)  # Small delay to ensure time difference
        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time


class TestCLIResponseCache(unittest.TestCase):
    """Test CLI response cache functionality."""

    def setUp(self):
        """Set up test cache with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CLIResponseCache(cache_dir=self.temp_dir, max_entries=10)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        key = "test_key"
        value = {"data": "test_value"}

        # Set value
        self.cache.set(key, value, ttl_seconds=300)

        # Get value
        retrieved = self.cache.get(key)
        assert retrieved == value

    def test_cache_miss(self):
        """Test cache miss scenarios."""
        # Non-existent key
        result = self.cache.get("non_existent")
        assert result is None

        # Expired key
        self.cache.set("expired_key", "value", ttl_seconds=1)
        time.sleep(1.1)
        result = self.cache.get("expired_key")
        assert result is None

    def test_cache_invalidation(self):
        """Test cache invalidation patterns."""
        # Set multiple entries
        self.cache.set("prefix_key1", "value1", ttl_seconds=300)
        self.cache.set("prefix_key2", "value2", ttl_seconds=300)
        self.cache.set("other_key", "value3", ttl_seconds=300)

        # Invalidate all entries
        count = self.cache.invalidate()
        assert count == 3
        assert self.cache.get("prefix_key1") is None

        # Set entries again for pattern test
        self.cache.set("prefix_key1", "value1", ttl_seconds=300)
        self.cache.set("prefix_key2", "value2", ttl_seconds=300)
        self.cache.set("other_key", "value3", ttl_seconds=300)

        # Invalidate by pattern
        count = self.cache.invalidate("prefix_")
        assert count == 2
        assert self.cache.get("prefix_key1") is None
        assert self.cache.get("prefix_key2") is None
        assert self.cache.get("other_key") is not None

    def test_cache_cleanup_expired(self):
        """Test expired entry cleanup."""
        # Set entries with different TTLs
        self.cache.set("short_ttl", "value1", ttl_seconds=1)
        self.cache.set("long_ttl", "value2", ttl_seconds=300)

        time.sleep(1.1)

        # Cleanup expired entries
        cleaned_count = self.cache.cleanup_expired()
        assert cleaned_count == 1

        # Verify only non-expired entry remains
        assert self.cache.get("short_ttl") is None
        assert self.cache.get("long_ttl") is not None

    def test_cache_eviction(self):
        """Test cache entry eviction when max_entries is reached."""
        # Fill cache to capacity
        for i in range(12):  # More than max_entries (10)
            self.cache.set(f"key_{i}", f"value_{i}", ttl_seconds=300)

        # Should have evicted some entries
        stats = self.cache.get_stats()
        assert stats["entries"] <= 10
        assert stats["evictions"] > 0

    def test_cache_stats(self):
        """Test cache statistics reporting."""
        # Perform cache operations
        self.cache.set("key1", "value1", ttl_seconds=300)
        self.cache.get("key1")  # Hit
        self.cache.get("non_existent")  # Miss

        stats = self.cache.get_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "entries" in stats
        assert "evictions" in stats
        assert "expired_entries" in stats

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1


class TestLazyImportManager(unittest.TestCase):
    """Test lazy import manager functionality."""

    def setUp(self):
        """Set up lazy import manager."""
        self.manager = LazyImportManager()

    def test_lazy_import_module(self):
        """Test lazy importing of modules."""
        # Import a standard library module
        os_module = self.manager.lazy_import("os")
        assert os_module == os

        # Should be cached on second call
        os_module2 = self.manager.lazy_import("os")
        assert os_module is os_module2

    def test_lazy_import_attribute(self):
        """Test lazy importing of module attributes."""
        # Import specific attribute from module
        path_class = self.manager.lazy_import("pathlib", "Path")
        assert path_class == Path

        # Should be cached
        path_class2 = self.manager.lazy_import("pathlib", "Path")
        assert path_class is path_class2

    def test_lazy_import_nonexistent(self):
        """Test lazy import of nonexistent modules."""
        with pytest.raises(ImportError):
            self.manager.lazy_import("nonexistent_module_12345")

    def test_import_stats(self):
        """Test import statistics tracking."""
        # Perform some imports
        self.manager.lazy_import("os")
        self.manager.lazy_import("pathlib", "Path")

        stats = self.manager.get_import_stats()

        assert "imported_modules" in stats
        assert "total_import_time" in stats
        assert "import_times" in stats

        assert stats["imported_modules"] == 2
        assert isinstance(stats["import_times"], dict)


class TestCLIPerformanceOptimizer(unittest.TestCase):
    """Test CLI performance optimizer functionality."""

    def setUp(self):
        """Set up performance optimizer."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = CLIPerformanceOptimizer(cache_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cached_command_decorator(self):
        """Test cached command decorator functionality."""
        call_count = 0

        @self.optimizer.cached_command(ttl_seconds=300)
        def test_function(arg1, arg2="default"):
            nonlocal call_count
            call_count += 1
            return f"result_{arg1}_{arg2}"

        # First call should execute function
        result1 = test_function("test", arg2="value")
        assert call_count == 1
        assert result1 == "result_test_value"

        # Second call with same args should use cache
        result2 = test_function("test", arg2="value")
        assert call_count == 1  # Should not increment
        assert result2 == "result_test_value"

        # Call with different args should execute function
        result3 = test_function("different")
        assert call_count == 2
        assert result3 == "result_different_default"

    def test_lazy_import_proxy(self):
        """Test lazy import through optimizer."""
        # Test module import
        os_module = self.optimizer.lazy_import("os")
        assert os_module == os

        # Test attribute import
        path_class = self.optimizer.lazy_import("pathlib", "Path")
        assert path_class == Path

    def test_command_cache_invalidation(self):
        """Test command-specific cache invalidation."""

        @self.optimizer.cached_command(ttl_seconds=300)
        def test_command():
            return "test_result"

        # Execute command to populate cache
        test_command()

        # Invalidate command cache
        count = self.optimizer.invalidate_command_cache("test_command")
        assert count >= 0

    def test_performance_report(self):
        """Test performance report generation."""

        # Execute some operations to generate data
        @self.optimizer.cached_command(ttl_seconds=300)
        def test_cmd():
            time.sleep(0.01)
            return "result"

        test_cmd()
        self.optimizer.lazy_import("os")

        report = self.optimizer.get_performance_report()

        assert "uptime" in report
        assert "cache_stats" in report
        assert "import_stats" in report
        assert "command_stats" in report
        assert "optimization_recommendations" in report

    def test_startup_optimization(self):
        """Test startup optimization functionality."""
        results = self.optimizer.optimize_startup()

        assert "cache_cleanup" in results
        assert "lazy_imports" in results
        assert "startup_time" in results

        assert isinstance(results["cache_cleanup"], int)
        assert isinstance(results["lazy_imports"], int)
        assert isinstance(results["startup_time"], float)


class TestGlobalFunctions(unittest.TestCase):
    """Test global convenience functions."""

    def test_get_cli_optimizer(self):
        """Test global optimizer instance creation."""
        optimizer1 = get_cli_optimizer()
        optimizer2 = get_cli_optimizer()

        # Should return same instance (singleton pattern)
        assert optimizer1 is optimizer2
        assert isinstance(optimizer1, CLIPerformanceOptimizer)

    @patch("vexy_co_model_catalog.core.cli_optimization.get_cli_optimizer")
    def test_cached_command_global(self, mock_get_optimizer):
        """Test global cached_command decorator."""
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer

        # Use global cached_command decorator
        @cached_command(ttl_seconds=60)
        def test_func():
            return "test"

        # Should call optimizer's cached_command method
        mock_optimizer.cached_command.assert_called_once_with(60, None)

    @patch("vexy_co_model_catalog.core.cli_optimization.get_cli_optimizer")
    def test_lazy_import_global(self, mock_get_optimizer):
        """Test global lazy_import function."""
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer
        mock_optimizer.lazy_import.return_value = "mocked_module"

        result = lazy_import("test_module", "test_attr")

        mock_optimizer.lazy_import.assert_called_once_with("test_module", "test_attr")
        assert result == "mocked_module"

    def test_get_cached_provider_list(self):
        """Test cached provider list function."""
        # Should work even without actual providers (returns empty list)
        providers = get_cached_provider_list()
        assert isinstance(providers, list)

    def test_get_cached_help_text(self):
        """Test cached help text function."""
        # Test known command
        help_text = get_cached_help_text("providers")
        assert "provider" in help_text.lower()

        # Test unknown command
        help_text = get_cached_help_text("unknown_command")
        assert "unknown_command" in help_text

    @patch("vexy_co_model_catalog.core.cli_optimization.get_cli_optimizer")
    def test_optimize_cli_startup_global(self, mock_get_optimizer):
        """Test global optimize_cli_startup function."""
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer
        mock_optimizer.optimize_startup.return_value = {"test": "result"}

        result = optimize_cli_startup()

        mock_optimizer.optimize_startup.assert_called_once()
        assert result == {"test": "result"}

    @patch("vexy_co_model_catalog.core.cli_optimization.get_cli_optimizer")
    def test_get_cli_performance_report_global(self, mock_get_optimizer):
        """Test global performance report function."""
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer
        mock_optimizer.get_performance_report.return_value = {"uptime": "1.0s"}

        result = get_cli_performance_report()

        mock_optimizer.get_performance_report.assert_called_once()
        assert result == {"uptime": "1.0s"}

    @patch("vexy_co_model_catalog.core.cli_optimization.get_cli_optimizer")
    def test_invalidate_cli_cache_global(self, mock_get_optimizer):
        """Test global cache invalidation function."""
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer
        mock_optimizer.response_cache.invalidate.return_value = 5

        result = invalidate_cli_cache("test_pattern")

        mock_optimizer.response_cache.invalidate.assert_called_once_with("test_pattern")
        assert result == 5

    @patch("vexy_co_model_catalog.core.cli_optimization.get_cli_optimizer")
    def test_persist_cli_cache_global(self, mock_get_optimizer):
        """Test global cache persistence function."""
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer
        mock_optimizer.response_cache.persist.return_value = True

        result = persist_cli_cache()

        mock_optimizer.response_cache.persist.assert_called_once()
        assert result


if __name__ == "__main__":
    unittest.main()
