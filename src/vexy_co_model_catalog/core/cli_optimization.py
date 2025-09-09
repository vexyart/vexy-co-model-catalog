"""
this_file: src/vexy_co_model_catalog/core/cli_optimization.py

CLI performance optimization with startup time improvements and response caching.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger
from vexy_co_model_catalog.core.provider import get_all_providers

F = TypeVar("F", bound=Callable[..., Any])

# CLI optimization performance thresholds and limits
MAX_COMMAND_EXECUTION_HISTORY = 100     # Maximum command execution history to keep
MAX_CACHE_KEYS_PER_COMMAND = 50         # Maximum cache keys per command to keep  
CACHE_HIT_RATE_THRESHOLD = 50           # Cache hit rate threshold percentage
SLOW_COMMAND_THRESHOLD_SECONDS = 2.0    # Slow command threshold in seconds
EXPIRED_CACHE_ENTRIES_THRESHOLD = 20    # Expired entries threshold for recommendations


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata."""

    data: Any
    created_at: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds <= 0:
            return False  # Never expires

        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self) -> None:
        """Update access information."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)


class CLIResponseCache:
    """High-performance response cache for CLI commands with TTL and persistence."""

    def __init__(self, cache_dir: str | Path | None = None, max_entries: int = 1000) -> None:
        """
        Initialize CLI response cache.

        Args:
            cache_dir: Directory for persistent cache storage
            max_entries: Maximum number of cache entries
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "vexy-co-model-catalog"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries

        # In-memory cache for fast access
        self._memory_cache: dict[str, CacheEntry] = {}

        # Persistent cache file
        self.cache_file = self.cache_dir / "cli_cache.pkl"

        # Load existing cache
        self._load_cache()

        # Performance metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired():
                entry.touch()
                self._hits += 1
                return entry.data
            # Remove expired entry
            del self._memory_cache[key]

        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set value in cache with TTL."""
        # Check if we need to evict entries
        if len(self._memory_cache) >= self.max_entries:
            self._evict_entries()

        entry = CacheEntry(data=value, created_at=datetime.now(timezone.utc), ttl_seconds=ttl_seconds)

        self._memory_cache[key] = entry

    def invalidate(self, pattern: str | None = None) -> int:
        """Invalidate cache entries. If pattern provided, invalidate matching keys."""
        if pattern is None:
            count = len(self._memory_cache)
            self._memory_cache.clear()
            return count

        # Simple pattern matching (startswith)
        keys_to_remove = [key for key in self._memory_cache if key.startswith(pattern)]
        for key in keys_to_remove:
            del self._memory_cache[key]

        return len(keys_to_remove)

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache."""
        expired_keys = [key for key, entry in self._memory_cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self._memory_cache[key]

        return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "entries": len(self._memory_cache),
            "evictions": self._evictions,
            "expired_entries": sum(1 for entry in self._memory_cache.values() if entry.is_expired()),
        }

    def persist(self) -> bool:
        """Persist cache to disk."""
        try:
            # Clean up expired entries before persisting
            self.cleanup_expired()

            with open(self.cache_file, "wb") as f:
                pickle.dump(self._memory_cache, f)
            return True
        except Exception as e:
            logger.warning(f"Failed to persist CLI cache: {e}")
            return False

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, "rb") as f:
                self._memory_cache = pickle.load(f)

            # Clean up expired entries
            self.cleanup_expired()

        except Exception as e:
            logger.warning(f"Failed to load CLI cache: {e}")
            self._memory_cache = {}

    def _evict_entries(self) -> None:
        """Evict least recently used entries."""
        if len(self._memory_cache) < self.max_entries:
            return

        # Sort by last accessed time (LRU)
        sorted_entries = sorted(self._memory_cache.items(), key=lambda x: x[1].last_accessed)

        # Remove oldest 25% of entries
        entries_to_remove = len(sorted_entries) // 4
        if entries_to_remove == 0:
            entries_to_remove = 1

        for i in range(entries_to_remove):
            key = sorted_entries[i][0]
            del self._memory_cache[key]
            self._evictions += 1


class LazyImportManager:
    """Manages lazy imports to improve CLI startup time."""

    def __init__(self) -> None:
        """Initialize lazy import manager."""
        self._imported_modules: dict[str, Any] = {}
        self._import_times: dict[str, float] = {}

    def lazy_import(self, module_name: str, attribute: str | None = None) -> Any:
        """
        Lazily import a module or attribute.

        Args:
            module_name: Name of module to import
            attribute: Specific attribute to import from module

        Returns:
            Imported module or attribute
        """
        cache_key = f"{module_name}.{attribute}" if attribute else module_name

        if cache_key in self._imported_modules:
            return self._imported_modules[cache_key]

        start_time = time.time()

        try:
            module = __import__(module_name, fromlist=[attribute] if attribute else [])

            result = getattr(module, attribute) if attribute else module

            self._imported_modules[cache_key] = result
            self._import_times[cache_key] = time.time() - start_time

            return result

        except ImportError as e:
            logger.warning(f"Failed to lazy import {cache_key}: {e}")
            raise

    def get_import_stats(self) -> dict[str, Any]:
        """Get import performance statistics."""
        total_time = sum(self._import_times.values())
        return {
            "imported_modules": len(self._imported_modules),
            "total_import_time": f"{total_time:.3f}s",
            "import_times": {
                module: f"{time_taken:.3f}s"
                for module, time_taken in sorted(self._import_times.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]  # Top 10 slowest imports
            },
        }


class CLIPerformanceOptimizer:
    """Main CLI performance optimizer with caching and lazy loading."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize CLI performance optimizer."""
        self.response_cache = CLIResponseCache(cache_dir)
        self.lazy_import_manager = LazyImportManager()
        self.startup_time = time.time()

        # Performance tracking
        self._command_times: dict[str, list[float]] = {}
        self._command_cache_keys: dict[str, list[str]] = {}

    def cached_command(self, ttl_seconds: int = 300, cache_key_func: Callable | None = None) -> Callable:
        """
        Decorator for caching command results.

        Args:
            ttl_seconds: Time to live for cache entries
            cache_key_func: Function to generate cache key from arguments
        """

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)

                # Try to get from cache
                cached_result = self.response_cache.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Cache the result
                self.response_cache.set(cache_key, result, ttl_seconds)

                # Track performance
                self._track_command_performance(func.__name__, execution_time)
                self._track_cache_key(func.__name__, cache_key)

                return result

            return wrapper

        return decorator

    def lazy_import(self, module_name: str, attribute: str | None = None) -> Any:
        """Proxy to lazy import manager."""
        return self.lazy_import_manager.lazy_import(module_name, attribute)

    def invalidate_command_cache(self, command_name: str) -> int:
        """Invalidate cache for a specific command."""
        pattern = f"cmd_{command_name}_"
        return self.response_cache.invalidate(pattern)

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        uptime = time.time() - self.startup_time

        # Calculate command statistics
        command_stats = {}
        for command, times in self._command_times.items():
            if times:
                command_stats[command] = {
                    "executions": len(times),
                    "avg_time": f"{sum(times) / len(times):.3f}s",
                    "min_time": f"{min(times):.3f}s",
                    "max_time": f"{max(times):.3f}s",
                    "total_time": f"{sum(times):.3f}s",
                }

        return {
            "uptime": f"{uptime:.3f}s",
            "cache_stats": self.response_cache.get_stats(),
            "import_stats": self.lazy_import_manager.get_import_stats(),
            "command_stats": command_stats,
            "optimization_recommendations": self._get_optimization_recommendations(),
        }

    def optimize_startup(self) -> dict[str, Any]:
        """Apply startup optimizations and return results."""
        optimizations = {"cache_cleanup": 0, "lazy_imports": 0, "startup_time": 0}

        # Clean up expired cache entries
        optimizations["cache_cleanup"] = self.response_cache.cleanup_expired()

        # Count lazy imports (they're already optimized by being lazy)
        optimizations["lazy_imports"] = len(self.lazy_import_manager._imported_modules)

        # Calculate startup time improvement (this is the baseline)
        optimizations["startup_time"] = time.time() - self.startup_time

        return optimizations

    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function name and arguments."""
        # Create a hash of the arguments for consistent caching
        arg_str = str(args) + str(sorted(kwargs.items()))
        arg_hash = hashlib.md5(arg_str.encode()).hexdigest()[:8]

        return f"cmd_{func_name}_{arg_hash}"

    def _track_command_performance(self, command_name: str, execution_time: float) -> None:
        """Track command execution performance."""
        if command_name not in self._command_times:
            self._command_times[command_name] = []

        self._command_times[command_name].append(execution_time)

        # Keep only last executions per command
        if len(self._command_times[command_name]) > MAX_COMMAND_EXECUTION_HISTORY:
            self._command_times[command_name] = self._command_times[command_name][-MAX_COMMAND_EXECUTION_HISTORY:]

    def _track_cache_key(self, command_name: str, cache_key: str) -> None:
        """Track cache keys used by commands."""
        if command_name not in self._command_cache_keys:
            self._command_cache_keys[command_name] = []

        self._command_cache_keys[command_name].append(cache_key)

        # Keep only last cache keys per command
        if len(self._command_cache_keys[command_name]) > MAX_CACHE_KEYS_PER_COMMAND:
            cache_keys = self._command_cache_keys[command_name]
            self._command_cache_keys[command_name] = cache_keys[-MAX_CACHE_KEYS_PER_COMMAND:]

    def _get_optimization_recommendations(self) -> list[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []

        cache_stats = self.response_cache.get_stats()

        # Check cache hit rate
        hit_rate = float(cache_stats["hit_rate"].rstrip("%"))
        if hit_rate < CACHE_HIT_RATE_THRESHOLD:
            recommendations.append("Consider increasing cache TTL for better hit rates")

        # Check for slow commands
        for command, times in self._command_times.items():
            if times:
                avg_time = sum(times) / len(times)
                if avg_time > SLOW_COMMAND_THRESHOLD_SECONDS:
                    recommendations.append(
                        f"Command '{command}' is slow (avg: {avg_time:.2f}s) - consider optimization"
                    )

        # Check expired entries
        expired_entries = cache_stats.get("expired_entries", 0)
        if expired_entries > EXPIRED_CACHE_ENTRIES_THRESHOLD:
            recommendations.append("Many expired cache entries - consider cleanup or TTL adjustment")

        # Check import performance
        import_stats = self.lazy_import_manager.get_import_stats()
        total_import_time = float(import_stats["total_import_time"].rstrip("s"))
        if total_import_time > 1.0:
            recommendations.append("Import time is high - consider more aggressive lazy loading")

        if not recommendations:
            recommendations.append("Performance is optimal - no recommendations")

        return recommendations


# Global optimizer instance
_cli_optimizer: CLIPerformanceOptimizer | None = None


def get_cli_optimizer() -> CLIPerformanceOptimizer:
    """Get or create global CLI optimizer instance."""
    global _cli_optimizer
    if _cli_optimizer is None:
        _cli_optimizer = CLIPerformanceOptimizer()
    return _cli_optimizer


def cached_command(ttl_seconds: int = 300, cache_key_func: Callable | None = None) -> Callable:
    """Decorator for caching command results. Uses global optimizer."""
    return get_cli_optimizer().cached_command(ttl_seconds, cache_key_func)


def lazy_import(module_name: str, attribute: str | None = None) -> Any:
    """Lazy import function using global optimizer."""
    return get_cli_optimizer().lazy_import(module_name, attribute)


@lru_cache(maxsize=128)
def get_cached_provider_list() -> list[str]:
    """Get cached provider list for fast access."""
    # This would normally fetch from the actual provider system
    # Using LRU cache for very fast repeated access
    try:
        providers = get_all_providers()
        return [provider.name for provider in providers]
    except ImportError:
        return []


@lru_cache(maxsize=64)
def get_cached_help_text(command: str) -> str:
    """Get cached help text for commands."""
    # This would normally generate help text dynamically
    # Using LRU cache for instant help display
    help_texts = {
        "providers": "Manage AI model providers and configurations",
        "fetch": "Fetch model catalogs from providers",
        "stats": "Show system statistics and health information",
        "clean": "Clean up temporary files and caches",
        "validate": "Validate provider configurations",
        "health": "Check system health and diagnostics",
        "performance": "Show performance metrics and optimization recommendations",
    }

    return help_texts.get(command, f"Help for command: {command}")


def optimize_cli_startup() -> dict[str, Any]:
    """Apply CLI startup optimizations."""
    optimizer = get_cli_optimizer()
    return optimizer.optimize_startup()


def get_cli_performance_report() -> dict[str, Any]:
    """Get comprehensive CLI performance report."""
    optimizer = get_cli_optimizer()
    return optimizer.get_performance_report()


def invalidate_cli_cache(pattern: str | None = None) -> int:
    """Invalidate CLI response cache."""
    optimizer = get_cli_optimizer()
    return optimizer.response_cache.invalidate(pattern)


def persist_cli_cache() -> bool:
    """Persist CLI cache to disk."""
    optimizer = get_cli_optimizer()
    return optimizer.response_cache.persist()
