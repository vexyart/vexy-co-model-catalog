# this_file: src/vexy_co_model_catalog/core/caching.py

"""
Intelligent TTL-based caching system with persistence and auto-invalidation.
Provides performance optimization for model metadata, validation results, and API responses.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class CacheStrategy(Enum):
    """Cache invalidation and persistence strategies."""

    MEMORY_ONLY = "memory_only"  # In-memory only, no persistence
    PERSISTENT = "persistent"  # Persist to disk with TTL
    WRITE_THROUGH = "write_through"  # Immediate disk writes
    WRITE_BACK = "write_back"  # Batched disk writes


class CacheEvictionPolicy(Enum):
    """Cache eviction policies when memory limits are reached."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL_BASED = "ttl_based"  # Evict by TTL expiration
    SIZE_BASED = "size_based"  # Evict by entry size


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl_seconds: float
    access_count: int = 0
    size_bytes: int = 0
    tags: list[str] = None

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()

    def _calculate_size(self) -> int:
        """Fast memory size estimation optimized for performance."""
        try:
            if isinstance(self.value, str):
                return len(self.value.encode('utf-8'))
            if isinstance(self.value, bytes):
                return len(self.value)
            if isinstance(self.value, dict):
                # Fast approximation: avoid expensive JSON serialization
                return self._fast_dict_size(self.value)
            if isinstance(self.value, list):
                return self._fast_list_size(self.value)
            if isinstance(self.value, int | float | bool):
                return 24  # Approximate Python object overhead + value
            # For other types, use a reasonable estimate to avoid pickle overhead
            return 1024
        except Exception:
            return 1024  # Safe fallback

    def _fast_dict_size(self, d: dict) -> int:
        """Fast dictionary size estimation without serialization."""
        size = 240  # Base dict overhead
        for k, v in d.items():
            size += len(str(k)) * 4  # Key estimate
            if isinstance(v, str):
                size += len(v.encode('utf-8'))
            elif isinstance(v, dict | list):
                size += 100  # Nested structure estimate
            else:
                size += 50  # Other types estimate
            if size > 10000:  # Cap expensive calculations
                break
        return min(size, 100000)  # Reasonable upper bound

    def _fast_list_size(self, lst: list) -> int:
        """Fast list size estimation."""
        if not lst:
            return 64
        size = 64 + len(lst) * 8  # Base list + pointer overhead
        for item in lst[:10]:  # Sample first 10 items
            if isinstance(item, str):
                size += len(item.encode('utf-8'))
            else:
                size += 50  # Estimate for other types
        return size * (len(lst) // max(len(lst[:10]), 1))  # Extrapolate

    def is_expired(self, current_time: float | None = None) -> bool:
        """Check if the cache entry has expired."""
        if current_time is None:
            current_time = time.time()
        return (current_time - self.created_at) >= self.ttl_seconds

    def touch(self) -> None:
        """Update access time and increment access count."""
        self.accessed_at = time.time()
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "ttl_seconds": self.ttl_seconds,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Create from dictionary for persistence."""
        return cls(**data)


@dataclass
class CacheConfiguration:
    """Configuration for cache behavior."""

    # Basic settings
    max_memory_mb: int = 100
    default_ttl_seconds: int = 3600  # 1 hour
    max_entries: int = 10000

    # Persistence settings
    cache_directory: Path | None = None
    strategy: CacheStrategy = CacheStrategy.PERSISTENT
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU

    # Performance settings - optimized for sub-100ms responses
    cleanup_interval_seconds: int = 600  # 10 minutes (reduced cleanup frequency)
    batch_write_size: int = 500  # Larger batches for efficiency
    enable_compression: bool = False  # Disable for speed (use disk space instead)
    lazy_loading: bool = True  # Load cache entries on demand
    memory_optimization: bool = True  # Enable memory usage optimizations

    # Advanced settings
    enable_statistics: bool = True
    enable_automatic_warming: bool = True
    warming_patterns: list[str] = None

    def __post_init__(self) -> None:
        if self.warming_patterns is None:
            self.warming_patterns = []
        if self.cache_directory is None:
            self.cache_directory = Path.cwd() / ".cache" / "vexy_model_catalog"


class IntelligentCache:
    """
    High-performance TTL-based cache with persistence and intelligent invalidation.
    Designed for model metadata, validation results, and API response caching.
    """

    def __init__(self, name: str, config: CacheConfiguration | None = None) -> None:
        """Initialize the intelligent cache."""
        self.name = name
        self.config = config or CacheConfiguration()

        # Core cache storage
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

        # Statistics tracking
        self._stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "evictions": 0,
            "invalidations": 0,
            "disk_reads": 0,
            "disk_writes": 0,
        }

        # Cleanup and persistence
        self._last_cleanup = time.time()
        self._dirty_keys: set = set()
        self._cleanup_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Performance optimization caches
        self._cached_memory_usage = 0.0
        self._memory_cache_valid = False
        self._memory_cache_lock = threading.Lock()
        self._access_order: dict[str, float] = {}  # Fast LRU tracking

        # Setup persistence
        if self.config.strategy != CacheStrategy.MEMORY_ONLY:
            self._setup_persistence()

        # Start background cleanup
        self._start_cleanup_thread()

        logger.debug(f"Initialized intelligent cache '{name}' with {self.config.strategy.value} strategy")

    def _setup_persistence(self) -> None:
        """Setup persistent storage directory and load existing data."""
        if not self.config.cache_directory:
            return

        self.cache_file = self.config.cache_directory / f"{self.name}.cache"
        self.config.cache_directory.mkdir(parents=True, exist_ok=True)

        # Load existing cache data
        if self.cache_file.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load cache entries from persistent storage."""
        try:
            with open(self.cache_file, "rb") as f:
                data = pickle.load(f)

            current_time = time.time()
            loaded_count = 0
            expired_count = 0

            for entry_data in data.get("entries", []):
                entry = CacheEntry.from_dict(entry_data)

                if not entry.is_expired(current_time):
                    self._entries[entry.key] = entry
                    loaded_count += 1
                else:
                    expired_count += 1

            logger.debug(f"Loaded {loaded_count} cache entries, skipped {expired_count} expired entries")
            self._stats["disk_reads"] += 1

        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")

    def _save_to_disk(self) -> None:
        """Save cache entries to persistent storage."""
        if self.config.strategy == CacheStrategy.MEMORY_ONLY or not self.config.cache_directory:
            return

        try:
            # Prepare data for serialization
            entries_data = []
            current_time = time.time()

            for entry in self._entries.values():
                if not entry.is_expired(current_time):
                    entries_data.append(entry.to_dict())

            data = {
                "cache_name": self.name,
                "saved_at": current_time,
                "entries": entries_data,
                "stats": self._stats.copy(),
            }

            # Atomic write
            temp_file = self.cache_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                pickle.dump(data, f)

            temp_file.replace(self.cache_file)
            self._stats["disk_writes"] += 1
            self._dirty_keys.clear()

            logger.debug(f"Saved {len(entries_data)} cache entries to disk")

        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker, name=f"cache-cleanup-{self.name}", daemon=True
        )
        self._cleanup_thread.start()

    def _cleanup_worker(self) -> None:
        """Background worker for cache cleanup and persistence."""
        while not self._shutdown_event.wait(self.config.cleanup_interval_seconds):
            try:
                self._perform_cleanup()

                # Handle persistence based on strategy
                if self.config.strategy == CacheStrategy.WRITE_BACK:
                    if len(self._dirty_keys) >= self.config.batch_write_size:
                        self._save_to_disk()
                elif self.config.strategy == CacheStrategy.PERSISTENT:
                    # Periodic saves for persistent strategy
                    if len(self._dirty_keys) > 0:
                        self._save_to_disk()

            except Exception as e:
                logger.warning(f"Cache cleanup error: {e}")

    def _perform_cleanup(self) -> None:
        """Perform cache cleanup - remove expired entries and handle eviction."""
        with self._lock:
            current_time = time.time()

            # Remove expired entries
            expired_keys = [key for key, entry in self._entries.items() if entry.is_expired(current_time)]

            for key in expired_keys:
                del self._entries[key]
                self._dirty_keys.discard(key)

                # Clean up fast access tracking
                if self.config.memory_optimization:
                    self._access_order.pop(key, None)

                self._stats["evictions"] += 1

            if expired_keys:
                # Invalidate memory usage cache after cleanup
                if self.config.memory_optimization:
                    with self._memory_cache_lock:
                        self._memory_cache_valid = False

                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            # Handle memory pressure
            current_memory_mb = self._estimate_memory_usage()
            if current_memory_mb > self.config.max_memory_mb or len(self._entries) > self.config.max_entries:
                self._handle_eviction()

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB with caching for performance."""
        if self.config.memory_optimization:
            # Use cached value when available
            with self._memory_cache_lock:
                if self._memory_cache_valid:
                    return self._cached_memory_usage

                # Recalculate and cache
                total_bytes = sum(entry.size_bytes for entry in self._entries.values())
                self._cached_memory_usage = total_bytes / (1024 * 1024)
                self._memory_cache_valid = True
                return self._cached_memory_usage
        else:
            # Fallback to original method
            total_bytes = sum(entry.size_bytes for entry in self._entries.values())
            return total_bytes / (1024 * 1024)

    def _handle_eviction(self) -> None:
        """Handle cache eviction based on configured policy with performance optimizations."""
        if self.config.eviction_policy == CacheEvictionPolicy.LRU:
            # Use fast access order tracking if available
            if self.config.memory_optimization and self._access_order:
                sorted_entries = sorted(self._access_order.items(), key=lambda x: x[1])
                sorted_entries = [(key, self._entries[key]) for key, _ in sorted_entries if key in self._entries]
            else:
                # Fallback to original method
                sorted_entries = sorted(self._entries.items(), key=lambda x: x[1].accessed_at)
        elif self.config.eviction_policy == CacheEvictionPolicy.LFU:
            # Sort by access count, remove least frequent
            sorted_entries = sorted(self._entries.items(), key=lambda x: x[1].access_count)
        elif self.config.eviction_policy == CacheEvictionPolicy.SIZE_BASED:
            # Sort by size, remove largest first
            sorted_entries = sorted(self._entries.items(), key=lambda x: x[1].size_bytes, reverse=True)
        else:  # TTL_BASED
            # Sort by expiration time - entries expiring soonest first
            current_time = time.time()
            sorted_entries = sorted(
                self._entries.items(), key=lambda x: (x[1].created_at + x[1].ttl_seconds) - current_time
            )

        # Remove entries until under limits
        entries_to_remove = max(len(self._entries) - int(self.config.max_entries * 0.8), 0)

        for i in range(min(entries_to_remove, len(sorted_entries))):
            key = sorted_entries[i][0]
            del self._entries[key]
            self._dirty_keys.discard(key)

            # Clean up fast access tracking
            if self.config.memory_optimization:
                self._access_order.pop(key, None)

            self._stats["evictions"] += 1

        if entries_to_remove > 0:
            # Invalidate memory cache after evictions
            if self.config.memory_optimization:
                with self._memory_cache_lock:
                    self._memory_cache_valid = False

            logger.debug(f"Evicted {entries_to_remove} cache entries due to memory pressure")

    def get(self, key: str, default: T = None) -> T | None:
        """Get value from cache with optimized access tracking for sub-100ms performance."""
        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return default

            # Check expiration
            if entry.is_expired():
                del self._entries[key]
                self._dirty_keys.discard(key)

                # Clean up fast access tracking
                if self.config.memory_optimization:
                    self._access_order.pop(key, None)

                self._stats["misses"] += 1
                self._stats["evictions"] += 1
                return default

            # Update access tracking with performance optimization
            current_time = time.time()
            entry.touch()

            # Fast access order tracking for LRU optimization
            if self.config.memory_optimization:
                self._access_order[key] = current_time

            self._stats["hits"] += 1
            return entry.value

    def put(self, key: str, value: Any, ttl_seconds: int | None = None, tags: list[str] | None = None) -> None:
        """Store value in cache with optimized performance tracking."""
        if ttl_seconds is None:
            ttl_seconds = self.config.default_ttl_seconds

        if tags is None:
            tags = []

        with self._lock:
            current_time = time.time()

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                accessed_at=current_time,
                ttl_seconds=ttl_seconds,
                tags=tags,
            )

            self._entries[key] = entry
            self._dirty_keys.add(key)

            # Update fast access tracking for LRU optimization
            if self.config.memory_optimization:
                self._access_order[key] = current_time
                # Invalidate memory usage cache
                with self._memory_cache_lock:
                    self._memory_cache_valid = False

            self._stats["writes"] += 1

            # Immediate persistence for write-through strategy
            if self.config.strategy == CacheStrategy.WRITE_THROUGH:
                self._save_to_disk()

    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache with optimized cleanup."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                self._dirty_keys.discard(key)

                # Clean up fast access tracking
                if self.config.memory_optimization:
                    self._access_order.pop(key, None)
                    # Invalidate memory usage cache
                    with self._memory_cache_lock:
                        self._memory_cache_valid = False

                self._stats["invalidations"] += 1
                return True
            return False

    def invalidate_by_tags(self, tags: list[str]) -> int:
        """Invalidate all entries with matching tags."""
        with self._lock:
            keys_to_remove = []

            for key, entry in self._entries.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._entries[key]
                self._dirty_keys.discard(key)
                self._stats["invalidations"] += 1

            return len(keys_to_remove)

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate entries with keys matching pattern."""
        with self._lock:
            regex = re.compile(pattern)
            keys_to_remove = [key for key in self._entries if regex.match(key)]

            for key in keys_to_remove:
                del self._entries[key]
                self._dirty_keys.discard(key)
                self._stats["invalidations"] += 1

            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._dirty_keys.clear()
            self._stats["invalidations"] += count

            logger.debug(f"Cleared {count} cache entries")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                **self._stats,
                "entry_count": len(self._entries),
                "memory_usage_mb": self._estimate_memory_usage(),
                "hit_rate_percent": hit_rate,
                "dirty_keys_count": len(self._dirty_keys),
            }

    def warm_cache(self, key_value_pairs: dict[str, Any], ttl_seconds: int | None = None) -> None:
        """Pre-populate cache with key-value pairs."""
        for key, value in key_value_pairs.items():
            self.put(key, value, ttl_seconds)

        logger.debug(f"Warmed cache with {len(key_value_pairs)} entries")

    def memoize(self, ttl_seconds: int | None = None, tags: list[str] | None = None) -> Callable:
        """Decorator for function memoization with TTL."""

        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Create cache key from function name and arguments
                key_data = {"func": func.__name__, "args": args, "kwargs": sorted(kwargs.items())}
                cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()

                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.put(cache_key, result, ttl_seconds, tags)
                return result

            return wrapper

        return decorator

    def shutdown(self) -> None:
        """Gracefully shutdown the cache."""
        logger.debug(f"Shutting down cache '{self.name}'")

        # Signal cleanup thread to stop
        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        # Final save to disk
        if self.config.strategy != CacheStrategy.MEMORY_ONLY:
            self._save_to_disk()

        # Clear memory
        with self._lock:
            self._entries.clear()
            self._dirty_keys.clear()


# Performance-optimized cache configurations for sub-100ms response times
FAST_CACHE_CONFIG = CacheConfiguration(
    max_memory_mb=50,  # Smaller memory footprint for speed
    default_ttl_seconds=1800,  # 30 minutes
    max_entries=5000,  # Reasonable limit
    cleanup_interval_seconds=900,  # 15 minutes - less frequent cleanup
    batch_write_size=1000,  # Larger batches
    enable_compression=False,  # Disabled for speed
    lazy_loading=True,
    memory_optimization=True,
    strategy=CacheStrategy.MEMORY_ONLY,  # Pure memory for fastest access
    eviction_policy=CacheEvictionPolicy.LRU,
)

PROVIDER_CACHE_CONFIG = CacheConfiguration(
    max_memory_mb=100,  # Larger for provider data
    default_ttl_seconds=3600,  # 1 hour
    max_entries=10000,
    cleanup_interval_seconds=600,  # 10 minutes
    batch_write_size=500,
    enable_compression=False,
    lazy_loading=True,
    memory_optimization=True,
    strategy=CacheStrategy.PERSISTENT,  # Persistent for provider data
    eviction_policy=CacheEvictionPolicy.LRU,
)

VALIDATION_CACHE_CONFIG = CacheConfiguration(
    max_memory_mb=25,  # Small for validation results
    default_ttl_seconds=900,  # 15 minutes
    max_entries=2000,
    cleanup_interval_seconds=1200,  # 20 minutes
    batch_write_size=200,
    enable_compression=False,
    lazy_loading=True,
    memory_optimization=True,
    strategy=CacheStrategy.MEMORY_ONLY,  # Fast access for validations
    eviction_policy=CacheEvictionPolicy.SIZE_BASED,  # Evict large validation data first
)

# Global cache manager
_cache_instances: dict[str, IntelligentCache] = {}
_cache_lock = threading.Lock()


def get_cache(name: str, config: CacheConfiguration | None = None) -> IntelligentCache:
    """Get or create a named cache instance."""
    with _cache_lock:
        if name not in _cache_instances:
            _cache_instances[name] = IntelligentCache(name, config)
        return _cache_instances[name]


def get_model_cache() -> IntelligentCache:
    """Get cache optimized for model metadata with sub-100ms performance."""
    # Use optimized provider cache config for model metadata
    config = CacheConfiguration(
        max_memory_mb=80,  # Increased for model metadata
        default_ttl_seconds=7200,  # 2 hours - model metadata changes infrequently
        max_entries=15000,  # Higher limit for models
        cleanup_interval_seconds=600,  # 10 minutes
        batch_write_size=500,
        enable_compression=False,  # Disabled for speed
        lazy_loading=True,
        memory_optimization=True,
        strategy=CacheStrategy.PERSISTENT,  # Keep provider data
        eviction_policy=CacheEvictionPolicy.LRU,
    )
    return get_cache("model_metadata", config)


def get_validation_cache() -> IntelligentCache:
    """Get cache optimized for validation results with fast performance."""
    # Use pre-defined optimized validation cache config
    return get_cache("validation_results", VALIDATION_CACHE_CONFIG)


def get_api_response_cache() -> IntelligentCache:
    """Get cache optimized for API responses with fastest performance."""
    # Use fast cache config with API-specific TTL
    config = CacheConfiguration(
        max_memory_mb=40,  # Moderate size for API responses
        default_ttl_seconds=600,  # 10 minutes - API responses change frequently
        max_entries=3000,
        cleanup_interval_seconds=900,  # 15 minutes
        batch_write_size=1000,
        enable_compression=False,  # Disabled for speed
        lazy_loading=True,
        memory_optimization=True,
        strategy=CacheStrategy.MEMORY_ONLY,  # Pure memory for fastest access
        eviction_policy=CacheEvictionPolicy.LFU,  # Evict least frequent for API calls
    )
    return get_cache("api_responses", config)


# Convenience functions
def cache_model_data(provider_name: str, data: Any, ttl_seconds: int | None = None) -> None:
    """Cache model data for a provider."""
    cache = get_model_cache()
    cache.put(f"models:{provider_name}", data, ttl_seconds, tags=["models", provider_name])


def get_cached_model_data(provider_name: str) -> Any | None:
    """Get cached model data for a provider."""
    cache = get_model_cache()
    return cache.get(f"models:{provider_name}")


def get_provider_validation_cache() -> IntelligentCache:
    """Get ultra-fast cache for provider validation results - optimized for sub-100ms access."""
    # Use the fastest possible configuration for provider validation
    config = CacheConfiguration(
        max_memory_mb=15,  # Small size for validation results only
        default_ttl_seconds=300,  # 5 minutes - validation results change quickly
        max_entries=1000,  # Reasonable limit for provider count
        cleanup_interval_seconds=1800,  # 30 minutes - infrequent cleanup
        batch_write_size=100,
        enable_compression=False,  # Disabled for maximum speed
        lazy_loading=True,
        memory_optimization=True,
        strategy=CacheStrategy.MEMORY_ONLY,  # Fastest possible - pure memory
        eviction_policy=CacheEvictionPolicy.TTL_BASED,  # Remove stale validations quickly
    )
    return get_cache("provider_validation", config)


# Fast provider validation caching functions
def cache_provider_validation(provider_name: str, is_valid: bool, error_message: str | None = None) -> None:
    """Cache provider validation result for ultra-fast lookup."""
    cache = get_provider_validation_cache()
    validation_data = {
        "is_valid": is_valid,
        "error_message": error_message,
        "validated_at": time.time(),
    }
    cache.put(f"validation:{provider_name}", validation_data, tags=["validation", provider_name])


def get_cached_provider_validation(provider_name: str) -> dict | None:
    """Get cached provider validation result with sub-100ms performance."""
    cache = get_provider_validation_cache()
    return cache.get(f"validation:{provider_name}")


def invalidate_provider_cache(provider_name: str) -> None:
    """Invalidate all caches for a specific provider."""
    # Invalidate across all cache types
    get_model_cache().invalidate(f"models:{provider_name}")
    get_validation_cache().invalidate_by_tags([provider_name])
    get_provider_validation_cache().invalidate(f"validation:{provider_name}")
    get_api_response_cache().invalidate_by_tags([provider_name])




def shutdown_all_caches() -> None:
    """Shutdown all cache instances."""
    with _cache_lock:
        for cache in _cache_instances.values():
            cache.shutdown()
        _cache_instances.clear()
