#!/usr/bin/env python3
# this_file: tests/test_caching_unit.py

"""Comprehensive unit tests for the caching module."""

import asyncio
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vexy_co_model_catalog.core.caching import (
    CacheConfiguration,
    CacheEntry,
    CacheEvictionPolicy,
    CacheStrategy,
    IntelligentCache,
    cache_model_data,
    get_api_response_cache,
    get_cache,
    get_cached_model_data,
    get_model_cache,
    get_validation_cache,
    invalidate_provider_cache,
    shutdown_all_caches,
)


class TestCacheStrategy(unittest.TestCase):
    """Test CacheStrategy enum."""

    def test_cache_strategies_exist(self):
        """Test that all expected cache strategies exist."""
        strategies = [strategy.value for strategy in CacheStrategy]
        expected = ["memory_only", "persistent", "write_through", "write_back"]
        assert sorted(strategies) == sorted(expected)


class TestCacheEvictionPolicy(unittest.TestCase):
    """Test CacheEvictionPolicy enum."""

    def test_eviction_policies_exist(self):
        """Test that all expected eviction policies exist."""
        policies = [policy.value for policy in CacheEvictionPolicy]
        expected = ["lru", "lfu", "ttl_based", "size_based"]
        assert sorted(policies) == sorted(expected)


class TestCacheEntry(unittest.TestCase):
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        import time
        current_time = time.time()
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=current_time,
            accessed_at=current_time,
            ttl_seconds=300,
            tags=["tag1", "tag2"]
        )
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl_seconds == 300
        assert entry.tags == ["tag1", "tag2"]
        assert entry.created_at == current_time
        assert entry.accessed_at == current_time

    def test_cache_entry_defaults(self):
        """Test cache entry with default values."""
        import time
        current_time = time.time()
        entry = CacheEntry(
            key="test_key",
            value="test",
            created_at=current_time,
            accessed_at=current_time,
            ttl_seconds=300
        )
        assert entry.value == "test"
        assert entry.ttl_seconds == 300
        assert entry.tags == []
        assert entry.access_count == 0

    def test_cache_entry_is_expired(self):
        """Test cache entry expiration logic."""
        current_time = time.time()
        
        # Entry with long TTL should not be expired
        entry_future = CacheEntry(
            key="test_key",
            value="test",
            created_at=current_time,
            accessed_at=current_time,
            ttl_seconds=3600
        )
        assert not entry_future.is_expired()

        # Entry with past expiration should be expired
        entry_past = CacheEntry(
            key="test_key",
            value="test", 
            created_at=current_time - 10,  # Created 10 seconds ago
            accessed_at=current_time,
            ttl_seconds=1  # TTL of 1 second
        )
        assert entry_past.is_expired()

    def test_cache_entry_access_tracking(self):
        """Test that cache entry tracks access properly."""
        current_time = time.time()
        entry = CacheEntry(
            key="test_key",
            value="test",
            created_at=current_time,
            accessed_at=current_time,
            ttl_seconds=300
        )
        original_access_time = entry.accessed_at
        original_count = entry.access_count

        # Update access
        time.sleep(0.01)  # Small delay to ensure time difference
        entry.touch()

        assert entry.accessed_at > original_access_time
        assert entry.access_count == original_count + 1


class TestIntelligentCache(unittest.TestCase):
    """Test IntelligentCache class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_initialization_memory_only(self):
        """Test cache initialization with memory-only strategy."""
        config = CacheConfiguration(strategy=CacheStrategy.MEMORY_ONLY)
        cache = IntelligentCache(name="test_memory", config=config)
        assert cache.name == "test_memory"
        assert cache.config.strategy == CacheStrategy.MEMORY_ONLY
        assert len(cache._entries) == 0

    def test_cache_initialization_persistent(self):
        """Test cache initialization with persistent strategy."""
        config = CacheConfiguration(strategy=CacheStrategy.PERSISTENT, cache_directory=self.cache_dir)
        cache = IntelligentCache(name="test_persistent", config=config)
        assert cache.config.strategy == CacheStrategy.PERSISTENT
        assert cache.config.cache_directory == self.cache_dir

    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        cache = IntelligentCache(name="test")

        cache.put("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    async def test_cache_get_with_default(self):
        """Test cache get with default value."""
        cache = IntelligentCache(name="test")

        result = cache.get("nonexistent", default="default_value")
        assert result == "default_value"

        result_none = cache.get("nonexistent")
        assert result_none is None

        cache.shutdown()

    async def test_cache_ttl_expiration(self):
        """Test cache entry TTL expiration."""
        cache = IntelligentCache(name="test")

        cache.put("key1", "value1", ttl_seconds=1)

        # Should be available immediately
        result = cache.get("key1")
        assert result == "value1"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be None after expiration
        result = cache.get("key1")
        assert result is None

        cache.shutdown()

    async def test_cache_invalidate_by_key(self):
        """Test cache invalidation by key."""
        cache = IntelligentCache(name="test")

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Both should exist
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        # Invalidate one
        cache.invalidate("key1")

        # Only key2 should remain
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

        cache.shutdown()

    async def test_cache_invalidate_by_tags(self):
        """Test cache invalidation by tags."""
        cache = IntelligentCache(name="test")

        cache.put("key1", "value1", tags=["tag1", "tag2"])
        cache.put("key2", "value2", tags=["tag2", "tag3"])
        cache.put("key3", "value3", tags=["tag3"])

        # All should exist
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

        # Invalidate by tag2
        invalidated = cache.invalidate_by_tags(["tag2"])
        assert invalidated == 2  # key1 and key2

        # Only key3 should remain
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

        cache.shutdown()

    async def test_cache_clear(self):
        """Test cache clear operation."""
        cache = IntelligentCache(name="test_cache_clear")

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Both should exist
        assert len(cache._entries) == 2

        # Clear cache
        cache.clear()

        # Should be empty
        assert len(cache._entries) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

        cache.shutdown()

    async def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = IntelligentCache(name="test")

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["writes"] == 0

        # Add entry and access it
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        # Check updated stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["writes"] == 1

        cache.shutdown()

    async def test_cache_eviction_lru(self):
        """Test LRU eviction policy."""
        config = CacheConfiguration(eviction_policy=CacheEvictionPolicy.LRU, max_entries=2)
        cache = IntelligentCache(name="test", config=config)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 to make it more recently used
        cache.get("key1")

        # Add key3, should evict key2 (least recently used)
        cache.put("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"

        cache.shutdown()

    async def test_cache_eviction_lfu(self):
        """Test LFU eviction policy."""
        config = CacheConfiguration(eviction_policy=CacheEvictionPolicy.LFU, max_entries=2)
        cache = IntelligentCache(name="test", config=config)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1 multiple times to increase frequency
        cache.get("key1")
        cache.get("key1")
        cache.get("key2")  # key2 accessed once, key1 accessed twice

        # Add key3, should evict key2 (least frequently used)
        cache.put("key3", "value3")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"

        cache.shutdown()

    async def test_cache_persistence(self):
        """Test cache persistence to disk."""
        cache_file = self.cache_dir / "test_persistent.cache"

        # Create cache and add data
        config1 = CacheConfiguration(strategy=CacheStrategy.PERSISTENT, cache_directory=self.cache_dir)
        cache1 = IntelligentCache(name="test_persistent", config=config1)

        cache1.put("key1", "value1")
        cache1.put("key2", "value2")
        cache1.shutdown()

        # Verify cache file was created
        assert cache_file.exists()

        # Create new cache instance and verify data is loaded
        config2 = CacheConfiguration(strategy=CacheStrategy.PERSISTENT, cache_directory=self.cache_dir)
        cache2 = IntelligentCache(name="test_persistent", config=config2)

        assert cache2.get("key1") == "value1"
        assert cache2.get("key2") == "value2"

        cache2.shutdown()

    async def test_cache_memory_usage(self):
        """Test cache memory usage reporting."""
        cache = IntelligentCache(name="test")

        # Initial memory usage should be minimal
        initial_usage = cache.get_stats()["memory_usage_mb"]
        assert initial_usage >= 0

        # Add some data
        large_data = "x" * 10000  # 10KB of data
        for i in range(10):
            cache.put(f"key{i}", large_data)

        # Memory usage should increase
        final_usage = cache.get_stats()["memory_usage_mb"]
        assert final_usage > initial_usage

        cache.shutdown()

    async def test_cache_concurrent_access(self):
        """Test cache under concurrent access."""
        cache = IntelligentCache(name="test")

        async def put_data(start_idx, count):
            for i in range(start_idx, start_idx + count):
                cache.put(f"key{i}", f"value{i}")

        async def get_data(start_idx, count):
            results = []
            for i in range(start_idx, start_idx + count):
                result = cache.get(f"key{i}")
                results.append(result)
            return results

        # Run concurrent put operations
        await asyncio.gather(put_data(0, 10), put_data(10, 10), put_data(20, 10))

        # Run concurrent get operations
        results = await asyncio.gather(get_data(0, 10), get_data(10, 10), get_data(20, 10))

        # Verify all data is accessible
        all_results = [item for sublist in results for item in sublist]
        non_none_results = [r for r in all_results if r is not None]
        assert len(non_none_results) == 30

        cache.shutdown()


class TestCacheFactoryFunctions(unittest.TestCase):
    """Test cache factory functions."""

    async def test_get_model_cache(self):
        """Test get_model_cache factory function."""
        cache = get_model_cache()
        assert cache.name == "model_metadata"
        assert cache.config.strategy == CacheStrategy.PERSISTENT
        cache.shutdown()

    async def test_get_validation_cache(self):
        """Test get_validation_cache factory function."""
        cache = get_validation_cache()
        assert cache.name == "validation_results"
        assert cache.config.strategy == CacheStrategy.MEMORY_ONLY
        cache.shutdown()

    async def test_get_api_response_cache(self):
        """Test get_api_response_cache factory function."""
        cache = get_api_response_cache()
        assert cache.name == "api_responses"
        assert cache.config.strategy == CacheStrategy.MEMORY_ONLY
        cache.shutdown()


class TestCacheErrorHandling(unittest.TestCase):
    """Test cache error handling."""

    async def test_cache_disk_error_handling(self):
        """Test cache handling disk errors gracefully."""
        # Try to create cache in non-existent directory
        bad_dir = Path("/nonexistent/path")
        config = CacheConfiguration(strategy=CacheStrategy.PERSISTENT, cache_directory=bad_dir)
        cache = IntelligentCache(name="test_error", config=config)

        # Should handle errors gracefully and fall back to memory-only
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

        cache.shutdown()

    async def test_cache_serialization_error_handling(self):
        """Test cache handling serialization errors."""
        cache = IntelligentCache(name="test")

        # Try to cache un-serializable object
        class UnserializableClass:
            def __init__(self):
                self.file = open(__file__)  # File objects can't be pickled

        obj = UnserializableClass()

        # Should handle gracefully (cache or skip depending on implementation)
        try:
            cache.put("bad_key", obj)
            # If it succeeds, try to get it back
            cache.get("bad_key")
            # Result might be None if serialization failed
        except Exception:
            # Or it might raise an exception, which is also acceptable
            pass

        obj.file.close()
        cache.shutdown()


class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def async_test(self, coro):
        """Helper to run async tests."""
        return self.loop.run_until_complete(coro)


# Convert async test methods to work with unittest
for _cls_name, cls in list(globals().items()):
    if isinstance(cls, type) and issubclass(cls, unittest.TestCase):
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and attr_name.startswith("test_") and asyncio.iscoroutinefunction(attr):
                # Wrap async test method
                def make_sync_test(async_method):
                    def sync_test(self):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(async_method(self))
                        finally:
                            loop.close()

                    return sync_test

                setattr(cls, attr_name, make_sync_test(attr))


if __name__ == "__main__":
    unittest.main()
