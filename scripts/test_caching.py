#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["loguru", "rich"]
# ///
# this_file: scripts/test_caching.py

"""
Test script to verify the intelligent caching system functionality.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console

from vexy_co_model_catalog.core.caching import (
    CacheConfiguration,
    CacheEvictionPolicy,
    CacheStrategy,
    IntelligentCache,
    cache_model_data,
    get_cached_model_data,
    get_model_cache,
    get_validation_cache,
)

console = Console()


def test_basic_cache_operations():
    """Test basic cache put/get operations."""
    console.print("[bold cyan]Testing Basic Cache Operations[/bold cyan]")

    config = CacheConfiguration(
        max_memory_mb=10,
        default_ttl_seconds=5,  # Short TTL for testing
        strategy=CacheStrategy.MEMORY_ONLY,
    )

    cache = IntelligentCache("test_basic", config)

    # Test put and get
    cache.put("test_key", {"data": "test_value"})
    result = cache.get("test_key")

    console.print(f"Put/Get test: {'✅ PASS' if result and result['data'] == 'test_value' else '❌ FAIL'}")

    # Test TTL expiration
    time.sleep(6)  # Wait for TTL to expire
    expired_result = cache.get("test_key")

    console.print(f"TTL expiration test: {'✅ PASS' if expired_result is None else '❌ FAIL'}")

    cache.shutdown()
    console.print()


def test_cache_tags_and_invalidation():
    """Test cache tagging and invalidation."""
    console.print("[bold cyan]Testing Cache Tags and Invalidation[/bold cyan]")

    config = CacheConfiguration(strategy=CacheStrategy.MEMORY_ONLY)
    cache = IntelligentCache("test_tags", config)

    # Add entries with tags
    cache.put("openai:models", {"models": ["gpt-4"]}, tags=["models", "openai"])
    cache.put("anthropic:models", {"models": ["claude-3"]}, tags=["models", "anthropic"])
    cache.put("openai:config", {"api_key": "test"}, tags=["config", "openai"])

    # Test tag-based invalidation
    invalidated = cache.invalidate_by_tags(["openai"])

    # Check results
    openai_models = cache.get("openai:models")
    anthropic_models = cache.get("anthropic:models")
    openai_config = cache.get("openai:config")

    success = openai_models is None and anthropic_models is not None and openai_config is None and invalidated == 2

    console.print(f"Tag invalidation test: {'✅ PASS' if success else '❌ FAIL'}")
    console.print(f"  Invalidated {invalidated} entries")

    cache.shutdown()
    console.print()


def test_cache_statistics():
    """Test cache statistics tracking."""
    console.print("[bold cyan]Testing Cache Statistics[/bold cyan]")

    config = CacheConfiguration(strategy=CacheStrategy.MEMORY_ONLY)
    cache = IntelligentCache("test_stats", config)

    # Generate some cache activity
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")

    # Generate hits and misses
    cache.get("key1")  # hit
    cache.get("key1")  # hit
    cache.get("key2")  # hit
    cache.get("nonexistent")  # miss
    cache.get("another_miss")  # miss

    stats = cache.get_stats()

    expected_hits = 3
    expected_misses = 2
    expected_writes = 3
    expected_entries = 3

    success = (
        stats["hits"] == expected_hits
        and stats["misses"] == expected_misses
        and stats["writes"] == expected_writes
        and stats["entry_count"] == expected_entries
    )

    console.print(f"Statistics tracking test: {'✅ PASS' if success else '❌ FAIL'}")
    console.print(f"  Stats: {stats['hits']} hits, {stats['misses']} misses, {stats['writes']} writes")
    console.print(f"  Hit rate: {stats['hit_rate_percent']:.1f}%")

    cache.shutdown()
    console.print()


def test_model_cache_integration():
    """Test integration with model cache functions."""
    console.print("[bold cyan]Testing Model Cache Integration[/bold cyan]")

    # Test the convenience functions
    test_data = {"models": [{"name": "gpt-4", "id": "gpt-4"}, {"name": "gpt-3.5-turbo", "id": "gpt-3.5-turbo"}]}

    # Cache some model data
    cache_model_data("test_provider", test_data, ttl_seconds=10)

    # Retrieve it
    cached_result = get_cached_model_data("test_provider")

    success = (
        cached_result is not None
        and len(cached_result["models"]) == 2
        and cached_result["models"][0]["name"] == "gpt-4"
    )

    console.print(f"Model cache integration test: {'✅ PASS' if success else '❌ FAIL'}")

    # Test cache stats
    model_cache = get_model_cache()
    stats = model_cache.get_stats()

    console.print(f"  Model cache entries: {stats['entry_count']}")
    console.print(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")
    console.print()


def test_persistence_simulation():
    """Test cache persistence behavior (simulated)."""
    console.print("[bold cyan]Testing Cache Persistence (Simulated)[/bold cyan]")

    # Create a temporary directory for cache files
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CacheConfiguration(strategy=CacheStrategy.PERSISTENT, cache_directory=Path(temp_dir))

        # Create cache and add data
        cache1 = IntelligentCache("test_persist", config)
        cache1.put("persistent_key", {"data": "persistent_value"})

        # Force save and shutdown
        cache1._save_to_disk()
        cache1.shutdown()

        # Create new cache instance (simulating restart)
        cache2 = IntelligentCache("test_persist", config)

        # Try to retrieve the data
        result = cache2.get("persistent_key")

        success = result is not None and result["data"] == "persistent_value"

        console.print(f"Persistence simulation test: {'✅ PASS' if success else '❌ FAIL'}")

        cache2.shutdown()

    console.print()


def test_memory_pressure_handling():
    """Test cache behavior under memory pressure."""
    console.print("[bold cyan]Testing Memory Pressure Handling[/bold cyan]")

    config = CacheConfiguration(
        max_memory_mb=1,  # Very small to trigger eviction
        max_entries=5,
        strategy=CacheStrategy.MEMORY_ONLY,
        eviction_policy=CacheEvictionPolicy.LRU,
    )

    cache = IntelligentCache("test_pressure", config)

    # Add more entries than the limit
    large_data = "x" * 1000  # 1KB strings
    for i in range(10):
        cache.put(f"key_{i}", large_data)

    stats = cache.get_stats()

    # Should have evicted some entries
    success = stats["entry_count"] <= config.max_entries and stats["evictions"] > 0

    console.print(f"Memory pressure test: {'✅ PASS' if success else '❌ FAIL'}")
    console.print(f"  Final entries: {stats['entry_count']}/{config.max_entries}")
    console.print(f"  Evictions: {stats['evictions']}")

    cache.shutdown()
    console.print()


if __name__ == "__main__":
    console.print("[bold green]Intelligent Caching System Test Suite[/bold green]")
    console.print()

    test_basic_cache_operations()
    test_cache_tags_and_invalidation()
    test_cache_statistics()
    test_model_cache_integration()
    test_persistence_simulation()
    test_memory_pressure_handling()

    console.print("[bold green]✅ Caching system testing complete![/bold green]")
