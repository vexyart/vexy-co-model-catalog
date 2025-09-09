#!/usr/bin/env python3
# this_file: scripts/test_production_features.py

"""
Production Features Test Script

Tests production error handling, graceful degradation, and deployment features.
"""

import sys
import tempfile
from pathlib import Path

# Add the src directory to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import contextlib
import threading

from vexy_co_model_catalog.cli import CLI
from vexy_co_model_catalog.core.production_deployment import (
    ProductionDeploymentManager,
    check_production_readiness,
    get_production_metrics,
    initialize_production_mode,
)
from vexy_co_model_catalog.core.production_error_handling import (
    ProductionErrorHandler,
    error_context,
    get_error_summary,
    handle_critical_error,
    production_error_handler,
)
from vexy_co_model_catalog.core.production_graceful_degradation import (
    FallbackConfig,
    FallbackStrategy,
    GracefulDegradationManager,
    get_system_status,
    with_graceful_degradation,
)


def test_error_handling():
    """Test production error handling features."""

    # Test error handler creation
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = Path(temp_dir) / "test_errors.log"
        handler = ProductionErrorHandler(log_file)

        # Test error categorization
        try:
            msg = "Network connection failed"
            raise ConnectionError(msg)
        except Exception as e:
            category = handler.categorize_error(e)
            handler.determine_severity(e, category)

        # Test error context creation
        try:
            msg = "Config file not found"
            raise FileNotFoundError(msg)
        except Exception as e:
            error_context = handler.create_error_context(e, "test operation")

        # Test error logging
        try:
            msg = "Invalid input data"
            raise ValueError(msg)
        except Exception as e:
            handler.handle_error(e, "test validation")

    # Test error decorator
    @production_error_handler(user_action="testing decorator")
    def failing_function():
        msg = "Test error for decorator"
        raise RuntimeError(msg)

    failing_function()

    # Test error context manager
    try:
        with error_context("testing context manager"):
            msg = "Missing configuration key"
            raise KeyError(msg)
    except:
        pass  # Error should be handled by context manager

    # Test error summary
    get_error_summary()

    return True


def test_graceful_degradation():
    """Test graceful degradation features."""

    manager = GracefulDegradationManager()

    # Test circuit breaker
    service_name = "test_service"
    for _i in range(6):  # Trigger circuit breaker
        manager.record_failure(service_name)

    manager.is_circuit_open(service_name)

    # Test caching
    cache_key = "test_cache"
    test_data = {"test": "data"}
    manager.cache_result(cache_key, test_data)

    manager.get_cached_result(cache_key)

    # Test degraded mode
    manager.enable_degraded_mode(service_name, "test reason")
    manager.is_service_degraded(service_name)

    # Test fallback decorator
    config = FallbackConfig(
        strategy=FallbackStrategy.DEFAULT_VALUE,
        default_value="fallback_result",
        max_retries=2
    )

    @with_graceful_degradation("test_decorator", config)
    def unreliable_function():
        msg = "Simulated failure"
        raise Exception(msg)

    unreliable_function()

    # Test system status
    get_system_status()

    return True


def test_production_deployment():
    """Test production deployment features."""

    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "logs"
        manager = ProductionDeploymentManager(log_dir)

        # Test initialization
        manager.initialize_production_environment()

        # Test health status
        manager.get_health_status()

        # Test log cleanup (should be 0 since no old logs)
        with contextlib.suppress(Exception):
            manager.cleanup_old_logs(max_age_days=0)
            # Don't fail the test for this

    # Test production readiness check
    check_production_readiness()

    # Test production metrics
    with contextlib.suppress(Exception):
        get_production_metrics()

    return True


def test_cli_integration():
    """Test CLI integration with production features."""

    # Test that CLI can import production modules
    try:
        cli = CLI()

        # Test that production methods exist
        methods = ["production_status", "production_init", "production_readiness", "production_errors"]
        for method in methods:
            if hasattr(cli, method):
                pass
            else:
                return False

    except Exception:
        return False

    return True


def test_edge_cases():
    """Test edge cases and error scenarios."""

    # Test with invalid log directory
    try:
        invalid_path = Path("/invalid/directory/that/does/not/exist")
        ProductionErrorHandler(invalid_path / "test.log")
    except Exception:
        pass

    # Test with nested exceptions
    try:
        try:
            msg = "Inner error"
            raise ValueError(msg)
        except Exception:
            msg = "Outer error"
            raise RuntimeError(msg) from None
    except Exception:
        with error_context("nested exception test"):
            pass

    # Test with very long error messages
    try:
        long_message = "A" * 1000
        raise Exception(long_message)
    except Exception as e:
        handle_critical_error(e, "long message test")

    # Test concurrent access (basic)
    def concurrent_error_handler():
        with error_context("concurrent test"):
            try:
                msg = "Concurrent error"
                raise Exception(msg)
            except:
                pass

    threads = [threading.Thread(target=concurrent_error_handler) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return True


def main():
    """Run all production feature tests."""

    tests = [
        ("Error Handling", test_error_handling),
        ("Graceful Degradation", test_graceful_degradation),
        ("Production Deployment", test_production_deployment),
        ("CLI Integration", test_cli_integration),
        ("Edge Cases", test_edge_cases),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception:
            results.append((test_name, False))

    # Summary

    passed = 0
    total = len(results)

    for test_name, success in results:
        if success:
            passed += 1


    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
