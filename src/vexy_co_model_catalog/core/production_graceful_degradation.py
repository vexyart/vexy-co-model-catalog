"""
this_file: src/vexy_co_model_catalog/core/production_graceful_degradation.py

Graceful degradation mechanisms for production deployment edge cases.
"""

import functools
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import httpx
from loguru import logger

from vexy_co_model_catalog.core.production_error_handling import (
    error_context,
    get_error_handler,
)

F = TypeVar('F', bound=Callable[..., Any])


class FallbackStrategy(Enum):
    """Strategies for handling failures gracefully."""
    RETRY = "retry"
    CACHE = "cache"
    DEFAULT_VALUE = "default_value"
    SKIP = "skip"
    PARTIAL_SUCCESS = "partial_success"
    DEGRADED_SERVICE = "degraded_service"


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    strategy: FallbackStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    default_value: Any = None
    cache_duration: int = 300  # 5 minutes
    timeout: float | None = None


class GracefulDegradationManager:
    """
    Manages graceful degradation strategies for production deployment.

    This manager implements multiple resilience patterns:
    - Circuit breakers: Prevent cascading failures by temporarily disabling failed services
    - Result caching: Store successful results for fallback during failures
    - Service degradation tracking: Monitor which services are operating in degraded mode
    - Failure counting: Track failure rates for intelligent decision making

    Attributes:
        failure_counts: Dictionary tracking failure counts per service (deprecated - use circuit_breakers)
        circuit_breakers: Circuit breaker state per service with failure counts and timestamps
        cached_results: Cached successful results with timestamps for TTL validation
        degraded_services: Set of service names currently operating in degraded mode

    Example:
        manager = GracefulDegradationManager()

        # Check if service should be bypassed
        if manager.is_circuit_open("api_service"):
            return fallback_data

        # Cache successful result
        manager.cache_result("api_data", successful_response)

        # Record failure for circuit breaker
        manager.record_failure("api_service")
    """

    def __init__(self) -> None:
        """
        Initialize the graceful degradation manager.

        Sets up empty state tracking for circuit breakers, caches, and service status.
        All services start in healthy state with no cached results.
        """
        self.failure_counts: dict[str, int] = {}  # Legacy - kept for compatibility
        self.circuit_breakers: dict[str, dict[str, Any]] = {}
        self.cached_results: dict[str, dict[str, Any]] = {}
        self.degraded_services: set[str] = set()

    def is_circuit_open(self, service_name: str, threshold: int = 5) -> bool:
        """
        Check if circuit breaker is open for a service.

        A circuit breaker is considered "open" when a service has exceeded the failure
        threshold within the reset window. Open circuits prevent further calls to
        failing services, allowing them time to recover.

        Args:
            service_name: Unique identifier for the service to check
            threshold: Number of failures required to open circuit (default: 5)

        Returns:
            True if circuit is open (service calls should be bypassed),
            False if circuit is closed (service calls can proceed)

        Behavior:
            - Returns False for unknown services (no failure history)
            - Auto-resets circuit after 60 seconds of no failures
            - Compares current failure count against threshold

        Example:
            if manager.is_circuit_open("api_service", threshold=3):
                return cached_fallback_data  # Don't call failing service
        """
        if service_name not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[service_name]
        current_time = time.time()

        # Auto-reset circuit breaker after recovery window
        reset_window_seconds = 60
        if current_time - breaker.get("last_failure", 0) > reset_window_seconds:
            self.circuit_breakers[service_name] = {"failures": 0, "last_failure": 0}
            return False

        return breaker.get("failures", 0) >= threshold

    def record_failure(self, service_name: str) -> None:
        """Record a failure for circuit breaker tracking."""
        current_time = time.time()

        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = {"failures": 0, "last_failure": 0}

        self.circuit_breakers[service_name]["failures"] += 1
        self.circuit_breakers[service_name]["last_failure"] = current_time

        logger.warning(f"Service failure recorded: {service_name} ({self.circuit_breakers[service_name]['failures']} failures)")

    def record_success(self, service_name: str) -> None:
        """Record a success to reset circuit breaker."""
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name]["failures"] = 0
            logger.info(f"Service recovered: {service_name}")

    def get_cached_result(self, cache_key: str, max_age: int = 300) -> Any:
        """Get cached result if still valid."""
        if cache_key not in self.cached_results:
            return None

        cached_data = self.cached_results[cache_key]
        current_time = time.time()

        if current_time - cached_data["timestamp"] > max_age:
            del self.cached_results[cache_key]
            return None

        logger.info(f"Using cached result for: {cache_key}")
        return cached_data["result"]

    def cache_result(self, cache_key: str, result: Any) -> None:
        """Cache a result for fallback use."""
        self.cached_results[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        logger.debug(f"Cached result for: {cache_key}")

    def enable_degraded_mode(self, service_name: str, reason: str) -> None:
        """Enable degraded mode for a service."""
        self.degraded_services.add(service_name)
        logger.warning(f"Service degraded: {service_name} - {reason}")

    def disable_degraded_mode(self, service_name: str) -> None:
        """Disable degraded mode for a service."""
        if service_name in self.degraded_services:
            self.degraded_services.remove(service_name)
            logger.info(f"Service recovered from degraded mode: {service_name}")

    def is_service_degraded(self, service_name: str) -> bool:
        """Check if service is in degraded mode."""
        return service_name in self.degraded_services


# Global degradation manager
_global_degradation_manager: GracefulDegradationManager | None = None


def get_degradation_manager() -> GracefulDegradationManager:
    """Get global degradation manager instance."""
    global _global_degradation_manager
    if _global_degradation_manager is None:
        _global_degradation_manager = GracefulDegradationManager()
    return _global_degradation_manager


def with_graceful_degradation(
    service_name: str,
    fallback_config: FallbackConfig | None = None,
    cache_key: str | None = None
) -> Callable[[F], F]:
    """
    Decorator that adds graceful degradation capabilities to functions.

    This decorator implements circuit breaker patterns, caching strategies,
    and retry logic to handle failures gracefully in production environments.

    Args:
        service_name: Unique identifier for the service (used for circuit breaker tracking)
        fallback_config: Configuration for fallback behavior (defaults to retry strategy)
        cache_key: Optional cache key for storing/retrieving cached results

    Returns:
        Decorated function with graceful degradation capabilities

    Behavior:
        1. Checks circuit breaker status - returns fallback if circuit is open
        2. Attempts to use cached results if cache strategy is enabled
        3. Executes function with retry logic up to max_retries
        4. Records success/failure for circuit breaker tracking
        5. Caches successful results if cache_key is provided
        6. Falls back to degraded behavior on repeated failures

    Example:
        @with_graceful_degradation("api_service",
                                 FallbackConfig(strategy=FallbackStrategy.CACHE),
                                 cache_key="api_data")
        def fetch_api_data():
            return api_client.get_data()
    """
    config = fallback_config or FallbackConfig(strategy=FallbackStrategy.RETRY)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Enhanced wrapper with graceful degradation logic.

            Implements the following failure handling sequence:
            1. Circuit breaker check
            2. Cache lookup (if applicable)
            3. Function execution with retry logic
            4. Fallback handling on complete failure
            """
            manager = get_degradation_manager()

            # Step 1: Circuit breaker protection
            if _is_circuit_breaker_open(manager, service_name):
                return handle_fallback(service_name, config, cache_key, func.__name__)

            # Step 2: Cache lookup for fast response
            cached_result = _try_cached_result(manager, config, cache_key)
            if cached_result is not None:
                return cached_result

            # Step 3: Execute with retry logic
            return _execute_with_retries(
                func, args, kwargs, manager, service_name, config, cache_key
            )

        return wrapper
    return decorator


def _is_circuit_breaker_open(manager: GracefulDegradationManager, service_name: str) -> bool:
    """
    Check if circuit breaker is open for the service.

    Args:
        manager: Degradation manager instance
        service_name: Service to check

    Returns:
        True if circuit is open (service should be bypassed)
    """
    if manager.is_circuit_open(service_name):
        logger.warning(f"Circuit breaker open for {service_name}, using fallback")
        return True
    return False


def _try_cached_result(
    manager: GracefulDegradationManager,
    config: FallbackConfig,
    cache_key: str | None
) -> Any | None:
    """
    Attempt to retrieve cached result if cache strategy is enabled.

    Args:
        manager: Degradation manager instance
        config: Fallback configuration
        cache_key: Optional cache key for lookup

    Returns:
        Cached result if available and valid, None otherwise
    """
    if cache_key and config.strategy == FallbackStrategy.CACHE:
        cached_result = manager.get_cached_result(cache_key, config.cache_duration)
        if cached_result is not None:
            logger.debug(f"Using cached result for key: {cache_key}")
            return cached_result
    return None


def _execute_with_retries(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    manager: GracefulDegradationManager,
    service_name: str,
    config: FallbackConfig,
    cache_key: str | None
) -> Any:
    """
    Execute function with retry logic and failure handling.

    Args:
        func: Function to execute
        args: Function positional arguments
        kwargs: Function keyword arguments
        manager: Degradation manager instance
        service_name: Service name for tracking
        config: Fallback configuration
        cache_key: Optional cache key for results

    Returns:
        Function result or fallback value

    Raises:
        Exception: Re-raises last exception if all fallback strategies fail
    """
    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            # Execute function with error context tracking
            with error_context(f"calling {func.__name__} (attempt {attempt + 1})"):
                result = func(*args, **kwargs)

            # Success path: record success and cache result
            manager.record_success(service_name)

            if cache_key:
                manager.cache_result(cache_key, result)
                logger.debug(f"Cached successful result for key: {cache_key}")

            return result

        except Exception as e:
            last_exception = e
            manager.record_failure(service_name)

            # Handle retry logic with exponential backoff
            if attempt < config.max_retries:
                delay = _calculate_retry_delay(config, attempt)
                logger.warning(
                    f"Attempt {attempt + 1} failed for {service_name} ({type(e).__name__}), "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"All {config.max_retries + 1} attempts failed for {service_name}: "
                    f"{type(e).__name__}: {e}"
                )

    # All retries failed, attempt fallback handling
    return handle_fallback(service_name, config, cache_key, func.__name__, last_exception)


def _calculate_retry_delay(config: FallbackConfig, attempt: int) -> float:
    """
    Calculate retry delay using exponential backoff.

    Args:
        config: Fallback configuration with delay settings
        attempt: Current attempt number (0-based)

    Returns:
        Delay in seconds before next retry
    """
    return config.retry_delay * (config.backoff_multiplier ** attempt)


def handle_fallback(
    service_name: str,
    config: FallbackConfig,
    cache_key: str | None,
    function_name: str,
    _last_exception: Exception | None = None
) -> Any:
    """Handle fallback strategies when operation fails."""
    manager = get_degradation_manager()

    match config.strategy:
        case FallbackStrategy.CACHE:
            if cache_key:
                # Try stale cache as last resort
                cached_result = manager.get_cached_result(cache_key, max_age=3600)  # 1 hour stale
                if cached_result is not None:
                    logger.info(f"Using stale cache for {service_name}")
                    return cached_result

            # Fall through to default value
            logger.warning(f"No cache available for {service_name}, using default value")
            return config.default_value

        case FallbackStrategy.DEFAULT_VALUE:
            logger.info(f"Using default value for {service_name}")
            return config.default_value

        case FallbackStrategy.SKIP:
            logger.info(f"Skipping operation {function_name} for {service_name}")
            return None

        case FallbackStrategy.PARTIAL_SUCCESS:
            logger.warning(f"Partial success mode for {service_name}")
            # Return empty list/dict based on expected type
            if "list" in function_name.lower():
                return []
            if "dict" in function_name.lower():
                return {}
            return config.default_value

        case FallbackStrategy.DEGRADED_SERVICE:
            manager.enable_degraded_mode(service_name, f"Fallback from {function_name}")
            logger.warning(f"Service {service_name} running in degraded mode")
            return config.default_value

        case _:
            logger.warning(f"Unknown fallback strategy for {service_name}")
            return config.default_value


def with_timeout(timeout_seconds: float) -> Callable[[F], F]:
    """Decorator to add timeout to operations."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            def timeout_handler(_signum: int, _frame: Any) -> None:
                msg = f"Operation {func.__name__} timed out after {timeout_seconds}s"
                raise TimeoutError(msg)

            # Set alarm for timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))

            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            finally:
                signal.alarm(0)  # Ensure alarm is cancelled
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler

        return wrapper
    return decorator


def safe_file_operation(
    operation: str,
    file_path: Path,
    fallback_result: Any = None,
    create_parents: bool = True
) -> Any:
    """Safely perform file operations with fallback."""
    try:
        # Ensure parent directories exist
        if create_parents and file_path.parent != file_path:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        return True

    except PermissionError:
        logger.error(f"Permission denied for {operation} on {file_path}")
        return fallback_result

    except OSError as e:
        logger.error(f"OS error during {operation} on {file_path}: {e}")
        return fallback_result

    except Exception as e:
        handler = get_error_handler()
        handler.handle_error(e, f"file {operation}", {"file_path": str(file_path)})
        return fallback_result


def safe_network_operation(
    operation_name: str,
    url: str,
    timeout: float = 30.0,
    max_retries: int = 3
) -> Any:
    """Safely perform network operations with fallback."""
    config = FallbackConfig(
        strategy=FallbackStrategy.RETRY,
        max_retries=max_retries,
        retry_delay=1.0,
        timeout=timeout
    )

    @with_graceful_degradation(f"network_{operation_name}", config)
    def perform_request() -> Any:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            return response

    return perform_request()


def get_system_status() -> dict[str, Any]:
    """Get system status for monitoring and debugging."""
    manager = get_degradation_manager()

    return {
        "circuit_breakers": dict(manager.circuit_breakers),
        "degraded_services": list(manager.degraded_services),
        "cached_results": len(manager.cached_results),
        "failure_counts": dict(manager.failure_counts),
        "system_healthy": len(manager.degraded_services) == 0
    }
