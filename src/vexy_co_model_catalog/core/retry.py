"""
this_file: src/vexy_co_model_catalog/core/retry.py

Comprehensive error recovery and retry logic for transient network failures.
"""

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
from loguru import logger


class RetryStrategy(Enum):
    """Retry strategy options."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures detected, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    backoff_multiplier: float = 2.0

    # Transient error detection
    transient_status_codes: list[int] = field(
        default_factory=lambda: [
            408,  # Request Timeout
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
            520,  # Unknown Error (Cloudflare)
            521,  # Web Server Is Down (Cloudflare)
            522,  # Connection Timed Out (Cloudflare)
            523,  # Origin Is Unreachable (Cloudflare)
            524,  # A Timeout Occurred (Cloudflare)
        ]
    )

    transient_exceptions: list[type[Exception]] = field(
        default_factory=lambda: [
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ConnectError,
            httpx.ReadError,
            ConnectionResetError,
            OSError,
        ]
    )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 30.0  # Seconds to wait before testing recovery
    success_threshold: int = 2  # Successes needed to close circuit
    half_open_max_calls: int = 3  # Max calls to allow in half-open state


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    circuit_breaker_trips: int = 0
    total_delay_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_attempts / self.total_attempts) * 100.0

    @property
    def average_delay(self) -> float:
        """Calculate average delay per attempt."""
        if self.failed_attempts == 0:
            return 0.0
        return self.total_delay_time / self.failed_attempts


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""

    def __init__(self, config: CircuitBreakerConfig) -> None:
        """Initialize circuit breaker with configuration."""
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.success_count = 0
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit state."""
        current_time = time.time()

        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if current_time - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker OPENED - service still failing")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")

    def record_call(self) -> None:
        """Record a call attempt in half-open state."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1


class EnhancedRetryHandler:
    """Comprehensive retry handler with circuit breaker and advanced error recovery."""

    def __init__(self, retry_config: RetryConfig | None = None, circuit_config: CircuitBreakerConfig | None = None) -> None:
        """Initialize retry handler with configuration."""
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.circuit_breaker = CircuitBreaker(self.circuit_config)
        self.metrics = RetryMetrics()

    async def execute_with_retry(self, operation: Callable, operation_name: str = "operation", *args, **kwargs) -> Any:
        """
        Execute an operation with comprehensive retry logic.

        Args:
            operation: Async function to execute
            operation_name: Name for logging and metrics
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of successful operation

        Raises:
            Exception: Final exception after all retries exhausted
        """
        last_exception = None

        for attempt in range(1, self.retry_config.max_attempts + 1):
            self.metrics.total_attempts += 1

            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                logger.warning(f"Circuit breaker OPEN - blocking {operation_name}")
                self.metrics.circuit_breaker_trips += 1
                msg = f"Circuit breaker is OPEN for {operation_name}"
                raise Exception(msg)

            try:
                # Record call attempt for half-open state
                self.circuit_breaker.record_call()

                # Execute the operation
                start_time = time.time()
                result = await operation(*args, **kwargs)
                execution_time = time.time() - start_time

                # Success!
                self.metrics.successful_attempts += 1
                self.circuit_breaker.record_success()

                if attempt > 1:
                    logger.info(
                        f"{operation_name} succeeded on attempt {attempt} (execution time: {execution_time:.2f}s)"
                    )

                return result

            except Exception as e:
                last_exception = e
                self.metrics.failed_attempts += 1

                # Check if this is a transient error we should retry
                should_retry = self._is_transient_error(e, attempt)

                if not should_retry or attempt >= self.retry_config.max_attempts:
                    # Final failure
                    self.circuit_breaker.record_failure()
                    logger.error(f"{operation_name} failed permanently after {attempt} attempts: {e}")
                    break

                # Calculate delay before retry
                delay = self._calculate_delay(attempt)
                self.metrics.total_delay_time += delay

                logger.warning(
                    f"{operation_name} failed on attempt {attempt}/{self.retry_config.max_attempts}, "
                    f"retrying in {delay:.2f}s: {e}"
                )

                # Wait before retry
                await asyncio.sleep(delay)

        # All retries exhausted
        self.circuit_breaker.record_failure()
        raise last_exception or Exception(f"Failed to execute {operation_name}")

    def _is_transient_error(self, error: Exception, _attempt: int) -> bool:
        """
        Determine if an error is transient and should trigger retry.

        Args:
            error: Exception that occurred
            attempt: Current attempt number

        Returns:
            True if error appears transient and we should retry
        """
        # Check exception types
        if any(isinstance(error, exc_type) for exc_type in self.retry_config.transient_exceptions):
            return True

        # Check HTTP status codes
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            if error.response.status_code in self.retry_config.transient_status_codes:
                return True

        # Check for specific httpx errors with status codes
        if isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code in self.retry_config.transient_status_codes:
                return True

        # Check error message for transient indicators
        error_str = str(error).lower()
        transient_indicators = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "busy",
            "overloaded",
            "unavailable",
            "reset",
            "refused",
        ]

        return bool(any(indicator in error_str for indicator in transient_indicators))

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay before retry based on strategy.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Delay in seconds
        """
        if self.retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.retry_config.base_delay * (self.retry_config.backoff_multiplier ** (attempt - 1))
        elif self.retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.retry_config.base_delay * attempt
        elif self.retry_config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.retry_config.base_delay
        else:  # IMMEDIATE
            delay = 0.0

        # Apply maximum delay limit
        delay = min(delay, self.retry_config.max_delay)

        # Add jitter if enabled
        if self.retry_config.jitter and delay > 0:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)  # Ensure non-negative

        return delay

    def get_metrics(self) -> RetryMetrics:
        """Get current retry metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset retry metrics."""
        self.metrics = RetryMetrics()

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state."""
        self.circuit_breaker = CircuitBreaker(self.circuit_config)
        logger.info("Circuit breaker manually reset to CLOSED state")


# Global retry handler instance for convenience
default_retry_handler = EnhancedRetryHandler()


async def retry_async(
    operation: Callable,
    operation_name: str = "async_operation",
    retry_config: RetryConfig | None = None,
    *args,
    **kwargs,
) -> Any:
    """
    Convenience function for retrying async operations.

    Args:
        operation: Async function to execute
        operation_name: Name for logging
        retry_config: Optional custom retry configuration
        *args: Arguments to pass to operation
        **kwargs: Keyword arguments to pass to operation

    Returns:
        Result of successful operation
    """
    handler = EnhancedRetryHandler(retry_config) if retry_config else default_retry_handler

    return await handler.execute_with_retry(operation, operation_name, *args, **kwargs)


def create_resilient_http_client(
    timeout: float = 30.0, max_connections: int = 100, max_keepalive_connections: int = 20
) -> httpx.AsyncClient:
    """
    Create an HTTP client configured for resilience.

    Args:
        timeout: Request timeout in seconds
        max_connections: Maximum number of connections in pool
        max_keepalive_connections: Maximum keepalive connections

    Returns:
        Configured AsyncClient instance
    """
    # Configure connection limits for better resilience
    limits = httpx.Limits(
        max_connections=max_connections, max_keepalive_connections=max_keepalive_connections, keepalive_expiry=30.0
    )

    # Configure timeouts
    timeout_config = httpx.Timeout(
        timeout=timeout,
        connect=10.0,  # Connection timeout
        read=timeout,  # Read timeout
        write=10.0,  # Write timeout
        pool=5.0,  # Pool timeout
    )

    # Create client with resilient configuration
    return httpx.AsyncClient(
        timeout=timeout_config,
        limits=limits,
        follow_redirects=True,
        max_redirects=5,
        http2=True,  # Enable HTTP/2 for better performance
        verify=True,  # Enable SSL verification
    )
