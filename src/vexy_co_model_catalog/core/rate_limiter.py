"""
this_file: src/vexy_co_model_catalog/core/rate_limiter.py

Intelligent rate limiting system with provider-specific throttling patterns.
Prevents API rate limit violations through adaptive request scheduling and smart backoff.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class ThrottleStrategy(Enum):
    """Different throttling strategies for various provider patterns."""

    FIXED_RATE = "fixed_rate"  # Fixed requests per second
    BURST_THEN_THROTTLE = "burst_then_throttle"  # Allow bursts, then throttle
    ADAPTIVE = "adaptive"  # Adapt based on response headers
    TOKEN_BUCKET = "token_bucket"  # Token bucket algorithm
    SLIDING_WINDOW = "sliding_window"  # Sliding window rate limiting


@dataclass
class ProviderLimits:
    """Rate limit configuration for a specific provider."""

    provider_name: str
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    burst_capacity: int = 5
    strategy: ThrottleStrategy = ThrottleStrategy.ADAPTIVE

    # Provider-specific patterns
    has_strict_limits: bool = False  # Providers like OpenAI with strict enforcement
    supports_burst: bool = True  # Can handle burst requests initially
    rate_limit_headers: set[str] = field(
        default_factory=lambda: {
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
            "ratelimit-limit",
            "ratelimit-remaining",
            "ratelimit-reset",
        }
    )

    # Adaptive adjustments
    backoff_multiplier: float = 1.5  # How much to slow down when hitting limits
    recovery_factor: float = 0.9  # How quickly to recover after backoff

    @classmethod
    def for_provider(cls, provider_name: str) -> ProviderLimits:
        """Create provider-specific rate limits based on known patterns."""
        provider_configs = {
            # Major providers with known strict limits
            "openai": cls(
                provider_name="openai",
                requests_per_second=0.5,  # Conservative for API stability
                requests_per_minute=20,
                requests_per_hour=500,
                burst_capacity=3,
                has_strict_limits=True,
                strategy=ThrottleStrategy.ADAPTIVE,
            ),
            "anthropic": cls(
                provider_name="anthropic",
                requests_per_second=0.8,
                requests_per_minute=30,
                requests_per_hour=1000,
                burst_capacity=5,
                has_strict_limits=True,
                strategy=ThrottleStrategy.SLIDING_WINDOW,
            ),
            "groq": cls(
                provider_name="groq",
                requests_per_second=2.0,  # Generally more permissive
                requests_per_minute=100,
                requests_per_hour=3000,
                burst_capacity=10,
                strategy=ThrottleStrategy.BURST_THEN_THROTTLE,
            ),
            # Gateway providers (often more permissive)
            "openrouter": cls(
                provider_name="openrouter",
                requests_per_second=1.5,
                requests_per_minute=80,
                requests_per_hour=2000,
                burst_capacity=8,
                strategy=ThrottleStrategy.TOKEN_BUCKET,
            ),
            "together": cls(
                provider_name="together",
                requests_per_second=1.2,
                requests_per_minute=60,
                requests_per_hour=1500,
                burst_capacity=6,
                strategy=ThrottleStrategy.ADAPTIVE,
            ),
            # URL providers (unknown limits, be conservative)
            "url_provider": cls(
                provider_name="url_provider",
                requests_per_second=0.3,
                requests_per_minute=10,
                requests_per_hour=200,
                burst_capacity=2,
                has_strict_limits=True,
                strategy=ThrottleStrategy.FIXED_RATE,
            ),
        }

        return provider_configs.get(provider_name.lower(), provider_configs["url_provider"])


@dataclass
class RateLimitState:
    """Current state of rate limiting for a provider."""

    provider_name: str
    last_request_time: float = 0.0
    request_count_minute: int = 0
    request_count_hour: int = 0
    burst_tokens: int = 0

    # Adaptive state
    current_delay: float = 0.0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_reset_minute: float = 0.0
    last_reset_hour: float = 0.0

    # Rate limit info from headers
    api_limit_remaining: int | None = None
    api_limit_reset: float | None = None

    # Sliding window for more precise tracking
    request_timestamps: list[float] = field(default_factory=list)


class IntelligentRateLimiter:
    """Advanced rate limiter with provider-specific throttling patterns."""

    def __init__(self) -> None:
        """Initialize the intelligent rate limiter."""
        self.provider_limits: dict[str, ProviderLimits] = {}
        self.provider_states: dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()

    def configure_provider(self, provider_name: str, custom_limits: ProviderLimits | None = None) -> None:
        """Configure rate limits for a specific provider."""
        if custom_limits:
            self.provider_limits[provider_name] = custom_limits
        else:
            self.provider_limits[provider_name] = ProviderLimits.for_provider(provider_name)

        if provider_name not in self.provider_states:
            self.provider_states[provider_name] = RateLimitState(provider_name=provider_name)
            # Initialize burst tokens
            limits = self.provider_limits[provider_name]
            self.provider_states[provider_name].burst_tokens = limits.burst_capacity

        logger.debug(
            f"Configured rate limits for {provider_name}: "
            f"{self.provider_limits[provider_name].requests_per_second} req/s"
        )

    async def acquire_permit(self, provider_name: str) -> float:
        """
        Acquire permission to make a request, returning the delay needed.

        Returns:
            float: Seconds to wait before making the request
        """
        async with self._lock:
            # Ensure provider is configured
            if provider_name not in self.provider_limits:
                self.configure_provider(provider_name)

            limits = self.provider_limits[provider_name]
            state = self.provider_states[provider_name]
            current_time = time.time()

            # Clean up old timestamps for sliding window
            self._cleanup_old_timestamps(state, current_time)

            # Calculate required delay based on strategy
            delay = await self._calculate_delay(limits, state, current_time)

            # Update state after calculating delay
            self._update_state(state, current_time, delay)

            return delay

    async def record_response(self, provider_name: str, response_headers: dict[str, Any], success: bool) -> None:
        """Record response to adapt rate limiting based on actual API behavior."""
        async with self._lock:
            if provider_name not in self.provider_states:
                return

            state = self.provider_states[provider_name]
            limits = self.provider_limits[provider_name]

            # Update success/failure tracking
            if success:
                state.consecutive_successes += 1
                state.consecutive_failures = 0
            else:
                state.consecutive_failures += 1
                state.consecutive_successes = 0

            # Parse rate limit headers
            self._parse_rate_limit_headers(state, response_headers)

            # Adaptive adjustment
            await self._adjust_rate_limits(limits, state, success)

    async def _calculate_delay(self, limits: ProviderLimits, state: RateLimitState, current_time: float) -> float:
        """Calculate the required delay before next request."""
        if limits.strategy == ThrottleStrategy.FIXED_RATE:
            return self._fixed_rate_delay(limits, state, current_time)

        if limits.strategy == ThrottleStrategy.BURST_THEN_THROTTLE:
            return self._burst_then_throttle_delay(limits, state, current_time)

        if limits.strategy == ThrottleStrategy.TOKEN_BUCKET:
            return self._token_bucket_delay(limits, state, current_time)

        if limits.strategy == ThrottleStrategy.SLIDING_WINDOW:
            return self._sliding_window_delay(limits, state, current_time)

        # ADAPTIVE
        return self._adaptive_delay(limits, state, current_time)

    def _fixed_rate_delay(self, limits: ProviderLimits, state: RateLimitState, current_time: float) -> float:
        """Calculate delay for fixed rate strategy."""
        min_interval = 1.0 / limits.requests_per_second
        time_since_last = current_time - state.last_request_time

        return max(0, min_interval - time_since_last + state.current_delay)

    def _burst_then_throttle_delay(self, limits: ProviderLimits, state: RateLimitState, current_time: float) -> float:
        """Allow burst initially, then apply throttling."""
        if state.burst_tokens > 0:
            return 0.0  # No delay for burst requests

        # After burst, apply normal rate limiting
        return self._fixed_rate_delay(limits, state, current_time)

    def _token_bucket_delay(self, limits: ProviderLimits, state: RateLimitState, current_time: float) -> float:
        """Token bucket algorithm with burst capacity."""
        # Replenish tokens based on time passed
        time_passed = current_time - state.last_request_time
        tokens_to_add = time_passed * limits.requests_per_second
        state.burst_tokens = min(limits.burst_capacity, state.burst_tokens + tokens_to_add)

        if state.burst_tokens >= 1.0:
            return 0.0  # Token available, no delay

        # Calculate delay until next token is available
        return (1.0 - state.burst_tokens) / limits.requests_per_second

    def _sliding_window_delay(self, limits: ProviderLimits, state: RateLimitState, current_time: float) -> float:
        """Sliding window rate limiting for precise control."""
        window_size = 60.0  # 1 minute window
        requests_in_window = len([t for t in state.request_timestamps if current_time - t < window_size])

        if requests_in_window < limits.requests_per_minute:
            return state.current_delay  # Apply any adaptive delay

        # Find oldest request in window and delay until it expires
        oldest_in_window = min(t for t in state.request_timestamps if current_time - t < window_size)
        return (oldest_in_window + window_size - current_time) + state.current_delay

    def _adaptive_delay(self, limits: ProviderLimits, state: RateLimitState, current_time: float) -> float:
        """Adaptive strategy that learns from API responses."""
        base_delay = self._fixed_rate_delay(limits, state, current_time)

        # Adapt based on recent failures
        if state.consecutive_failures > 0:
            failure_multiplier = limits.backoff_multiplier**state.consecutive_failures
            adaptive_delay = base_delay * min(failure_multiplier, 10.0)  # Cap at 10x
        else:
            # Gradually reduce delay after successes
            success_factor = limits.recovery_factor ** min(state.consecutive_successes, 10)
            adaptive_delay = base_delay * success_factor

        # Consider API rate limit headers if available
        if state.api_limit_remaining is not None and state.api_limit_remaining < 5:
            # Very few requests remaining, be more conservative
            adaptive_delay *= 2.0

        return adaptive_delay + state.current_delay

    def _update_state(self, state: RateLimitState, current_time: float, delay: float) -> None:
        """Update provider state after calculating delay."""
        state.last_request_time = current_time
        state.current_delay = delay

        # Update counters
        self._update_request_counters(state, current_time)

        # Update burst tokens
        if state.burst_tokens > 0:
            state.burst_tokens -= 1

        # Add to sliding window
        state.request_timestamps.append(current_time)

    def _update_request_counters(self, state: RateLimitState, current_time: float) -> None:
        """Update minute and hourly request counters."""
        # Reset minute counter if needed
        if current_time - state.last_reset_minute > 60.0:
            state.request_count_minute = 0
            state.last_reset_minute = current_time

        # Reset hour counter if needed
        if current_time - state.last_reset_hour > 3600.0:
            state.request_count_hour = 0
            state.last_reset_hour = current_time

        state.request_count_minute += 1
        state.request_count_hour += 1

    def _cleanup_old_timestamps(self, state: RateLimitState, current_time: float) -> None:
        """Remove timestamps older than 1 hour for memory efficiency."""
        cutoff_time = current_time - 3600.0  # 1 hour ago
        state.request_timestamps = [t for t in state.request_timestamps if t > cutoff_time]

    def _parse_rate_limit_headers(self, state: RateLimitState, headers: dict[str, Any]) -> None:
        """Parse rate limit information from response headers."""
        headers_lower = {k.lower(): v for k, v in headers.items()}

        # Try different header patterns
        remaining_headers = ["x-ratelimit-remaining", "ratelimit-remaining", "x-rate-limit-remaining"]
        reset_headers = ["x-ratelimit-reset", "ratelimit-reset", "x-rate-limit-reset"]

        for header in remaining_headers:
            if header in headers_lower:
                try:
                    state.api_limit_remaining = int(headers_lower[header])
                    break
                except (ValueError, TypeError):
                    continue

        for header in reset_headers:
            if header in headers_lower:
                try:
                    state.api_limit_reset = float(headers_lower[header])
                    break
                except (ValueError, TypeError):
                    continue

    async def _adjust_rate_limits(self, limits: ProviderLimits, state: RateLimitState, success: bool) -> None:
        """Adaptively adjust rate limits based on API responses."""
        if not success and state.consecutive_failures >= 2:
            # Increase delay after multiple failures
            state.current_delay = min(state.current_delay * limits.backoff_multiplier, 30.0)
            logger.debug(f"Increased delay for {state.provider_name} to {state.current_delay:.2f}s")

        elif success and state.consecutive_successes >= 5:
            # Decrease delay after sustained success
            state.current_delay = max(state.current_delay * limits.recovery_factor, 0.0)
            if state.current_delay < 0.1:
                state.current_delay = 0.0

    def get_provider_stats(self, provider_name: str) -> dict[str, Any]:
        """Get current rate limiting statistics for a provider."""
        if provider_name not in self.provider_states:
            return {}

        state = self.provider_states[provider_name]
        limits = self.provider_limits.get(provider_name)

        return {
            "provider_name": provider_name,
            "strategy": limits.strategy.value if limits else "unknown",
            "requests_per_second_limit": limits.requests_per_second if limits else None,
            "current_delay": state.current_delay,
            "burst_tokens": state.burst_tokens,
            "requests_this_minute": state.request_count_minute,
            "requests_this_hour": state.request_count_hour,
            "consecutive_successes": state.consecutive_successes,
            "consecutive_failures": state.consecutive_failures,
            "api_limit_remaining": state.api_limit_remaining,
            "recent_request_count": len(state.request_timestamps),
        }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get rate limiting statistics for all providers."""
        return {name: self.get_provider_stats(name) for name in self.provider_states}


# Global rate limiter instance
_global_rate_limiter: IntelligentRateLimiter | None = None


def get_rate_limiter() -> IntelligentRateLimiter:
    """Get the global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = IntelligentRateLimiter()
    return _global_rate_limiter


def configure_provider_rate_limits(provider_name: str, custom_limits: ProviderLimits | None = None) -> None:
    """Configure rate limits for a provider using the global rate limiter."""
    limiter = get_rate_limiter()
    limiter.configure_provider(provider_name, custom_limits)
