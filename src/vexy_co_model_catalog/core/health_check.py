"""
this_file: src/vexy_co_model_catalog/core/health_check.py

Provider health check and connectivity validation utilities.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx
from loguru import logger

from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind

# HTTP Status Code Constants for Health Assessment
HTTP_OK_START = 200
HTTP_OK_END = 300
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_CLIENT_ERROR_START = 400
HTTP_CLIENT_ERROR_END = 500
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_START = 500
HTTP_SERVER_ERROR_END = 600


class HealthStatus(Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a provider health check."""

    provider_name: str
    status: HealthStatus
    response_time_ms: int
    error_message: str | None = None
    status_code: int | None = None
    endpoint_tested: str | None = None
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class HealthSummary:
    """Summary of multiple provider health checks."""

    total_providers: int
    healthy_providers: int
    unhealthy_providers: int
    degraded_providers: int
    unknown_providers: int
    results: list[HealthCheckResult]

    @property
    def health_percentage(self) -> float:
        """Calculate overall health percentage."""
        if self.total_providers == 0:
            return 0.0
        return (self.healthy_providers / self.total_providers) * 100.0


class ProviderHealthChecker:
    """Validates provider health and connectivity."""

    def __init__(self, timeout: float = 10.0, max_concurrency: int = 5) -> None:
        """
        Initialize health checker with performance settings.

        Args:
            timeout: HTTP request timeout in seconds
            max_concurrency: Maximum concurrent health checks
        """
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ProviderHealthChecker":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def check_provider_health(self, provider: ProviderConfig) -> HealthCheckResult:
        """
        Check health of a single provider.

        Args:
            provider: Provider configuration to check

        Returns:
            HealthCheckResult with status and metrics
        """
        async with self._semaphore:
            start_time = time.time()

            # Determine the endpoint to test
            endpoint = self._get_health_check_endpoint(provider)
            if not endpoint:
                return HealthCheckResult(
                    provider_name=provider.name,
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=0,
                    error_message="No testable endpoint available",
                    endpoint_tested=None,
                )

            try:
                # Prepare headers for the request
                headers = self._get_headers_for_provider(provider)

                # Make the health check request
                if not self._client:
                    msg = "Health checker not properly initialized"
                    raise RuntimeError(msg)

                response = await self._client.get(endpoint, headers=headers)

                response_time = int((time.time() - start_time) * 1000)

                # Analyze response to determine health status
                status = self._analyze_response_status(response)

                return HealthCheckResult(
                    provider_name=provider.name,
                    status=status,
                    response_time_ms=response_time,
                    status_code=response.status_code,
                    endpoint_tested=endpoint,
                    error_message=None if status == HealthStatus.HEALTHY else f"HTTP {response.status_code}",
                )

            except httpx.TimeoutException:
                response_time = int((time.time() - start_time) * 1000)
                return HealthCheckResult(
                    provider_name=provider.name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    error_message="Request timeout",
                    endpoint_tested=endpoint,
                )
            except httpx.NetworkError as e:
                response_time = int((time.time() - start_time) * 1000)
                return HealthCheckResult(
                    provider_name=provider.name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    error_message=f"Network error: {e!s}",
                    endpoint_tested=endpoint,
                )
            except Exception as e:
                response_time = int((time.time() - start_time) * 1000)
                return HealthCheckResult(
                    provider_name=provider.name,
                    status=HealthStatus.UNKNOWN,
                    response_time_ms=response_time,
                    error_message=f"Unexpected error: {e!s}",
                    endpoint_tested=endpoint,
                )

    async def check_multiple_providers(self, providers: list[ProviderConfig]) -> HealthSummary:
        """
        Check health of multiple providers concurrently.

        Args:
            providers: List of provider configurations to check

        Returns:
            HealthSummary with aggregated results
        """
        if not providers:
            return HealthSummary(
                total_providers=0,
                healthy_providers=0,
                unhealthy_providers=0,
                degraded_providers=0,
                unknown_providers=0,
                results=[],
            )

        # Run health checks concurrently
        tasks = [self.check_provider_health(provider) for provider in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle any exceptions
        health_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Health check failed for {providers[i].name}: {result}")
                health_results.append(
                    HealthCheckResult(
                        provider_name=providers[i].name,
                        status=HealthStatus.UNKNOWN,
                        response_time_ms=0,
                        error_message=f"Check failed: {result!s}",
                    )
                )
            else:
                health_results.append(result)

        # Calculate summary statistics
        healthy = sum(1 for r in health_results if r.status == HealthStatus.HEALTHY)
        unhealthy = sum(1 for r in health_results if r.status == HealthStatus.UNHEALTHY)
        degraded = sum(1 for r in health_results if r.status == HealthStatus.DEGRADED)
        unknown = sum(1 for r in health_results if r.status == HealthStatus.UNKNOWN)

        return HealthSummary(
            total_providers=len(health_results),
            healthy_providers=healthy,
            unhealthy_providers=unhealthy,
            degraded_providers=degraded,
            unknown_providers=unknown,
            results=health_results,
        )

    def _get_health_check_endpoint(self, provider: ProviderConfig) -> str | None:
        """
        Get appropriate health check endpoint for provider.

        Args:
            provider: Provider configuration

        Returns:
            URL to test for health check, or None if no suitable endpoint
        """
        # Use environment variable for base_url if specified
        if provider.base_url_env:
            base_url = os.environ.get(provider.base_url_env)
            if base_url:
                return base_url

        # Use configured base_url
        if provider.base_url:
            return provider.base_url

        # Provider-specific health check endpoints
        health_endpoints = {
            "openai": "https://api.openai.com/v1/models",
            "anthropic": "https://api.anthropic.com/v1/messages",
            "groq": "https://api.groq.com/openai/v1/models",
            "fireworks": "https://api.fireworks.ai/inference/v1/models",
            "togetherai": "https://api.together.xyz/v1/models",
            "mistral": "https://api.mistral.ai/v1/models",
            "deepinfra": "https://api.deepinfra.com/v1/openai/models",
            "cerebras": "https://api.cerebras.ai/v1/models",
            "openrouter": "https://openrouter.ai/api/v1/models",
        }

        return health_endpoints.get(provider.name)

    def _get_headers_for_provider(self, provider: ProviderConfig) -> dict[str, str]:
        """
        Get appropriate headers for provider health check.

        Args:
            provider: Provider configuration

        Returns:
            Headers dictionary for the request
        """
        headers = {"User-Agent": "vexy-co-model-catalog/health-check", "Accept": "application/json"}

        # Add authentication if available (but don't fail if missing)
        if provider.api_key_env:
            api_key = os.environ.get(provider.api_key_env)
            if api_key:
                if provider.kind == ProviderKind.ANTHROPIC:
                    headers["x-api-key"] = api_key
                    headers["anthropic-version"] = "2023-06-01"
                else:
                    headers["Authorization"] = f"Bearer {api_key}"

        return headers

    def _analyze_response_status(self, response: httpx.Response) -> HealthStatus:
        """
        Analyze HTTP response to determine health status.

        Args:
            response: HTTP response object

        Returns:
            HealthStatus based on response analysis
        """
        status_code = response.status_code

        # Success codes indicate healthy service
        if HTTP_OK_START <= status_code < HTTP_OK_END:
            return HealthStatus.HEALTHY

        # Authentication errors suggest service is up but auth is needed
        if status_code in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
            return HealthStatus.DEGRADED

        # Rate limiting suggests service is up but overloaded
        if status_code == HTTP_TOO_MANY_REQUESTS:
            return HealthStatus.DEGRADED

        # Client errors that aren't auth-related
        if HTTP_CLIENT_ERROR_START <= status_code < HTTP_CLIENT_ERROR_END:
            return HealthStatus.DEGRADED

        # Server errors indicate unhealthy service
        if HTTP_SERVER_ERROR_START <= status_code < HTTP_SERVER_ERROR_END:
            return HealthStatus.UNHEALTHY

        # Unknown status codes
        return HealthStatus.UNKNOWN
