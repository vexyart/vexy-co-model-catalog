"""
this_file: tests/test_health_check.py

Test provider health check functionality.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from vexy_co_model_catalog.core.health_check import (
    HealthCheckResult,
    HealthStatus,
    HealthSummary,
    ProviderHealthChecker,
)
from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind


class TestProviderHealthChecker:
    """Test suite for provider health checking."""

    @pytest.fixture
    def health_checker(self):
        """Create a health checker instance."""
        return ProviderHealthChecker(timeout=5.0, max_concurrency=2)

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider config."""
        return ProviderConfig(
            name="openai",
            kind=ProviderKind.OPENAI,
            api_key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1/models",
        )

    @pytest.fixture
    def url_provider(self):
        """Create URL provider config."""
        return ProviderConfig(name="test_url", kind=ProviderKind.URL, base_url="https://api.example.com/models")

    @pytest.fixture
    def provider_without_endpoint(self):
        """Create provider without testable endpoint."""
        return ProviderConfig(name="unknown_provider", kind=ProviderKind.OPENAI, api_key_env="UNKNOWN_API_KEY")

    @pytest.mark.asyncio
    async def test_check_provider_health_success(self, health_checker, openai_provider):
        """Test successful health check."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            async with health_checker:
                result = await health_checker.check_provider_health(openai_provider)

        assert result.provider_name == "openai"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms > 0
        assert result.status_code == 200
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_check_provider_health_auth_error(self, health_checker, openai_provider):
        """Test health check with authentication error (degraded)."""
        mock_response = AsyncMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            async with health_checker:
                result = await health_checker.check_provider_health(openai_provider)

        assert result.provider_name == "openai"
        assert result.status == HealthStatus.DEGRADED
        assert result.status_code == 401
        assert "HTTP 401" in result.error_message

    @pytest.mark.asyncio
    async def test_check_provider_health_server_error(self, health_checker, openai_provider):
        """Test health check with server error (unhealthy)."""
        mock_response = AsyncMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            async with health_checker:
                result = await health_checker.check_provider_health(openai_provider)

        assert result.provider_name == "openai"
        assert result.status == HealthStatus.UNHEALTHY
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_check_provider_health_timeout(self, health_checker, openai_provider):
        """Test health check with timeout."""
        with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Request timeout")):
            async with health_checker:
                result = await health_checker.check_provider_health(openai_provider)

        assert result.provider_name == "openai"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Request timeout" in result.error_message

    @pytest.mark.asyncio
    async def test_check_provider_health_network_error(self, health_checker, openai_provider):
        """Test health check with network error."""
        with patch("httpx.AsyncClient.get", side_effect=httpx.NetworkError("Connection failed")):
            async with health_checker:
                result = await health_checker.check_provider_health(openai_provider)

        assert result.provider_name == "openai"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Network error" in result.error_message

    @pytest.mark.asyncio
    async def test_check_provider_health_no_endpoint(self, health_checker, provider_without_endpoint):
        """Test health check for provider without testable endpoint."""
        async with health_checker:
            result = await health_checker.check_provider_health(provider_without_endpoint)

        assert result.provider_name == "unknown_provider"
        assert result.status == HealthStatus.UNKNOWN
        assert "No testable endpoint" in result.error_message
        assert result.endpoint_tested is None

    @pytest.mark.asyncio
    async def test_check_multiple_providers(self, health_checker, openai_provider, url_provider):
        """Test checking multiple providers."""
        providers = [openai_provider, url_provider]

        mock_response = AsyncMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            async with health_checker:
                summary = await health_checker.check_multiple_providers(providers)

        assert summary.total_providers == 2
        assert summary.healthy_providers == 2
        assert summary.unhealthy_providers == 0
        assert summary.health_percentage == 100.0
        assert len(summary.results) == 2

    @pytest.mark.asyncio
    async def test_check_multiple_providers_mixed_results(self, health_checker, openai_provider, url_provider):
        """Test checking multiple providers with mixed results."""
        providers = [openai_provider, url_provider]

        def mock_get(url, **kwargs):
            if "openai" in url:
                response = AsyncMock()
                response.status_code = 200
                return response
            response = AsyncMock()
            response.status_code = 500
            return response

        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            async with health_checker:
                summary = await health_checker.check_multiple_providers(providers)

        assert summary.total_providers == 2
        assert summary.healthy_providers == 1
        assert summary.unhealthy_providers == 1
        assert summary.health_percentage == 50.0

    @pytest.mark.asyncio
    async def test_check_empty_provider_list(self, health_checker):
        """Test checking empty provider list."""
        async with health_checker:
            summary = await health_checker.check_multiple_providers([])

        assert summary.total_providers == 0
        assert summary.healthy_providers == 0
        assert summary.health_percentage == 0.0
        assert len(summary.results) == 0

    def test_health_check_result_dataclass(self):
        """Test HealthCheckResult dataclass."""
        result = HealthCheckResult(
            provider_name="test",
            status=HealthStatus.HEALTHY,
            response_time_ms=150,
            status_code=200,
            endpoint_tested="https://api.test.com",
        )

        assert result.provider_name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms == 150
        assert result.timestamp > 0  # Should be set by __post_init__

    def test_health_summary_calculations(self):
        """Test HealthSummary calculations."""
        results = [
            HealthCheckResult("provider1", HealthStatus.HEALTHY, 100),
            HealthCheckResult("provider2", HealthStatus.DEGRADED, 200),
            HealthCheckResult("provider3", HealthStatus.UNHEALTHY, 0),
            HealthCheckResult("provider4", HealthStatus.UNKNOWN, 0),
        ]

        summary = HealthSummary(
            total_providers=4,
            healthy_providers=1,
            unhealthy_providers=1,
            degraded_providers=1,
            unknown_providers=1,
            results=results,
        )

        assert summary.health_percentage == 25.0  # 1 healthy out of 4

    def test_get_health_check_endpoint(self, health_checker, openai_provider):
        """Test endpoint resolution logic."""
        # Test with base_url
        endpoint = health_checker._get_health_check_endpoint(openai_provider)
        assert endpoint == "https://api.openai.com/v1/models"

        # Test with known provider
        openai_provider.base_url = None
        endpoint = health_checker._get_health_check_endpoint(openai_provider)
        assert endpoint == "https://api.openai.com/v1/models"

    def test_get_headers_for_provider(self, health_checker, openai_provider):
        """Test header generation for providers."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}):
            headers = health_checker._get_headers_for_provider(openai_provider)

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test123"
        assert "User-Agent" in headers
        assert "Accept" in headers

    def test_analyze_response_status(self, health_checker):
        """Test response status analysis."""
        mock_response = AsyncMock()

        # Test healthy (200)
        mock_response.status_code = 200
        assert health_checker._analyze_response_status(mock_response) == HealthStatus.HEALTHY

        # Test degraded (401)
        mock_response.status_code = 401
        assert health_checker._analyze_response_status(mock_response) == HealthStatus.DEGRADED

        # Test degraded (429)
        mock_response.status_code = 429
        assert health_checker._analyze_response_status(mock_response) == HealthStatus.DEGRADED

        # Test unhealthy (500)
        mock_response.status_code = 500
        assert health_checker._analyze_response_status(mock_response) == HealthStatus.UNHEALTHY

        # Test unknown (600)
        mock_response.status_code = 600
        assert health_checker._analyze_response_status(mock_response) == HealthStatus.UNKNOWN
