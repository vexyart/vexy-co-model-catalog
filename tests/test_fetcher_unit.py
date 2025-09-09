#!/usr/bin/env python3
"""
this_file: tests/test_fetcher_unit.py

Comprehensive unit tests for the ModelFetcher class ensuring HTTP and networking reliability.
"""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from vexy_co_model_catalog.core.failure_tracker import FailureTracker
from vexy_co_model_catalog.core.fetcher import ModelFetcher
from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind
from vexy_co_model_catalog.utils.exceptions import AuthenticationError, FetchError, RateLimitError


class TestModelFetcher(unittest.TestCase):
    """Comprehensive unit tests for ModelFetcher HTTP operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_failure_tracker = Mock(spec=FailureTracker)
        self.mock_failure_tracker.is_provider_failed.return_value = False
        self.mock_failure_tracker.mark_provider_failed = Mock()
        self.mock_failure_tracker.mark_provider_success = Mock()

    def tearDown(self):
        """Clean up after tests."""
        # Clear any environment variables set during tests
        env_vars_to_clear = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CHUTES_API_KEY"]
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    def test_fetcher_initialization(self):
        """Test ModelFetcher initialization with various parameters."""
        # Default initialization
        fetcher = ModelFetcher()
        assert fetcher._semaphore._value == 8  # Default max_concurrency
        assert fetcher._client is not None
        assert fetcher._request_count == 0
        assert fetcher._error_count == 0
        assert fetcher._failure_tracker is None

        # Custom initialization
        fetcher_custom = ModelFetcher(max_concurrency=4, timeout=60.0, failure_tracker=self.mock_failure_tracker)
        assert fetcher_custom._semaphore._value == 4
        assert fetcher_custom._failure_tracker == self.mock_failure_tracker

    @patch("vexy_co_model_catalog.core.fetcher.get_cached_model_data")
    @patch("vexy_co_model_catalog.core.fetcher.cache_model_data")
    def test_fetcher_context_manager(self, mock_cache, mock_get_cache):
        """Test ModelFetcher as async context manager."""

        async def test_context():
            fetcher = ModelFetcher()
            async with fetcher as f:
                assert f is fetcher
                assert f._client is not None
            # After context exit, client should be closed
            # Note: We can't easily test if client is closed without implementation details

        asyncio.run(test_context())

    def test_build_provider_headers(self):
        """Test header building for different provider types."""
        fetcher = ModelFetcher()

        # Test OpenAI provider without API key
        openai_provider = ProviderConfig(
            name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com", models_path="/v1/models"
        )
        headers = fetcher._build_provider_headers(openai_provider)
        expected = {"User-Agent": "ModelDumper/1.0"}
        assert headers == expected

        # Test OpenAI provider with API key
        os.environ["OPENAI_API_KEY"] = "test-key-123"
        openai_provider.api_key_env = "OPENAI_API_KEY"
        headers = fetcher._build_provider_headers(openai_provider)
        expected = {"User-Agent": "ModelDumper/1.0", "Authorization": "Bearer test-key-123"}
        assert headers == expected

        # Test Anthropic provider
        os.environ["ANTHROPIC_API_KEY"] = "ant-key-456"
        anthropic_provider = ProviderConfig(
            name="anthropic",
            kind=ProviderKind.ANTHROPIC,
            base_url="https://api.anthropic.com",
            models_path="/v1/models",
            api_key_env="ANTHROPIC_API_KEY",
        )
        headers = fetcher._build_provider_headers(anthropic_provider)
        expected = {"User-Agent": "ModelDumper/1.0", "x-api-key": "ant-key-456", "anthropic-version": "2023-06-01"}
        assert headers == expected

        # Test URL provider (no auth)
        url_provider = ProviderConfig(
            name="litellm", kind=ProviderKind.URL, base_url="https://registry.litellm.ai", models_path="/models"
        )
        headers = fetcher._build_provider_headers(url_provider)
        expected = {"User-Agent": "ModelDumper/1.0"}
        assert headers == expected

        # Test provider with custom headers
        custom_provider = ProviderConfig(
            name="custom",
            kind=ProviderKind.OPENAI,
            base_url="https://api.custom.com",
            models_path="/models",
            headers={"Custom-Header": "custom-value", "X-API-Version": "v2"},
        )
        headers = fetcher._build_provider_headers(custom_provider)
        expected = {"User-Agent": "ModelDumper/1.0", "Custom-Header": "custom-value", "X-API-Version": "v2"}
        assert headers == expected

    @patch("vexy_co_model_catalog.core.fetcher.get_cached_model_data")
    @patch("vexy_co_model_catalog.core.fetcher.cache_model_data")
    @patch("vexy_co_model_catalog.core.fetcher.get_rate_limiter")
    def test_fetch_provider_models_success(self, mock_get_rate_limiter, mock_cache, mock_get_cache):
        """Test successful model fetching."""
        # Setup
        mock_get_cache.return_value = None  # No cache hit
        mock_rate_limiter = Mock()
        mock_rate_limiter.acquire_permit = AsyncMock(return_value=0)
        mock_rate_limiter.record_response = AsyncMock()
        mock_rate_limiter.configure_provider = Mock()
        mock_get_rate_limiter.return_value = mock_rate_limiter

        # Create test data
        test_data = {"data": [{"id": "gpt-3.5-turbo", "object": "model"}, {"id": "gpt-4", "object": "model"}]}

        async def test_fetch():
            with patch.object(ModelFetcher, "_retry_handler") as mock_retry:
                # Setup mock response
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.is_success = True
                mock_response.json.return_value = test_data
                mock_response.headers = {"content-type": "application/json"}

                mock_retry.execute_with_retry = AsyncMock(return_value=mock_response)

                fetcher = ModelFetcher(failure_tracker=self.mock_failure_tracker)
                provider = ProviderConfig(
                    name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com", models_path="/v1/models"
                )

                result = await fetcher.fetch_provider_models(provider)

                assert result == test_data
                self.mock_failure_tracker.mark_provider_success.assert_called_once()
                mock_cache.assert_called_once()  # Data should be cached

        asyncio.run(test_fetch())

    @patch("vexy_co_model_catalog.core.fetcher.get_cached_model_data")
    def test_fetch_provider_models_cache_hit(self, mock_get_cache):
        """Test cache hit scenario."""
        # Setup cache hit
        cached_data = {"data": [{"id": "cached-model", "object": "model"}]}
        mock_get_cache.return_value = cached_data

        async def test_cache_hit():
            fetcher = ModelFetcher()
            provider = ProviderConfig(
                name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com", models_path="/v1/models"
            )

            result = await fetcher.fetch_provider_models(provider)
            assert result == cached_data

        asyncio.run(test_cache_hit())

    @patch("vexy_co_model_catalog.core.fetcher.get_cached_model_data")
    @patch("vexy_co_model_catalog.core.fetcher.get_rate_limiter")
    def test_fetch_provider_models_authentication_error(self, mock_get_rate_limiter, mock_get_cache):
        """Test authentication error handling."""
        mock_get_cache.return_value = None
        mock_rate_limiter = Mock()
        mock_rate_limiter.acquire_permit = AsyncMock(return_value=0)
        mock_rate_limiter.record_response = AsyncMock()
        mock_rate_limiter.configure_provider = Mock()
        mock_get_rate_limiter.return_value = mock_rate_limiter

        async def test_auth_error():
            with patch.object(ModelFetcher, "_retry_handler") as mock_retry:
                # Setup 401 response
                mock_response = Mock()
                mock_response.status_code = 401
                mock_response.is_success = False
                mock_response.headers = {"content-type": "application/json"}

                mock_retry.execute_with_retry = AsyncMock(return_value=mock_response)

                fetcher = ModelFetcher(failure_tracker=self.mock_failure_tracker)
                provider = ProviderConfig(
                    name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com", models_path="/v1/models"
                )

                with pytest.raises(AuthenticationError):
                    await fetcher.fetch_provider_models(provider)

                self.mock_failure_tracker.mark_provider_failed.assert_called_once()

        asyncio.run(test_auth_error())

    @patch("vexy_co_model_catalog.core.fetcher.get_cached_model_data")
    @patch("vexy_co_model_catalog.core.fetcher.get_rate_limiter")
    def test_fetch_provider_models_rate_limit_error(self, mock_get_rate_limiter, mock_get_cache):
        """Test rate limit error handling."""
        mock_get_cache.return_value = None
        mock_rate_limiter = Mock()
        mock_rate_limiter.acquire_permit = AsyncMock(return_value=0)
        mock_rate_limiter.record_response = AsyncMock()
        mock_rate_limiter.configure_provider = Mock()
        mock_get_rate_limiter.return_value = mock_rate_limiter

        async def test_rate_limit():
            with patch.object(ModelFetcher, "_retry_handler") as mock_retry:
                # Setup 429 response
                mock_response = Mock()
                mock_response.status_code = 429
                mock_response.is_success = False
                mock_response.headers = {
                    "content-type": "application/json",
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": "1640995200",
                }

                mock_retry.execute_with_retry = AsyncMock(return_value=mock_response)

                fetcher = ModelFetcher(failure_tracker=self.mock_failure_tracker)
                provider = ProviderConfig(
                    name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com", models_path="/v1/models"
                )

                with pytest.raises(RateLimitError):
                    await fetcher.fetch_provider_models(provider)

                self.mock_failure_tracker.mark_provider_failed.assert_called_once()

        asyncio.run(test_rate_limit())

    @patch("vexy_co_model_catalog.core.fetcher.get_cached_model_data")
    @patch("vexy_co_model_catalog.core.fetcher.get_rate_limiter")
    def test_fetch_provider_models_http_error(self, mock_get_rate_limiter, mock_get_cache):
        """Test general HTTP error handling."""
        mock_get_cache.return_value = None
        mock_rate_limiter = Mock()
        mock_rate_limiter.acquire_permit = AsyncMock(return_value=0)
        mock_rate_limiter.record_response = AsyncMock()
        mock_rate_limiter.configure_provider = Mock()
        mock_get_rate_limiter.return_value = mock_rate_limiter

        async def test_http_error():
            with patch.object(ModelFetcher, "_retry_handler") as mock_retry:
                # Setup 500 response
                mock_response = Mock()
                mock_response.status_code = 500
                mock_response.is_success = False
                mock_response.text = "Internal Server Error"
                mock_response.headers = {"content-type": "text/plain"}

                mock_retry.execute_with_retry = AsyncMock(return_value=mock_response)

                fetcher = ModelFetcher(failure_tracker=self.mock_failure_tracker)
                provider = ProviderConfig(
                    name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com", models_path="/v1/models"
                )

                with pytest.raises(FetchError):
                    await fetcher.fetch_provider_models(provider)

                self.mock_failure_tracker.mark_provider_failed.assert_called_once()

        asyncio.run(test_http_error())

    @patch("vexy_co_model_catalog.core.fetcher.get_cached_model_data")
    @patch("vexy_co_model_catalog.core.fetcher.get_rate_limiter")
    def test_fetch_provider_models_json_decode_error(self, mock_get_rate_limiter, mock_get_cache):
        """Test JSON decode error handling."""
        mock_get_cache.return_value = None
        mock_rate_limiter = Mock()
        mock_rate_limiter.acquire_permit = AsyncMock(return_value=0)
        mock_rate_limiter.record_response = AsyncMock()
        mock_rate_limiter.configure_provider = Mock()
        mock_get_rate_limiter.return_value = mock_rate_limiter

        async def test_json_error():
            with patch.object(ModelFetcher, "_retry_handler") as mock_retry:
                # Setup response with invalid JSON
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.is_success = True
                mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
                mock_response.headers = {"content-type": "application/json"}

                mock_retry.execute_with_retry = AsyncMock(return_value=mock_response)

                fetcher = ModelFetcher(failure_tracker=self.mock_failure_tracker)
                provider = ProviderConfig(
                    name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com", models_path="/v1/models"
                )

                with pytest.raises(FetchError):
                    await fetcher.fetch_provider_models(provider)

                self.mock_failure_tracker.mark_provider_failed.assert_called_once()

        asyncio.run(test_json_error())

    def test_log_rate_limit_info(self):
        """Test rate limit info logging."""
        fetcher = ModelFetcher()

        # Test with rate limit headers
        headers = {"X-RateLimit-Remaining": "99", "X-RateLimit-Reset": "1640995200", "Content-Type": "application/json"}

        # This should not raise an exception
        fetcher._log_rate_limit_info("test_provider", headers)

        # Test with no rate limit headers
        headers_no_rate = {"Content-Type": "application/json"}
        fetcher._log_rate_limit_info("test_provider", headers_no_rate)

        # Test with case-insensitive headers
        headers_mixed_case = {"x-ratelimit-remaining": "50", "X-RATELIMIT-RESET": "1640995300"}
        fetcher._log_rate_limit_info("test_provider", headers_mixed_case)

    def test_sort_json_data(self):
        """Test JSON data sorting functionality."""
        fetcher = ModelFetcher()

        # Test OpenAI format with data array
        openai_data = {
            "data": [
                {"id": "gpt-4", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "ada", "object": "model"},
            ]
        }
        sorted_data = fetcher.sort_json_data(openai_data)
        expected_order = ["ada", "gpt-3.5-turbo", "gpt-4"]
        actual_order = [item["id"] for item in sorted_data["data"]]
        assert actual_order == expected_order

        # Test direct array format
        array_data = [{"id": "model-c"}, {"id": "model-a"}, {"id": "model-b"}]
        sorted_array = fetcher.sort_json_data(array_data)
        expected_array_order = ["model-a", "model-b", "model-c"]
        actual_array_order = [item["id"] for item in sorted_array]
        assert actual_array_order == expected_array_order

        # Test dict format
        dict_data = {"zebra": "value", "alpha": "value", "beta": "value"}
        sorted_dict = fetcher.sort_json_data(dict_data)
        expected_keys = ["alpha", "beta", "zebra"]
        assert list(sorted_dict.keys()) == expected_keys

        # Test primitive types (should return unchanged)
        assert fetcher.sort_json_data("string") == "string"
        assert fetcher.sort_json_data(123) == 123
        assert fetcher.sort_json_data(None) is None

    def test_extract_model_ids(self):
        """Test model ID extraction from various data formats."""
        fetcher = ModelFetcher()

        # Test OpenAI format
        openai_data = {
            "data": [
                {"id": "gpt-3.5-turbo", "object": "model"},
                {"id": "gpt-4", "object": "model"},
                {"id": "", "object": "model"},  # Empty ID should be filtered
                {"object": "model"},  # Missing ID should be filtered
            ]
        }
        ids = fetcher.extract_model_ids(openai_data)
        assert ids == ["gpt-3.5-turbo", "gpt-4"]

        # Test direct array format
        array_data = [
            {"id": "model-1"},
            {"id": "model-2"},
            {"no_id_field": "value"},  # Missing ID should be filtered
        ]
        ids = fetcher.extract_model_ids(array_data)
        assert ids == ["model-1", "model-2"]

        # Test object format (keys are model IDs)
        object_data = {
            "model-a": {"description": "Model A"},
            "model-b": {"description": "Model B"},
            "sample_spec": {"spec": "value"},  # Should be filtered out
        }
        ids = fetcher.extract_model_ids(object_data)
        assert set(ids) == {"model-a", "model-b"}

        # Test empty/invalid data
        assert fetcher.extract_model_ids({}) == []
        assert fetcher.extract_model_ids([]) == []
        assert fetcher.extract_model_ids("invalid") == []
        assert fetcher.extract_model_ids(None) == []

    def test_stats(self):
        """Test statistics tracking."""
        fetcher = ModelFetcher()

        # Test initial stats
        stats = fetcher.stats()
        expected = {"requests": 0, "errors": 0, "success_rate": None}
        assert stats == expected

        # Test stats with requests
        fetcher._request_count = 10
        fetcher._error_count = 2
        stats = fetcher.stats()
        expected = {"requests": 10, "errors": 2, "success_rate": 0.8}
        assert stats == expected

        # Test stats with only errors
        fetcher._request_count = 0
        fetcher._error_count = 5
        stats = fetcher.stats()
        expected = {"requests": 0, "errors": 5, "success_rate": None}
        assert stats == expected

    def test_merge_chutes_data(self):
        """Test chutes data merging functionality."""
        fetcher = ModelFetcher()

        # Test normal merge
        openai_data = {"data": [{"id": "model-1", "object": "model"}, {"id": "model-2", "object": "model"}]}

        chutes_data = {
            "items": [
                {
                    "model": "model-1",
                    "max_model_len": 4096,
                    "description": "Enhanced model 1",
                    "tags": ["chat", "completion"],
                },
                {
                    "model": "model-3",  # Not in OpenAI data
                    "max_model_len": 8192,
                },
            ]
        }

        merged = fetcher._merge_chutes_data(openai_data, chutes_data)

        # Check that model-1 was enhanced
        model_1 = next(m for m in merged["data"] if m["id"] == "model-1")
        assert model_1["max_model_len"] == 4096
        assert model_1["chute_description"] == "Enhanced model 1"
        assert model_1["chute_tags"] == ["chat", "completion"]

        # Check that model-2 remains unchanged
        model_2 = next(m for m in merged["data"] if m["id"] == "model-2")
        assert "max_model_len" not in model_2

        # Test with None chutes data
        merged_none = fetcher._merge_chutes_data(openai_data, None)
        assert merged_none == openai_data

        # Test with empty chutes data
        merged_empty = fetcher._merge_chutes_data(openai_data, {})
        assert merged_empty == openai_data

        # Test with chutes data missing items
        merged_no_items = fetcher._merge_chutes_data(openai_data, {"other": "data"})
        assert merged_no_items == openai_data


if __name__ == "__main__":
    unittest.main()
