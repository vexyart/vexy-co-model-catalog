"""
this_file: tests/test_provider_types.py

Unit tests for all provider types and special cases including error scenarios.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from vexy_co_model_catalog.core.config import AichatConfigParser, CodexConfigParser, ConfigManager
from vexy_co_model_catalog.core.fetcher import ModelFetcher
from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind, get_all_providers, get_provider_by_name
from vexy_co_model_catalog.core.storage import StorageManager
from vexy_co_model_catalog.utils.exceptions import AuthenticationError, FetchError, RateLimitError


class TestProviderConfig:
    """Test ProviderConfig creation and validation."""

    def test_openai_provider_creation(self):
        """Test creating OpenAI-compatible provider config."""
        provider = ProviderConfig(
            name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY"
        )

        assert provider.name == "openai"
        assert provider.kind == ProviderKind.OPENAI
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.api_key_env == "OPENAI_API_KEY"

    def test_anthropic_provider_creation(self):
        """Test creating Anthropic provider config."""
        provider = ProviderConfig(
            name="anthropic",
            kind=ProviderKind.ANTHROPIC,
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
        )

        assert provider.name == "anthropic"
        assert provider.kind == ProviderKind.ANTHROPIC
        assert provider.base_url == "https://api.anthropic.com/v1"
        assert provider.api_key_env == "ANTHROPIC_API_KEY"

    def test_url_provider_creation(self):
        """Test creating URL provider config."""
        provider = ProviderConfig(
            name="litellm",
            kind=ProviderKind.URL,
            base_url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            api_key_env="",
        )

        assert provider.name == "litellm"
        assert provider.kind == ProviderKind.URL
        assert provider.base_url.startswith("https://raw.githubusercontent.com")
        assert provider.api_key_env == ""


class TestProviderDiscovery:
    """Test provider discovery and lookup functionality."""

    def test_get_all_providers(self):
        """Test getting all registered providers."""
        providers = get_all_providers()

        assert len(providers) >= 40, "Should have at least 40 providers"

        # Check we have the expected provider types
        provider_names = [p.name for p in providers]
        assert "openai" in provider_names
        assert "anthropic" in provider_names
        assert "groq" in provider_names
        assert "chutes" in provider_names

    def test_get_provider_by_name_success(self):
        """Test getting provider by name when it exists."""
        provider = get_provider_by_name("openai")

        assert provider is not None
        assert provider.name == "openai"
        assert provider.kind == ProviderKind.OPENAI

    def test_get_provider_by_name_not_found(self):
        """Test getting provider by name when it doesn't exist."""
        provider = get_provider_by_name("nonexistent_provider")

        assert provider is None

    def test_provider_kinds_distribution(self):
        """Test that we have providers of all expected kinds."""
        providers = get_all_providers()

        openai_count = sum(1 for p in providers if p.kind == ProviderKind.OPENAI)
        anthropic_count = sum(1 for p in providers if p.kind == ProviderKind.ANTHROPIC)
        url_count = sum(1 for p in providers if p.kind == ProviderKind.URL)

        assert openai_count >= 35, "Should have many OpenAI-compatible providers"
        assert anthropic_count >= 1, "Should have Anthropic provider"
        assert url_count >= 1, "Should have URL providers"


class TestModelFetcher:
    """Test ModelFetcher with different provider types."""

    @pytest.fixture
    def storage_manager(self, tmp_path):
        """Create temporary storage manager for testing."""
        return StorageManager(tmp_path)

    @pytest.fixture
    def fetcher(self, storage_manager):
        """Create ModelFetcher for testing."""
        from vexy_co_model_catalog.core.failure_tracker import FailureTracker

        failure_tracker = FailureTracker(storage_manager)
        return ModelFetcher(failure_tracker=failure_tracker)

    @pytest.mark.asyncio
    async def test_openai_provider_fetching(self, fetcher):
        """Test fetching from OpenAI-compatible provider."""
        provider = ProviderConfig(
            name="test_openai", kind=ProviderKind.OPENAI, base_url="https://api.test.com/v1", api_key_env="TEST_API_KEY"
        )

        mock_response = {"data": [{"id": "gpt-4", "object": "model"}, {"id": "gpt-3.5-turbo", "object": "model"}]}

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response
            mock_get.return_value = mock_resp

            async with fetcher:
                result = await fetcher.fetch_provider_models(provider)

                assert "data" in result
                assert len(result["data"]) == 2
                assert result["data"][0]["id"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_anthropic_provider_fetching(self, fetcher):
        """Test fetching from Anthropic provider with version headers."""
        provider = ProviderConfig(
            name="anthropic",
            kind=ProviderKind.ANTHROPIC,
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
        )

        mock_response = [
            {"name": "claude-3-haiku-20240307", "type": "model"},
            {"name": "claude-3-sonnet-20240229", "type": "model"},
        ]

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response
            mock_get.return_value = mock_resp

            async with fetcher:
                result = await fetcher.fetch_provider_models(provider)

                assert isinstance(result, list)
                assert len(result) == 2
                assert result[0]["name"] == "claude-3-haiku-20240307"

                # Verify Anthropic headers were used
                call_args = mock_get.call_args
                headers = call_args[1]["headers"]
                assert "anthropic-version" in headers
                assert "x-api-key" in headers

    @pytest.mark.asyncio
    async def test_url_provider_fetching(self, fetcher):
        """Test fetching from URL provider (direct JSON)."""
        provider = ProviderConfig(
            name="test_url", kind=ProviderKind.URL, base_url="https://example.com/models.json", api_key_env=""
        )

        mock_response = {"models": ["model-1", "model-2", "model-3"]}

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response
            mock_get.return_value = mock_resp

            async with fetcher:
                result = await fetcher.fetch_provider_models(provider)

                assert "models" in result
                assert len(result["models"]) == 3

    @pytest.mark.asyncio
    async def test_chutes_special_handling(self, fetcher):
        """Test Chutes provider dual-API merge functionality."""
        provider = get_provider_by_name("chutes")
        if not provider:
            pytest.skip("Chutes provider not found in configuration")

        # Mock both API responses that should be merged
        main_response = {"data": [{"id": "main-model-1", "object": "model"}]}

        supplemental_response = {"data": [{"id": "supp-model-1", "object": "model"}]}

        with patch("httpx.AsyncClient.get") as mock_get:
            # Set up different responses for different URLs
            def side_effect(url, **kwargs):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                if "chutes.ai" in str(url):
                    mock_resp.json.return_value = main_response
                else:
                    mock_resp.json.return_value = supplemental_response
                return mock_resp

            mock_get.side_effect = side_effect

            async with fetcher:
                result = await fetcher.fetch_provider_models(provider)

                assert "data" in result
                # Should have merged results from both APIs
                assert len(result["data"]) >= 1


class TestErrorHandling:
    """Test error handling scenarios for all provider types."""

    @pytest.fixture
    def storage_manager(self, tmp_path):
        return StorageManager(tmp_path)

    @pytest.fixture
    def fetcher(self, storage_manager):
        from vexy_co_model_catalog.core.failure_tracker import FailureTracker

        failure_tracker = FailureTracker(storage_manager)
        return ModelFetcher(failure_tracker=failure_tracker)

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self, fetcher):
        """Test handling of authentication errors (401/403)."""
        provider = ProviderConfig(
            name="test_auth_fail",
            kind=ProviderKind.OPENAI,
            base_url="https://api.test.com/v1",
            api_key_env="INVALID_KEY",
        )

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_resp
            )
            mock_get.return_value = mock_resp

            async with fetcher:
                with pytest.raises(AuthenticationError):
                    await fetcher.fetch_provider_models(provider)

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, fetcher):
        """Test handling of rate limit errors (429)."""
        provider = ProviderConfig(
            name="test_rate_limit", kind=ProviderKind.OPENAI, base_url="https://api.test.com/v1", api_key_env="TEST_KEY"
        )

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Too Many Requests", request=MagicMock(), response=mock_resp
            )
            mock_get.return_value = mock_resp

            async with fetcher:
                with pytest.raises(RateLimitError):
                    await fetcher.fetch_provider_models(provider)

    @pytest.mark.asyncio
    async def test_general_fetch_error_handling(self, fetcher):
        """Test handling of general fetch errors."""
        provider = ProviderConfig(
            name="test_fetch_fail", kind=ProviderKind.OPENAI, base_url="https://api.test.com/v1", api_key_env="TEST_KEY"
        )

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.RequestError("Network error", request=MagicMock())

            async with fetcher:
                with pytest.raises(FetchError):
                    await fetcher.fetch_provider_models(provider)

    @pytest.mark.asyncio
    async def test_retry_logic_success_after_failure(self, fetcher):
        """Test retry logic succeeds after initial failures."""
        provider = ProviderConfig(
            name="test_retry", kind=ProviderKind.OPENAI, base_url="https://api.test.com/v1", api_key_env="TEST_KEY"
        )

        success_response = {"data": [{"id": "test-model", "object": "model"}]}

        call_count = 0

        def mock_get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count < 3:  # Fail first 2 attempts
                mock_resp = MagicMock()
                mock_resp.status_code = 500
                raise httpx.HTTPStatusError(
                    "Server Error", request=MagicMock(), response=mock_resp
                )
            else:  # Succeed on 3rd attempt
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = success_response
                return mock_resp

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = mock_get_side_effect

            async with fetcher:
                result = await fetcher.fetch_provider_models(provider, _max_attempts=3, use_cache=False)

                assert call_count == 3  # Should have retried 3 times
                assert "data" in result
                assert result["data"][0]["id"] == "test-model"


class TestConfigGeneration:
    """Test config generation for different tools and providers."""

    @pytest.fixture
    def storage_manager(self, tmp_path):
        return StorageManager(tmp_path)

    @pytest.fixture
    def config_manager(self, storage_manager):
        return ConfigManager(storage_manager.config_dir)

    def test_aichat_config_generation(self, config_manager):
        """Test aichat YAML config generation."""
        from vexy_co_model_catalog.core.config import ConfigTemplate

        provider = ProviderConfig(
            name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY"
        )

        parser = AichatConfigParser()
        template = ConfigTemplate(tool_name="aichat", provider_id="openai", provider_config=provider)
        config = parser.generate(template)

        assert "openai" in config
        assert config["openai"]["type"] == "openai"
        assert config["openai"]["api_base"] == "https://api.openai.com/v1"
        assert config["openai"]["api_key"] == "${OPENAI_API_KEY}"

    def test_codex_config_generation(self, config_manager):
        """Test codex TOML config generation."""
        from vexy_co_model_catalog.core.config import ConfigTemplate
        from vexy_co_model_catalog.core.provider import Model

        provider = ProviderConfig(
            name="anthropic",
            kind=ProviderKind.ANTHROPIC,
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
        )

        # Create a sample model for testing
        test_model = Model(
            id="claude-3-haiku-20240307", provider="anthropic", max_input_tokens=200000, max_output_tokens=4096
        )

        parser = CodexConfigParser()
        template = ConfigTemplate(
            tool_name="codex", provider_id="anthropic", provider_config=provider, models=[test_model]
        )
        config = parser.generate(template)

        assert "profiles" in config
        assert "anthropic-claude-3-haiku-20240307" in config["profiles"]
        profile = config["profiles"]["anthropic-claude-3-haiku-20240307"]
        assert profile["base_url"] == "https://api.anthropic.com/v1"
        assert profile["api_key_env"] == "ANTHROPIC_API_KEY"
        assert profile["model"] == "claude-3-haiku-20240307"

    def test_config_template_creation(self, config_manager):
        """Test creating config templates for providers."""
        from vexy_co_model_catalog.core.config import ConfigTemplate

        provider = ProviderConfig(
            name="groq", kind=ProviderKind.OPENAI, base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY"
        )

        template = ConfigTemplate(tool_name="test_tool", provider_id="groq", provider_config=provider)

        assert template.provider_id == "groq"
        assert template.provider_config == provider
        assert template.models == []  # Empty until populated


def test_provider_import_coverage():
    """Test that all expected providers from dump_models.py are imported."""
    providers = get_all_providers()
    provider_names = {p.name for p in providers}

    # Key providers that should be present
    expected_providers = {
        "openai",
        "anthropic",
        "groq",
        "cerebras",
        "deepinfra",
        "fireworks",
        "togetherai",
        "openrouter",
        "huggingface",
        "chutes",
        "mistral",
    }

    missing_providers = expected_providers - provider_names
    assert not missing_providers, f"Missing expected providers: {missing_providers}"

    # Check total count is reasonable
    assert len(providers) >= 35, f"Expected at least 35 providers, got {len(providers)}"


def test_provider_kind_mapping():
    """Test that provider kind mapping works correctly."""
    providers = get_all_providers()

    # Test specific mappings
    openai_provider = get_provider_by_name("openai")
    assert openai_provider.kind == ProviderKind.OPENAI

    anthropic_provider = get_provider_by_name("anthropic")
    assert anthropic_provider.kind == ProviderKind.ANTHROPIC

    # Test that URL providers exist
    url_providers = [p for p in providers if p.kind == ProviderKind.URL]
    assert len(url_providers) >= 1, "Should have at least one URL provider"


if __name__ == "__main__":
    pytest.main([__file__])
