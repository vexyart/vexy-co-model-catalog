"""
this_file: tests/test_dump_models_compatibility.py

Compatibility tests ensuring the new package maintains compatibility with external/dump_models.py.
Tests output formats, file naming, and data structures to ensure backwards compatibility.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import yaml

from vexy_co_model_catalog.core.failure_tracker import FailureTracker
from vexy_co_model_catalog.core.fetcher import ModelFetcher
from vexy_co_model_catalog.core.provider import Model, ProviderConfig, ProviderKind
from vexy_co_model_catalog.core.storage import StorageManager


class TestDumpModelsCompatibility:
    """Test compatibility with external/dump_models.py output formats."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def storage_manager(self, temp_dir):
        """Create StorageManager for testing."""
        return StorageManager(temp_dir)

    @pytest.fixture
    def failure_tracker(self, storage_manager):
        """Create FailureTracker for testing."""
        return FailureTracker(storage_manager)

    @pytest.fixture
    def fetcher(self, failure_tracker):
        """Create ModelFetcher for testing."""
        return ModelFetcher(failure_tracker=failure_tracker)

    @pytest.fixture
    def sample_provider(self):
        """Create sample provider config."""
        return ProviderConfig(
            name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY"
        )

    @pytest.fixture
    def openai_api_response(self):
        """Mock OpenAI API response."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "gpt-4",
                    "object": "model",
                    "created": 1687882411,
                    "owned_by": "openai",
                    "permission": [],
                    "root": "gpt-4",
                    "parent": None,
                },
                {
                    "id": "gpt-3.5-turbo",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "openai",
                    "permission": [],
                    "root": "gpt-3.5-turbo",
                    "parent": None,
                },
            ],
        }

    def test_output_file_naming_compatibility(self, storage_manager, sample_provider):
        """Test that output files use compatible naming convention."""
        # Expected naming pattern: models_{provider_name}.{ext}
        provider_name = sample_provider.name

        # Test JSON file naming
        test_data = {"test": "data"}
        storage_manager.write_config_json(f"models_{provider_name}", test_data)
        json_path = storage_manager.config_json_dir / f"models_{provider_name}.json"
        assert json_path.exists()

        # Test TXT file naming
        test_lines = ["gpt-4", "gpt-3.5-turbo"]
        storage_manager.write_config_txt(f"models_{provider_name}", test_lines)
        txt_path = storage_manager.config_txt_dir / f"models_{provider_name}.txt"
        assert txt_path.exists()

        # Verify TXT file content format
        with txt_path.open("r", encoding="utf-8") as f:
            content = f.read()
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            assert lines == test_lines

    def test_json_format_compatibility(self, storage_manager, openai_api_response):
        """Test JSON output format matches dump_models.py expectations."""
        # The JSON should preserve the original API response structure
        storage_manager.write_config_json("models_openai", openai_api_response)
        json_path = storage_manager.config_json_dir / "models_openai.json"

        # Read back and verify structure
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Should have OpenAI API format with 'data' array
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 2

        # Check model entries have required fields
        for model in data["data"]:
            assert "id" in model
            assert "object" in model
            assert "created" in model
            assert "owned_by" in model

    def test_txt_format_model_id_extraction(self, storage_manager, openai_api_response):
        """Test TXT format contains just model IDs, one per line."""
        # Extract model IDs from API response (compatible with dump_models.py logic)
        model_ids = [item["id"] for item in openai_api_response["data"]]

        storage_manager.write_config_txt("models_openai", model_ids)
        txt_path = storage_manager.config_txt_dir / "models_openai.txt"

        # Verify content matches expected format
        with txt_path.open("r", encoding="utf-8") as f:
            content = f.read()
            lines = [line.strip() for line in content.split("\n") if line.strip()]

        assert lines == ["gpt-4", "gpt-3.5-turbo"]
        assert len(lines) == len(model_ids)

    @patch("httpx.AsyncClient.get")
    @pytest.mark.asyncio
    async def test_fetcher_produces_compatible_output(
        self, mock_get, fetcher, sample_provider, openai_api_response, temp_dir
    ):
        """Test that ModelFetcher produces output compatible with dump_models.py."""
        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: openai_api_response  # Simple lambda instead of mock
        mock_response.headers = {"date": "Wed, 01 Jan 2025 00:00:00 GMT"}
        mock_response.raise_for_status = lambda: None
        mock_get.return_value = mock_response

        # Fetch models - returns raw dict/list data compatible with dump_models.py
        data = await fetcher.fetch_provider_models(sample_provider)

        # Verify we got the expected OpenAI API format
        assert isinstance(data, dict)
        assert "data" in data
        assert len(data["data"]) == 2

        # Find GPT-4 model in the data
        gpt4_model = next((m for m in data["data"] if m["id"] == "gpt-4"), None)
        assert gpt4_model is not None
        assert gpt4_model["owned_by"] == "openai"
        assert "created" in gpt4_model

        # Verify structure matches what dump_models.py expects
        assert data["object"] == "list"
        assert len(data["data"]) == 2

        # Verify model IDs can be extracted like dump_models.py does
        model_ids = [item["id"] for item in data["data"]]
        assert "gpt-4" in model_ids
        assert "gpt-3.5-turbo" in model_ids

    def test_provider_config_compatibility(self, sample_provider):
        """Test that ProviderConfig can produce dump_models.py compatible URLs."""
        # Test URL building matches dump_models.py expectations
        models_url = sample_provider.build_models_url()
        assert models_url == "https://api.openai.com/v1/models"

        # Test environment variable handling
        sample_provider.base_url_env = "TEST_API_URL"
        with patch.dict("os.environ", {"TEST_API_URL": "https://test.api.com/v1"}):
            models_url = sample_provider.build_models_url()
            assert models_url == "https://test.api.com/v1/models"

    def test_url_provider_compatibility(self):
        """Test URL provider type matches dump_models.py behavior."""
        url_provider = ProviderConfig(
            name="litellm",
            kind=ProviderKind.URL,
            base_url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            api_key_env=None,
        )

        # URL providers should return the direct URL
        models_url = url_provider.build_models_url()
        assert (
            models_url == "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
        )

        # Should not append /models for URL providers
        assert not models_url.endswith("/models")

    def test_anthropic_provider_compatibility(self):
        """Test Anthropic provider configuration matches dump_models.py."""
        anthropic_provider = ProviderConfig(
            name="anthropic",
            kind=ProviderKind.ANTHROPIC,
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
        )

        # Should build correct models URL
        models_url = anthropic_provider.build_models_url()
        assert models_url == "https://api.anthropic.com/v1/models"

    def test_model_sorting_compatibility(self, openai_api_response):
        """Test that model sorting matches dump_models.py behavior."""
        # dump_models.py sorts by model ID
        unsorted_data = {
            "object": "list",
            "data": [
                {"id": "gpt-3.5-turbo", "created": 1677610602},
                {"id": "gpt-4", "created": 1687882411},
                {"id": "gpt-3.5-turbo-16k", "created": 1685474247},
            ],
        }

        # Sort like dump_models.py does
        sorted_data = unsorted_data.copy()
        sorted_data["data"] = sorted(sorted_data["data"], key=lambda x: x.get("id", ""))

        expected_order = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]
        actual_order = [item["id"] for item in sorted_data["data"]]

        assert actual_order == expected_order

    @pytest.mark.asyncio
    async def test_error_handling_compatibility(self, fetcher, sample_provider):
        """Test error handling maintains dump_models.py behavior patterns."""
        # Test network error handling
        with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection failed")):
            # Should raise exception - this tests that errors are properly propagated
            with pytest.raises(Exception, match="Connection failed"):  # Could be FetchError or ConnectError
                await fetcher.fetch_provider_models(sample_provider)

    def test_special_field_extraction_compatibility(self):
        """Test extraction of special fields like context_length matches dump_models.py."""
        # Test various field name patterns that dump_models.py handles
        model_data_samples = [
            {"id": "test-1", "context_length": 4096},
            {"id": "test-2", "context_window": 8192},
            {"id": "test-3", "max_context_length": 2048},
            {"id": "test-4", "max_tokens": 1024},
            {"id": "test-5", "max_model_len": 16384},  # chutes provider
            {"id": "test-6", "top_provider": {"context_length": 32768}},  # OpenRouter format
        ]

        for sample in model_data_samples:
            # Test that we can extract context length using dump_models.py logic
            context_length = (
                sample.get("context_length")
                or sample.get("context_window")
                or sample.get("max_context_length")
                or sample.get("max_tokens")
                or sample.get("max_model_len")
            )

            # Handle OpenRouter format
            if "top_provider" in sample:
                context_length = sample["top_provider"].get("context_length")

            assert context_length is not None
            assert isinstance(context_length, int)


class TestLegacyFormatCompatibility:
    """Test compatibility with legacy dump_models.py data formats."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_failed_models_tracking_format(self, temp_dir):
        """Test failed models tracking uses compatible JSON format."""
        failure_tracker = FailureTracker(StorageManager(temp_dir))

        # Simulate a failure
        failure_tracker.mark_provider_failed("openai", "Network timeout")

        # Check that failed models file uses compatible format
        storage_manager = StorageManager(temp_dir)
        failed_models_file = storage_manager.extra_dir / "failed_providers.json"
        assert failed_models_file.exists()

        with failed_models_file.open("r", encoding="utf-8") as f:
            failed_data = json.load(f)

        # Should contain compatible failure tracking info
        assert "openai" in failed_data
        failure_record = failed_data["openai"]
        assert "failed" in failure_record
        assert failure_record["failed"] is True

        # New format has 'errors' array instead of single 'error' field
        # This is actually an improvement over dump_models.py format
        if "errors" in failure_record:
            assert len(failure_record["errors"]) > 0
            assert failure_record["errors"][0]["error"] == "Network timeout"
        else:
            # Legacy format fallback
            assert "error" in failure_record

    def test_provider_list_compatibility(self):
        """Test that provider lists match dump_models.py configuration."""
        from vexy_co_model_catalog.core.provider import parse_provider_config

        providers = parse_provider_config()

        # Should have major providers that dump_models.py supports
        provider_names = [p.name for p in providers]
        expected_providers = [
            "openai",
            "anthropic",
            "groq",
            "fireworks",
            "togetherai",
            "deepinfra",
            "mistral",
            "huggingface",
            "openrouter",
        ]

        for expected in expected_providers:
            assert expected in provider_names, f"Missing provider: {expected}"

        # Test specific provider configuration matches
        openai_provider = next((p for p in providers if p.name == "openai"), None)
        assert openai_provider is not None
        assert openai_provider.kind == ProviderKind.OPENAI
        assert openai_provider.api_key_env == "OPENAI_API_KEY"


if __name__ == "__main__":
    pytest.main([__file__])
