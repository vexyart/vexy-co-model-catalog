"""
this_file: tests/test_validation.py

Test provider configuration validation functionality.
"""

import os
from unittest.mock import patch

import pytest

from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind
from vexy_co_model_catalog.core.validator import ProviderValidator, ValidationResult, ValidationSummary


class TestProviderValidator:
    """Test suite for provider configuration validation."""

    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return ProviderValidator()

    @pytest.fixture
    def valid_openai_provider(self):
        """Create a valid OpenAI provider config."""
        return ProviderConfig(
            name="openai", kind=ProviderKind.OPENAI, api_key_env="OPENAI_API_KEY", base_url="https://api.openai.com/v1"
        )

    @pytest.fixture
    def invalid_provider(self):
        """Create an invalid provider config."""
        return ProviderConfig(
            name="broken",
            kind=ProviderKind.OPENAI,
            api_key_env="",  # Missing API key env
            base_url="invalid-url",  # Invalid URL
        )

    def test_validate_valid_provider(self, validator, valid_openai_provider):
        """Test validation of a valid provider."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test1234567890123456789012345678901234567890123456"}):
            result = validator.validate_provider(valid_openai_provider)

            assert result.is_valid
            assert result.provider_name == "openai"
            assert len(result.issues) == 0
            assert result.guidance  # Should have guidance

    def test_validate_invalid_provider(self, validator, invalid_provider):
        """Test validation of an invalid provider."""
        result = validator.validate_provider(invalid_provider)

        assert not result.is_valid
        assert result.provider_name == "broken"
        assert len(result.issues) > 0
        assert any("Missing API key" in issue for issue in result.issues)

    def test_validate_missing_api_key(self, validator, valid_openai_provider):
        """Test validation when API key environment variable is missing."""
        with patch.dict(os.environ, {}, clear=True):
            result = validator.validate_provider(valid_openai_provider)

            assert not result.is_valid
            assert any("OPENAI_API_KEY is not set" in issue for issue in result.issues)

    def test_validate_multiple_providers(self, validator, valid_openai_provider, invalid_provider):
        """Test validation of multiple providers."""
        providers = [valid_openai_provider, invalid_provider]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test1234567890123456789012345678901234567890123456"}):
            summary = validator.validate_providers(providers)

            assert summary.total_providers == 2
            assert summary.valid_providers == 1
            assert summary.invalid_providers == 1
            assert summary.success_rate == 50.0

    def test_url_provider_validation(self, validator):
        """Test validation of URL provider (should not require API key)."""
        url_provider = ProviderConfig(name="test_url", kind=ProviderKind.URL, base_url="https://api.example.com/models")

        result = validator.validate_provider(url_provider)

        # Should be valid even without API key for URL providers
        assert result.is_valid
        assert len(result.issues) == 0

    def test_provider_guidance(self, validator):
        """Test that provider-specific guidance is provided."""
        guidance = validator.get_setup_guidance("openai")

        assert len(guidance) > 0
        assert any("openai" in guide.lower() for guide in guidance)
        assert any("OPENAI_API_KEY" in guide for guide in guidance)

    def test_validation_result_dataclass(self):
        """Test ValidationResult dataclass."""
        result = ValidationResult(
            is_valid=True, provider_name="test", issues=[], warnings=["test warning"], guidance=["test guidance"]
        )

        assert result.is_valid
        assert result.provider_name == "test"
        assert len(result.warnings) == 1
        assert len(result.guidance) == 1

    def test_validation_summary_success_rate(self):
        """Test ValidationSummary success rate calculation."""
        summary = ValidationSummary(total_providers=4, valid_providers=3, invalid_providers=1, results=[])

        assert summary.success_rate == 75.0

        # Test zero providers case
        empty_summary = ValidationSummary(total_providers=0, valid_providers=0, invalid_providers=0, results=[])

        assert empty_summary.success_rate == 0.0
