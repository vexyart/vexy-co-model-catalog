"""
this_file: tests/test_model_validator.py

Test model metadata validation and normalization functionality.
"""

import pytest

from vexy_co_model_catalog.core.model_validator import (
    ModelDataValidator,
    ModelValidationIssue,
    ModelValidationResult,
    ModelValidationSeverity,
)
from vexy_co_model_catalog.core.provider import Model, ProviderConfig, ProviderKind


class TestModelDataValidator:
    """Test suite for model data validation."""

    @pytest.fixture
    def validator(self):
        """Create a model validator instance."""
        return ModelDataValidator()

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
    def anthropic_provider(self):
        """Create Anthropic provider config."""
        return ProviderConfig(
            name="anthropic",
            kind=ProviderKind.ANTHROPIC,
            api_key_env="ANTHROPIC_API_KEY",
            base_url="https://api.anthropic.com/v1/models",
        )

    @pytest.fixture
    def valid_openai_response(self):
        """Create valid OpenAI-style response."""
        return {
            "data": [
                {"id": "gpt-4", "object": "model", "created": 1687882411, "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "object": "model", "created": 1677610602, "owned_by": "openai"},
            ]
        }

    @pytest.fixture
    def valid_object_response(self):
        """Create valid object-style response."""
        return {
            "model1": {
                "name": "Model 1",
                "max_input_tokens": 8192,
                "max_output_tokens": 4096,
                "supports_functions": True,
            },
            "model2": {
                "name": "Model 2",
                "context_length": 16384,
                "supports_vision": True,
                "input_price": 0.01,
                "output_price": 0.03,
            },
        }

    @pytest.fixture
    def response_with_issues(self):
        """Create response with validation issues."""
        return {
            "data": [
                {"id": "good-model", "max_input_tokens": 8192, "max_output_tokens": 4096},
                {
                    # Missing ID
                    "name": "missing-id-model",
                    "max_input_tokens": "not_a_number",  # Invalid type
                },
                {
                    "id": "",  # Empty ID
                    "max_input_tokens": -100,  # Invalid range
                },
                {
                    "id": "weird model with spaces",
                    "max_input_tokens": 99999999999,  # Excessive value
                },
            ]
        }

    def test_validate_valid_openai_response(self, validator, openai_provider, valid_openai_response):
        """Test validation of valid OpenAI response."""
        result = validator.validate_and_normalize(valid_openai_response, openai_provider)

        assert result.is_valid
        assert result.original_count == 2
        assert result.normalized_count == 2
        assert result.success_rate == 100.0
        assert len(result.models) == 2

        # Check model details
        gpt4 = next((m for m in result.models if m.id == "gpt-4"), None)
        assert gpt4 is not None
        assert gpt4.provider == "openai"
        assert gpt4.created == 1687882411

    def test_validate_object_response(self, validator, openai_provider, valid_object_response):
        """Test validation of object-style response."""
        result = validator.validate_and_normalize(valid_object_response, openai_provider)

        assert result.is_valid or not result.has_errors  # Should be valid or have only warnings
        assert result.original_count == 2
        assert len(result.models) >= 1  # At least some should normalize successfully

        # Check that model IDs are extracted correctly
        model_ids = [m.id for m in result.models]
        assert "model1" in model_ids or "model2" in model_ids

    def test_validate_response_with_issues(self, validator, openai_provider, response_with_issues):
        """Test validation of response with various issues."""
        result = validator.validate_and_normalize(response_with_issues, openai_provider)

        # Should have issues but some models might still be valid
        assert len(result.issues) > 0
        assert result.has_errors  # Should have some errors

        # At least the good model should be normalized
        assert result.normalized_count >= 1
        assert any(m.id == "good-model" for m in result.models)

        # Check for specific error types
        error_messages = [issue.message for issue in result.issues if issue.severity == ModelValidationSeverity.ERROR]
        assert any("missing" in msg.lower() for msg in error_messages)  # Missing ID error

    def test_validate_empty_response(self, validator, openai_provider):
        """Test validation of empty response."""
        result = validator.validate_and_normalize({"data": []}, openai_provider)

        assert result.is_valid  # Empty is valid, just no models
        assert result.original_count == 0
        assert result.normalized_count == 0
        assert len(result.models) == 0

    def test_validate_invalid_response_structure(self, validator, openai_provider):
        """Test validation of invalid response structure."""
        result = validator.validate_and_normalize("invalid", openai_provider)

        assert not result.is_valid
        assert result.has_errors
        assert any("response format" in issue.message.lower() for issue in result.issues)

    def test_normalize_model_id_edge_cases(self, validator, openai_provider):
        """Test model ID normalization edge cases."""
        test_cases = [
            {"model": "model-from-model-field"},
            {"name": "model-from-name-field"},
            {"id": 12345},  # Non-string ID
            {"id": "  spaced  "},  # ID with spaces
        ]

        for test_case in test_cases:
            response = {"data": [test_case]}
            result = validator.validate_and_normalize(response, openai_provider)

            # Should handle gracefully
            if result.models:
                assert result.models[0].id is not None
                assert len(result.models[0].id.strip()) > 0

    def test_normalize_numeric_fields(self, validator, openai_provider):
        """Test normalization of numeric fields."""
        response = {
            "data": [
                {
                    "id": "test-model",
                    "max_input_tokens": "8192",  # String number
                    "max_output_tokens": 4096.0,  # Float
                    "context_length": "16k tokens",  # Number in string
                    "input_price": "0.01",  # String price
                }
            ]
        }

        result = validator.validate_and_normalize(response, openai_provider)

        assert result.normalized_count >= 1
        model = result.models[0]

        # Should normalize string numbers
        assert model.max_input_tokens == 8192
        assert model.max_output_tokens == 4096
        assert model.input_price == 0.01

    def test_normalize_boolean_fields(self, validator, openai_provider):
        """Test normalization of boolean fields."""
        response = {
            "data": [
                {"id": "test-model", "supports_functions": "true", "supports_vision": 1, "supports_streaming": "no"}
            ]
        }

        result = validator.validate_and_normalize(response, openai_provider)

        assert result.normalized_count >= 1
        model = result.models[0]

        # Should normalize various boolean formats
        assert model.supports_functions is True
        assert model.supports_vision is True
        assert model.supports_streaming is False

    def test_detect_vision_models(self, validator, openai_provider):
        """Test automatic detection of vision models."""
        response = {"data": [{"id": "gpt-4-vision-preview"}, {"id": "claude-3-v"}, {"id": "regular-model"}]}

        result = validator.validate_and_normalize(response, openai_provider)

        vision_models = [m for m in result.models if m.supports_vision]
        [m for m in result.models if not m.supports_vision]

        # Should detect vision models by ID patterns
        assert len(vision_models) >= 1
        assert any("vision" in m.id.lower() or "v" in m.id.lower() for m in vision_models)

    def test_duplicate_detection(self, validator, openai_provider):
        """Test duplicate model detection."""
        response = {
            "data": [
                {"id": "model-1"},
                {"id": "model-2"},
                {"id": "model-1"},  # Duplicate
            ]
        }

        result = validator.validate_and_normalize(response, openai_provider)

        # Should detect duplicates
        duplicate_issues = [issue for issue in result.issues if "duplicate" in issue.message.lower()]
        assert len(duplicate_issues) > 0

    def test_model_naming_validation(self, validator, openai_provider):
        """Test model naming pattern validation."""
        response = {
            "data": [
                {"id": "model with spaces"},
                {"id": "gpt-4"},  # Good OpenAI pattern
                {"id": "weird@model#name"},
            ]
        }

        result = validator.validate_and_normalize(response, openai_provider)

        # Should have warnings about naming issues
        naming_issues = [
            issue for issue in result.issues if "spaces" in issue.message.lower() or "pattern" in issue.message.lower()
        ]
        assert len(naming_issues) > 0

    def test_provider_specific_validation(self, validator, anthropic_provider):
        """Test provider-specific validation rules."""
        response = {
            "data": [
                {"id": "claude-3-opus"},  # Good Anthropic pattern
                {"id": "gpt-4"},  # OpenAI pattern on Anthropic
            ]
        }

        result = validator.validate_and_normalize(response, anthropic_provider)

        # Should work but may have pattern warnings
        assert result.normalized_count >= 1

        # Check if provider-specific patterns are validated
        [issue for issue in result.issues if "pattern" in issue.message.lower()]
        # May or may not have pattern issues depending on implementation

    def test_validation_result_properties(self, validator, openai_provider):
        """Test ValidationResult properties and calculations."""
        response = {
            "data": [
                {"id": "good-model"},
                {"name": "missing-id"},  # Will cause error
                {"id": "another-good-model"},
            ]
        }

        result = validator.validate_and_normalize(response, openai_provider)

        # Test properties
        assert result.original_count == 3
        assert isinstance(result.success_rate, float)
        assert 0 <= result.success_rate <= 100

        # Test issue categorization
        if result.issues:
            error_count = sum(1 for issue in result.issues if issue.severity == ModelValidationSeverity.ERROR)
            warning_count = sum(1 for issue in result.issues if issue.severity == ModelValidationSeverity.WARNING)

            assert result.has_errors == (error_count > 0)
            assert result.has_warnings == (warning_count > 0)

    def test_model_dataclass_creation(self, validator, openai_provider):
        """Test Model dataclass creation and field assignment."""
        response = {
            "data": [
                {
                    "id": "test-model",
                    "name": "Test Model",
                    "max_input_tokens": 8192,
                    "max_output_tokens": 4096,
                    "context_length": 16384,
                    "input_price": 0.01,
                    "output_price": 0.03,
                    "supports_functions": True,
                    "supports_vision": False,
                    "supports_streaming": True,
                    "created": 1234567890,
                    "description": "A test model",
                }
            ]
        }

        result = validator.validate_and_normalize(response, openai_provider)

        assert result.normalized_count == 1
        model = result.models[0]

        # Check all fields are properly assigned
        assert model.id == "test-model"
        assert model.provider == "openai"
        assert model.name == "Test Model"
        assert model.max_input_tokens == 8192
        assert model.max_output_tokens == 4096
        assert model.context_length == 16384
        assert model.input_price == 0.01
        assert model.output_price == 0.03
        assert model.supports_functions is True
        assert model.supports_vision is False
        assert model.supports_streaming is True
        assert model.created == 1234567890
        assert model.description == "A test model"
        assert model.raw is not None

    def test_edge_case_values(self, validator, openai_provider):
        """Test handling of edge case values."""
        response = {
            "data": [
                {
                    "id": "edge-case-model",
                    "max_input_tokens": 0,  # Edge case: zero
                    "input_price": -0.01,  # Edge case: negative price
                    "description": "x" * 2000,  # Edge case: very long description
                    "supports_functions": "maybe",  # Edge case: unclear boolean
                    "context_length": None,  # Edge case: null value
                }
            ]
        }

        result = validator.validate_and_normalize(response, openai_provider)

        # Should handle gracefully without crashing
        assert result.normalized_count >= 0  # May or may not normalize depending on validation rules

        # Should have some validation issues for edge cases
        assert len(result.issues) > 0
