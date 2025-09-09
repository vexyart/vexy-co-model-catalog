"""
this_file: tests/test_config_validator.py

Test configuration file syntax validation functionality.
"""

import sys
import tempfile
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib as toml_read
else:
    import tomli as toml_read
import tomli_w as toml_write
import yaml

from vexy_co_model_catalog.core.config_validator import (
    ConfigFormat,
    ConfigValidationResult,
    ConfigValidator,
    ValidationIssue,
    ValidationSeverity,
)


class TestConfigValidator:
    """Test suite for configuration validation."""

    @pytest.fixture
    def validator(self):
        """Create a config validator instance."""
        return ConfigValidator()

    @pytest.fixture
    def valid_yaml_config(self):
        """Create valid YAML configuration."""
        return {
            "api_base": "https://api.openai.com/v1",
            "api_key": "${OPENAI_API_KEY}",
            "type": "openai",
            "models": [
                {"name": "gpt-4", "max_input_tokens": 8192},
                {"name": "gpt-3.5-turbo", "max_input_tokens": 4096},
            ],
        }

    @pytest.fixture
    def valid_toml_config(self):
        """Create valid TOML configuration."""
        return {"model": "gpt-4", "max_tokens": 2048, "temperature": 0.7}

    @pytest.fixture
    def invalid_yaml_content(self):
        """Create invalid YAML content."""
        return """
        api_base: https://api.openai.com/v1
        api_key: ${OPENAI_API_KEY}
          invalid_indent: value
        type: openai
        """

    @pytest.fixture
    def invalid_toml_content(self):
        """Create invalid TOML content."""
        return """
        model = "gpt-4
        max_tokens = 2048
        temperature = 0.7
        """

    def test_validate_valid_yaml_content(self, validator, valid_yaml_config):
        """Test validation of valid YAML content."""
        yaml_content = yaml.dump(valid_yaml_config)
        result = validator.validate_content(yaml_content, ConfigFormat.YAML, tool_name="aichat")

        assert result.is_valid
        assert result.format_type == ConfigFormat.YAML
        assert result.parsed_content is not None
        assert len(result.issues) == 0 or not result.has_errors

    def test_validate_valid_toml_content(self, validator, valid_toml_config):
        """Test validation of valid TOML content."""
        toml_content = toml_write.dumps(valid_toml_config)
        result = validator.validate_content(toml_content, ConfigFormat.TOML, tool_name="codex")

        assert result.is_valid
        assert result.format_type == ConfigFormat.TOML
        assert result.parsed_content is not None
        assert len(result.issues) == 0 or not result.has_errors

    def test_validate_invalid_yaml_syntax(self, validator, invalid_yaml_content):
        """Test validation of invalid YAML syntax."""
        result = validator.validate_content(invalid_yaml_content, ConfigFormat.YAML)

        assert not result.is_valid
        assert result.has_errors
        assert any("YAML syntax error" in issue.message for issue in result.issues)

    def test_validate_invalid_toml_syntax(self, validator, invalid_toml_content):
        """Test validation of invalid TOML syntax."""
        result = validator.validate_content(invalid_toml_content, ConfigFormat.TOML)

        assert not result.is_valid
        assert result.has_errors
        assert any("TOML syntax error" in issue.message for issue in result.issues)

    def test_validate_file_not_found(self, validator):
        """Test validation of non-existent file."""
        result = validator.validate_file("non_existent_file.yaml")

        assert not result.is_valid
        assert result.has_errors
        assert any("not found" in issue.message for issue in result.issues)

    def test_validate_file_with_temp_file(self, validator, valid_yaml_config):
        """Test validation with temporary file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_yaml_config, f)
            temp_path = f.name

        try:
            result = validator.validate_file(temp_path)
            assert result.is_valid or not result.has_errors
            assert result.format_type == ConfigFormat.YAML
        finally:
            Path(temp_path).unlink()

    def test_validate_generated_config_yaml(self, validator, valid_yaml_config):
        """Test validation of generated YAML config."""
        result = validator.validate_generated_config(valid_yaml_config, "aichat", ConfigFormat.YAML)

        assert result.is_valid or not result.has_errors
        assert result.format_type == ConfigFormat.YAML

    def test_validate_generated_config_toml(self, validator, valid_toml_config):
        """Test validation of generated TOML config."""
        result = validator.validate_generated_config(valid_toml_config, "codex", ConfigFormat.TOML)

        assert result.is_valid or not result.has_errors
        assert result.format_type == ConfigFormat.TOML

    def test_detect_format_from_path(self, validator):
        """Test format detection from file path."""
        assert validator._detect_format_from_path(Path("config.yaml")) == ConfigFormat.YAML
        assert validator._detect_format_from_path(Path("config.yml")) == ConfigFormat.YAML
        assert validator._detect_format_from_path(Path("config.toml")) == ConfigFormat.TOML
        assert validator._detect_format_from_path(Path("config.json")) == ConfigFormat.JSON
        assert validator._detect_format_from_path(Path("config.txt")) == ConfigFormat.YAML  # Default

    def test_detect_format_from_content(self, validator):
        """Test format detection from content."""
        json_content = '{"key": "value"}'
        yaml_content = "key: value"
        toml_content = 'key = "value"'

        assert validator._detect_format_from_content(json_content) == ConfigFormat.JSON
        assert validator._detect_format_from_content(yaml_content) == ConfigFormat.YAML
        assert validator._detect_format_from_content(toml_content) == ConfigFormat.TOML

    def test_validation_result_properties(self):
        """Test ValidationResult properties."""
        issues = [
            ValidationIssue(ValidationSeverity.ERROR, "Error message"),
            ValidationIssue(ValidationSeverity.WARNING, "Warning message"),
            ValidationIssue(ValidationSeverity.INFO, "Info message"),
        ]

        result = ConfigValidationResult(is_valid=False, format_type=ConfigFormat.YAML, issues=issues)

        assert result.has_errors
        assert result.has_warnings
        assert not result.is_valid

    def test_common_validation_checks(self, validator):
        """Test common validation issues detection."""
        problematic_config = {
            "api_key": "your_key_here",  # Should trigger placeholder warning
            "base_url": "invalid-url",  # Should trigger URL warning
            "very_long_string": "x" * 1500,  # Should trigger length warning
        }

        result = validator.validate_content(yaml.dump(problematic_config), ConfigFormat.YAML)

        # Should have warnings but still be valid YAML
        assert result.is_valid  # Valid syntax
        assert result.has_warnings  # But has warnings

    def test_tool_schema_validation_aichat(self, validator):
        """Test tool-specific schema validation for aichat."""
        config_with_invalid_type = {
            "type": "invalid_type",  # Should be one of allowed values
            "api_base": "https://api.example.com",
            "api_key": "${API_KEY}",
        }

        result = validator.validate_content(yaml.dump(config_with_invalid_type), ConfigFormat.YAML, tool_name="aichat")

        # Should have schema validation errors
        assert any("invalid value" in issue.message.lower() for issue in result.issues)

    def test_tool_schema_validation_codex(self, validator):
        """Test tool-specific schema validation for codex."""
        config_missing_model = {
            "max_tokens": 2048,
            "temperature": 0.7,
            # Missing required 'model' field
        }

        result = validator.validate_content(toml_write.dumps(config_missing_model), ConfigFormat.TOML, tool_name="codex")

        # Should have required field error
        assert any("required field" in issue.message.lower() for issue in result.issues)

    def test_validation_issue_dataclass(self):
        """Test ValidationIssue dataclass."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Test error",
            line_number=5,
            column_number=10,
            field_path="api.key",
            suggestion="Fix the API key",
        )

        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Test error"
        assert issue.line_number == 5
        assert issue.column_number == 10
        assert issue.field_path == "api.key"
        assert issue.suggestion == "Fix the API key"

    def test_nested_value_access(self, validator):
        """Test nested value access utility."""
        config = {"level1": {"level2": {"value": "test"}}}

        assert validator._get_nested_value(config, "level1.level2.value") == "test"
        assert validator._get_nested_value(config, "level1.level2") == {"value": "test"}
        assert validator._get_nested_value(config, "nonexistent") is None
        assert validator._get_nested_value(config, "level1.nonexistent") is None

    def test_type_checking(self, validator):
        """Test type checking utility."""
        assert validator._is_correct_type("string", "string")
        assert validator._is_correct_type(123, "int")
        assert validator._is_correct_type(12.3, "float")
        assert validator._is_correct_type(True, "bool")
        assert validator._is_correct_type([], "list")
        assert validator._is_correct_type({}, "dict")

        assert not validator._is_correct_type("string", "int")
        assert not validator._is_correct_type(123, "string")

    def test_empty_config_validation(self, validator):
        """Test validation of empty configuration."""
        result = validator.validate_content("{}", ConfigFormat.JSON)

        assert result.is_valid  # Valid JSON
        assert result.has_warnings  # But should warn about empty config
        assert any("empty" in issue.message.lower() for issue in result.issues)

    def test_unsupported_format(self, validator):
        """Test handling of unsupported format."""
        # This shouldn't happen in normal usage, but test defensive coding
        content = "some content"

        # Force an invalid format by directly calling _validate_syntax
        syntax_result = validator._validate_syntax(content, None)

        # Should handle gracefully and not crash
        assert not syntax_result.is_valid
