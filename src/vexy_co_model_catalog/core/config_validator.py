"""
this_file: src/vexy_co_model_catalog/core/config_validator.py

Configuration file syntax validation and schema checking.
"""

import errno
import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib as toml_read
else:
    import tomli as toml_read
import tomli_w as toml_write
import yaml

# Constants
MAX_STRING_LENGTH = 1000  # Maximum reasonable length for config strings


class ConfigFormat(Enum):
    """Configuration file format enumeration."""

    YAML = "yaml"
    TOML = "toml"
    JSON = "json"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Individual configuration validation issue."""

    severity: ValidationSeverity
    message: str
    line_number: int | None = None
    column_number: int | None = None
    field_path: str | None = None
    suggestion: str | None = None


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    format_type: ConfigFormat
    file_path: str | None = None
    issues: list[ValidationIssue] = None
    parsed_content: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize issues list if not provided."""
        if self.issues is None:
            self.issues = []

    @property
    def has_errors(self) -> bool:
        """Check if validation result has any errors."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if validation result has any warnings."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)


class ConfigValidator:
    """Validates configuration file syntax and schema."""

    def __init__(self) -> None:
        """Initialize config validator."""
        self.tool_schemas = self._get_tool_schemas()

    def validate_file(self, file_path: str | Path) -> ConfigValidationResult:
        """
        Validate configuration file from filesystem.

        Args:
            file_path: Path to configuration file

        Returns:
            ConfigValidationResult with validation details
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return ConfigValidationResult(
                is_valid=False,
                format_type=self._detect_format_from_path(file_path),
                file_path=str(file_path),
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR, message=f"Configuration file not found: {file_path}"
                    )
                ],
            )

        try:
            content = file_path.read_text(encoding="utf-8")
            return self.validate_content(content, file_path=str(file_path))
        except UnicodeDecodeError as e:
            return ConfigValidationResult(
                is_valid=False,
                format_type=self._detect_format_from_path(file_path),
                file_path=str(file_path),
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Failed to read configuration file {file_path}: File encoding error at byte {e.start}. "
                            f"Expected UTF-8 encoding but found invalid characters. "
                            f"Try: 1) Re-save file as UTF-8, 2) Check for binary content, "
                            f"3) Verify file wasn't corrupted during transfer."
                        ),
                    )
                ],
            )
        except OSError as e:
            error_code = getattr(e, 'errno', 'unknown')
            if error_code == errno.EACCES:
                msg = (
                    f"Failed to read configuration file {file_path}: Permission denied. "
                    f"Try: 1) Check file permissions with 'ls -la {file_path}', "
                    f"2) Run 'chmod 644 {file_path}' to fix permissions, "
                    f"3) Verify you have read access to the parent directory."
                )
            elif error_code == errno.EISDIR:
                msg = (
                    f"Failed to read configuration file {file_path}: Target is a directory, not a file. "
                    f"Try: 1) Check path spelling, 2) Verify you meant to specify a config file, not directory, "
                    f"3) Use 'ls {file_path}' to see directory contents."
                )
            elif error_code == errno.ENAMETOOLONG:
                msg = (
                    f"Failed to read configuration file {file_path}: File path is too long. "
                    f"Try: 1) Move file to shorter path, 2) Use relative path, 3) Create symlink with shorter name."
                )
            else:
                msg = (
                    f"Failed to read configuration file {file_path}: {e}. "
                    f"Try: 1) Verify file exists and is readable, 2) Check file permissions, "
                    f"3) Ensure parent directories are accessible."
                )

            return ConfigValidationResult(
                is_valid=False,
                format_type=self._detect_format_from_path(file_path),
                file_path=str(file_path),
                issues=[ValidationIssue(severity=ValidationSeverity.ERROR, message=msg)],
            )
        except Exception as e:
            return ConfigValidationResult(
                is_valid=False,
                format_type=self._detect_format_from_path(file_path),
                file_path=str(file_path),
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Unexpected error reading configuration file {file_path}: {e}. "
                            f"Try: 1) Verify file integrity, 2) Check available disk space, "
                            f"3) Restart and try again."
                        ),
                    )
                ],
            )

    def validate_content(
        self,
        content: str,
        format_type: ConfigFormat | None = None,
        file_path: str | None = None,
        tool_name: str | None = None,
    ) -> ConfigValidationResult:
        """
        Validate configuration content string.

        Args:
            content: Configuration file content
            format_type: Override format detection
            file_path: Optional file path for context
            tool_name: Tool name for schema validation (aichat, codex, mods)

        Returns:
            ConfigValidationResult with validation details
        """
        # Detect format if not provided
        if format_type is None:
            if file_path:
                format_type = self._detect_format_from_path(Path(file_path))
            else:
                format_type = self._detect_format_from_content(content)

        # Validate basic syntax
        syntax_result = self._validate_syntax(content, format_type)

        if not syntax_result.is_valid:
            return ConfigValidationResult(
                is_valid=False,
                format_type=format_type,
                file_path=file_path,
                issues=syntax_result.issues,
                parsed_content=None,
            )

        # Perform schema validation if tool name is provided
        schema_issues = []
        if tool_name and syntax_result.parsed_content:
            schema_issues = self._validate_tool_schema(syntax_result.parsed_content, tool_name, format_type)

        # Perform common validation checks
        common_issues = self._validate_common_issues(syntax_result.parsed_content, format_type)

        all_issues = syntax_result.issues + schema_issues + common_issues
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in all_issues)

        return ConfigValidationResult(
            is_valid=not has_errors,
            format_type=format_type,
            file_path=file_path,
            issues=all_issues,
            parsed_content=syntax_result.parsed_content,
        )

    def validate_generated_config(
        self, config_data: dict[str, Any], tool_name: str, format_type: ConfigFormat
    ) -> ConfigValidationResult:
        """
        Validate programmatically generated configuration.

        Args:
            config_data: Configuration data dictionary
            tool_name: Tool name (aichat, codex, mods)
            format_type: Target configuration format

        Returns:
            ConfigValidationResult with validation details
        """
        # Convert to string representation for full validation
        try:
            if format_type == ConfigFormat.YAML:
                content = yaml.dump(config_data, default_flow_style=False, allow_unicode=True)
            elif format_type == ConfigFormat.TOML:
                content = toml_write.dumps(config_data)
            else:
                content = json.dumps(config_data, indent=2)
        except TypeError as e:
            data_type = type(config_data).__name__
            return ConfigValidationResult(
                is_valid=False,
                format_type=format_type,
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Failed to serialize {data_type} config data to {format_type.value.upper()}: {e}. "
                            f"Data contains non-serializable types. "
                            f"Try: 1) Check for function objects or complex types in config, "
                            f"2) Convert custom objects to dictionaries, 3) Remove non-JSON-compatible values."
                        ),
                    )
                ],
            )
        except ValueError as e:
            return ConfigValidationResult(
                is_valid=False,
                format_type=format_type,
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Failed to serialize config data to {format_type.value.upper()}: Invalid data structure. {e}. "
                            f"Try: 1) Ensure all keys are strings for TOML format, "
                            f"2) Check for circular references, 3) Verify data structure compatibility."
                        ),
                    )
                ],
            )
        except Exception as e:
            problematic_keys = []
            if isinstance(config_data, dict):
                for key, value in config_data.items():
                    if not isinstance(value, str | int | float | bool | list | dict | type(None)):
                        problematic_keys.append(f"{key} ({type(value).__name__})")

            key_info = f" Problematic keys: {', '.join(problematic_keys[:3])}" if problematic_keys else ""

            return ConfigValidationResult(
                is_valid=False,
                format_type=format_type,
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=(
                            f"Failed to serialize config data to {format_type.value.upper()}: {e}.{key_info} "
                            f"Try: 1) Verify data types are compatible with {format_type.value} format, "
                            f"2) Check for special characters in keys, 3) Ensure nested structures are properly formatted."
                        ),
                    )
                ],
            )

        # Validate the serialized content
        return self.validate_content(content=content, format_type=format_type, tool_name=tool_name)

    def _validate_syntax(self, content: str, format_type: ConfigFormat) -> ConfigValidationResult:
        """Validate basic syntax for the given format."""
        issues = []
        parsed_content = None

        # Data-driven format configuration
        format_configs = {
            ConfigFormat.YAML: {
                "parser": yaml.safe_load,
                "exceptions": (yaml.YAMLError,),
                "error_handler": self._handle_yaml_error,
            },
            ConfigFormat.TOML: {
                "parser": toml_read.loads,
                "exceptions": (OSError, ValueError),
                "error_handler": self._handle_toml_error,
            },
            ConfigFormat.JSON: {
                "parser": self._parse_json,
                "exceptions": (json.JSONDecodeError, ValueError),
                "error_handler": self._handle_json_error,
            },
        }

        config = format_configs.get(format_type)
        if not config:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Unsupported configuration format: {format_type.value}",
                )
            )
            return ConfigValidationResult(is_valid=False, format_type=format_type, issues=issues)

        try:
            parsed_content = config["parser"](content)
        except config["exceptions"] as e:
            issues.append(config["error_handler"](e))
        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR, message=f"{format_type.value.upper()} parsing error: {e}"
                )
            )

        return ConfigValidationResult(
            is_valid=len(issues) == 0, format_type=format_type, issues=issues, parsed_content=parsed_content
        )

    def _parse_json(self, content: str) -> Any:
        """Parse JSON content."""
        return json.loads(content)

    def _handle_yaml_error(self, error: yaml.YAMLError) -> ValidationIssue:
        """Handle YAML parsing errors with detailed recovery guidance."""
        line_num = getattr(error, "problem_mark", None)
        error_str = str(error).lower()

        # Provide specific suggestions based on error type
        if "found character" in error_str and "that cannot start" in error_str:
            suggestion = (
                "YAML syntax error - invalid character. "
                "Try: 1) Check for tabs (use spaces for indentation), "
                "2) Ensure colons have spaces after them, 3) Quote strings with special characters."
            )
        elif "mapping values are not allowed here" in error_str:
            suggestion = (
                "YAML indentation error. "
                "Try: 1) Check indentation consistency (use 2 or 4 spaces), "
                "2) Ensure child items are indented further than parent, 3) Verify no tabs are used."
            )
        elif "expected" in error_str and "but found" in error_str:
            suggestion = (
                "YAML structure error. "
                "Try: 1) Check for missing quotes around strings, "
                "2) Verify bracket/brace matching, 3) Ensure proper key:value format."
            )
        else:
            suggestion = (
                "YAML syntax error. "
                "Try: 1) Validate at yamllint.com, 2) Check indentation (use spaces, not tabs), "
                "3) Quote strings with special characters, 4) Ensure proper key:value format."
            )

        return ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message=f"YAML syntax error: {error}",
            line_number=line_num.line + 1 if line_num else None,
            column_number=line_num.column + 1 if line_num else None,
            suggestion=suggestion,
        )

    def _handle_toml_error(self, error: Exception) -> ValidationIssue:
        """Handle TOML parsing errors with detailed recovery guidance."""
        error_str = str(error).lower()

        # Provide specific suggestions based on error type
        if "unterminated string" in error_str:
            suggestion = (
                "TOML unterminated string error. "
                "Try: 1) Check for missing closing quotes, 2) Escape quotes inside strings with \\, "
                "3) Use triple quotes for multi-line strings."
            )
        elif "invalid character" in error_str:
            suggestion = (
                "TOML invalid character error. "
                "Try: 1) Check for unsupported characters, 2) Quote string values properly, "
                "3) Escape special characters with backslash."
            )
        elif "expected" in error_str and "found" in error_str:
            suggestion = (
                "TOML structure error. "
                "Try: 1) Check section headers use [brackets], 2) Ensure key = value format, "
                "3) Verify array syntax uses [square brackets]."
            )
        elif "duplicate key" in error_str:
            suggestion = (
                "TOML duplicate key error. "
                "Try: 1) Remove or rename duplicate keys, 2) Use different section names, "
                "3) Check for case-sensitive key conflicts."
            )
        else:
            suggestion = (
                "TOML syntax error. "
                "Try: 1) Validate at toml-lint online, 2) Check key=value format, "
                "3) Ensure section headers use [brackets], 4) Quote string values properly."
            )

        return ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message=f"TOML syntax error: {error}",
            suggestion=suggestion,
        )

    def _handle_json_error(self, error: Exception) -> ValidationIssue:
        """Handle JSON parsing errors with detailed recovery guidance."""
        error_str = str(error).lower()
        line_col_info = ""

        # Extract line/column information if available (for JSONDecodeError)
        if hasattr(error, 'lineno') and hasattr(error, 'colno'):
            line_col_info = f" at line {error.lineno}, column {error.colno}"

        # Provide specific suggestions based on error type
        if "expecting property name" in error_str:
            suggestion = (
                "JSON expects property name. "
                "Try: 1) Ensure object keys are quoted with double quotes, "
                "2) Remove trailing commas after last property, 3) Check for missing commas between properties."
            )
        elif "expecting value" in error_str:
            suggestion = (
                "JSON expects a value. "
                "Try: 1) Check for trailing commas in arrays/objects, "
                "2) Ensure all values are properly formatted, 3) Remove empty values or add null."
            )
        elif "expecting ',' delimiter" in error_str:
            suggestion = (
                "JSON missing comma delimiter. "
                "Try: 1) Add commas between array elements, 2) Add commas between object properties, "
                "3) Check for extra commas at the end."
            )
        elif "unterminated string" in error_str:
            suggestion = (
                "JSON unterminated string error. "
                "Try: 1) Check for missing closing quotes, 2) Escape quotes inside strings with \\, "
                "3) Escape backslashes with \\\\."
            )
        elif "invalid \\escape" in error_str:
            suggestion = (
                "JSON invalid escape sequence. "
                "Try: 1) Use valid escapes: \\n, \\t, \\r, \\\\, \\\", "
                "2) Escape literal backslashes as \\\\, 3) Use Unicode escapes \\uXXXX for special characters."
            )
        else:
            suggestion = (
                "JSON syntax error. "
                "Try: 1) Validate at jsonlint.com, 2) Use double quotes for strings, "
                "3) Check bracket/brace matching, 4) Remove trailing commas."
            )

        return ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message=f"JSON syntax error{line_col_info}: {error}",
            line_number=getattr(error, 'lineno', None),
            column_number=getattr(error, 'colno', None),
            suggestion=suggestion,
        )

    def _validate_tool_schema(
        self, config_data: dict[str, Any], tool_name: str, _format_type: ConfigFormat
    ) -> list[ValidationIssue]:
        """Validate configuration against tool-specific schema."""
        issues = []
        schema = self.tool_schemas.get(tool_name)

        if not schema:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING, message=f"No schema validation available for tool: {tool_name}"
                )
            )
            return issues

        # Basic schema validation
        issues.extend(self._check_required_fields(config_data, schema.get("required", []), tool_name))
        issues.extend(self._check_field_types(config_data, schema.get("types", {}), tool_name))
        issues.extend(self._check_allowed_values(config_data, schema.get("allowed_values", {}), tool_name))

        return issues

    def _validate_common_issues(self, config_data: dict[str, Any], _format_type: ConfigFormat) -> list[ValidationIssue]:
        """Check for common configuration issues."""
        issues = []

        if not isinstance(config_data, dict):
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR, message="Configuration must be a dictionary/object at top level"
                )
            )
            return issues

        # Check for empty configuration
        if not config_data:
            issues.append(ValidationIssue(severity=ValidationSeverity.WARNING, message="Configuration is empty"))

        # Check for suspiciously long strings (potential copy-paste errors)
        self._check_string_lengths(config_data, issues)

        # Check for common API key patterns
        self._check_api_key_patterns(config_data, issues)

        # Check for valid URLs
        self._check_url_patterns(config_data, issues)

        return issues

    def _check_required_fields(
        self, config_data: dict[str, Any], required_fields: list[str], tool_name: str
    ) -> list[ValidationIssue]:
        """Check for required configuration fields."""
        issues = []

        for field_path in required_fields:
            if not self._get_nested_value(config_data, field_path):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Required field missing: {field_path}",
                        field_path=field_path,
                        suggestion=f"Add required field for {tool_name} configuration",
                    )
                )

        return issues

    def _check_field_types(
        self, config_data: dict[str, Any], type_specs: dict[str, str], _tool_name: str
    ) -> list[ValidationIssue]:
        """Check field types against expected types."""
        issues = []

        for field_path, expected_type in type_specs.items():
            value = self._get_nested_value(config_data, field_path)
            if value is not None and not self._is_correct_type(value, expected_type):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Field {field_path} should be {expected_type}, got {type(value).__name__}",
                        field_path=field_path,
                        suggestion=f"Ensure {field_path} is a valid {expected_type}",
                    )
                )

        return issues

    def _check_allowed_values(
        self, config_data: dict[str, Any], allowed_specs: dict[str, list[str]], _tool_name: str
    ) -> list[ValidationIssue]:
        """Check field values against allowed values."""
        issues = []

        for field_path, allowed_values in allowed_specs.items():
            value = self._get_nested_value(config_data, field_path)
            if value is not None and value not in allowed_values:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Field {field_path} has invalid value '{value}'. Allowed: {allowed_values}",
                        field_path=field_path,
                        suggestion=f"Use one of the allowed values: {', '.join(allowed_values)}",
                    )
                )

        return issues

    def _check_string_lengths(self, config_data: dict[str, Any], issues: list[ValidationIssue]) -> None:
        """Check for suspiciously long strings."""

        def check_recursive(data: Any, path: str = "") -> None:
            if isinstance(data, dict):
                for key, value in data.items():
                    check_recursive(value, f"{path}.{key}" if path else key)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    check_recursive(item, f"{path}[{i}]")
            elif isinstance(data, str) and len(data) > MAX_STRING_LENGTH:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Unusually long string in {path} ({len(data)} characters)",
                        field_path=path,
                        suggestion="Verify this string is correct and not a copy-paste error",
                    )
                )

        check_recursive(config_data)

    def _check_api_key_patterns(self, config_data: dict[str, Any], issues: list[ValidationIssue]) -> None:
        """Check for potentially invalid API key patterns."""

        def check_recursive(data: Any, path: str = "") -> None:
            if isinstance(data, dict):
                for key, value in data.items():
                    check_recursive(value, f"{path}.{key}" if path else key)
            elif isinstance(data, str):
                # Check for common API key field names
                if any(keyword in path.lower() for keyword in ["api_key", "token", "key"]):
                    if not value.startswith("${") and not value.startswith("$"):
                        # Check for placeholder patterns
                        if "your_key" in value.lower() or "replace" in value.lower():
                            issues.append(
                                ValidationIssue(
                                    severity=ValidationSeverity.WARNING,
                                    message=f"API key field {path} appears to contain placeholder text",
                                    field_path=path,
                                    suggestion="Replace with actual API key or environment variable reference",
                                )
                            )

        check_recursive(config_data)

    def _check_url_patterns(self, config_data: dict[str, Any], issues: list[ValidationIssue]) -> None:
        """Check for potentially invalid URL patterns."""
        url_pattern = re.compile(r"^https?://", re.IGNORECASE)

        def check_recursive(data: Any, path: str = "") -> None:
            if isinstance(data, dict):
                for key, value in data.items():
                    check_recursive(value, f"{path}.{key}" if path else key)
            elif isinstance(data, str):
                # Check for URL-like field names
                field_name = path.split(".")[-1] if path else ""
                if any(keyword in field_name.lower() for keyword in ["url", "endpoint", "base"]):
                    if data and not url_pattern.match(data) and not data.startswith("${"):
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"URL field {path} may not be a valid URL: {data}",
                                field_path=path,
                                suggestion="Ensure URL starts with http:// or https://",
                            )
                        )

        check_recursive(config_data)

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _is_correct_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type string."""
        type_mapping = {"string": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict}

        expected_python_type = type_mapping.get(expected_type.lower())
        if expected_python_type:
            return isinstance(value, expected_python_type)

        return True  # Unknown type, allow it

    def _detect_format_from_path(self, file_path: Path) -> ConfigFormat:
        """Detect configuration format from file extension."""
        suffix = file_path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            return ConfigFormat.YAML
        if suffix == ".toml":
            return ConfigFormat.TOML
        if suffix == ".json":
            return ConfigFormat.JSON
        return ConfigFormat.YAML  # Default assumption

    def _detect_format_from_content(self, content: str) -> ConfigFormat:
        """Attempt to detect format from content analysis."""
        content = content.strip()

        # Simple heuristics
        if content.startswith("{") or content.startswith("["):
            return ConfigFormat.JSON
        if "[" in content and "=" in content:
            return ConfigFormat.TOML
        return ConfigFormat.YAML

    def _get_tool_schemas(self) -> dict[str, dict[str, Any]]:
        """Get validation schemas for supported tools."""
        return {
            "aichat": {
                "required": [],
                "types": {"api_base": "string", "api_key": "string", "models": "list"},
                "allowed_values": {"type": ["openai", "claude", "gemini", "localai"]},
            },
            "codex": {
                "required": ["model"],
                "types": {"model": "string", "max_tokens": "int", "temperature": "float"},
                "allowed_values": {},
            },
            "mods": {"required": [], "types": {"base_url": "string", "api_key": "string"}, "allowed_values": {}},
        }
