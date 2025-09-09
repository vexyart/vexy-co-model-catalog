"""
this_file: src/vexy_co_model_catalog/core/model_validator.py

Model metadata validation and normalization across provider responses.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger

from vexy_co_model_catalog.core.provider import Model, ProviderConfig, ProviderKind


class ModelValidationSeverity(Enum):
    """Model validation issue severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ModelValidationIssue:
    """Individual model validation issue."""

    severity: ModelValidationSeverity
    message: str
    model_id: str | None = None
    field_name: str | None = None
    suggestion: str | None = None
    original_value: Any | None = None
    normalized_value: Any | None = None


@dataclass
class ModelValidationResult:
    """Result of model validation and normalization."""

    is_valid: bool
    original_count: int
    normalized_count: int
    models: list[Model]
    issues: list[ModelValidationIssue]

    @property
    def has_errors(self) -> bool:
        """Check if validation result has any errors."""
        return any(issue.severity == ModelValidationSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if validation result has any warnings."""
        return any(issue.severity == ModelValidationSeverity.WARNING for issue in self.issues)

    @property
    def success_rate(self) -> float:
        """Calculate model normalization success rate."""
        if self.original_count == 0:
            return 0.0
        return (self.normalized_count / self.original_count) * 100.0


class ModelDataValidator:
    """Validates and normalizes model metadata across provider responses."""

    def __init__(self) -> None:
        """Initialize model validator with provider-specific rules."""
        self.provider_schemas = self._get_provider_schemas()
        self.common_model_patterns = self._get_common_model_patterns()

    def validate_and_normalize(self, response_data: Any, provider_config: ProviderConfig) -> ModelValidationResult:
        """
        Validate and normalize model data from provider response.

        Args:
            response_data: Raw response data from provider
            provider_config: Provider configuration

        Returns:
            ModelValidationResult with normalized models and issues
        """
        issues = []
        normalized_models = []

        # Extract raw model data
        raw_models = self._extract_raw_models(response_data, provider_config, issues)
        original_count = len(raw_models)

        # Validate and normalize each model
        for raw_model in raw_models:
            try:
                normalized_model = self._normalize_single_model(raw_model, provider_config, issues)
                if normalized_model:
                    normalized_models.append(normalized_model)
            except Exception as e:
                model_id = self._safe_get_model_id(raw_model)
                issues.append(
                    ModelValidationIssue(
                        severity=ModelValidationSeverity.ERROR,
                        message=f"Failed to normalize model: {e}",
                        model_id=model_id,
                    )
                )
                logger.warning(f"Failed to normalize model {model_id} for {provider_config.name}: {e}")

        # Validate overall response structure
        self._validate_response_structure(response_data, provider_config, issues)

        # Check for duplicate models
        self._check_for_duplicates(normalized_models, issues)

        # Validate model naming patterns
        self._validate_model_naming(normalized_models, provider_config, issues)

        has_errors = any(issue.severity == ModelValidationSeverity.ERROR for issue in issues)

        return ModelValidationResult(
            is_valid=not has_errors,
            original_count=original_count,
            normalized_count=len(normalized_models),
            models=normalized_models,
            issues=issues,
        )

    def _extract_raw_models(
        self, response_data: Any, provider_config: ProviderConfig, issues: list[ModelValidationIssue]
    ) -> list[dict[str, Any]]:
        """Extract raw model data from provider response."""
        if isinstance(response_data, dict):
            if "data" in response_data and isinstance(response_data["data"], list):
                # OpenAI-style format
                return response_data["data"]
            if "models" in response_data and isinstance(response_data["models"], list):
                # Alternative models array format
                return response_data["models"]
            if isinstance(response_data, dict) and len(response_data) > 0:
                # Object format where keys are model IDs
                return [
                    {"id": key, **value} if isinstance(value, dict) else {"id": key, "raw": value}
                    for key, value in response_data.items()
                    if key != "sample_spec"  # Skip chutes sample_spec
                ]
        elif isinstance(response_data, list):
            # Direct array format
            return response_data

        issues.append(
            ModelValidationIssue(
                severity=ModelValidationSeverity.ERROR,
                message=f"Unexpected response format for {provider_config.name}: {type(response_data)}",
            )
        )
        return []

    def _normalize_single_model(
        self, raw_model: dict[str, Any], provider_config: ProviderConfig, issues: list[ModelValidationIssue]
    ) -> Model | None:
        """Normalize a single model from raw data."""
        if not isinstance(raw_model, dict):
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.ERROR,
                    message="Model data must be a dictionary",
                    original_value=raw_model,
                )
            )
            return None

        # Extract and validate required fields
        model_id = self._normalize_model_id(raw_model, issues)
        if not model_id:
            return None

        # Extract optional fields with normalization
        name = self._normalize_model_name(raw_model, model_id, issues)
        context_length = self._normalize_context_length(raw_model, model_id, issues)
        max_input_tokens = self._normalize_max_input_tokens(raw_model, model_id, issues)
        max_output_tokens = self._normalize_max_output_tokens(raw_model, model_id, issues)
        max_tokens = self._normalize_max_tokens(raw_model, model_id, issues)
        input_price = self._normalize_input_price(raw_model, model_id, issues)
        output_price = self._normalize_output_price(raw_model, model_id, issues)
        supports_functions = self._normalize_supports_functions(raw_model, model_id, issues)
        supports_vision = self._normalize_supports_vision(raw_model, model_id, issues)
        supports_streaming = self._normalize_supports_streaming(raw_model, model_id, issues)
        created = self._normalize_created_timestamp(raw_model, model_id, issues)
        description = self._normalize_description(raw_model, model_id, issues)

        return Model(
            id=model_id,
            provider=provider_config.name,
            name=name,
            context_length=context_length,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_tokens=max_tokens,
            input_price=input_price,
            output_price=output_price,
            supports_functions=supports_functions,
            supports_vision=supports_vision,
            supports_streaming=supports_streaming,
            created=created,
            description=description,
            raw=raw_model,
        )

    def _normalize_model_id(self, raw_model: dict[str, Any], issues: list[ModelValidationIssue]) -> str | None:
        """Normalize and validate model ID."""
        model_id = raw_model.get("id") or raw_model.get("model") or raw_model.get("name")

        if not model_id:
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.ERROR, message="Model ID is missing", field_name="id"
                )
            )
            return None

        if not isinstance(model_id, str):
            original_id = model_id
            model_id = str(model_id)
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.WARNING,
                    message="Model ID converted to string",
                    field_name="id",
                    original_value=original_id,
                    normalized_value=model_id,
                )
            )

        # Validate ID format
        if len(model_id.strip()) == 0:
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.ERROR,
                    message="Model ID is empty",
                    field_name="id",
                    model_id=model_id,
                )
            )
            return None

        # Check for suspicious patterns
        if len(model_id) > 200:
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.WARNING,
                    message=f"Model ID is unusually long ({len(model_id)} characters)",
                    field_name="id",
                    model_id=model_id,
                )
            )

        return model_id.strip()

    def _normalize_model_name(
        self, raw_model: dict[str, Any], _model_id: str, _issues: list[ModelValidationIssue]
    ) -> str | None:
        """Normalize model name field."""
        name = raw_model.get("name") or raw_model.get("display_name")
        if name and not isinstance(name, str):
            name = str(name)
        return name.strip() if name else None

    def _normalize_context_length(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> int | None:
        """Normalize context length field."""
        context_length = (
            raw_model.get("context_length") or raw_model.get("context_window") or raw_model.get("max_context_tokens")
        )
        return self._normalize_integer_field(
            context_length, "context_length", model_id, issues, min_value=1, max_value=2000000
        )

    def _normalize_max_input_tokens(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> int | None:
        """Normalize max input tokens field."""
        max_input = (
            raw_model.get("max_input_tokens") or raw_model.get("input_tokens") or raw_model.get("max_prompt_tokens")
        )
        return self._normalize_integer_field(
            max_input, "max_input_tokens", model_id, issues, min_value=1, max_value=2000000
        )

    def _normalize_max_output_tokens(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> int | None:
        """Normalize max output tokens field."""
        max_output = (
            raw_model.get("max_output_tokens")
            or raw_model.get("output_tokens")
            or raw_model.get("max_completion_tokens")
        )
        return self._normalize_integer_field(
            max_output, "max_output_tokens", model_id, issues, min_value=1, max_value=100000
        )

    def _normalize_max_tokens(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> int | None:
        """Normalize legacy max tokens field."""
        max_tokens = raw_model.get("max_tokens")
        return self._normalize_integer_field(max_tokens, "max_tokens", model_id, issues, min_value=1, max_value=2000000)

    def _normalize_input_price(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> float | None:
        """Normalize input price field."""
        price = raw_model.get("input_price") or raw_model.get("price_input") or raw_model.get("input_cost")
        return self._normalize_float_field(price, "input_price", model_id, issues, min_value=0.0)

    def _normalize_output_price(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> float | None:
        """Normalize output price field."""
        price = raw_model.get("output_price") or raw_model.get("price_output") or raw_model.get("output_cost")
        return self._normalize_float_field(price, "output_price", model_id, issues, min_value=0.0)

    def _normalize_supports_functions(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> bool:
        """Normalize function calling support field."""
        supports = raw_model.get("supports_functions") or raw_model.get("function_calling") or raw_model.get("tools")
        return self._normalize_boolean_field(supports, "supports_functions", model_id, issues, default=False)

    def _normalize_supports_vision(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> bool:
        """Normalize vision support field."""
        supports = raw_model.get("supports_vision") or raw_model.get("vision") or raw_model.get("multimodal")
        # Check model ID for vision indicators
        if supports is None and any(keyword in model_id.lower() for keyword in ["vision", "v", "multimodal"]):
            supports = True
        return self._normalize_boolean_field(supports, "supports_vision", model_id, issues, default=False)

    def _normalize_supports_streaming(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> bool:
        """Normalize streaming support field."""
        supports = raw_model.get("supports_streaming") or raw_model.get("streaming")
        return self._normalize_boolean_field(supports, "supports_streaming", model_id, issues, default=True)

    def _normalize_created_timestamp(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> int | None:
        """Normalize created timestamp field."""
        created = raw_model.get("created") or raw_model.get("created_at") or raw_model.get("timestamp")
        return self._normalize_integer_field(created, "created", model_id, issues, min_value=0)

    def _normalize_description(
        self, raw_model: dict[str, Any], model_id: str, issues: list[ModelValidationIssue]
    ) -> str | None:
        """Normalize description field."""
        description = raw_model.get("description") or raw_model.get("summary")
        if description and not isinstance(description, str):
            description = str(description)
        if description and len(description) > 1000:
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.WARNING,
                    message="Description is unusually long",
                    field_name="description",
                    model_id=model_id,
                )
            )
        return description.strip() if description else None

    def _normalize_integer_field(
        self,
        value: Any,
        field_name: str,
        model_id: str,
        issues: list[ModelValidationIssue],
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int | None:
        """Normalize integer field with validation."""
        if value is None:
            return None

        original_value = value

        try:
            if isinstance(value, str):
                # Try to extract number from string
                number_match = re.search(r"(\d+)", value)
                if number_match:
                    value = int(number_match.group(1))
                else:
                    msg = f"No number found in string: {value}"
                    raise ValueError(msg)
            else:
                value = int(value)
        except (ValueError, TypeError):
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.WARNING,
                    message=f"Could not convert {field_name} to integer",
                    field_name=field_name,
                    model_id=model_id,
                    original_value=original_value,
                )
            )
            return None

        # Validate range
        if min_value is not None and value < min_value:
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.WARNING,
                    message=f"{field_name} value {value} is below minimum {min_value}",
                    field_name=field_name,
                    model_id=model_id,
                    original_value=original_value,
                )
            )

        if max_value is not None and value > max_value:
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.WARNING,
                    message=f"{field_name} value {value} exceeds maximum {max_value}",
                    field_name=field_name,
                    model_id=model_id,
                    original_value=original_value,
                )
            )

        if original_value != value:
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.INFO,
                    message=f"{field_name} normalized from {original_value} to {value}",
                    field_name=field_name,
                    model_id=model_id,
                    original_value=original_value,
                    normalized_value=value,
                )
            )

        return value

    def _normalize_float_field(
        self,
        value: Any,
        field_name: str,
        model_id: str,
        issues: list[ModelValidationIssue],
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float | None:
        """Normalize float field with validation."""
        if value is None:
            return None

        original_value = value

        try:
            value = float(value)
        except (ValueError, TypeError):
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.WARNING,
                    message=f"Could not convert {field_name} to float",
                    field_name=field_name,
                    model_id=model_id,
                    original_value=original_value,
                )
            )
            return None

        # Validate range
        if min_value is not None and value < min_value:
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.WARNING,
                    message=f"{field_name} value {value} is below minimum {min_value}",
                    field_name=field_name,
                    model_id=model_id,
                )
            )

        if max_value is not None and value > max_value:
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.WARNING,
                    message=f"{field_name} value {value} exceeds maximum {max_value}",
                    field_name=field_name,
                    model_id=model_id,
                )
            )

        return value

    def _normalize_boolean_field(
        self, value: Any, field_name: str, model_id: str, issues: list[ModelValidationIssue], default: bool = False
    ) -> bool:
        """Normalize boolean field with validation."""
        if value is None:
            return default

        if isinstance(value, bool):
            return value

        # Try to interpret various formats
        if isinstance(value, str):
            value_lower = value.lower().strip()
            if value_lower in ("true", "yes", "1", "on", "enabled"):
                return True
            if value_lower in ("false", "no", "0", "off", "disabled"):
                return False
        elif isinstance(value, int | float):
            return bool(value)

        issues.append(
            ModelValidationIssue(
                severity=ModelValidationSeverity.WARNING,
                message=f"Could not interpret {field_name} value: {value}, using default {default}",
                field_name=field_name,
                model_id=model_id,
                original_value=value,
                normalized_value=default,
            )
        )

        return default

    def _validate_response_structure(
        self, response_data: Any, provider_config: ProviderConfig, issues: list[ModelValidationIssue]
    ) -> None:
        """Validate overall response structure."""
        if not isinstance(response_data, dict | list):
            issues.append(
                ModelValidationIssue(
                    severity=ModelValidationSeverity.ERROR,
                    message=f"Response must be dict or list, got {type(response_data)}",
                )
            )

        # Provider-specific validations
        if provider_config.kind == ProviderKind.OPENAI:
            if isinstance(response_data, dict) and "data" in response_data:
                if not isinstance(response_data["data"], list):
                    issues.append(
                        ModelValidationIssue(
                            severity=ModelValidationSeverity.ERROR,
                            message="OpenAI response 'data' field must be a list",
                        )
                    )

    def _check_for_duplicates(self, models: list[Model], issues: list[ModelValidationIssue]) -> None:
        """Check for duplicate model IDs."""
        seen_ids = set()
        for model in models:
            if model.id in seen_ids:
                issues.append(
                    ModelValidationIssue(
                        severity=ModelValidationSeverity.WARNING,
                        message="Duplicate model ID found",
                        model_id=model.id,
                        suggestion="Remove duplicate entries or ensure model IDs are unique",
                    )
                )
            else:
                seen_ids.add(model.id)

    def _validate_model_naming(
        self, models: list[Model], provider_config: ProviderConfig, issues: list[ModelValidationIssue]
    ) -> None:
        """Validate model naming patterns."""
        for model in models:
            # Check for common naming issues
            if " " in model.id:
                issues.append(
                    ModelValidationIssue(
                        severity=ModelValidationSeverity.WARNING,
                        message="Model ID contains spaces",
                        model_id=model.id,
                        suggestion="Model IDs typically use hyphens or underscores instead of spaces",
                    )
                )

            # Check for provider-specific patterns
            expected_patterns = self.common_model_patterns.get(provider_config.name, [])
            if expected_patterns and not any(pattern in model.id.lower() for pattern in expected_patterns):
                issues.append(
                    ModelValidationIssue(
                        severity=ModelValidationSeverity.INFO,
                        message=f"Model ID doesn't match expected patterns for {provider_config.name}",
                        model_id=model.id,
                        suggestion=f"Expected patterns: {', '.join(expected_patterns)}",
                    )
                )

    def _safe_get_model_id(self, raw_model: Any) -> str:
        """Safely extract model ID for error reporting."""
        if isinstance(raw_model, dict):
            return str(raw_model.get("id", raw_model.get("model", raw_model.get("name", "unknown"))))
        return "unknown"

    def _get_provider_schemas(self) -> dict[str, dict[str, Any]]:
        """Get provider-specific validation schemas."""
        return {
            "openai": {
                "required_fields": ["id"],
                "expected_fields": ["created", "object", "owned_by"],
                "id_patterns": [r"^gpt-", r"^text-", r"^code-", r"^whisper-", r"^dall-e-"],
            },
            "anthropic": {
                "required_fields": ["id"],
                "expected_fields": ["display_name", "created_at"],
                "id_patterns": [r"^claude-"],
            },
            "groq": {
                "required_fields": ["id"],
                "expected_fields": ["context_window", "active"],
                "id_patterns": [r"^llama", r"^mixtral", r"^gemma"],
            },
        }

    def _get_common_model_patterns(self) -> dict[str, list[str]]:
        """Get common model naming patterns by provider."""
        return {
            "openai": ["gpt", "text", "code", "whisper", "dall-e"],
            "anthropic": ["claude"],
            "groq": ["llama", "mixtral", "gemma"],
            "mistral": ["mistral", "mixtral"],
            "togetherai": ["meta", "mistral", "upstage"],
            "fireworks": ["accounts", "models"],
        }
