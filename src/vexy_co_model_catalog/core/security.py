# this_file: src/vexy_co_model_catalog/core/security.py

"""
Security utilities for protecting sensitive data in logs, CLI output, and error messages.
Provides comprehensive API key masking, header sanitization, and sensitive data redaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Constants for token masking
MIN_TOKEN_LENGTH_FOR_PARTIAL_DISPLAY = 8
MIN_TOKEN_LENGTH_FOR_MEDIUM_DISPLAY = 16
SHORT_TOKEN_DISPLAY_CHARS = 2
LONG_TOKEN_DISPLAY_CHARS = 4
MIN_SENSITIVE_FIELD_LENGTH_FOR_PARTIAL_DISPLAY = 4


class SensitivityLevel(Enum):
    """Sensitivity levels for different types of data."""

    PUBLIC = "public"  # No masking needed
    INTERNAL = "internal"  # Light masking for internal logs
    CONFIDENTIAL = "confidential"  # Heavy masking for external output
    SECRET = "secret"  # Complete redaction


@dataclass
class MaskingRule:
    """Defines how to mask sensitive data."""

    pattern: str
    replacement: str = "***[REDACTED]"
    description: str | None = None
    sensitivity: SensitivityLevel = SensitivityLevel.CONFIDENTIAL

    def apply(self, text: str) -> str:
        """Apply this masking rule to text."""
        return re.sub(self.pattern, self.replacement, text, flags=re.IGNORECASE)


class SensitiveDataProtector:
    """Comprehensive protection for sensitive data in logs, CLI output, and API responses."""

    def __init__(self) -> None:
        """Initialize with predefined masking rules for common sensitive patterns."""
        self.masking_rules = self._create_default_rules()

        # Known sensitive header names (case-insensitive)
        self.sensitive_headers = {
            "authorization",
            "x-api-key",
            "api-key",
            "x-auth-token",
            "auth-token",
            "bearer",
            "x-access-token",
            "access-token",
            "x-api-token",
            "api-token",
            "x-secret",
            "secret-key",
            "x-anthropic-api-key",
            "anthropic-version",
            "openai-api-key",
        }

        # Environment variable patterns that contain sensitive data
        self.sensitive_env_patterns = [
            r".*API_KEY.*",
            r".*TOKEN.*",
            r".*SECRET.*",
            r".*PASSWORD.*",
            r".*PRIVATE.*",
            r".*CREDENTIALS.*",
            r".*AUTH.*",
        ]

    def _create_default_rules(self) -> list[MaskingRule]:
        """Create default masking rules for common sensitive patterns."""
        return [
            # API Keys - various formats
            MaskingRule(
                pattern=r"\b(sk-[a-zA-Z0-9]{16,})",
                replacement=r"sk-***[REDACTED]",
                description="OpenAI-style API keys (sk-...)",
                sensitivity=SensitivityLevel.SECRET,
            ),
            MaskingRule(
                pattern=r"\b(ant-[a-zA-Z0-9_-]{40,})",
                replacement=r"ant-***[REDACTED]",
                description="Anthropic API keys (ant-...)",
                sensitivity=SensitivityLevel.SECRET,
            ),
            MaskingRule(
                pattern=r"\b(gsk_[a-zA-Z0-9]{50,})",
                replacement=r"gsk_***[REDACTED]",
                description="Groq API keys (gsk_...)",
                sensitivity=SensitivityLevel.SECRET,
            ),
            MaskingRule(
                pattern=r"\b([a-zA-Z0-9]{32,})",
                replacement=r"***[REDACTED]",
                description="Generic long tokens (32+ chars)",
                sensitivity=SensitivityLevel.CONFIDENTIAL,
            ),
            # Bearer tokens in Authorization headers
            MaskingRule(
                pattern=r"(Bearer\s+)([a-zA-Z0-9._-]{10,})",
                replacement=r"\1***[REDACTED]",
                description="Bearer tokens in Authorization headers",
                sensitivity=SensitivityLevel.SECRET,
            ),
            # API key in query parameters or JSON
            MaskingRule(
                pattern=r"(['\"]?api[\s_-]?key['\"]?\s*[=:]\s*['\"]?)([^\s'\"]{3,})(['\"]?)",
                replacement=r"\1***[REDACTED]\3",
                description="API keys in key-value pairs (supports 'API key:', 'api_key=', etc.)",
                sensitivity=SensitivityLevel.SECRET,
            ),
            # Token patterns in key-value pairs
            MaskingRule(
                pattern=r"(['\"]?token['\"]?\s*[=:]\s*['\"]?)([^\s'\"]{3,})(['\"]?)",
                replacement=r"\1***[REDACTED]\3",
                description="Tokens in key-value pairs (supports 'Token:', 'token=', etc.)",
                sensitivity=SensitivityLevel.SECRET,
            ),
            # Password patterns in key-value pairs
            MaskingRule(
                pattern=r"(['\"]?password['\"]?\s*[=:]\s*['\"]?)([^\s'\"]{3,})(['\"]?)",
                replacement=r"\1***[REDACTED]\3",
                description="Passwords in key-value pairs (supports 'password=', etc.)",
                sensitivity=SensitivityLevel.SECRET,
            ),
            # Environment variable assignments
            MaskingRule(
                pattern=r"(export\s+[A-Z_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)[A-Z_]*\s*=\s*['\"]?)([^\s'\"]{8,})(['\"]?)",
                replacement=r"\1***[REDACTED]\3",
                description="Environment variable assignments",
                sensitivity=SensitivityLevel.SECRET,
            ),
            # URLs with credentials
            MaskingRule(
                pattern=r"(https?://[^:@\s]+):([^@\s]+)@",
                replacement=r"\1:***[REDACTED]@",
                description="URLs with embedded credentials",
                sensitivity=SensitivityLevel.SECRET,
            ),
        ]

    def _mask_long_token(self, token: str) -> str:
        """Mask a long token showing only first/last few characters."""
        if len(token) < MIN_TOKEN_LENGTH_FOR_PARTIAL_DISPLAY:
            return "***[REDACTED]"
        if len(token) < MIN_TOKEN_LENGTH_FOR_MEDIUM_DISPLAY:
            return f"{token[:SHORT_TOKEN_DISPLAY_CHARS]}***[REDACTED]{token[-SHORT_TOKEN_DISPLAY_CHARS:]}"
        return f"{token[:LONG_TOKEN_DISPLAY_CHARS]}***[REDACTED]{token[-LONG_TOKEN_DISPLAY_CHARS:]}"

    def mask_api_key(self, api_key: str, show_chars: int = 4) -> str:
        """Mask an API key showing only the first/last few characters."""
        if not api_key or len(api_key) < show_chars * 2:
            return "***[REDACTED]"

        if len(api_key) <= show_chars * 2 + 3:
            return "***[REDACTED]"

        return f"{api_key[:show_chars]}***[REDACTED]{api_key[-show_chars:]}"

    def sanitize_headers(
        self, headers: dict[str, Any], sensitivity: SensitivityLevel = SensitivityLevel.CONFIDENTIAL
    ) -> dict[str, Any]:
        """Sanitize HTTP headers by masking sensitive values."""
        if headers is None:
            return {}
            
        sanitized = {}

        for key, value in headers.items():
            key_lower = key.lower()

            if any(sensitive in key_lower for sensitive in self.sensitive_headers):
                # This is a sensitive header
                if sensitivity == SensitivityLevel.SECRET:
                    sanitized[key] = "[REDACTED]"
                elif sensitivity == SensitivityLevel.CONFIDENTIAL:
                    sanitized[key] = self.mask_api_key(str(value)) if value else "[REDACTED]"
                else:
                    sanitized[key] = f"***{str(value)[-LONG_TOKEN_DISPLAY_CHARS:] if value and len(str(value)) > MIN_SENSITIVE_FIELD_LENGTH_FOR_PARTIAL_DISPLAY else '[REDACTED]'}"
            else:
                # Non-sensitive header, include as-is
                sanitized[key] = value

        return sanitized

    def mask_text(self, text: str, sensitivity: SensitivityLevel = SensitivityLevel.CONFIDENTIAL) -> str:
        """Apply all masking rules to protect sensitive data in text."""
        if not text:
            return text

        masked_text = text

        for rule in self.masking_rules:
            if rule.sensitivity.value in [SensitivityLevel.SECRET.value, sensitivity.value]:
                masked_text = rule.apply(masked_text)

        return masked_text

    def mask_environment_vars(
        self, env_dict: dict[str, Any], sensitivity: SensitivityLevel = SensitivityLevel.CONFIDENTIAL
    ) -> dict[str, Any]:
        """Mask sensitive environment variable values."""
        masked_env = {}

        for key, value in env_dict.items():
            is_sensitive = any(re.match(pattern, key, re.IGNORECASE) for pattern in self.sensitive_env_patterns)

            if is_sensitive:
                if sensitivity == SensitivityLevel.SECRET:
                    masked_env[key] = "[REDACTED]"
                elif sensitivity == SensitivityLevel.CONFIDENTIAL:
                    masked_env[key] = self.mask_api_key(str(value)) if value else "[REDACTED]"
                else:
                    masked_env[key] = "***[SET]" if value else "[UNSET]"
            else:
                masked_env[key] = value

        return masked_env

    def mask_response_data(self, data: Any, sensitivity: SensitivityLevel = SensitivityLevel.CONFIDENTIAL) -> Any:
        """Recursively mask sensitive data in API response data."""
        if isinstance(data, dict):
            return {key: self.mask_response_data(value, sensitivity) for key, value in data.items()}
        if isinstance(data, list):
            return [self.mask_response_data(item, sensitivity) for item in data]
        if isinstance(data, str):
            return self.mask_text(data, sensitivity)
        return data

    def create_safe_log_context(
        self,
        provider_name: str,
        url: str,
        headers: dict[str, Any] | None = None,
        response_data: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Create a logging context with all sensitive data properly masked."""
        context = {
            "provider": provider_name,
            "url": self.mask_text(url, SensitivityLevel.INTERNAL),  # URLs might contain tokens
        }

        if headers:
            context["headers"] = self.sanitize_headers(headers, SensitivityLevel.INTERNAL)

        if response_data:
            context["response_data"] = self.mask_response_data(response_data, SensitivityLevel.INTERNAL)

        if error_message:
            context["error"] = self.mask_text(error_message, SensitivityLevel.INTERNAL)

        return context

    def create_safe_cli_output(self, data: Any, mask_level: SensitivityLevel = SensitivityLevel.CONFIDENTIAL) -> Any:
        """Create CLI output with appropriate masking for user display."""
        if isinstance(data, dict):
            return {key: self.create_safe_cli_output(value, mask_level) for key, value in data.items()}
        if isinstance(data, list):
            return [self.create_safe_cli_output(item, mask_level) for item in data]
        if isinstance(data, str):
            return self.mask_text(data, mask_level)
        return data

    def is_sensitive_key(self, key: str) -> bool:
        """Check if a key name suggests it contains sensitive data."""
        key_lower = key.lower()
        sensitive_patterns = [
            "key",
            "token",
            "secret",
            "password",
            "auth",
            "credential",
            "private",
            "bearer",
            "api",
            "access",
        ]
        return any(pattern in key_lower for pattern in sensitive_patterns)


# Global instance for easy access
_global_protector: SensitiveDataProtector | None = None


def get_protector() -> SensitiveDataProtector:
    """Get the global sensitive data protector instance."""
    global _global_protector
    if _global_protector is None:
        _global_protector = SensitiveDataProtector()
    return _global_protector


# Convenience functions for common operations
def mask_api_key(api_key: str, show_chars: int = 4) -> str:
    """Mask an API key showing only first/last few characters."""
    return get_protector().mask_api_key(api_key, show_chars)


def sanitize_headers(headers: dict[str, Any]) -> dict[str, Any]:
    """Sanitize HTTP headers for safe logging."""
    return get_protector().sanitize_headers(headers, SensitivityLevel.INTERNAL)


def sanitize_cli_headers(headers: dict[str, Any]) -> dict[str, Any]:
    """Sanitize HTTP headers for CLI display."""
    return get_protector().sanitize_headers(headers, SensitivityLevel.CONFIDENTIAL)


def mask_text_for_logs(text: str) -> str:
    """Mask sensitive data in text for internal logging."""
    return get_protector().mask_text(text, SensitivityLevel.INTERNAL)


def mask_text_for_cli(text: str) -> str:
    """Mask sensitive data in text for CLI output."""
    return get_protector().mask_text(text, SensitivityLevel.CONFIDENTIAL)


def create_safe_log_context(provider_name: str, url: str, **kwargs) -> dict[str, Any]:
    """Create a safe logging context with masked sensitive data."""
    return get_protector().create_safe_log_context(provider_name, url, **kwargs)
