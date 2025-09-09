"""
this_file: src/vexy_co_model_catalog/core/validator.py

Provider configuration validation utilities.
Validates provider configurations, environment variables, and provides actionable guidance.
"""

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from vexy_co_model_catalog.core.caching import get_validation_cache
from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind

# Constants for validation
MIN_API_KEY_LENGTH = 10  # Minimum expected API key length


@dataclass
class ValidationResult:
    """Result of provider validation."""

    is_valid: bool
    provider_name: str
    issues: list[str]
    warnings: list[str]
    guidance: list[str]


@dataclass
class ValidationSummary:
    """Summary of validation results for multiple providers."""

    total_providers: int
    valid_providers: int
    invalid_providers: int
    results: list[ValidationResult]

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_providers == 0:
            return 0.0
        return (self.valid_providers / self.total_providers) * 100


class ProviderValidator:
    """Validates provider configurations and environment setup."""

    def __init__(self) -> None:
        """Initialize the validator with caching for performance."""
        self.common_env_patterns = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "groq": ["GROQ_API_KEY"],
            "fireworks": ["FIREWORKS_API_KEY"],
            "togetherai": ["TOGETHERAI_API_KEY"],
            "mistral": ["MISTRAL_API_KEY"],
            "deepinfra": ["DEEPINFRA_API_KEY"],
            "huggingface": ["HUGGINGFACEHUB_API_TOKEN"],
            "cerebras": ["CEREBRAS_API_KEY"],
            "chutes": ["CHUTES_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
        }
        # Use intelligent caching for validation results and environment variables
        self._validation_cache = get_validation_cache()
        self._env_cache = get_validation_cache()  # Use same cache for environment variables

    def validate_provider(self, provider: ProviderConfig) -> ValidationResult:
        """
        Validate a single provider configuration with caching.

        Args:
            provider: The provider configuration to validate

        Returns:
            ValidationResult with issues, warnings, and guidance
        """
        # Create cache key based on provider configuration
        cache_key = f"validation:{provider.name}:{provider.kind.value}:{provider.api_key_env}:{provider.base_url}:{provider.base_url_env}"

        # Check cache first
        cached_result = self._validation_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = ValidationResult(is_valid=True, provider_name=provider.name, issues=[], warnings=[], guidance=[])

        # Validate provider kind
        self._validate_provider_kind(provider, result)

        # Validate API key configuration
        self._validate_api_key(provider, result)

        # Validate base URL configuration
        self._validate_base_url(provider, result)

        # Validate environment variables
        self._validate_environment_variables(provider, result)

        # Add provider-specific guidance
        self._add_provider_guidance(provider, result)

        # Set overall validity
        result.is_valid = len(result.issues) == 0

        # Cache the result for future calls (30 minute TTL for validation results)
        self._validation_cache.put(cache_key, result, ttl_seconds=1800, tags=["validation", provider.name])

        return result

    def _get_cached_env_var(self, var_name: str) -> str | None:
        """Get environment variable with caching for performance."""
        cache_key = f"env:{var_name}"
        cached_value = self._env_cache.get(cache_key)

        if cached_value is None:
            env_value = os.environ.get(var_name)
            # Cache environment variables for 10 minutes (they rarely change during a session)
            self._env_cache.put(cache_key, env_value, ttl_seconds=600, tags=["environment"])
            return env_value

        return cached_value

    def clear_cache(self) -> None:
        """Clear validation and environment variable caches."""
        self._validation_cache.clear()
        # Note: _env_cache is the same instance as _validation_cache

    def validate_providers(self, providers: list[ProviderConfig]) -> ValidationSummary:
        """
        Validate multiple provider configurations.

        Args:
            providers: List of provider configurations to validate

        Returns:
            ValidationSummary with overall results
        """
        results = []
        valid_count = 0

        for provider in providers:
            result = self.validate_provider(provider)
            results.append(result)
            if result.is_valid:
                valid_count += 1

        return ValidationSummary(
            total_providers=len(providers),
            valid_providers=valid_count,
            invalid_providers=len(providers) - valid_count,
            results=results,
        )

    def _validate_provider_kind(self, provider: ProviderConfig, result: ValidationResult) -> None:
        """Validate provider kind configuration."""
        if not isinstance(provider.kind, ProviderKind):
            result.issues.append(f"Invalid provider kind: {provider.kind}")
            result.guidance.append("Provider kind must be one of: openai, anthropic, url")
        # Valid ProviderKind enum values - no issues to add

    def _validate_api_key(self, provider: ProviderConfig, result: ValidationResult) -> None:
        """Validate API key configuration."""
        if provider.kind == ProviderKind.URL:
            # URL providers typically don't need API keys
            if provider.api_key_env:
                result.warnings.append("URL provider has API key configured (usually not needed)")
            return

        if not provider.api_key_env:
            result.issues.append("Missing API key environment variable name")
            result.guidance.append("Add api_key_env to provider configuration")
            return

        # Check if environment variable is set (using cache)
        api_key = self._get_cached_env_var(provider.api_key_env)
        if not api_key:
            result.issues.append(f"Environment variable {provider.api_key_env} is not set")
            result.guidance.append(f"Set environment variable: export {provider.api_key_env}='your-api-key'")
        elif len(api_key.strip()) < MIN_API_KEY_LENGTH:
            result.warnings.append(f"API key in {provider.api_key_env} seems too short")
            result.guidance.append("Verify your API key is correct and complete")
        elif api_key.startswith("sk-") and provider.name != "openai":
            result.warnings.append("API key format looks like OpenAI but provider is different")
        elif provider.name == "openai" and not api_key.startswith("sk-"):
            result.warnings.append("OpenAI API keys typically start with 'sk-'")

    def _validate_base_url(self, provider: ProviderConfig, result: ValidationResult) -> None:
        """Validate base URL configuration."""
        # Check base_url if provided directly
        if provider.base_url:
            if not self._is_valid_url(provider.base_url):
                result.issues.append(f"Invalid base URL: {provider.base_url}")
                result.guidance.append("Ensure base URL is a valid HTTP/HTTPS URL")

        # Check base_url_env if specified
        if provider.base_url_env:
            base_url = self._get_cached_env_var(provider.base_url_env)
            if not base_url:
                result.warnings.append(f"Environment variable {provider.base_url_env} is not set")
                result.guidance.append(
                    f"Set environment variable: export {provider.base_url_env}='https://api.example.com'"
                )
            elif not self._is_valid_url(base_url):
                result.issues.append(f"Invalid base URL in {provider.base_url_env}: {base_url}")
                result.guidance.append("Ensure base URL in environment variable is valid")

    def _validate_environment_variables(self, provider: ProviderConfig, result: ValidationResult) -> None:
        """Validate environment variable patterns."""
        # Check for common environment variable patterns
        expected_vars = self.common_env_patterns.get(provider.name, [])

        for var in expected_vars:
            if var != provider.api_key_env and not self._get_cached_env_var(var):
                result.warnings.append(f"Common environment variable {var} is not set")
                result.guidance.append(f"Consider setting: export {var}='your-api-key'")

    def _add_provider_guidance(self, provider: ProviderConfig, result: ValidationResult) -> None:
        """Add provider-specific guidance."""
        guidance_map = {
            "openai": [
                "Get API key from: https://platform.openai.com/api-keys",
                "API keys start with 'sk-' and are 51 characters long",
            ],
            "anthropic": ["Get API key from: https://console.anthropic.com/", "Requires Claude API access"],
            "groq": ["Get API key from: https://console.groq.com/keys", "Groq provides ultra-fast inference"],
            "fireworks": ["Get API key from: https://fireworks.ai/api-keys", "Supports many open-source models"],
            "huggingface": [
                "Get API token from: https://huggingface.co/settings/tokens",
                "Use HUGGINGFACEHUB_API_TOKEN environment variable",
            ],
        }

        if provider.name in guidance_map:
            result.guidance.extend(guidance_map[provider.name])

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def get_setup_guidance(self, provider_name: str) -> list[str]:
        """Get setup guidance for a specific provider."""
        guidance = []

        if provider_name in self.common_env_patterns:
            env_vars = self.common_env_patterns[provider_name]
            for var in env_vars:
                guidance.append(f"Set {var} environment variable with your API key")

        provider_guidance = {
            "openai": [
                "Sign up at: https://platform.openai.com/",
                "Create API key: https://platform.openai.com/api-keys",
                "Usage: export OPENAI_API_KEY='sk-your-key-here'",
            ],
            "anthropic": [
                "Sign up at: https://console.anthropic.com/",
                "Get API access and create key",
                "Usage: export ANTHROPIC_API_KEY='your-key-here'",
            ],
            "groq": [
                "Sign up at: https://console.groq.com/",
                "Create API key: https://console.groq.com/keys",
                "Usage: export GROQ_API_KEY='your-key-here'",
            ],
        }

        if provider_name in provider_guidance:
            guidance.extend(provider_guidance[provider_name])

        return guidance
