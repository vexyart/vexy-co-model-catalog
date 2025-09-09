"""
this_file: src/vexy_co_model_catalog/core/fetcher.py

Async HTTP JSON fetcher with provider-specific handling.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import TYPE_CHECKING, Any, Self

from loguru import logger

from vexy_co_model_catalog.core.caching import (
    cache_model_data,
    get_cached_model_data,
)
from vexy_co_model_catalog.core.model_validator import ModelDataValidator, ModelValidationSeverity
from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind
from vexy_co_model_catalog.core.rate_limiter import get_rate_limiter
from vexy_co_model_catalog.core.retry import (
    CircuitBreakerConfig,
    EnhancedRetryHandler,
    RetryConfig,
    create_resilient_http_client,
)
from vexy_co_model_catalog.core.security import create_safe_log_context, mask_text_for_logs, sanitize_headers
from vexy_co_model_catalog.core.security_enhanced import get_enhanced_security_validator, validate_network_url
from vexy_co_model_catalog.utils.exceptions import AuthenticationError, FetchError, RateLimitError

if TYPE_CHECKING:
    import httpx

    from vexy_co_model_catalog.core.failure_tracker import FailureTracker

# HTTP Status Code Constants
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_TOO_MANY_REQUESTS = 429
PROGRESS_DISPLAY_THRESHOLD = 100


class ModelFetcher:
    """Async HTTP client for fetching model catalogs from AI providers."""

    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(
        self, max_concurrency: int = 8, timeout: float = 30.0, failure_tracker: FailureTracker | None = None
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = create_resilient_http_client(timeout=timeout, max_connections=max_concurrency * 2)
        self._request_count = 0
        self._error_count = 0
        self._failure_tracker = failure_tracker
        self._model_validator = ModelDataValidator()

        # Enhanced retry configuration
        retry_config = RetryConfig(max_attempts=5, base_delay=1.0, max_delay=30.0, backoff_multiplier=2.0, jitter=True)

        circuit_config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60.0, success_threshold=2)

        self._retry_handler = EnhancedRetryHandler(retry_config, circuit_config)

        # Initialize intelligent rate limiter
        self._rate_limiter = get_rate_limiter()

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def fetch_provider_models(
        self, provider: ProviderConfig, _max_attempts: int = 3, use_cache: bool = True
    ) -> dict[str, Any] | list[Any]:
        """Fetch models for a specific provider with provider-appropriate headers and enhanced retry logic."""
        try:
            # Check cache first if enabled
            if use_cache:
                cached_data = get_cached_model_data(provider.name)
                if cached_data is not None:
                    logger.debug(f"Using cached model data for {provider.name}")
                    return cached_data

            models_url = provider.build_models_url()

            # Validate URL for security compliance
            allow_localhost = provider.name in ["lmstudio", "ollama", "tabby", "text-generation-webui"]
            if not validate_network_url(models_url, allow_localhost=allow_localhost):
                msg = f"SECURITY: URL failed security validation for {provider.name}"
                raise FetchError(msg)

            headers = self._build_provider_headers(provider)

            logger.info(f"Fetching models from {provider.name}: {mask_text_for_logs(models_url)}")

            # Check if provider is currently marked as failed
            if self._failure_tracker and self._failure_tracker.is_provider_failed(provider.name):
                logger.warning(f"Provider {provider.name} is currently marked as failed, attempting fetch anyway")

            # Configure rate limiting for this provider
            self._rate_limiter.configure_provider(provider.name)

            # Use enhanced retry logic for the HTTP request with rate limiting
            response = await self._retry_handler.execute_with_retry(
                lambda url, hdrs: self._make_http_request_with_rate_limiting(provider.name, url, hdrs),
                f"fetch_{provider.name}_models",
                models_url,
                headers,
            )

            # Handle different error conditions
            if response.status_code == HTTP_UNAUTHORIZED or response.status_code == HTTP_FORBIDDEN:
                error_msg = f"Authentication failed ({response.status_code})"
                if self._failure_tracker:
                    safe_headers = sanitize_headers(dict(response.headers))
                    self._failure_tracker.mark_provider_failed(
                        provider.name, error_msg, response.status_code, safe_headers
                    )
                logger.warning(
                    create_safe_log_context(
                        provider.name, models_url, headers=dict(response.headers), error_message=error_msg
                    )
                )
                raise AuthenticationError(error_msg)

            if response.status_code == HTTP_TOO_MANY_REQUESTS:
                error_msg = "Rate limited"
                if self._failure_tracker:
                    safe_headers = sanitize_headers(dict(response.headers))
                    self._failure_tracker.mark_provider_failed(
                        provider.name, error_msg, response.status_code, safe_headers
                    )
                logger.warning(
                    create_safe_log_context(
                        provider.name, models_url, headers=dict(response.headers), error_message=error_msg
                    )
                )
                raise RateLimitError(error_msg)

            if not response.is_success:
                error_msg = f"HTTP {response.status_code}: {mask_text_for_logs(response.text[:200])}"
                if self._failure_tracker:
                    safe_headers = sanitize_headers(dict(response.headers))
                    self._failure_tracker.mark_provider_failed(
                        provider.name, error_msg, response.status_code, safe_headers
                    )
                logger.warning(
                    create_safe_log_context(
                        provider.name, models_url, headers=dict(response.headers), error_message=error_msg
                    )
                )
                raise FetchError(error_msg)

            # Parse JSON response with enhanced error context
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                # Enhanced error context with actionable guidance
                content_preview = response.text[:200] if hasattr(response, 'text') else "<unavailable>"
                error_msg = (
                    f"Invalid JSON response from {provider.name}: {e}. "
                    f"Response preview: {content_preview[:PROGRESS_DISPLAY_THRESHOLD]}{'...' if len(content_preview) > PROGRESS_DISPLAY_THRESHOLD else ''}. "
                    f"This usually indicates: 1) API endpoint changes, 2) Server error response, "
                    f"3) Rate limiting with HTML error page. Try: --validate {provider.name} to check status."
                )
                if self._failure_tracker:
                    self._failure_tracker.mark_provider_failed(provider.name, error_msg)
                logger.error(f"JSON parsing failed for {provider.name}: Response status {response.status_code}, Content-Type: {response.headers.get('content-type', 'unknown')}")
                raise FetchError(error_msg)

            # Special handling for chutes provider
            if provider.name == "chutes":
                chutes_data = await self._fetch_chutes_data()
                data = self._merge_chutes_data(data, chutes_data)

            # Mark provider as successful
            if self._failure_tracker:
                safe_headers = sanitize_headers(dict(response.headers))
                self._failure_tracker.mark_provider_success(provider.name, safe_headers)

            logger.debug(create_safe_log_context(provider.name, models_url, headers=dict(response.headers)))

            # Cache the successful result
            if use_cache:
                # Use longer TTL for stable model metadata
                cache_ttl = (
                    7200 if provider.name in ["openai", "anthropic", "groq"] else 3600
                )  # 2h for major providers, 1h for others
                cache_model_data(provider.name, data, cache_ttl)
                logger.debug(f"Cached model data for {provider.name} with TTL {cache_ttl}s")

            return data

        except (AuthenticationError, RateLimitError, FetchError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Enhanced error context with diagnostic information and recovery suggestions
            error_type = type(e).__name__
            error_details = mask_text_for_logs(str(e))

            # Provide specific guidance based on error type
            recovery_suggestions = []
            if "timeout" in error_details.lower():
                recovery_suggestions.extend([
                    "Try increasing timeout with --timeout option",
                    "Check your internet connection",
                    "Verify provider endpoint availability"
                ])
            elif "connection" in error_details.lower():
                recovery_suggestions.extend([
                    "Verify internet connectivity",
                    "Check if provider API endpoint is accessible",
                    "Try again in a few minutes (temporary outage possible)"
                ])
            elif "ssl" in error_details.lower() or "certificate" in error_details.lower():
                recovery_suggestions.extend([
                    "Check system time/date accuracy",
                    "Update certificates: pip install --upgrade certifi",
                    "Try with --no-verify-ssl if appropriate"
                ])
            else:
                recovery_suggestions.append("Run with --verbose for detailed debugging information")

            error_msg = (
                f"Unexpected {error_type} while fetching {provider.name}: {error_details}. "
                f"Suggested actions: {'; '.join(recovery_suggestions)}"
            )

            if self._failure_tracker:
                self._failure_tracker.mark_provider_failed(provider.name, error_msg)
            logger.error(f"Failed to fetch models for {provider.name}: {error_msg}")
            raise FetchError(error_msg)

    async def _make_http_request_with_rate_limiting(
        self, provider_name: str, url: str, headers: dict[str, str]
    ) -> httpx.Response:
        """Make HTTP request with intelligent rate limiting and response tracking."""
        # Acquire rate limit permit
        delay = await self._rate_limiter.acquire_permit(provider_name)
        if delay > 0:
            logger.debug(f"Rate limiting {provider_name}: waiting {delay:.2f}s")
            await asyncio.sleep(delay)

        # Make the actual request
        async with self._semaphore:
            self._request_count += 1
            time.time()

            try:
                response = await self._client.get(url, headers=headers)

                # Record successful response for rate limiting adaptation
                # Note: rate limiter needs actual headers for rate limiting logic,
                # but we'll mask them in any logging it does internally
                await self._rate_limiter.record_response(
                    provider_name, dict(response.headers), success=response.is_success
                )

                # Log rate limiting info if present in headers
                self._log_rate_limit_info(provider_name, response.headers)

                return response

            except Exception:
                # Record failed response for rate limiting adaptation
                await self._rate_limiter.record_response(provider_name, {}, success=False)
                raise

    async def _make_http_request(self, url: str, headers: dict[str, str]) -> httpx.Response:
        """Make HTTP request with semaphore control - used by retry handler (legacy)."""
        async with self._semaphore:
            self._request_count += 1
            return await self._client.get(url, headers=headers)

    def _log_rate_limit_info(self, provider_name: str, headers: dict[str, str]) -> None:
        """Log rate limit information if present in response headers."""
        headers_lower = {k.lower(): v for k, v in headers.items()}

        rate_info = []
        if "x-ratelimit-remaining" in headers_lower:
            rate_info.append(f"remaining: {headers_lower['x-ratelimit-remaining']}")
        if "x-ratelimit-reset" in headers_lower:
            rate_info.append(f"reset: {headers_lower['x-ratelimit-reset']}")

        if rate_info:
            logger.debug(f"{provider_name} rate limits: {', '.join(rate_info)}")

    def _build_provider_headers(self, provider: ProviderConfig) -> dict[str, str]:
        """Build HTTP headers appropriate for the provider type."""
        headers = {"User-Agent": "ModelDumper/1.0"}

        # Get API key if needed
        api_key = None
        if provider.api_key_env:
            api_key = os.environ.get(provider.api_key_env)

            # Validate API key format for security compliance
            if api_key:
                validator = get_enhanced_security_validator()
                if not validator.validate_api_key_format(api_key, provider.name):
                    logger.warning(f"SECURITY: API key format validation failed for {provider.name}")
                    # Continue but log the warning - don't block execution for format issues

        if provider.kind == ProviderKind.ANTHROPIC:
            # Anthropic API requires specific headers
            if api_key:
                headers["x-api-key"] = api_key
            headers["anthropic-version"] = self.ANTHROPIC_VERSION

        elif provider.kind == ProviderKind.OPENAI:
            # OpenAI-compatible APIs use Bearer token
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        elif provider.kind == ProviderKind.URL:
            # URL providers don't need authentication headers
            pass

        # Add any custom headers from provider config
        if provider.headers:
            headers.update(provider.headers)

        return headers

    async def _fetch_chutes_data(self) -> dict[str, Any] | None:
        """
        Fetch additional chutes data from the Chutes API.

        This supplements the OpenAI-compatible API response with additional
        model information from the chutes-specific API endpoint.
        """
        chutes_api_url = "https://api.chutes.ai/chutes/?include_public=true&include_schemas=false"

        # Validate chutes URL for security compliance
        if not validate_network_url(chutes_api_url, allow_localhost=False):
            logger.warning("SECURITY: Chutes URL failed security validation, skipping supplemental data")
            return None

        try:
            api_key = os.environ.get("CHUTES_API_KEY")
            headers = {"Content-Type": "application/json"}
            if api_key:
                # Follow standard Bearer token format for consistency
                headers["Authorization"] = f"Bearer {api_key}" if not api_key.startswith("Bearer ") else api_key

            logger.debug(f"Fetching Chutes supplemental data: {mask_text_for_logs(chutes_api_url)}")

            # Configure rate limiting for chutes as well
            self._rate_limiter.configure_provider("chutes")

            # Use enhanced retry logic for chutes data with rate limiting
            response = await self._retry_handler.execute_with_retry(
                lambda url, hdrs: self._make_http_request_with_rate_limiting("chutes", url, hdrs),
                "fetch_chutes_supplemental_data",
                chutes_api_url,
                headers,
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            # Enhanced Chutes error handling with fallback guidance
            error_details = mask_text_for_logs(str(e))
            logger.warning(
                f"Failed to fetch Chutes supplemental data: {error_details}. "
                f"Falling back to primary data only. This may result in incomplete model listings. "
                f"To resolve: 1) Check chutes.ai connectivity, 2) Verify API endpoint status, "
                f"3) Consider using --no-chutes flag to skip supplemental data."
            )
            return None

    def _merge_chutes_data(self, openai_data: dict, chutes_data: dict | None) -> dict:
        """
        Merge chutes API data with OpenAI API model data.

        Enhances OpenAI model entries with additional information from chutes API.
        """
        if not chutes_data or "items" not in chutes_data:
            return openai_data

        # Create a lookup map from chutes data by model name
        chutes_lookup = {}
        for chute in chutes_data["items"]:
            if "model" in chute:
                model_name = chute["model"]
                chutes_lookup[model_name] = chute

        # Enhance OpenAI data with chutes information
        if "data" in openai_data and isinstance(openai_data["data"], list):
            for model_entry in openai_data["data"]:
                if "id" in model_entry:
                    model_id = model_entry["id"]
                    if model_id in chutes_lookup:
                        chute_info = chutes_lookup[model_id]
                        # Add chute-specific fields
                        if "max_model_len" in chute_info:
                            model_entry["max_model_len"] = chute_info["max_model_len"]
                        if "description" in chute_info:
                            model_entry["chute_description"] = chute_info["description"]
                        if "tags" in chute_info:
                            model_entry["chute_tags"] = chute_info["tags"]

        return openai_data

    def sort_json_data(self, data: Any) -> Any:
        """Sort JSON data for consistent output."""
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                # OpenAI format with .data array
                data["data"] = sorted(data["data"], key=lambda x: x.get("id", ""))
                return data
            # Object format (sort by keys)
            return dict(sorted(data.items()))
        if isinstance(data, list):
            # Direct array format
            return sorted(data, key=lambda x: x.get("id", ""))
        return data

    def extract_model_ids(self, data: Any) -> list[str]:
        """Extract model IDs from JSON data."""
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                # OpenAI format
                return [item.get("id", "") for item in data["data"] if item.get("id")]
            # Object format (keys are model IDs)
            return [key for key in data if key != "sample_spec"]
        if isinstance(data, list):
            # Array format
            return [item.get("id", "") for item in data if item.get("id")]
        return []

    def validate_and_normalize_models(self, data: Any, provider: ProviderConfig) -> dict[str, Any]:
        """
        Validate and normalize model data from provider response.

        Args:
            data: Raw response data from provider
            provider: Provider configuration

        Returns:
            Dictionary with validation results and normalized models
        """
        validation_result = self._model_validator.validate_and_normalize(data, provider)

        # Log validation issues
        if validation_result.issues:
            error_count = sum(
                1 for issue in validation_result.issues if issue.severity == ModelValidationSeverity.ERROR
            )
            warning_count = sum(
                1 for issue in validation_result.issues if issue.severity == ModelValidationSeverity.WARNING
            )

            if error_count > 0:
                logger.warning(f"Model validation for {provider.name}: {error_count} errors, {warning_count} warnings")
                for issue in validation_result.issues:
                    if issue.severity == ModelValidationSeverity.ERROR:
                        logger.warning(f"  Error: {issue.message} (model: {issue.model_id})")
            elif warning_count > 0:
                logger.debug(f"Model validation for {provider.name}: {warning_count} warnings")
                for issue in validation_result.issues:
                    if issue.severity == ModelValidationSeverity.WARNING:
                        logger.debug(f"  Warning: {issue.message} (model: {issue.model_id})")

        logger.info(
            f"Model normalization for {provider.name}: "
            f"{validation_result.normalized_count}/{validation_result.original_count} models "
            f"({validation_result.success_rate:.1f}% success rate)"
        )

        return {
            "validation_result": validation_result,
            "models": validation_result.models,
            "model_ids": [model.id for model in validation_result.models],
            "original_data": data,
            "is_valid": validation_result.is_valid,
            "issues": validation_result.issues,
        }

    def stats(self) -> dict[str, float | int | None]:
        if self._request_count > 0:
            return {
                "requests": self._request_count,
                "errors": self._error_count,
                "success_rate": (self._request_count - self._error_count) / self._request_count,
            }
        return {"requests": 0, "errors": self._error_count, "success_rate": None}
