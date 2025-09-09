"""
this_file: src/vexy_co_model_catalog/core/graceful_degradation.py

Graceful degradation system for handling partial provider failures during batch operations.
Provides intelligent failure categorization, fallback strategies, and detailed progress reporting.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable


class FailureType(Enum):
    """Categorizes different types of failures for appropriate handling."""

    TEMPORARY_NETWORK = "temporary_network"  # Rate limits, timeouts, temporary outages
    AUTHENTICATION = "authentication"  # API key issues, auth failures
    CONFIGURATION = "configuration"  # Missing env vars, invalid config
    PERMANENT_API = "permanent_api"  # Provider discontinued, endpoint moved
    UNKNOWN = "unknown"  # Unclassified errors


class OperationResult(Enum):
    """Result status for individual operations."""

    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED_TEMPORARY = "failed_temporary"
    FAILED_PERMANENT = "failed_permanent"
    RETRYING = "retrying"


@dataclass
class ProviderResult:
    """Result of processing a single provider."""

    provider_name: str
    result: OperationResult
    failure_type: FailureType | None = None
    error_message: str | None = None
    retry_count: int = 0
    execution_time_ms: float = 0.0
    model_count: int = 0
    fallback_used: bool = False

    @property
    def is_success(self) -> bool:
        """Check if the operation was successful."""
        return self.result == OperationResult.SUCCESS

    @property
    def should_retry(self) -> bool:
        """Check if this failure should be retried."""
        return (
            self.result == OperationResult.FAILED_TEMPORARY
            and self.failure_type in {FailureType.TEMPORARY_NETWORK, FailureType.UNKNOWN}
            and self.retry_count < 3
        )


@dataclass
class BatchResult:
    """Aggregated results from a batch operation."""

    total_providers: int = 0
    successful: int = 0
    skipped: int = 0
    failed_permanent: int = 0
    failed_temporary: int = 0
    retries_attempted: int = 0
    total_models: int = 0
    execution_time_ms: float = 0.0

    provider_results: list[ProviderResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_providers == 0:
            return 0.0
        return (self.successful / self.total_providers) * 100

    @property
    def processed_providers(self) -> int:
        """Number of providers that were attempted (not skipped)."""
        return self.successful + self.failed_permanent + self.failed_temporary

    @property
    def has_partial_success(self) -> bool:
        """Check if we have some successes even with some failures."""
        return self.successful > 0 and (self.failed_permanent > 0 or self.failed_temporary > 0)


class FailureClassifier:
    """Classifies failures to determine appropriate handling strategy."""

    @staticmethod
    def classify_error(error: Exception, provider_name: str) -> FailureType:
        """Classify an error to determine the failure type and appropriate response."""
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Authentication failures
        if any(
            keyword in error_str
            for keyword in ["authentication", "unauthorized", "api key", "invalid key", "forbidden", "401", "403"]
        ):
            return FailureType.AUTHENTICATION

        # Temporary network issues
        if any(
            keyword in error_str
            for keyword in [
                "rate limit",
                "429",
                "timeout",
                "connection",
                "network",
                "temporary",
                "service unavailable",
                "503",
                "502",
                "504",
            ]
        ):
            return FailureType.TEMPORARY_NETWORK

        # Configuration issues
        if any(
            keyword in error_str
            for keyword in ["configuration", "missing", "environment", "not found", "invalid config", "setup"]
        ):
            return FailureType.CONFIGURATION

        # Permanent API issues
        if any(
            keyword in error_str
            for keyword in [
                "not found",
                "404",
                "moved",
                "discontinued",
                "deprecated",
                "no longer available",
                "endpoint not found",
            ]
        ):
            return FailureType.PERMANENT_API

        # Default to unknown for further analysis
        logger.debug(f"Unclassified error for {provider_name}: {error_type}: {error}")
        return FailureType.UNKNOWN


class GracefulDegradationManager:
    """Manages graceful degradation strategies for batch operations."""

    def __init__(self, max_retries: int = 2, retry_delay_base: float = 1.0) -> None:
        """
        Initialize graceful degradation manager.

        Args:
            max_retries: Maximum number of retries for temporary failures
            retry_delay_base: Base delay for exponential backoff (seconds)
        """
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        self.classifier = FailureClassifier()

    async def execute_batch_operation(
        self,
        providers: list[Any],
        operation: Callable,
        operation_name: str = "batch_operation",
        progress_callback: Callable[[str, ProviderResult], None] | None = None,
    ) -> BatchResult:
        """
        Execute a batch operation with graceful degradation.

        Args:
            providers: List of providers to process
            operation: Async function to execute for each provider
            operation_name: Name of the operation for logging
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult with detailed success/failure information
        """
        start_time = time.time()
        batch_result = BatchResult(total_providers=len(providers))

        logger.info(f"Starting {operation_name} for {len(providers)} providers")

        # Process providers with retry logic
        for provider in providers:
            provider_result = await self._process_provider_with_retries(provider, operation, progress_callback)

            batch_result.provider_results.append(provider_result)

            # Update counters
            if provider_result.result == OperationResult.SUCCESS:
                batch_result.successful += 1
                batch_result.total_models += provider_result.model_count
            elif provider_result.result == OperationResult.SKIPPED:
                batch_result.skipped += 1
            elif provider_result.result == OperationResult.FAILED_TEMPORARY:
                batch_result.failed_temporary += 1
            elif provider_result.result == OperationResult.FAILED_PERMANENT:
                batch_result.failed_permanent += 1

            batch_result.retries_attempted += provider_result.retry_count

        batch_result.execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Completed {operation_name}: {batch_result.successful}/{batch_result.total_providers} "
            f"successful ({batch_result.success_rate:.1f}%), "
            f"{batch_result.retries_attempted} retries attempted"
        )

        return batch_result

    async def _process_provider_with_retries(
        self, provider: Any, operation: Callable, progress_callback: Callable[[str, ProviderResult], None] | None
    ) -> ProviderResult:
        """Process a single provider with retry logic."""
        provider_name = getattr(provider, "name", str(provider))
        start_time = time.time()

        result = ProviderResult(provider_name=provider_name, result=OperationResult.RETRYING)

        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(
                        f"Processing {provider_name}" + (f" (retry {attempt})" if attempt > 0 else ""), result
                    )

                # Execute the operation
                operation_result = await operation(provider)

                # Handle different result types
                if operation_result is None:
                    # Operation was skipped
                    result.result = OperationResult.SKIPPED
                    break
                if isinstance(operation_result, dict) and "model_count" in operation_result:
                    # Successful with model count
                    result.result = OperationResult.SUCCESS
                    result.model_count = operation_result["model_count"]
                    break
                # Generic success
                result.result = OperationResult.SUCCESS
                break

            except Exception as e:
                result.retry_count = attempt
                result.error_message = str(e)
                result.failure_type = self.classifier.classify_error(e, provider_name)

                # Determine if we should retry
                if attempt < self.max_retries and self._should_retry(result.failure_type):
                    result.result = OperationResult.RETRYING
                    delay = self.retry_delay_base * (2**attempt)  # Exponential backoff

                    logger.debug(
                        f"Retrying {provider_name} after {result.failure_type.value} "
                        f"(attempt {attempt + 1}/{self.max_retries + 1}) in {delay:.1f}s"
                    )

                    await asyncio.sleep(delay)
                    continue
                # No more retries or permanent failure
                if result.failure_type in {
                    FailureType.AUTHENTICATION,
                    FailureType.CONFIGURATION,
                    FailureType.PERMANENT_API,
                }:
                    result.result = OperationResult.FAILED_PERMANENT
                else:
                    result.result = OperationResult.FAILED_TEMPORARY
                break

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    def _should_retry(self, failure_type: FailureType) -> bool:
        """Determine if a failure type should be retried."""
        return failure_type in {FailureType.TEMPORARY_NETWORK, FailureType.UNKNOWN}

    def generate_fallback_strategies(self, batch_result: BatchResult) -> list[str]:
        """Generate fallback strategies based on batch results."""
        strategies = []

        if batch_result.has_partial_success:
            strategies.append(f"✓ Partial success: {batch_result.successful} providers completed successfully")

        # Analyze failure patterns
        temp_failures = [r for r in batch_result.provider_results if r.result == OperationResult.FAILED_TEMPORARY]
        perm_failures = [r for r in batch_result.provider_results if r.result == OperationResult.FAILED_PERMANENT]

        if temp_failures:
            strategies.append(
                f"⏰ {len(temp_failures)} temporary failures - retry later with: "
                f"--providers {','.join(r.provider_name for r in temp_failures)}"
            )

        if perm_failures:
            auth_failures = [r for r in perm_failures if r.failure_type == FailureType.AUTHENTICATION]
            config_failures = [r for r in perm_failures if r.failure_type == FailureType.CONFIGURATION]

            if auth_failures:
                strategies.append(
                    f"🔑 {len(auth_failures)} authentication issues - check API keys for: "
                    f"{', '.join(r.provider_name for r in auth_failures)}"
                )

            if config_failures:
                strategies.append(
                    f"⚙️ {len(config_failures)} configuration issues - review settings for: "
                    f"{', '.join(r.provider_name for r in config_failures)}"
                )

        if batch_result.success_rate < 50 and batch_result.successful > 0:
            strategies.append(
                "💡 Consider running with only successful providers for now: "
                f"--providers {','.join(r.provider_name for r in batch_result.provider_results if r.is_success)}"
            )

        return strategies

    def create_detailed_report(self, batch_result: BatchResult) -> dict[str, Any]:
        """Create a detailed report of the batch operation."""
        return {
            "summary": {
                "total_providers": batch_result.total_providers,
                "successful": batch_result.successful,
                "failed": batch_result.failed_permanent + batch_result.failed_temporary,
                "skipped": batch_result.skipped,
                "success_rate": batch_result.success_rate,
                "total_models": batch_result.total_models,
                "execution_time_ms": batch_result.execution_time_ms,
                "retries_attempted": batch_result.retries_attempted,
            },
            "failures_by_type": {
                failure_type.value: len([r for r in batch_result.provider_results if r.failure_type == failure_type])
                for failure_type in FailureType
            },
            "provider_details": [
                {
                    "name": result.provider_name,
                    "result": result.result.value,
                    "failure_type": result.failure_type.value if result.failure_type else None,
                    "error": result.error_message,
                    "retries": result.retry_count,
                    "execution_time_ms": result.execution_time_ms,
                    "model_count": result.model_count,
                }
                for result in batch_result.provider_results
            ],
            "fallback_strategies": self.generate_fallback_strategies(batch_result),
        }
