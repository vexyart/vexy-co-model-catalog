"""
this_file: src/vexy_co_model_catalog/core/enhanced_integration.py

Integration patterns for enhanced logging and error recovery with existing modules.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from vexy_co_model_catalog.core.caching import get_cached_model_data

from vexy_co_model_catalog.core.enhanced_logging import (
    EnhancedErrorRecovery,
    ErrorCategory,
    ErrorContext,
    StructuredLogger,
    operation_context,
)
from vexy_co_model_catalog.core.fetcher import ModelFetcher
from vexy_co_model_catalog.core.storage import StorageManager

if TYPE_CHECKING:
    from vexy_co_model_catalog.core.provider import ProviderConfig


class EnhancedModelFetcher:
    """Enhanced ModelFetcher with structured logging and error recovery."""

    def __init__(self, max_concurrency: int = 8, timeout: float = 30.0, failure_tracker=None) -> None:
        """Initialize enhanced fetcher with structured logging."""
        self.base_fetcher = ModelFetcher(max_concurrency, timeout, failure_tracker)
        self.logger = StructuredLogger("model_fetcher")
        self.error_recovery = EnhancedErrorRecovery(self.logger)

        # Register fallback handlers
        self._register_fallback_handlers()

    def _register_fallback_handlers(self) -> None:
        """Register fallback handlers for common error scenarios."""

        # Network error fallback: retry with exponential backoff
        async def network_fallback(_error: Exception, provider: ProviderConfig, **_kwargs) -> dict[str, Any] | None:
            """Fallback for network errors with reduced data."""
            self.logger.warning(f"Network fallback activated for provider {provider.name}")

            # Try with reduced timeout and simpler request
            simple_fetcher = ModelFetcher(max_concurrency=1, timeout=10.0)
            try:
                return await simple_fetcher.fetch_provider_models(provider, max_attempts=1, use_cache=True)
            finally:
                await simple_fetcher.close()

        # Authentication error fallback: return cached data if available
        async def auth_fallback(error: Exception, provider: ProviderConfig, **_kwargs) -> dict[str, Any] | None:
            """Fallback for authentication errors using cached data."""
            cached_data = get_cached_model_data(provider.name)
            if cached_data:
                self.logger.info(f"Using cached data for {provider.name} due to auth error")
                return cached_data

            raise error  # No cached data available

        # Rate limit fallback: wait and retry once
        async def rate_limit_fallback(_error: Exception, provider: ProviderConfig, **_kwargs) -> dict[str, Any] | None:
            """Fallback for rate limit errors with extended wait."""
            self.logger.info(f"Rate limit fallback: waiting 60s before retry for {provider.name}")
            await asyncio.sleep(60)

            return await self.base_fetcher.fetch_provider_models(provider, max_attempts=1, use_cache=False)

        self.error_recovery.register_fallback_handler(ErrorCategory.NETWORK, network_fallback)
        self.error_recovery.register_fallback_handler(ErrorCategory.AUTHENTICATION, auth_fallback)
        self.error_recovery.register_fallback_handler(ErrorCategory.RATE_LIMITING, rate_limit_fallback)

    async def fetch_provider_models_enhanced(
        self, provider: ProviderConfig, max_attempts: int = 3, use_cache: bool = True, enable_fallback: bool = True
    ) -> dict[str, Any]:
        """
        Fetch provider models with enhanced error recovery and structured logging.

        Args:
            provider: Provider configuration
            max_attempts: Maximum retry attempts
            use_cache: Whether to use cached data
            enable_fallback: Whether to enable fallback mechanisms

        Returns:
            Provider model data or fallback result
        """
        with operation_context(
            f"fetch_models_{provider.name}", "model_fetcher", provider_name=provider.name, use_cache=use_cache
        ):
            # Determine error category based on provider type
            error_category = ErrorCategory.EXTERNAL_SERVICE

            return await self.error_recovery.execute_with_fallback(
                operation=self.base_fetcher.fetch_provider_models,
                operation_name=f"fetch_{provider.name}_models",
                error_category=error_category,
                fallback_value={"data": [], "error": "All fetch attempts failed"},
                enable_fallback=enable_fallback,
                provider=provider,
                max_attempts=max_attempts,
                use_cache=use_cache,
            )

    async def fetch_multiple_providers_enhanced(
        self, providers: list[ProviderConfig], max_concurrency: int = 5, continue_on_error: bool = True
    ) -> dict[str, Any]:
        """
        Fetch from multiple providers with enhanced error handling and partial failure tolerance.

        Args:
            providers: List of provider configurations
            max_concurrency: Maximum concurrent requests
            continue_on_error: Whether to continue processing other providers if one fails

        Returns:
            Dictionary with results and failure summary
        """
        with operation_context(
            "fetch_multiple_providers", "model_fetcher", provider_count=len(providers), max_concurrency=max_concurrency
        ):
            semaphore = asyncio.Semaphore(max_concurrency)
            results = {}
            failures = {}

            async def fetch_single_provider(provider: ProviderConfig) -> None:
                """Fetch models from a single provider with error handling."""
                async with semaphore:
                    try:
                        result = await self.fetch_provider_models_enhanced(provider, enable_fallback=continue_on_error)
                        results[provider.name] = result

                    except Exception as e:
                        error_context = ErrorContext(
                            category=ErrorCategory.EXTERNAL_SERVICE,
                            operation=f"fetch_{provider.name}",
                            provider=provider.name,
                            metadata={"provider_type": provider.kind.value},
                        )

                        self.logger.error(f"Failed to fetch models from {provider.name}", error_context=error_context)

                        failures[provider.name] = {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "timestamp": error_context.timestamp,
                        }

                        if not continue_on_error:
                            raise

            # Execute all fetches
            tasks = [fetch_single_provider(provider) for provider in providers]

            if continue_on_error:
                # Gather with return_exceptions to continue on individual failures
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Fail fast on first error
                await asyncio.gather(*tasks)

            # Log summary
            success_count = len(results)
            failure_count = len(failures)
            total_count = len(providers)

            self.logger.info(
                "Batch fetch completed",
                success_count=success_count,
                failure_count=failure_count,
                total_count=total_count,
                success_rate=f"{(success_count / total_count) * 100:.1f}%",
            )

            return {
                "results": results,
                "failures": failures,
                "summary": {
                    "total_providers": total_count,
                    "successful": success_count,
                    "failed": failure_count,
                    "success_rate": (success_count / total_count) * 100,
                },
            }

    async def close(self) -> None:
        """Close the enhanced fetcher."""
        await self.base_fetcher.close()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics including recovery stats."""
        base_stats = self.base_fetcher.stats()
        recovery_stats = self.error_recovery.get_recovery_stats()
        error_history = self.logger.get_error_history(10)

        return {
            "base_stats": base_stats,
            "recovery_stats": recovery_stats,
            "recent_errors": [error.to_dict() for error in error_history],
            "error_count": len(error_history),
        }


class EnhancedStorageManager:
    """Enhanced StorageManager with structured logging and error recovery."""

    def __init__(self, root_path: str | None = None) -> None:
        """Initialize enhanced storage with structured logging."""
        self.base_storage = StorageManager(root_path)
        self.logger = StructuredLogger("storage_manager")
        self.error_recovery = EnhancedErrorRecovery(self.logger)

        # Register fallback handlers
        self._register_fallback_handlers()

    def _register_fallback_handlers(self) -> None:
        """Register fallback handlers for storage operations."""

        def file_operation_fallback(error: Exception, **kwargs) -> Any:
            """Fallback for file operation errors."""
            # Try to create missing directories and retry
            if "file_path" in kwargs:
                file_path = Path(kwargs["file_path"])
                try:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created missing directory: {file_path.parent}")
                    return None  # Signal to retry original operation
                except Exception as mkdir_error:
                    self.logger.warning(f"Failed to create directory: {mkdir_error}")

            raise error  # Cannot recover

        self.error_recovery.register_fallback_handler(ErrorCategory.FILE_OPERATION, file_operation_fallback)

    def write_json_enhanced(self, filename: str, data: Any, **kwargs) -> bool:
        """Write JSON with enhanced error recovery."""
        return self.error_recovery.execute_with_fallback(
            operation=self.base_storage.write_json,
            operation_name="write_json",
            error_category=ErrorCategory.FILE_OPERATION,
            fallback_value=False,
            filename=filename,
            data=data,
            **kwargs,
        )

    def write_config_json_enhanced(self, filename: str, data: Any) -> bool:
        """Write config JSON with enhanced error recovery."""
        return self.error_recovery.execute_with_fallback(
            operation=self.base_storage.write_config_json,
            operation_name="write_config_json",
            error_category=ErrorCategory.FILE_OPERATION,
            fallback_value=False,
            filename=filename,
            data=data,
        )

    def read_json_enhanced(self, filename: str, directory: str) -> Any | None:
        """Read JSON with enhanced error recovery."""
        return self.error_recovery.execute_with_fallback(
            operation=self.base_storage.read_json,
            operation_name="read_json",
            error_category=ErrorCategory.FILE_OPERATION,
            fallback_value=None,
            filename=filename,
            directory=directory,
        )

    def batch_write_enhanced(self, operations: list[dict[str, Any]], continue_on_error: bool = True) -> dict[str, Any]:
        """
        Perform batch write operations with enhanced error handling.

        Args:
            operations: List of operation dictionaries with 'type', 'filename', 'data', etc.
            continue_on_error: Whether to continue processing other operations if one fails

        Returns:
            Dictionary with results and failure summary
        """
        with operation_context("batch_write", "storage_manager", operation_count=len(operations)):
            successful_operations = []
            failed_operations = []

            for i, operation in enumerate(operations):
                try:
                    op_type = operation.get("type")
                    filename = operation.get("filename")
                    data = operation.get("data")

                    if op_type == "json":
                        self.write_json_enhanced(filename, data, **operation.get("kwargs", {}))
                    elif op_type == "config_json":
                        self.write_config_json_enhanced(filename, data)
                    elif op_type == "yaml":
                        directory = operation.get("directory", "config")
                        self.base_storage.write_yaml(filename, data, directory=directory)
                    else:
                        msg = f"Unsupported operation type: {op_type}"
                        raise ValueError(msg)

                    successful_operations.append(
                        {"index": i, "type": op_type, "filename": filename, "status": "success"}
                    )

                except Exception as e:
                    error_context = ErrorContext(
                        category=ErrorCategory.FILE_OPERATION,
                        operation=f"batch_write_{op_type}",
                        metadata={"filename": filename, "operation_index": i, "operation_type": op_type},
                    )

                    self.logger.error(f"Batch operation {i} failed", error_context=error_context)

                    failed_operations.append(
                        {
                            "index": i,
                            "type": op_type,
                            "filename": filename,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                    )

                    if not continue_on_error:
                        break

            # Log summary
            success_count = len(successful_operations)
            failure_count = len(failed_operations)
            total_count = len(operations)

            self.logger.info(
                "Batch write completed",
                success_count=success_count,
                failure_count=failure_count,
                total_count=total_count,
                success_rate=f"{(success_count / total_count) * 100:.1f}%",
            )

            return {
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "summary": {
                    "total_operations": total_count,
                    "successful": success_count,
                    "failed": failure_count,
                    "success_rate": (success_count / total_count) * 100,
                },
            }


def create_production_logger(log_file_path: str | None = None, log_level: str = "INFO") -> StructuredLogger:
    """
    Create a production-ready structured logger with file output.

    Args:
        log_file_path: Path to log file (optional)
        log_level: Minimum log level to record

    Returns:
        Configured StructuredLogger instance
    """
    # Remove default handler
    logger.remove()

    # Add console handler with structured format
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        colorize=True,
    )

    # Add file handler if path provided
    if log_file_path:
        logger.add(
            log_file_path,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="1 GB",
            retention="30 days",
            compression="gzip",
        )

    return StructuredLogger("production")


async def run_enhanced_diagnostics(
    storage_manager: EnhancedStorageManager, fetcher: EnhancedModelFetcher
) -> dict[str, Any]:
    """
    Run comprehensive diagnostics with enhanced logging and error recovery.

    Args:
        storage_manager: Enhanced storage manager instance
        fetcher: Enhanced model fetcher instance

    Returns:
        Diagnostic results with comprehensive metrics
    """
    with operation_context("system_diagnostics", "diagnostics"):
        logger = StructuredLogger("diagnostics")
        diagnostics = {}

        # Storage diagnostics
        try:
            storage_stats = storage_manager.base_storage.get_file_stats()
            diagnostics["storage"] = {
                "status": "healthy",
                "file_stats": storage_stats,
                "total_files": sum(storage_stats.values()),
            }
            logger.info("Storage diagnostics completed", **storage_stats)

        except Exception as e:
            diagnostics["storage"] = {"status": "error", "error": str(e)}
            logger.error(
                "Storage diagnostics failed", error_context=ErrorContext(category=ErrorCategory.FILE_OPERATION)
            )

        # Fetcher diagnostics
        try:
            fetcher_stats = fetcher.get_stats()
            diagnostics["fetcher"] = {"status": "healthy", **fetcher_stats}
            logger.info("Fetcher diagnostics completed", **fetcher_stats)

        except Exception as e:
            diagnostics["fetcher"] = {"status": "error", "error": str(e)}
            logger.error("Fetcher diagnostics failed", error_context=ErrorContext(category=ErrorCategory.INTERNAL))

        # System health summary
        healthy_components = sum(1 for component in diagnostics.values() if component.get("status") == "healthy")
        total_components = len(diagnostics)

        diagnostics["summary"] = {
            "overall_health": "healthy" if healthy_components == total_components else "degraded",
            "healthy_components": healthy_components,
            "total_components": total_components,
            "health_percentage": (healthy_components / total_components) * 100,
        }

        logger.info(
            "System diagnostics completed",
            overall_health=diagnostics["summary"]["overall_health"],
            health_percentage=diagnostics["summary"]["health_percentage"],
        )

        return diagnostics
