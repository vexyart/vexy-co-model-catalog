"""
this_file: src/vexy_co_model_catalog/core/enhanced_logging.py

Enhanced structured logging and error recovery patterns for production debugging and monitoring.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Generator


class LogLevel(Enum):
    """Enhanced log levels for structured logging."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorSeverity(Enum):
    """Error severity levels for production monitoring."""

    LOW = "low"  # Minor issues, graceful degradation possible
    MEDIUM = "medium"  # Notable issues, some functionality affected
    HIGH = "high"  # Major issues, significant functionality lost
    CRITICAL = "critical"  # System-wide issues, service degraded


class ErrorCategory(Enum):
    """Error categories for better classification and handling."""

    NETWORK = "network"  # HTTP, DNS, connectivity issues
    AUTHENTICATION = "authentication"  # API key, permission issues
    RATE_LIMITING = "rate_limiting"  # Rate limit exceeded
    DATA_VALIDATION = "data_validation"  # Invalid data format, schema issues
    FILE_OPERATION = "file_operation"  # File I/O, storage issues
    CONFIGURATION = "configuration"  # Config validation, missing settings
    EXTERNAL_SERVICE = "external_service"  # Third-party service issues
    INTERNAL = "internal"  # Internal logic errors
    UNKNOWN = "unknown"  # Unclassified errors


@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and monitoring."""

    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    operation: str | None = None
    provider: str | None = None
    component: str | None = None
    retry_attempt: int = 0
    max_retries: int = 0
    correlation_id: str | None = None
    user_action: str | None = None
    recovery_suggestion: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["category"] = self.category.value
        result["severity"] = self.severity.value
        return result


@dataclass
class OperationContext:
    """Context for tracking operations across the system."""

    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_name: str = ""
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    component: str | None = None
    provider: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# Context variables for maintaining operation context across async operations
_current_operation_context: contextvars.ContextVar[OperationContext | None] = contextvars.ContextVar(
    "current_operation_context", default=None
)
_current_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_correlation_id", default=None
)


class StructuredLogger:
    """Enhanced structured logger with context awareness and production debugging features."""

    def __init__(self, component: str = "model_catalog") -> None:
        """Initialize structured logger for a specific component."""
        self.component = component
        self._error_history: list[ErrorContext] = []
        self._max_error_history = 100

    def _get_structured_record(self, level: LogLevel, message: str, **kwargs) -> dict[str, Any]:
        """Create structured log record with full context."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "component": self.component,
            "message": message,
            "process_id": id(asyncio.current_task()) if asyncio.current_task() else None,
        }

        # Add operation context if available
        operation_ctx = _current_operation_context.get()
        if operation_ctx:
            record["operation"] = operation_ctx.to_dict()

        # Add correlation ID if available
        correlation_id = _current_correlation_id.get()
        if correlation_id:
            record["correlation_id"] = correlation_id

        # Add additional context
        if kwargs:
            record["context"] = kwargs

        return record

    def trace(self, message: str, **kwargs) -> None:
        """Log trace level message with structured context."""
        record = self._get_structured_record(LogLevel.TRACE, message, **kwargs)
        logger.trace(json.dumps(record, default=str))

    def debug(self, message: str, **kwargs) -> None:
        """Log debug level message with structured context."""
        record = self._get_structured_record(LogLevel.DEBUG, message, **kwargs)
        logger.debug(json.dumps(record, default=str))

    def info(self, message: str, **kwargs) -> None:
        """Log info level message with structured context."""
        record = self._get_structured_record(LogLevel.INFO, message, **kwargs)
        logger.info(json.dumps(record, default=str))

    def warning(self, message: str, **kwargs) -> None:
        """Log warning level message with structured context."""
        record = self._get_structured_record(LogLevel.WARNING, message, **kwargs)
        logger.warning(json.dumps(record, default=str))

    def error(self, message: str, error_context: ErrorContext | None = None, **kwargs) -> None:
        """Log error with enhanced context and recovery suggestions."""
        if error_context:
            # Add error context to history for debugging
            self._error_history.append(error_context)
            if len(self._error_history) > self._max_error_history:
                self._error_history.pop(0)

            kwargs["error_context"] = error_context.to_dict()

        record = self._get_structured_record(LogLevel.ERROR, message, **kwargs)
        logger.error(json.dumps(record, default=str))

    def critical(self, message: str, error_context: ErrorContext | None = None, **kwargs) -> None:
        """Log critical error with enhanced context."""
        if error_context:
            error_context.severity = ErrorSeverity.CRITICAL
            kwargs["error_context"] = error_context.to_dict()

        record = self._get_structured_record(LogLevel.CRITICAL, message, **kwargs)
        logger.critical(json.dumps(record, default=str))

    def log_operation_start(self, operation_name: str, **metadata) -> OperationContext:
        """Log the start of an operation and return context."""
        operation_ctx = OperationContext(operation_name=operation_name, component=self.component, metadata=metadata)

        self.info(f"Operation started: {operation_name}", operation_id=operation_ctx.operation_id, **metadata)

        return operation_ctx

    def log_operation_success(
        self, operation_ctx: OperationContext, result_summary: str | None = None, **metadata
    ) -> None:
        """Log successful operation completion."""
        duration = self._calculate_duration(operation_ctx.start_time)

        self.info(
            f"Operation completed successfully: {operation_ctx.operation_name}",
            operation_id=operation_ctx.operation_id,
            duration_ms=duration,
            result_summary=result_summary,
            **metadata,
        )

    def log_operation_failure(
        self, operation_ctx: OperationContext, error: Exception, error_context: ErrorContext | None = None, **metadata
    ) -> None:
        """Log failed operation with detailed error context."""
        duration = self._calculate_duration(operation_ctx.start_time)

        if not error_context:
            error_context = self._create_default_error_context(operation_ctx)

        self._enrich_error_context_with_details(error_context, error, duration, metadata)

        self.error(
            f"Operation failed: {operation_ctx.operation_name}",
            error_context=error_context,
            operation_id=operation_ctx.operation_id,
            duration_ms=duration,
        )

    def _create_default_error_context(self, operation_ctx: OperationContext) -> ErrorContext:
        """Create default error context from operation context."""
        return ErrorContext(
            operation=operation_ctx.operation_name,
            component=self.component,
            correlation_id=operation_ctx.operation_id,
        )

    def _enrich_error_context_with_details(
        self, error_context: ErrorContext, error: Exception, duration: float, metadata: dict
    ) -> None:
        """Enrich error context with error details and metadata."""
        error_context.metadata.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "operation_duration_ms": duration,
                **metadata,
            }
        )

    def get_error_history(self, limit: int = 50) -> list[ErrorContext]:
        """Get recent error history for debugging."""
        return self._error_history[-limit:]

    def _calculate_duration(self, start_time: str) -> float:
        """Calculate duration in milliseconds from ISO timestamp."""
        try:
            start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - start).total_seconds() * 1000
        except (ValueError, AttributeError):
            return 0.0


class EnhancedErrorRecovery:
    """Enhanced error recovery patterns with fallback mechanisms and structured logging."""

    def __init__(self, logger_instance: StructuredLogger | None = None) -> None:
        """Initialize error recovery with structured logging."""
        self.logger = logger_instance or StructuredLogger("error_recovery")
        self._fallback_handlers: dict[ErrorCategory, list[callable]] = {}
        self._recovery_stats: dict[str, int] = {}

    def register_fallback_handler(self, category: ErrorCategory, handler: callable) -> None:
        """Register a fallback handler for specific error categories."""
        if category not in self._fallback_handlers:
            self._fallback_handlers[category] = []
        self._fallback_handlers[category].append(handler)

        self.logger.debug(f"Registered fallback handler for {category.value}", handler_name=handler.__name__)

    async def execute_with_fallback(
        self,
        operation: callable,
        operation_name: str,
        error_category: ErrorCategory = ErrorCategory.UNKNOWN,
        fallback_value: Any = None,
        enable_fallback: bool = True,
        **operation_kwargs,
    ) -> Any:
        """
        Execute operation with comprehensive error recovery and fallback mechanisms.

        Args:
            operation: Function to execute
            operation_name: Name for logging and metrics
            error_category: Category of operation for targeted recovery
            fallback_value: Default value to return if all recovery fails
            enable_fallback: Whether to enable fallback mechanisms
            **operation_kwargs: Arguments to pass to operation

        Returns:
            Operation result or fallback value
        """
        operation_ctx = self.logger.log_operation_start(operation_name)
        correlation_id = operation_ctx.operation_id

        try:
            # Set correlation context
            _current_correlation_id.set(correlation_id)
            _current_operation_context.set(operation_ctx)

            # Execute primary operation
            result = (
                await operation(**operation_kwargs)
                if asyncio.iscoroutinefunction(operation)
                else operation(**operation_kwargs)
            )

            self.logger.log_operation_success(operation_ctx, "Primary operation succeeded")
            return result

        except Exception as primary_error:
            error_context = ErrorContext(
                category=error_category,
                operation=operation_name,
                correlation_id=correlation_id,
                component=self.logger.component,
            )

            # Classify error and determine severity
            self._classify_error(primary_error, error_context)

            self.logger.log_operation_failure(operation_ctx, primary_error, error_context)

            # Attempt fallback recovery if enabled
            if enable_fallback and error_category in self._fallback_handlers:
                try:
                    fallback_result = await self._attempt_fallback_recovery(
                        error_category, operation_name, primary_error, correlation_id, **operation_kwargs
                    )

                    self.logger.info(
                        f"Fallback recovery successful for {operation_name}",
                        correlation_id=correlation_id,
                        fallback_type="handler_recovery",
                    )

                    self._recovery_stats[f"fallback_success_{error_category.value}"] = (
                        self._recovery_stats.get(f"fallback_success_{error_category.value}", 0) + 1
                    )

                    return fallback_result

                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback recovery failed for {operation_name}",
                        error_context=ErrorContext(
                            category=ErrorCategory.INTERNAL,
                            operation=f"{operation_name}_fallback",
                            correlation_id=correlation_id,
                            metadata={"original_error": str(primary_error), "fallback_error": str(fallback_error)},
                        ),
                    )

            # Final fallback to default value
            if fallback_value is not None:
                self.logger.warning(
                    f"Using default fallback value for {operation_name}",
                    correlation_id=correlation_id,
                    fallback_value=str(fallback_value)[:100],
                )  # Truncate for logging

                self._recovery_stats[f"default_fallback_{error_category.value}"] = (
                    self._recovery_stats.get(f"default_fallback_{error_category.value}", 0) + 1
                )

                return fallback_value

            # No recovery possible, re-raise original error
            self._recovery_stats[f"recovery_failed_{error_category.value}"] = (
                self._recovery_stats.get(f"recovery_failed_{error_category.value}", 0) + 1
            )

            raise primary_error

        finally:
            # Clear context
            _current_correlation_id.set(None)
            _current_operation_context.set(None)

    async def _attempt_fallback_recovery(
        self, category: ErrorCategory, operation_name: str, original_error: Exception, correlation_id: str, **kwargs
    ) -> Any:
        """Attempt recovery using registered fallback handlers."""
        handlers = self._fallback_handlers.get(category, [])

        for i, handler in enumerate(handlers):
            try:
                self.logger.debug(
                    f"Attempting fallback handler {i + 1}/{len(handlers)} for {operation_name}",
                    correlation_id=correlation_id,
                    handler_name=handler.__name__,
                )

                if asyncio.iscoroutinefunction(handler):
                    result = await handler(original_error, **kwargs)
                else:
                    result = handler(original_error, **kwargs)

                return result

            except Exception as handler_error:
                self.logger.warning(
                    f"Fallback handler {handler.__name__} failed",
                    correlation_id=correlation_id,
                    handler_error=str(handler_error),
                )
                continue

        msg = f"All fallback handlers failed for {category.value}"
        raise Exception(msg)

    def _classify_error(self, error: Exception, error_context: ErrorContext) -> None:
        """Classify error and set appropriate severity and recovery suggestions."""
        error_message = str(error).lower()
        classification = self._find_error_classification(error_message)

        error_context.category = classification["category"]
        error_context.severity = classification["severity"]
        error_context.recovery_suggestion = classification["recovery_suggestion"]
        error_context.user_action = classification["user_action"]

    def _find_error_classification(self, error_message: str) -> dict:
        """Find the appropriate error classification based on error message keywords."""
        classifications = [
            {
                "keywords": ["connection", "timeout", "network", "dns"],
                "category": ErrorCategory.NETWORK,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestion": "Check network connectivity and retry operation",
                "user_action": "Verify internet connection and API endpoint availability",
            },
            {
                "keywords": ["unauthorized", "forbidden", "auth", "api key", "token"],
                "category": ErrorCategory.AUTHENTICATION,
                "severity": ErrorSeverity.HIGH,
                "recovery_suggestion": "Verify API credentials and permissions",
                "user_action": "Check API key configuration and validity",
            },
            {
                "keywords": ["rate limit", "too many requests", "429"],
                "category": ErrorCategory.RATE_LIMITING,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestion": "Implement exponential backoff and reduce request frequency",
                "user_action": "Wait before retrying and consider reducing concurrency",
            },
            {
                "keywords": ["file", "directory", "permission denied", "disk"],
                "category": ErrorCategory.FILE_OPERATION,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestion": "Check file permissions and disk space",
                "user_action": "Verify write permissions and available storage",
            },
            {
                "keywords": ["json", "yaml", "toml", "invalid", "decode", "parse"],
                "category": ErrorCategory.DATA_VALIDATION,
                "severity": ErrorSeverity.MEDIUM,
                "recovery_suggestion": "Validate data format and schema compliance",
                "user_action": "Check data format and fix any syntax errors",
            },
            {
                "keywords": ["config", "setting", "environment", "missing"],
                "category": ErrorCategory.CONFIGURATION,
                "severity": ErrorSeverity.HIGH,
                "recovery_suggestion": "Review configuration settings and environment variables",
                "user_action": "Check configuration files and required environment variables",
            },
        ]

        # Find first matching classification
        for classification in classifications:
            if any(keyword in error_message for keyword in classification["keywords"]):
                return classification

        # Default classification for unknown errors
        return {
            "category": ErrorCategory.UNKNOWN,
            "severity": ErrorSeverity.MEDIUM,
            "recovery_suggestion": "Review error details and check system logs",
            "user_action": "Contact support with error details if issue persists",
        }

    def get_recovery_stats(self) -> dict[str, int]:
        """Get recovery statistics for monitoring."""
        return dict(self._recovery_stats)

    def reset_recovery_stats(self) -> None:
        """Reset recovery statistics."""
        self._recovery_stats.clear()


@contextmanager
def operation_context(operation_name: str, component: str = "model_catalog", **metadata: Any) -> Generator[ErrorContext, None, None]:
    """Context manager for tracking operations with structured logging."""
    logger_instance = StructuredLogger(component)
    operation_ctx = logger_instance.log_operation_start(operation_name, **metadata)

    try:
        _current_operation_context.set(operation_ctx)
        _current_correlation_id.set(operation_ctx.operation_id)
        yield operation_ctx
        logger_instance.log_operation_success(operation_ctx)

    except Exception as e:
        logger_instance.log_operation_failure(operation_ctx, e)
        raise
    finally:
        _current_operation_context.set(None)
        _current_correlation_id.set(None)


def get_current_correlation_id() -> str | None:
    """Get current correlation ID for cross-component tracing."""
    return _current_correlation_id.get()


def get_current_operation_context() -> OperationContext | None:
    """Get current operation context."""
    return _current_operation_context.get()


# Default instances for easy use
default_logger = StructuredLogger()
default_error_recovery = EnhancedErrorRecovery(default_logger)
