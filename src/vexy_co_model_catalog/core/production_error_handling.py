"""
this_file: src/vexy_co_model_catalog/core/production_error_handling.py

Production-grade error handling with comprehensive logging and graceful degradation.
"""

import functools
import traceback
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger

F = TypeVar('F', bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """Error severity levels for categorization and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better organization and handling strategies."""
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    DEPENDENCY = "dependency"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Comprehensive error context information for production debugging."""
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    function_name: str
    file_path: str
    line_number: int
    timestamp: str
    user_action: str | None = None
    recovery_suggestion: str | None = None
    error_id: str | None = None
    additional_context: dict[str, Any] | None = None


class ProductionErrorHandler:
    """Production-grade error handler with logging, categorization, and recovery."""

    def __init__(self, log_file: Path | None = None) -> None:
        """Initialize error handler with optional log file."""
        self.error_counts: dict[str, int] = {}
        self.critical_errors: list[ErrorContext] = []

        if log_file:
            logger.add(
                log_file,
                rotation="10 MB",
                retention="30 days",
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                backtrace=True,
                diagnose=True
            )

    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on type and context."""
        error_type = type(error).__name__
        error_message = str(error).lower()

        # Network-related errors
        if any(keyword in error_message for keyword in [
            "connection", "timeout", "network", "dns", "http", "ssl", "certificate"
        ]) or error_type in ["ConnectionError", "TimeoutError", "HTTPError"]:
            return ErrorCategory.NETWORK

        # Filesystem errors
        if any(keyword in error_message for keyword in [
            "file not found", "permission denied", "disk", "directory", "path"
        ]) or error_type in ["FileNotFoundError", "PermissionError", "OSError"]:
            return ErrorCategory.FILESYSTEM

        # Configuration errors
        if any(keyword in error_message for keyword in [
            "config", "setting", "missing", "invalid", "malformed"
        ]) or error_type in ["KeyError", "AttributeError"]:
            return ErrorCategory.CONFIGURATION

        # Validation errors
        if any(keyword in error_message for keyword in [
            "validation", "invalid", "format", "schema", "type"
        ]) or error_type in ["ValueError", "TypeError"]:
            return ErrorCategory.VALIDATION

        # Dependency errors
        if any(keyword in error_message for keyword in [
            "import", "module", "package", "dependency"
        ]) or error_type in ["ImportError", "ModuleNotFoundError"]:
            return ErrorCategory.DEPENDENCY

        # System errors
        if any(keyword in error_message for keyword in [
            "memory", "system", "resource", "limit"
        ]) or error_type in ["MemoryError", "SystemError"]:
            return ErrorCategory.SYSTEM

        return ErrorCategory.UNKNOWN

    def determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on type and category."""
        error_type = type(error).__name__

        # Critical errors that could crash the application
        if error_type in ["MemoryError", "SystemError", "KeyboardInterrupt"]:
            return ErrorSeverity.CRITICAL

        # High severity errors that significantly impact functionality
        if category == ErrorCategory.DEPENDENCY or error_type in [
            "ImportError", "ModuleNotFoundError"
        ]:
            return ErrorSeverity.HIGH

        # Medium severity errors that impact specific operations
        if category in [ErrorCategory.NETWORK, ErrorCategory.FILESYSTEM]:
            return ErrorSeverity.MEDIUM

        # Low severity errors that can be handled gracefully
        return ErrorSeverity.LOW

    def generate_recovery_suggestion(self, error: Exception, category: ErrorCategory) -> str:
        """Generate user-friendly recovery suggestions."""
        error_message = str(error).lower()

        match category:
            case ErrorCategory.NETWORK:
                return "Check your internet connection and try again. If the issue persists, the service may be temporarily unavailable."

            case ErrorCategory.FILESYSTEM:
                if "permission denied" in error_message:
                    return "Check file permissions or run with appropriate privileges."
                if "not found" in error_message:
                    return "Verify the file or directory path exists and is accessible."
                return "Check file system permissions and available disk space."

            case ErrorCategory.CONFIGURATION:
                return "Review your configuration files for missing or invalid settings. Use the validate command to check configuration syntax."

            case ErrorCategory.VALIDATION:
                return "Check your input data format and ensure all required fields are provided with valid values."

            case ErrorCategory.DEPENDENCY:
                return "Install missing dependencies with 'uv add <package>' or check your Python environment."

            case ErrorCategory.SYSTEM:
                return "Check available system resources (memory, disk space) and restart if necessary."

            case ErrorCategory.USER_INPUT:
                return "Review the command syntax and provided arguments. Use --help for usage information."

            case _:
                return "Review the error details and try the operation again. Contact support if the issue persists."

    def create_error_context(
        self,
        error: Exception,
        user_action: str | None = None,
        additional_context: dict[str, Any] | None = None
    ) -> ErrorContext:
        """Create comprehensive error context."""
        # Get traceback information
        tb = traceback.extract_tb(error.__traceback__)
        if tb:
            last_frame = tb[-1]
            function_name = last_frame.name
            file_path = last_frame.filename
            line_number = last_frame.lineno
        else:
            function_name = "unknown"
            file_path = "unknown"
            line_number = 0

        category = self.categorize_error(error)
        severity = self.determine_severity(error, category)

        return ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            category=category,
            severity=severity,
            function_name=function_name,
            file_path=file_path,
            line_number=line_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_action=user_action,
            recovery_suggestion=self.generate_recovery_suggestion(error, category),
            error_id=str(uuid.uuid4())[:8],
            additional_context=additional_context or {}
        )

    def log_error(self, error_context: ErrorContext, _include_traceback: bool = True) -> None:
        """Log error with structured information."""
        # Update error counts
        error_key = f"{error_context.category.value}:{error_context.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Store critical errors
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.critical_errors.append(error_context)

        # Create structured log entry
        log_data = {
            "error_id": error_context.error_id,
            "error_type": error_context.error_type,
            "category": error_context.category.value,
            "severity": error_context.severity.value,
            "function": error_context.function_name,
            "file": Path(error_context.file_path).name,
            "line": error_context.line_number,
            "user_action": error_context.user_action,
            "recovery": error_context.recovery_suggestion,
            "count": self.error_counts[error_key],
            **error_context.additional_context
        }

        # Log with appropriate level based on severity
        log_message = f"[{error_context.error_id}] {error_context.error_message}"

        match error_context.severity:
            case ErrorSeverity.CRITICAL:
                logger.critical(log_message, extra=log_data)
            case ErrorSeverity.HIGH:
                logger.error(log_message, extra=log_data)
            case ErrorSeverity.MEDIUM:
                logger.warning(log_message, extra=log_data)
            case ErrorSeverity.LOW:
                logger.info(log_message, extra=log_data)

    def handle_error(
        self,
        error: Exception,
        user_action: str | None = None,
        additional_context: dict[str, Any] | None = None,
        reraise: bool = False
    ) -> ErrorContext | None:
        """Handle error with logging and context creation."""
        error_context = self.create_error_context(error, user_action, additional_context)
        self.log_error(error_context)

        if reraise:
            raise error

        return error_context


# Global error handler instance
_global_error_handler: ProductionErrorHandler | None = None


def get_error_handler() -> ProductionErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ProductionErrorHandler()
    return _global_error_handler


def production_error_handler(
    user_action: str | None = None,
    reraise: bool = False,
    additional_context: dict[str, Any] | None = None
):
    """Decorator for production error handling."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                context = additional_context or {}
                context.update({
                    "function_args": str(args)[:200] if args else None,
                    "function_kwargs": str(kwargs)[:200] if kwargs else None,
                })

                handler.handle_error(
                    e,
                    user_action or f"calling {func.__name__}",
                    context,
                    reraise=reraise
                )

                if not reraise:
                    # Return None or appropriate default for graceful degradation
                    return None

        return wrapper
    return decorator


@contextmanager
def error_context(
    operation: str,
    reraise: bool = False,
    additional_context: dict[str, Any] | None = None
) -> Iterator[None]:
    """Context manager for error handling."""
    try:
        yield
    except Exception as e:
        handler = get_error_handler()
        handler.handle_error(e, operation, additional_context, reraise=reraise)


def handle_critical_error(error: Exception, context: str = "unknown operation") -> None:
    """Handle critical errors that require immediate attention."""
    handler = get_error_handler()
    handler.handle_error(
        error,
        user_action=context,
        additional_context={"critical": True}
    )

    # Print user-friendly error message


def get_error_summary() -> dict[str, Any]:
    """Get summary of error statistics."""
    handler = get_error_handler()
    return {
        "error_counts": dict(handler.error_counts),
        "critical_errors": len(handler.critical_errors),
        "total_errors": sum(handler.error_counts.values()),
    }
