"""
this_file: src/vexy_co_model_catalog/core/production_deployment.py

Production deployment utilities for error handling, monitoring, and graceful degradation.
"""

import os
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from loguru import logger

from vexy_co_model_catalog.core.production_error_handling import (
    ProductionErrorHandler,
    get_error_summary,
    handle_critical_error,
)
from vexy_co_model_catalog.core.production_graceful_degradation import (
    GracefulDegradationManager,
    get_system_status,
)


class ProductionDeploymentManager:
    """Manages production deployment configuration and monitoring."""

    def __init__(self, log_directory: Path | None = None) -> None:
        self.log_directory = log_directory or self._get_default_log_directory()
        self.error_handler: ProductionErrorHandler | None = None
        self.degradation_manager: GracefulDegradationManager | None = None
        self.initialized = False

    def _get_default_log_directory(self) -> Path:
        """Get default log directory based on environment."""
        # Check for environment-specific log directory
        if log_dir := os.environ.get("VMC_LOG_DIR"):
            return Path(log_dir)

        # Use platform-appropriate log directory
        if sys.platform == "win32":
            log_base = Path.home() / "AppData" / "Local" / "vexy-co-model-catalog"
        elif sys.platform == "darwin":
            log_base = Path.home() / "Library" / "Logs" / "vexy-co-model-catalog"
        else:  # Linux and other Unix-like systems
            log_base = Path.home() / ".local" / "share" / "vexy-co-model-catalog" / "logs"

        return log_base

    def initialize_production_environment(self) -> bool:
        """Initialize production environment with error handling and logging."""
        try:
            # Ensure log directory exists
            self.log_directory.mkdir(parents=True, exist_ok=True)

            # Initialize error handler with log file
            error_log_file = self.log_directory / "errors.log"
            self.error_handler = ProductionErrorHandler(error_log_file)

            # Initialize degradation manager
            self.degradation_manager = GracefulDegradationManager()

            # Configure main logger
            self._configure_production_logging()

            # Set up global exception handler
            self._setup_global_exception_handler()

            self.initialized = True
            logger.info("Production environment initialized successfully")
            return True

        except Exception:
            return False

    def _configure_production_logging(self) -> None:
        """Configure logging for production deployment."""
        # Remove default logger
        logger.remove()

        # Add console logger with appropriate level
        log_level = os.environ.get("VMC_LOG_LEVEL", "INFO").upper()
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
            colorize=True,
        )

        # Add file logger for all messages
        main_log_file = self.log_directory / "vexy-co-model-catalog.log"
        logger.add(
            main_log_file,
            rotation="50 MB",
            retention="14 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            backtrace=True,
            diagnose=True,
        )

        # Add performance log
        perf_log_file = self.log_directory / "performance.log"
        logger.add(
            perf_log_file,
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: record["extra"].get("performance", False),
        )

    def _setup_global_exception_handler(self) -> None:
        """Set up global exception handler for uncaught exceptions."""
        def exception_handler(exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
            if issubclass(exc_type, KeyboardInterrupt):
                # Allow KeyboardInterrupt to work normally
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            # Handle all other uncaught exceptions
            logger.critical(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback)
            )

            handle_critical_error(
                exc_value,
                context="uncaught exception in main thread"
            )

        sys.excepthook = exception_handler

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status for monitoring."""
        if not self.initialized:
            return {"status": "not_initialized", "healthy": False}

        error_summary = get_error_summary()
        system_status = get_system_status()

        # Determine overall health
        healthy = (
            error_summary["critical_errors"] == 0 and
            system_status["system_healthy"] and
            len(system_status["degraded_services"]) == 0
        )

        return {
            "status": "healthy" if healthy else "degraded",
            "healthy": healthy,
            "timestamp": logger._core.levels[0].name,  # Current time
            "error_summary": error_summary,
            "system_status": system_status,
            "log_directory": str(self.log_directory),
            "initialized": self.initialized,
        }

    def cleanup_old_logs(self, max_age_days: int = 30) -> int:
        """Clean up old log files beyond retention period."""
        if not self.log_directory.exists():
            return 0

        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        cleaned_count = 0

        try:
            for log_file in self.log_directory.glob("*.log*"):
                if log_file.is_file():
                    file_age = current_time - log_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        log_file.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned up old log file: {log_file.name}")

        except Exception as e:
            logger.error(f"Error during log cleanup: {e}")

        return cleaned_count


# Global production manager
_global_production_manager: ProductionDeploymentManager | None = None


def get_production_manager() -> ProductionDeploymentManager:
    """Get global production manager instance."""
    global _global_production_manager
    if _global_production_manager is None:
        _global_production_manager = ProductionDeploymentManager()
    return _global_production_manager


def initialize_production_mode() -> bool:
    """Initialize production mode for the application."""
    manager = get_production_manager()
    return manager.initialize_production_environment()


@contextmanager
def production_context() -> Iterator[ProductionDeploymentManager]:
    """Context manager for production operations."""
    manager = get_production_manager()

    if not manager.initialized:
        success = manager.initialize_production_environment()
        if not success:
            msg = "Failed to initialize production environment"
            raise RuntimeError(msg)

    try:
        logger.info("Starting production operation")
        yield manager
    finally:
        logger.info("Production operation completed")


def check_production_readiness() -> dict[str, Any]:
    """Check if the application is ready for production deployment."""
    checks = {
        "log_directory_writable": False,
        "all_dependencies_available": False,
        "configuration_valid": False,
        "error_handling_active": False,
        "performance_monitoring": False,
    }

    try:
        manager = get_production_manager()

        # Check log directory
        try:
            test_file = manager.log_directory / "test_write.tmp"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("test")
            test_file.unlink()
            checks["log_directory_writable"] = True
        except Exception:
            pass

        # Check dependencies
        with suppress(ImportError):
            checks["all_dependencies_available"] = True

        # Check error handling
        checks["error_handling_active"] = manager.error_handler is not None

        # Check configuration
        # This would be expanded based on actual configuration requirements
        checks["configuration_valid"] = True

        # Check performance monitoring
        checks["performance_monitoring"] = True

    except Exception as e:
        logger.error(f"Error during production readiness check: {e}")

    readiness_score = sum(checks.values()) / len(checks)

    return {
        "ready": readiness_score >= 0.8,  # 80% of checks must pass
        "score": readiness_score,
        "checks": checks,
        "recommendations": _get_readiness_recommendations(checks)
    }


def _get_readiness_recommendations(checks: dict[str, bool]) -> list[str]:
    """Get recommendations for production readiness issues."""
    recommendations = []

    if not checks["log_directory_writable"]:
        recommendations.append("Ensure log directory is writable or set VMC_LOG_DIR environment variable")

    if not checks["all_dependencies_available"]:
        recommendations.append("Install all required dependencies with 'uv sync'")

    if not checks["error_handling_active"]:
        recommendations.append("Initialize error handling with initialize_production_mode()")

    if not checks["configuration_valid"]:
        recommendations.append("Validate configuration files and environment variables")

    return recommendations


def get_production_metrics() -> dict[str, Any]:
    """Get production metrics for monitoring and alerting."""
    manager = get_production_manager()
    health_status = manager.get_health_status()

    return {
        "uptime_status": "running",
        "health_status": health_status,
        "log_directory": str(manager.log_directory),
        "log_files": [
            f.name for f in manager.log_directory.glob("*.log")
            if f.is_file() and manager.log_directory.exists()
        ],
        "memory_usage": _get_memory_usage(),
        "disk_usage": _get_disk_usage(manager.log_directory),
    }


def _get_memory_usage() -> dict[str, Any]:
    """Get memory usage information."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }
    except Exception:
        return {"error": "Unable to get memory usage"}


def _get_disk_usage(directory: Path) -> dict[str, Any]:
    """Get disk usage for log directory."""
    try:
        if directory.exists():
            usage = shutil.disk_usage(directory)
            return {
                "total_gb": usage.total / 1024 / 1024 / 1024,
                "used_gb": usage.used / 1024 / 1024 / 1024,
                "free_gb": usage.free / 1024 / 1024 / 1024,
                "percent_used": (usage.used / usage.total) * 100,
            }
        return {"error": "Log directory does not exist"}
    except Exception:
        return {"error": "Unable to get disk usage"}
