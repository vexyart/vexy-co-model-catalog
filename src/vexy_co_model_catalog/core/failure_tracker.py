"""
this_file: src/vexy_co_model_catalog/core/failure_tracker.py

Failed provider tracking system to maintain reliability statistics.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from vexy_co_model_catalog.core.storage import StorageManager


class FailureTracker:
    """Tracks failed provider requests and maintains reliability statistics."""

    def __init__(self, storage: StorageManager) -> None:
        """Initialize failure tracker with storage manager."""
        self.storage = storage
        self._failed_providers: dict[str, dict[str, Any]] = {}
        self._load_failed_providers()

    def _load_failed_providers(self) -> None:
        """Load failed provider data from storage."""
        try:
            data = self.storage.read_json("failed_providers", directory="extra")
            if data:
                self._failed_providers = data
                logger.debug(f"Loaded failure tracking for {len(self._failed_providers)} providers")
            else:
                self._failed_providers = {}
        except Exception as e:
            logger.warning(f"Failed to load provider failure tracking: {e}")
            self._failed_providers = {}

    def _save_failed_providers(self) -> None:
        """Save failed provider data to storage."""
        try:
            self.storage.write_extra("failed_providers", self._failed_providers)
        except Exception as e:
            logger.error(f"Failed to save provider failure tracking: {e}")

    def mark_provider_failed(
        self,
        provider_name: str,
        error_message: str,
        response_status: int | None = None,
        response_headers: dict[str, str] | None = None,
    ) -> None:
        """Mark a provider as failed with error details."""
        timestamp = datetime.now(timezone.utc).isoformat()

        if provider_name not in self._failed_providers:
            self._failed_providers[provider_name] = {
                "failed": True,
                "first_failure": timestamp,
                "failure_count": 0,
                "last_success": None,
                "consecutive_failures": 0,
                "errors": [],
            }

        provider_data = self._failed_providers[provider_name]
        provider_data["failed"] = True
        provider_data["last_failure"] = timestamp
        provider_data["failure_count"] = provider_data.get("failure_count", 0) + 1
        provider_data["consecutive_failures"] = provider_data.get("consecutive_failures", 0) + 1

        # Add error details
        error_entry = {
            "timestamp": timestamp,
            "error": error_message,
            "status_code": response_status,
        }

        if response_headers and "date" in response_headers:
            error_entry["response_date"] = response_headers["date"]

        provider_data["errors"].append(error_entry)

        # Keep only the last 10 errors to avoid excessive storage
        if len(provider_data["errors"]) > 10:
            provider_data["errors"] = provider_data["errors"][-10:]

        logger.warning(
            f"Provider {provider_name} marked as failed (consecutive: "
            f"{provider_data['consecutive_failures']}, total: {provider_data['failure_count']}): "
            f"{error_message}"
        )

        self._save_failed_providers()

    def mark_provider_success(self, provider_name: str, response_headers: dict[str, str] | None = None) -> None:
        """Mark a provider as successful, clearing failure status."""
        timestamp = datetime.now(timezone.utc).isoformat()

        if provider_name not in self._failed_providers:
            self._failed_providers[provider_name] = {
                "failed": False,
                "first_success": timestamp,
                "failure_count": 0,
                "success_count": 0,
                "last_failure": None,
                "consecutive_failures": 0,
                "errors": [],
            }

        provider_data = self._failed_providers[provider_name]
        was_failed = provider_data.get("failed", False)

        provider_data["failed"] = False
        provider_data["last_success"] = timestamp
        provider_data["success_count"] = provider_data.get("success_count", 0) + 1
        provider_data["consecutive_failures"] = 0  # Reset consecutive failures

        if response_headers and "date" in response_headers:
            provider_data["last_success_date"] = response_headers["date"]

        if was_failed:
            consecutive_failures = provider_data.get("consecutive_failures", 0)
            logger.info(f"Provider {provider_name} recovered after {consecutive_failures} consecutive failures")

        self._save_failed_providers()

    def is_provider_failed(self, provider_name: str) -> bool:
        """Check if a provider is currently marked as failed."""
        return self._failed_providers.get(provider_name, {}).get("failed", False)

    def get_provider_failure_info(self, provider_name: str) -> dict[str, Any]:
        """Get failure information for a specific provider."""
        return self._failed_providers.get(
            provider_name,
            {
                "failed": False,
                "failure_count": 0,
                "success_count": 0,
                "consecutive_failures": 0,
                "last_success": None,
                "last_failure": None,
                "errors": [],
            },
        )

    def get_failed_providers(self) -> list[str]:
        """Get list of currently failed provider names."""
        return [name for name, data in self._failed_providers.items() if data.get("failed", False)]

    def get_failure_summary(self) -> dict[str, Any]:
        """Get summary statistics of provider failures."""
        total_providers = len(self._failed_providers)
        failed_providers = len(self.get_failed_providers())

        total_failures = sum(data.get("failure_count", 0) for data in self._failed_providers.values())

        total_successes = sum(data.get("success_count", 0) for data in self._failed_providers.values())

        return {
            "total_providers_tracked": total_providers,
            "currently_failed": failed_providers,
            "currently_working": total_providers - failed_providers,
            "total_failures": total_failures,
            "total_successes": total_successes,
            "overall_success_rate": (
                total_successes / (total_failures + total_successes) if (total_failures + total_successes) > 0 else 0.0
            ),
        }

    def cleanup_old_errors(self, days_to_keep: int = 7) -> None:
        """Remove error entries older than specified days."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        cleaned_count = 0

        for _provider_name, provider_data in self._failed_providers.items():
            if "errors" in provider_data:
                original_count = len(provider_data["errors"])

                provider_data["errors"] = [
                    error
                    for error in provider_data["errors"]
                    if datetime.fromisoformat(error["timestamp"]) > cutoff_date
                ]

                cleaned_count += original_count - len(provider_data["errors"])

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old error entries")
            self._save_failed_providers()

    def reset_provider_failures(self, provider_name: str) -> None:
        """Reset failure tracking for a specific provider."""
        if provider_name in self._failed_providers:
            del self._failed_providers[provider_name]
            logger.info(f"Reset failure tracking for provider {provider_name}")
            self._save_failed_providers()

    def reset_all_failures(self) -> None:
        """Reset all failure tracking data."""
        self._failed_providers.clear()
        logger.info("Reset all provider failure tracking")
        self._save_failed_providers()
