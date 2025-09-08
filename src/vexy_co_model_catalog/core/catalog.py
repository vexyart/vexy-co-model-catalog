"""
this_file: src/vexy_co_model_catalog/core/catalog.py

Minimal catalog and registry scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from .storage import StorageManager


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}

    def register(self, name: str, provider: Any) -> None:
        self._providers[name] = provider
        logger.debug(f"Registered provider: {name}")

    def get(self, name: str) -> Any:
        return self._providers[name]

    def list_names(self) -> list[str]:
        return list(self._providers.keys())

    def remove(self, name: str) -> bool:
        return self._providers.pop(name, None) is not None


@dataclass
class ModelCatalog:
    storage: StorageManager | None = None
    registry: ProviderRegistry | None = None

    def __post_init__(self) -> None:
        self.storage = self.storage or StorageManager()
        self.registry = self.registry or ProviderRegistry()
        self._failed_providers: dict[str, str] = {}

    def get_failed_providers(self) -> dict[str, str]:
        return dict(self._failed_providers)
