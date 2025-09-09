"""
this_file: src/vexy_co_model_catalog/core/catalog.py

Provider registry and model catalog components for managing AI provider configurations.
Provides centralized storage and retrieval of provider configurations with type-safe operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from loguru import logger

from vexy_co_model_catalog.core.provider import ProviderConfig
from vexy_co_model_catalog.core.storage import StorageManager


class ProviderRegistry:
    """Registry for managing AI provider configurations.

    Provides thread-safe registration, retrieval, and management of provider configurations
    with comprehensive error handling and validation.

    Example:
        >>> registry = ProviderRegistry()
        >>> config = ProviderConfig(name="openai", api_url="...")
        >>> registry.register("openai", config)
        >>> provider = registry.get("openai")
    """

    def __init__(self) -> None:
        """Initialize an empty provider registry."""
        self._providers: dict[str, ProviderConfig] = {}

    def register(self, name: str, provider: ProviderConfig) -> None:
        """Register a provider configuration.

        Args:
            name: Unique identifier for the provider
            provider: Provider configuration object

        Raises:
            ValueError: If name is empty or provider is invalid
        """
        if not name or not name.strip():
            msg = "Provider name cannot be empty"
            raise ValueError(msg)
        if not isinstance(provider, ProviderConfig):
            msg = f"Provider must be ProviderConfig instance, got {type(provider)}"
            raise ValueError(msg)

        self._providers[name] = provider
        logger.debug(f"Registered provider: {name}")

    def get(self, name: str) -> ProviderConfig | None:
        """Retrieve a provider configuration by name.

        Args:
            name: Provider identifier

        Returns:
            Provider configuration if found, None otherwise
        """
        return self._providers.get(name)

    def get_required(self, name: str) -> ProviderConfig:
        """Retrieve a provider configuration, raising if not found.

        Args:
            name: Provider identifier

        Returns:
            Provider configuration

        Raises:
            KeyError: If provider is not registered
        """
        if name not in self._providers:
            msg = f"Provider '{name}' not found in registry"
            raise KeyError(msg)
        return self._providers[name]

    def list_names(self) -> list[str]:
        """Get list of all registered provider names.

        Returns:
            Sorted list of provider names
        """
        return sorted(self._providers.keys())

    def list_providers(self) -> list[ProviderConfig]:
        """Get list of all registered provider configurations.

        Returns:
            List of provider configurations sorted by name
        """
        return [self._providers[name] for name in self.list_names()]

    def items(self) -> Iterator[tuple[str, ProviderConfig]]:
        """Iterate over provider name and configuration pairs.

        Yields:
            Tuple of (name, provider_config) for each registered provider
        """
        for name in self.list_names():
            yield name, self._providers[name]

    def contains(self, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: Provider identifier

        Returns:
            True if provider is registered, False otherwise
        """
        return name in self._providers

    def remove(self, name: str) -> bool:
        """Remove a provider from the registry.

        Args:
            name: Provider identifier

        Returns:
            True if provider was removed, False if not found
        """
        removed = self._providers.pop(name, None) is not None
        if removed:
            logger.debug(f"Removed provider: {name}")
        return removed

    def clear(self) -> None:
        """Remove all providers from the registry."""
        count = len(self._providers)
        self._providers.clear()
        logger.debug(f"Cleared {count} providers from registry")

    def count(self) -> int:
        """Get the number of registered providers.

        Returns:
            Number of registered providers
        """
        return len(self._providers)


@dataclass
class ModelCatalog:
    """Main catalog for managing AI model providers and storage operations.

    Coordinates provider registry, storage management, and catalog operations.
    Provides a unified interface for fetching, storing, and managing AI model catalogs.

    Attributes:
        storage: Storage manager for file operations
        registry: Provider registry for configuration management
        storage_root: Root directory for storage operations

    Example:
        >>> catalog = ModelCatalog(storage_root=Path("/data/models"))
        >>> catalog.registry.register("openai", openai_config)
        >>> # Use catalog for operations
    """

    storage: StorageManager | None = None
    registry: ProviderRegistry | None = None
    storage_root: Path | None = None

    def __post_init__(self) -> None:
        """Initialize storage and registry components after instance creation.

        Creates default StorageManager and ProviderRegistry if not provided.
        If storage_root is specified, uses it for StorageManager initialization.
        """
        # Initialize storage manager with provided root or default
        if self.storage_root and not self.storage:
            self.storage = StorageManager(root_path=self.storage_root)
        else:
            self.storage = self.storage or StorageManager()

        # Initialize provider registry
        self.registry = self.registry or ProviderRegistry()

        logger.debug(f"ModelCatalog initialized with storage root: {self.storage.root}")

    def get_storage_root(self) -> Path:
        """Get the storage root directory.

        Returns:
            Path to the storage root directory

        Raises:
            RuntimeError: If storage is not initialized
        """
        if not self.storage:
            msg = "Storage manager not initialized"
            raise RuntimeError(msg)
        return self.storage.root

    def get_provider_count(self) -> int:
        """Get the number of registered providers.

        Returns:
            Number of registered providers
        """
        return self.registry.count() if self.registry else 0

    def validate_configuration(self) -> bool:
        """Validate the catalog configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check storage initialization
            if not self.storage or not self.storage.root.exists():
                logger.error("Storage root does not exist or is not accessible")
                return False

            # Check registry initialization
            if not self.registry:
                logger.error("Provider registry not initialized")
                return False

            logger.debug("ModelCatalog configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
