"""
this_file: src/vexy_co_model_catalog/utils/exceptions.py

Exception hierarchy for Model Catalog Manager.
"""

from __future__ import annotations


class ModelCatalogError(Exception):
    """Base error for the package."""


class ProviderError(ModelCatalogError):
    """Provider-specific error."""


class FetchError(ProviderError):
    """Generic fetch error (HTTP or parsing)."""


class AuthenticationError(ProviderError):
    """Authentication or authorization failure."""


class RateLimitError(ProviderError):
    """Rate limiting encountered."""


class ConfigurationError(ModelCatalogError):
    """Invalid or missing configuration."""


class GeneratorError(ModelCatalogError):
    """Generation/conversion failure."""


class ValidationError(ModelCatalogError):
    """Input/output validation error."""


class StorageError(ModelCatalogError):
    """Storage-layer failure."""
