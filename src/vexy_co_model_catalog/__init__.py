"""
this_file: src/vexy_co_model_catalog/__init__.py

Package entry for vexy_co_model_catalog.

Keep imports minimal to ensure `import vexy_co_model_catalog` is lightweight and
robust even if optional submodules are incomplete. Tests only require the
`__version__` attribute to be exposed.
"""

from __future__ import annotations

from vexy_co_model_catalog._version import __version__  # re-export version

# Import main classes for user convenience
from vexy_co_model_catalog.core.catalog import ModelCatalog, ProviderRegistry

__author__ = "Vexy Software"
__email__ = "opensource@vexy.art"

__all__ = [
    "ModelCatalog",
    "ProviderRegistry",
    "__author__",
    "__email__",
    "__version__",
]
