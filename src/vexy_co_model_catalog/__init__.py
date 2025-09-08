"""
this_file: src/vexy_co_model_catalog/__init__.py

Package entry for vexy_co_model_catalog.

Keep imports minimal to ensure `import vexy_co_model_catalog` is lightweight and
robust even if optional submodules are incomplete. Tests only require the
`__version__` attribute to be exposed.
"""

from __future__ import annotations

from ._version import __version__  # re-export version

__author__ = "Vexy Software"
__email__ = "opensource@vexy.art"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]
