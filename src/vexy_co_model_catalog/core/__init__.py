"""
this_file: src/vexy_co_model_catalog/core/__init__.py

Core exports. Keep imports local to avoid side effects at package import time.
"""

from __future__ import annotations

from .storage import StorageManager

__all__ = [
    "StorageManager",
]
