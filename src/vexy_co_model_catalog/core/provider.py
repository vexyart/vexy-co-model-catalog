"""
this_file: src/vexy_co_model_catalog/core/provider.py

Minimal provider abstractions and data structures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProviderKind(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    URL = "url"


@dataclass
class Model:
    id: str
    provider: str
    name: str | None = None
    context_length: int | None = None
    max_tokens: int | None = None
    input_price: float | None = None
    output_price: float | None = None
    supports_functions: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    created: int | None = None
    description: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class ProviderConfig:
    name: str
    kind: ProviderKind
    api_key_env: str | None = None
    base_url_env: str | None = None
    base_url: str | None = None
    headers: dict[str, str] | None = None
    timeout: float = 15.0
    verify_ssl: bool = True

    def get_base_url(self) -> str | None:
        if self.base_url_env and os.getenv(self.base_url_env):
            return os.getenv(self.base_url_env)
        return self.base_url
