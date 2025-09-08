"""
this_file: src/vexy_co_model_catalog/core/fetcher.py

Minimal async HTTP JSON fetcher.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..utils.exceptions import AuthenticationError, FetchError, RateLimitError


class ModelFetcher:
    def __init__(self, max_concurrency: int = 8, timeout: float = 15.0) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._request_count = 0
        self._error_count = 0

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ModelFetcher":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.close()

    async def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any] | list[Any]:
        attempt = 0
        last_exc: Exception | None = None
        while attempt < max_attempts:
            attempt += 1
            try:
                async with self._semaphore:
                    self._request_count += 1
                    resp = await self._client.get(url, headers=headers)
                if resp.status_code == 401 or resp.status_code == 403:
                    raise AuthenticationError(f"auth failed ({resp.status_code})")
                if resp.status_code == 429:
                    self._error_count += 1
                    if attempt < max_attempts:
                        await asyncio.sleep(1.0 * attempt)
                        continue
                    raise RateLimitError("rate limited")
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, ValueError) as e:
                self._error_count += 1
                last_exc = e
                if attempt < max_attempts:
                    await asyncio.sleep(0.5 * attempt)
                    continue
        raise FetchError(str(last_exc))

    def stats(self) -> dict[str, float | int | None]:
        if self._request_count > 0:
            return {
                "requests": self._request_count,
                "errors": self._error_count,
                "success_rate": (self._request_count - self._error_count) / self._request_count,
            }
        return {"requests": 0, "errors": self._error_count, "success_rate": None}

