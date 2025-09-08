"""
this_file: src/vexy_co_model_catalog/core/storage.py

Atomic file storage utilities used by other components.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import tomli_w
from loguru import logger


class StorageError(Exception):
    pass


class StorageManager:
    """Manages file storage with atomic writes and a simple layout."""

    def __init__(self, root_path: str | os.PathLike | None = None) -> None:
        # Default to the repository root if not provided
        if root_path is None:
            root_path = Path.cwd()
        self.root = Path(root_path).resolve()
        self.json_dir = self.root / "models" / "json"
        self.text_dir = self.root / "models" / "text"
        self.extra_dir = self.root / "models" / "extra"
        self.config_dir = self.root / "config"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        for directory in (self.json_dir, self.text_dir, self.extra_dir, self.config_dir):
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    def _atomic_write(self, target_path: Path, data: bytes) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(target_path.parent), prefix=f".{target_path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            tmp_path.replace(target_path)
        except OSError as e:
            raise StorageError(f"Failed to write {target_path}: {e}") from e
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def write_json(
        self,
        filename: str,
        data: Any,
        *,
        indent: int | None = 2,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
    ) -> None:
        target_path = self.json_dir / f"{filename}.json"
        json_str = json.dumps(
            data,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            separators=(",", ": ") if indent else (",", ":"),
        )
        self._atomic_write(target_path, json_str.encode("utf-8"))
        logger.info(f"Wrote JSON file: {target_path}")

    def write_text(self, filename: str, lines: list[str]) -> None:
        target_path = self.text_dir / f"{filename}.txt"
        content = "\n".join(lines)
        if not content.endswith("\n"):
            content += "\n"
        self._atomic_write(target_path, content.encode("utf-8"))
        logger.info(f"Wrote text file: {target_path} ({len(lines)} lines)")

    def write_toml(self, filename: str, data: dict[str, Any]) -> None:
        target_path = self.json_dir / f"{filename}.toml"
        toml_bytes = tomli_w.dumps(data).encode("utf-8")
        self._atomic_write(target_path, toml_bytes)
        logger.info(f"Wrote TOML file: {target_path}")

    def write_extra(self, filename: str, data: Any) -> None:
        target_path = self.extra_dir / f"{filename}.json"
        json_bytes = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
        self._atomic_write(target_path, json_bytes)
        logger.debug(f"Wrote extra file: {target_path}")

    def read_json(self, filename: str, *, directory: str) -> Any | None:
        if directory == "json":
            file_path = self.json_dir / f"{filename}.json"
        elif directory == "extra":
            file_path = self.extra_dir / f"{filename}.json"
        else:
            raise ValueError(f"Invalid directory: {directory}")
        if not file_path.exists():
            return None
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except OSError as e:
            raise StorageError(f"Failed to read {file_path}: {e}") from e

    def list_files(self, directory: str, pattern: str = "*") -> list[Path]:
        if directory == "json":
            search_dir = self.json_dir
        elif directory == "text":
            search_dir = self.text_dir
        elif directory == "extra":
            search_dir = self.extra_dir
        else:
            raise ValueError(f"Invalid directory: {directory}")
        return list(search_dir.glob(pattern))

    def cleanup_temp_files(self) -> None:
        temp_count = 0
        for directory in (self.json_dir, self.text_dir, self.extra_dir):
            if not directory.exists():
                continue
            for temp_file in directory.glob(".*.tmp"):
                try:
                    temp_file.unlink()
                    temp_count += 1
                except OSError:
                    pass
        if temp_count > 0:
            logger.info(f"Cleaned up {temp_count} temporary files")

    def get_file_stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        for name, directory in (
            ("json", self.json_dir),
            ("text", self.text_dir),
            ("extra", self.extra_dir),
        ):
            stats[f"{name}_files"] = len(list(directory.glob("*"))) if directory.exists() else 0
        return stats

    def __repr__(self) -> str:  # pragma: no cover
        stats = self.get_file_stats()
        return (
            f"StorageManager(root={self.root}, json={stats.get('json_files', 0)}, "
            f"text={stats.get('text_files', 0)}, extra={stats.get('extra_files', 0)})"
        )


