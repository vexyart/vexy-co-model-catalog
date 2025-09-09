"""
this_file: src/vexy_co_model_catalog/core/storage.py

Atomic file storage utilities used by other components.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import tomli_w
import yaml
from loguru import logger

# Handle tomllib import for Python 3.11+ compatibility
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from vexy_co_model_catalog.core.integrity import (
    IntegrityLevel,
    create_critical_backup,
    get_integrity_manager,
)
from vexy_co_model_catalog.core.security_enhanced import (
    get_enhanced_security_validator,
)

# Common errno constants for better code readability
ENOENT = errno.ENOENT  # 2: File not found
EACCES = errno.EACCES  # 13: Permission denied
EISDIR = errno.EISDIR  # 21: Is a directory
ENOSPC = errno.ENOSPC  # 28: No space left on device
EEXIST = errno.EEXIST  # 17: File exists


class StorageError(Exception):
    pass


class StorageManager:
    """Manages file storage with atomic writes and a simple layout."""

    def __init__(self, root_path: str | os.PathLike | None = None) -> None:
        # Default to the repository root if not provided
        if root_path is None:
            root_path = Path.cwd()
        self.root = Path(root_path).resolve()

        # Legacy directories for backward compatibility
        self.json_dir = self.root / "models" / "json"
        self.text_dir = self.root / "models" / "text"
        self.extra_dir = self.root / "models" / "extra"

        # New config-based directory structure
        self.config_dir = self.root / "config"
        self.config_json_dir = self.config_dir / "json"
        self.config_txt_dir = self.config_dir / "txt"
        self.config_aichat_dir = self.config_dir / "aichat"
        self.config_codex_dir = self.config_dir / "codex"
        self.config_mods_dir = self.config_dir / "mods"

        # Directories will be created lazily when needed

    def _ensure_directory(self, directory: Path) -> None:
        """Ensure a specific directory exists (lazy creation)."""
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

    def _atomic_write(
        self,
        target_path: Path,
        data: bytes,
        enable_integrity: bool = False,
        integrity_level: IntegrityLevel = IntegrityLevel.STANDARD,
    ) -> None:
        tmp_path = None
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup for critical files before modification
            if (
                enable_integrity
                and target_path.exists()
                and integrity_level in [IntegrityLevel.CRITICAL, IntegrityLevel.IMPORTANT]
            ):
                try:
                    create_critical_backup(target_path)
                except Exception as e:
                    logger.warning(f"Failed to create backup for {target_path}: {e}")

            fd, tmp = tempfile.mkstemp(dir=str(target_path.parent), prefix=f".{target_path.name}.", suffix=".tmp")
            tmp_path = Path(tmp)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            tmp_path.replace(target_path)

            # Validate file permissions for security compliance
            try:
                # Use secure mode for critical files, standard mode for others
                secure_mode = enable_integrity and integrity_level == IntegrityLevel.CRITICAL
                validator = get_enhanced_security_validator()
                if not validator.validate_file_permissions(target_path, secure_mode):
                    logger.warning(f"SECURITY: File written with potentially insecure permissions: {target_path}")
            except Exception as e:
                logger.debug(f"Security validation skipped for {target_path}: {e}")

            # Add integrity tracking for critical files
            if enable_integrity:
                try:
                    integrity_manager = get_integrity_manager(self.root)
                    file_tags = [f"storage_{target_path.parent.name}", "model_catalog"]
                    integrity_manager.add_file_tracking(target_path, integrity_level, file_tags)
                except Exception as e:
                    logger.warning(f"Failed to add integrity tracking for {target_path}: {e}")

        except OSError as e:
            # Enhanced file I/O error context with recovery guidance
            error_code = getattr(e, 'errno', 'unknown')

            if error_code == ENOSPC:  # No space left on device
                msg = (
                    f"Failed to write {target_path}: Insufficient disk space. "
                    f"Free up space in {target_path.parent} and try again. "
                    f"Current target requires approximately {len(data) if 'data' in locals() else 'unknown'} bytes."
                )
            elif error_code == EACCES:  # Permission denied
                msg = (
                    f"Failed to write {target_path}: Permission denied. "
                    f"Check file/directory permissions: chmod 755 {target_path.parent}, "
                    f"or run with appropriate user privileges."
                )
            elif error_code == ENOENT:  # Directory doesn't exist
                msg = (
                    f"Failed to write {target_path}: Directory doesn't exist. "
                    f"Create parent directory: mkdir -p {target_path.parent}"
                )
            else:
                msg = (
                    f"Failed to write {target_path}: {e} (errno: {error_code}). "
                    f"Check: 1) Disk space availability, 2) File permissions, "
                    f"3) Directory existence, 4) File system integrity."
                )

            logger.error(f"File write error: {msg}")
            raise StorageError(msg) from e
        finally:
            if tmp_path and tmp_path.exists():
                with contextlib.suppress(OSError):
                    tmp_path.unlink()

    def write_critical_json(self, filename: str, data: Any, **kwargs) -> None:
        """Write JSON file with critical integrity protection."""
        target_path = self.json_dir / f"{filename}.json"
        json_str = json.dumps(
            data,
            indent=kwargs.get("indent", 2),
            sort_keys=kwargs.get("sort_keys", False),
            ensure_ascii=kwargs.get("ensure_ascii", False),
        )
        self._atomic_write(
            target_path, json_str.encode("utf-8"), enable_integrity=True, integrity_level=IntegrityLevel.CRITICAL
        )
        logger.info(f"Wrote critical JSON file with integrity protection: {target_path}")

    def write_important_config(self, filename: str, data: Any, config_type: str = "json") -> None:
        """Write configuration file with important integrity protection."""
        if config_type == "json":
            target_path = self.config_json_dir / f"{filename}.json"
            json_str = json.dumps(data, indent=2, sort_keys=False, ensure_ascii=False)
            content = json_str.encode("utf-8")
        elif config_type == "yaml":
            target_path = self.config_dir / f"{filename}.yaml"
            yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
            content = yaml_str.encode("utf-8")
        elif config_type == "toml":
            target_path = self.config_codex_dir / f"{filename}.toml"
            toml_bytes = tomli_w.dumps(data).encode("utf-8")
            content = toml_bytes
        else:
            msg = f"Unsupported config type: {config_type}"
            raise ValueError(msg)

        self._atomic_write(target_path, content, enable_integrity=True, integrity_level=IntegrityLevel.IMPORTANT)
        logger.info(f"Wrote important {config_type.upper()} config with integrity protection: {target_path}")

    def verify_file_integrity(self, filename: str, directory: str = "json") -> bool:
        """Verify integrity of a specific file."""
        if directory == "json":
            file_path = self.json_dir / f"{filename}.json"
        elif directory == "config":
            file_path = self.config_dir / f"{filename}.yaml"
        elif directory == "config_json":
            file_path = self.config_json_dir / f"{filename}.json"
        else:
            msg = f"Unsupported directory: {directory}"
            raise ValueError(msg)

        try:
            integrity_manager = get_integrity_manager(self.root)
            return integrity_manager.verify_file_integrity(file_path, auto_repair=True)
        except Exception as e:
            logger.warning(f"Failed to verify integrity of {file_path}: {e}")
            return False

    def get_integrity_report(self) -> dict[str, Any]:
        """Get comprehensive integrity report for all tracked files."""
        try:
            integrity_manager = get_integrity_manager(self.root)
            return integrity_manager.get_integrity_report()
        except Exception as e:
            logger.warning(f"Failed to get integrity report: {e}")
            return {"error": str(e)}

    def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """Clean up old backup files."""
        try:
            integrity_manager = get_integrity_manager(self.root)
            return integrity_manager.cleanup_old_backups(retention_days)
        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")
            return 0

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

    def write_yaml(self, filename: str, data: dict[str, Any], *, directory: str = "config") -> None:
        """Write YAML file to specified directory."""
        if directory == "config":
            self._ensure_directory(self.config_dir)
            target_path = self.config_dir / f"{filename}.yaml"
        elif directory == "aichat":
            self._ensure_directory(self.config_aichat_dir)
            target_path = self.config_aichat_dir / f"{filename}.yaml"
        elif directory == "mods":
            self._ensure_directory(self.config_mods_dir)
            target_path = self.config_mods_dir / f"{filename}.yml"
        else:
            msg = f"Invalid directory for YAML: {directory}"
            raise ValueError(msg)

        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False, indent=2)
        self._atomic_write(target_path, yaml_str.encode("utf-8"))
        logger.info(f"Wrote YAML file: {target_path}")

    def write_extra(self, filename: str, data: Any) -> None:
        self._ensure_directory(self.extra_dir)
        target_path = self.extra_dir / f"{filename}.json"
        json_bytes = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
        self._atomic_write(target_path, json_bytes)
        logger.debug(f"Wrote extra file: {target_path}")

    def write_config_json(self, filename: str, data: Any) -> None:
        """Write JSON file to config/json/ directory."""
        self._ensure_directory(self.config_json_dir)
        target_path = self.config_json_dir / f"{filename}.json"
        json_str = json.dumps(data, indent=2, sort_keys=False, ensure_ascii=False)
        self._atomic_write(target_path, json_str.encode("utf-8"))
        logger.info(f"Wrote config JSON file: {target_path}")

    def write_config_txt(self, filename: str, lines: list[str]) -> None:
        """Write TXT file to config/txt/ directory."""
        self._ensure_directory(self.config_txt_dir)
        target_path = self.config_txt_dir / f"{filename}.txt"
        content = "\n".join(lines)
        if not content.endswith("\n"):
            content += "\n"
        self._atomic_write(target_path, content.encode("utf-8"))
        logger.info(f"Wrote config TXT file: {target_path} ({len(lines)} lines)")

    def write_config_toml(self, filename: str, data: dict[str, Any], *, directory: str = "codex") -> None:
        """Write TOML file to specified config tool directory."""
        if directory == "codex":
            target_path = self.config_codex_dir / f"{filename}.toml"
        else:
            msg = f"Invalid directory for TOML: {directory}"
            raise ValueError(msg)

        toml_bytes = tomli_w.dumps(data).encode("utf-8")
        self._atomic_write(target_path, toml_bytes)
        logger.info(f"Wrote config TOML file: {target_path}")

    def read_json(self, filename: str, *, directory: str) -> Any | None:
        if directory == "json":
            file_path = self.json_dir / f"{filename}.json"
        elif directory == "extra":
            file_path = self.extra_dir / f"{filename}.json"
        elif directory == "config_json":
            file_path = self.config_json_dir / f"{filename}.json"
        else:
            msg = f"Invalid directory: {directory}"
            raise ValueError(msg)
        if not file_path.exists():
            return None
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except OSError as e:
            # Enhanced file read error context with recovery guidance
            error_code = getattr(e, 'errno', 'unknown')

            if error_code == ENOENT:  # File not found
                msg = (
                    f"Failed to read {file_path}: File not found. "
                    f"Generate the file using: vexy fetch {file_path.stem.replace('models_', '')} "
                    f"or check if the file path is correct."
                )
            elif error_code == EACCES:  # Permission denied
                msg = (
                    f"Failed to read {file_path}: Permission denied. "
                    f"Fix permissions: chmod 644 {file_path} or run with appropriate privileges."
                )
            elif error_code == EISDIR:  # Is a directory
                msg = (
                    f"Failed to read {file_path}: Path is a directory, not a file. "
                    f"Check the correct file path or list directory contents."
                )
            else:
                msg = (
                    f"Failed to read {file_path}: {e} (errno: {error_code}). "
                    f"Verify: 1) File exists, 2) Read permissions, 3) File integrity, 4) Disk health."
                )

            logger.error(f"File read error: {msg}")
            raise StorageError(msg) from e

    def read_yaml(self, filename: str, *, directory: str) -> Any | None:
        """Read YAML file from specified directory."""
        if directory == "aichat":
            file_path = self.config_aichat_dir / f"{filename}.yaml"
        elif directory == "mods":
            file_path = self.config_mods_dir / f"{filename}.yml"
        elif directory == "config":
            file_path = self.config_dir / f"{filename}.yaml"
        else:
            msg = f"Invalid directory for YAML: {directory}"
            raise ValueError(msg)

        if not file_path.exists():
            return None
        try:
            return yaml.safe_load(file_path.read_text(encoding="utf-8"))
        except OSError as e:
            # Enhanced file read error context with recovery guidance
            error_code = getattr(e, 'errno', 'unknown')

            if error_code == ENOENT:  # File not found
                msg = (
                    f"Failed to read {file_path}: File not found. "
                    f"Generate the file using: vexy fetch {file_path.stem.replace('models_', '')} "
                    f"or check if the file path is correct."
                )
            elif error_code == EACCES:  # Permission denied
                msg = (
                    f"Failed to read {file_path}: Permission denied. "
                    f"Fix permissions: chmod 644 {file_path} or run with appropriate privileges."
                )
            elif error_code == EISDIR:  # Is a directory
                msg = (
                    f"Failed to read {file_path}: Path is a directory, not a file. "
                    f"Check the correct file path or list directory contents."
                )
            else:
                msg = (
                    f"Failed to read {file_path}: {e} (errno: {error_code}). "
                    f"Verify: 1) File exists, 2) Read permissions, 3) File integrity, 4) Disk health."
                )

            logger.error(f"File read error: {msg}")
            raise StorageError(msg) from e

    def read_toml(self, filename: str, *, directory: str) -> Any | None:
        """Read TOML file from specified directory."""
        if directory == "codex":
            file_path = self.config_codex_dir / f"{filename}.toml"
        else:
            msg = f"Invalid directory for TOML: {directory}"
            raise ValueError(msg)

        if not file_path.exists():
            return None
        try:
            return tomllib.loads(file_path.read_text(encoding="utf-8"))
        except OSError as e:
            # Enhanced file read error context with recovery guidance
            error_code = getattr(e, 'errno', 'unknown')

            if error_code == ENOENT:  # File not found
                msg = (
                    f"Failed to read {file_path}: File not found. "
                    f"Generate the file using: vexy fetch {file_path.stem.replace('models_', '')} "
                    f"or check if the file path is correct."
                )
            elif error_code == EACCES:  # Permission denied
                msg = (
                    f"Failed to read {file_path}: Permission denied. "
                    f"Fix permissions: chmod 644 {file_path} or run with appropriate privileges."
                )
            elif error_code == EISDIR:  # Is a directory
                msg = (
                    f"Failed to read {file_path}: Path is a directory, not a file. "
                    f"Check the correct file path or list directory contents."
                )
            else:
                msg = (
                    f"Failed to read {file_path}: {e} (errno: {error_code}). "
                    f"Verify: 1) File exists, 2) Read permissions, 3) File integrity, 4) Disk health."
                )

            logger.error(f"File read error: {msg}")
            raise StorageError(msg) from e

    def list_files(self, directory: str, pattern: str = "*") -> list[Path]:
        if directory == "json":
            search_dir = self.json_dir
        elif directory == "text":
            search_dir = self.text_dir
        elif directory == "extra":
            search_dir = self.extra_dir
        elif directory == "config_json":
            search_dir = self.config_json_dir
        elif directory == "config_txt":
            search_dir = self.config_txt_dir
        elif directory == "config_aichat":
            search_dir = self.config_aichat_dir
        elif directory == "config_codex":
            search_dir = self.config_codex_dir
        elif directory == "config_mods":
            search_dir = self.config_mods_dir
        else:
            msg = f"Invalid directory: {directory}"
            raise ValueError(msg)
        return list(search_dir.glob(pattern))

    def cleanup_temp_files(self) -> None:
        temp_count = 0
        directories = (
            # Legacy directories
            self.json_dir,
            self.text_dir,
            self.extra_dir,
            # New config directories
            self.config_json_dir,
            self.config_txt_dir,
            self.config_aichat_dir,
            self.config_codex_dir,
            self.config_mods_dir,
        )
        for directory in directories:
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
        directory_mapping = (
            # Legacy directories
            ("json", self.json_dir),
            ("text", self.text_dir),
            ("extra", self.extra_dir),
            # New config directories
            ("config_json", self.config_json_dir),
            ("config_txt", self.config_txt_dir),
            ("config_aichat", self.config_aichat_dir),
            ("config_codex", self.config_codex_dir),
            ("config_mods", self.config_mods_dir),
        )
        for name, directory in directory_mapping:
            stats[f"{name}_files"] = len(list(directory.glob("*"))) if directory.exists() else 0
        return stats

    def __repr__(self) -> str:  # pragma: no cover
        stats = self.get_file_stats()
        legacy_count = stats.get("json_files", 0) + stats.get("text_files", 0) + stats.get("extra_files", 0)
        config_count = sum(stats.get(f"config_{key}_files", 0) for key in ["json", "txt", "aichat", "codex", "mods"])
        return f"StorageManager(root={self.root}, legacy_files={legacy_count}, config_files={config_count})"
