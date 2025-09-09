"""
this_file: src/vexy_co_model_catalog/core/config.py

Config file parsers and generators for external tools (aichat, codex, mods).
Provides classes to read, parse, and generate config files for different AI CLI tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w
import yaml
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from vexy_co_model_catalog.core.provider import Model, ProviderConfig


@dataclass
class ConfigTemplate:
    """Template for generating tool-specific configurations."""

    tool_name: str
    provider_id: str
    models: list[Model] = field(default_factory=list)
    base_config: dict[str, Any] = field(default_factory=dict)
    provider_config: ProviderConfig | None = None


class ConfigParser(ABC):
    """Abstract base class for tool-specific config parsers."""

    @abstractmethod
    def parse(self, config_path: Path) -> dict[str, Any]:
        """Parse an existing config file."""

    @abstractmethod
    def generate(self, template: ConfigTemplate) -> dict[str, Any]:
        """Generate config data from template."""

    @abstractmethod
    def merge_providers(self, base_config: dict[str, Any], new_providers: list[ConfigTemplate]) -> dict[str, Any]:
        """Merge new providers into existing config."""


class AichatConfigParser(ConfigParser):
    """Parser and generator for aichat YAML configurations."""

    def parse(self, config_path: Path) -> dict[str, Any]:
        """Parse aichat config.yaml file."""
        try:
            if not config_path.exists():
                logger.debug(f"Aichat config file not found: {config_path}")
                return self._get_default_config()

            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            logger.debug(f"Parsed aichat config from: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to parse aichat config {config_path}: {e}")
            return self._get_default_config()

    def generate(self, template: ConfigTemplate) -> dict[str, Any]:
        """Generate aichat provider configuration."""
        if not template.provider_config:
            msg = "Provider config required for aichat generation"
            raise ValueError(msg)

        provider_config = {
            "type": self._get_aichat_type(template.provider_config.kind),
            "api_base": template.provider_config.base_url,
            "api_key": f"${{{template.provider_config.api_key_env or f'{template.provider_id.upper()}_API_KEY'}}}",
        }

        # Add models if available
        if template.models:
            provider_config["models"] = []
            for model in template.models:
                model_entry = {"name": model.id}
                if model.max_input_tokens:
                    model_entry["max_input_tokens"] = model.max_input_tokens
                if model.max_output_tokens:
                    model_entry["max_output_tokens"] = model.max_output_tokens
                provider_config["models"].append(model_entry)

        return {template.provider_id: provider_config}

    def merge_providers(self, base_config: dict[str, Any], new_providers: list[ConfigTemplate]) -> dict[str, Any]:
        """Merge new providers into aichat config."""
        config = base_config.copy()

        # Ensure clients section exists
        if "clients" not in config:
            config["clients"] = []

        # Convert clients list to dict for easier merging
        clients_dict = {}
        for client in config.get("clients", []):
            if isinstance(client, dict):
                for provider_id, provider_config in client.items():
                    clients_dict[provider_id] = provider_config

        # Add new providers
        for template in new_providers:
            new_provider = self.generate(template)
            clients_dict.update(new_provider)

        # Convert back to list format expected by aichat
        config["clients"] = [{provider_id: provider_config} for provider_id, provider_config in clients_dict.items()]

        return config

    def _get_aichat_type(self, provider_kind: str) -> str:
        """Map provider kind to aichat client type."""
        mapping = {
            "openai": "openai",
            "anthropic": "claude",
            "url": "openai",  # Generic OpenAI-compatible for URL providers
        }
        return mapping.get(provider_kind, "openai")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default aichat configuration."""
        return {"model": "gpt-4", "temperature": 0.7, "clients": []}


class CodexConfigParser(ConfigParser):
    """Parser and generator for Codex TOML configurations."""

    def parse(self, config_path: Path) -> dict[str, Any]:
        """Parse codex config.toml file."""
        try:
            if not config_path.exists():
                logger.debug(f"Codex config file not found: {config_path}")
                return self._get_default_config()

            try:
                import tomllib

                toml_loads = tomllib.loads
            except ImportError:
                import tomli

                toml_loads = tomli.loads

            with config_path.open("r", encoding="utf-8") as f:
                config = toml_loads(f.read())

            logger.debug(f"Parsed codex config from: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to parse codex config {config_path}: {e}")
            return self._get_default_config()

    def generate(self, template: ConfigTemplate) -> dict[str, Any]:
        """Generate codex provider configuration with profiles."""
        if not template.provider_config:
            msg = "Provider config required for codex generation"
            raise ValueError(msg)

        profiles = {}

        # Generate profiles for each model
        for model in template.models:
            profile_name = f"{template.provider_id}-{model.id.replace('/', '-').replace(':', '-')}"
            profiles[profile_name] = {
                "name": profile_name,
                "base_url": template.provider_config.base_url,
                "api_key_env": template.provider_config.api_key_env or f"{template.provider_id.upper()}_API_KEY",
                "model": model.id,
                "max_tokens": model.max_output_tokens or 4096,
                "context_window": model.max_input_tokens or 8192,
            }

        return {"profiles": profiles}

    def merge_providers(self, base_config: dict[str, Any], new_providers: list[ConfigTemplate]) -> dict[str, Any]:
        """Merge new providers into codex config."""
        config = base_config.copy()

        # Ensure profiles section exists
        if "profiles" not in config:
            config["profiles"] = {}

        # Add new provider profiles
        for template in new_providers:
            new_profiles = self.generate(template)
            config["profiles"].update(new_profiles["profiles"])

        return config

    def _get_default_config(self) -> dict[str, Any]:
        """Get default codex configuration."""
        return {"default_model": "gpt-4", "profiles": {}}


class ModsConfigParser(ConfigParser):
    """Parser and generator for mods YAML configurations."""

    def parse(self, config_path: Path) -> dict[str, Any]:
        """Parse mods config.yml file."""
        try:
            if not config_path.exists():
                logger.debug(f"Mods config file not found: {config_path}")
                return self._get_default_config()

            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            logger.debug(f"Parsed mods config from: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to parse mods config {config_path}: {e}")
            return self._get_default_config()

    def generate(self, template: ConfigTemplate) -> dict[str, Any]:
        """Generate mods provider configuration."""
        if not template.provider_config:
            msg = "Provider config required for mods generation"
            raise ValueError(msg)

        # Mods uses a simpler model list format
        models = []
        for model in template.models:
            models.append(
                {"name": model.id, "provider": template.provider_id, "max_tokens": model.max_output_tokens or 4096}
            )

        return {
            "apis": {
                template.provider_id: {
                    "base_url": template.provider_config.base_url,
                    "api_key": f"${{{template.provider_config.api_key_env or f'{template.provider_id.upper()}_API_KEY'}}}",
                    "models": models,
                }
            }
        }

    def merge_providers(self, base_config: dict[str, Any], new_providers: list[ConfigTemplate]) -> dict[str, Any]:
        """Merge new providers into mods config."""
        config = base_config.copy()

        # Ensure apis section exists
        if "apis" not in config:
            config["apis"] = {}

        # Add new providers
        for template in new_providers:
            new_api = self.generate(template)
            config["apis"].update(new_api["apis"])

        return config

    def _get_default_config(self) -> dict[str, Any]:
        """Get default mods configuration."""
        return {"default_model": "gpt-4", "apis": {}}


class ConfigManager:
    """High-level manager for all tool configurations."""

    def __init__(self, config_root: Path) -> None:
        """Initialize with config root directory."""
        self.config_root = Path(config_root)
        self.parsers = {"aichat": AichatConfigParser(), "codex": CodexConfigParser(), "mods": ModsConfigParser()}

    def get_parser(self, tool: str) -> ConfigParser:
        """Get parser for specified tool."""
        if tool not in self.parsers:
            msg = f"Unsupported tool: {tool}. Supported: {list(self.parsers.keys())}"
            raise ValueError(msg)
        return self.parsers[tool]

    def backup_config(self, tool: str) -> Path | None:
        """Create timestamped backup of existing config."""
        # Validate tool first to ensure it's supported
        self.get_parser(tool)  # This will raise ValueError for unsupported tools
        config_path = self.get_config_path(tool)
        if not config_path.exists():
            logger.debug(f"No config to backup for {tool}")
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:17]  # Include microseconds, truncate to milliseconds
        backup_dir = self.config_root / "backups"
        backup_dir.mkdir(exist_ok=True)

        backup_path = backup_dir / f"{tool}_config_{timestamp}{config_path.suffix}"
        backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

        logger.info(f"Created backup: {backup_path}")
        return backup_path

    def get_config_path(self, tool: str) -> Path:
        """Get config file path for tool."""
        paths = {
            "aichat": self.config_root / "aichat" / "config.yaml",
            "codex": self.config_root / "codex" / "config.toml",
            "mods": self.config_root / "mods" / "config.yml",
        }
        return paths.get(tool, self.config_root / f"{tool}.yaml")

    def update_tool_config(self, tool: str, new_providers: list[ConfigTemplate], backup: bool = True) -> None:
        """Update tool config with new providers."""
        parser = self.get_parser(tool)
        config_path = self.get_config_path(tool)

        # Create backup if requested
        if backup:
            self.backup_config(tool)

        # Load existing config
        existing_config = parser.parse(config_path)

        # Merge new providers
        updated_config = parser.merge_providers(existing_config, new_providers)

        # Write updated config
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if tool in ["aichat", "mods"]:
            # YAML format
            with config_path.open("w", encoding="utf-8") as f:
                yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False, indent=2)
        elif tool == "codex":
            # TOML format
            with config_path.open("w", encoding="utf-8") as f:
                f.write(tomli_w.dumps(updated_config))

        logger.info(f"Updated {tool} config: {config_path}")

    def get_latest_backup(self, tool: str) -> Path | None:
        """Get the most recent backup for a tool."""
        backup_dir = self.config_root / "backups"
        if not backup_dir.exists():
            return None

        # Find all backups for this tool
        pattern = f"{tool}_config_*.{self._get_config_extension(tool)}"
        backups = list(backup_dir.glob(pattern))

        if not backups:
            return None

        # Sort by modification time (most recent first)
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return backups[0]

    def restore_from_backup(self, tool: str, backup_path: Path | None = None) -> bool:
        """
        Restore config from backup.

        Args:
            tool: Tool name (aichat, codex, mods)
            backup_path: Specific backup to restore from. If None, uses latest backup.

        Returns:
            True if restoration succeeded, False otherwise
        """
        if backup_path is None:
            backup_path = self.get_latest_backup(tool)

        if backup_path is None or not backup_path.exists():
            logger.warning(f"No backup found for {tool} to restore from")
            return False

        try:
            config_path = self.get_config_path(tool)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy backup content to config file
            config_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info(f"Restored {tool} config from backup: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore {tool} config from backup {backup_path}: {e}")
            return False

    @contextmanager
    def config_transaction(self, tool: str) -> Generator[None]:
        """
        Context manager for safe config modifications with automatic rollback.

        Creates a backup before modifications and automatically rolls back
        if any exception occurs within the context.

        Usage:
            with config_manager.config_transaction("aichat"):
                # Perform config modifications
                config_manager.update_tool_config("aichat", providers, backup=False)
        """
        backup_path = None
        try:
            # Create backup before modifications
            backup_path = self.backup_config(tool)
            logger.debug(f"Started config transaction for {tool} with backup: {backup_path}")

            yield

            logger.debug(f"Config transaction for {tool} completed successfully")

        except Exception as e:
            logger.error(f"Config transaction for {tool} failed: {e}")

            # Attempt automatic rollback
            if backup_path:
                logger.info(f"Attempting automatic rollback for {tool}")
                if self.restore_from_backup(tool, backup_path):
                    logger.info(f"Successfully rolled back {tool} config")
                else:
                    logger.error(f"Failed to rollback {tool} config - manual intervention required")

            # Re-raise the original exception
            raise

    def update_tool_config_safe(self, tool: str, new_providers: list[ConfigTemplate]) -> bool:
        """
        Safely update tool config with automatic backup and rollback on failure.

        Args:
            tool: Tool name to update
            new_providers: List of provider configurations to add

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            with self.config_transaction(tool):
                self.update_tool_config(tool, new_providers, backup=False)  # backup handled by transaction
            return True

        except Exception as e:
            logger.error(f"Safe config update failed for {tool}: {e}")
            return False

    def _get_config_extension(self, tool: str) -> str:
        """Get file extension for tool config file."""
        extensions = {"aichat": "yaml", "codex": "toml", "mods": "yml"}
        return extensions.get(tool, "yaml")
