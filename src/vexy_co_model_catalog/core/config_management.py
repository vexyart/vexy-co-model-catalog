"""
this_file: src/vexy_co_model_catalog/core/config_management.py

Comprehensive configuration management with validation, repair, and fallback mechanisms.
"""

from __future__ import annotations

import json
import os
import sys
import toml
from pathlib import Path
from typing import Any

import yaml

from vexy_co_model_catalog.core.config import ConfigManager
from vexy_co_model_catalog.core.enhanced_config_validation import (
    ConfigRepairResult,
    ConfigRepairStrategy,
    EnhancedConfigValidator,
)
from vexy_co_model_catalog.core.enhanced_logging import ErrorCategory, ErrorContext, StructuredLogger, operation_context
from vexy_co_model_catalog.core.storage import StorageManager


class ConfigurationManager:
    """Comprehensive configuration management with enhanced validation and recovery."""

    def __init__(self, root_path: str | Path | None = None, auto_repair: bool = True) -> None:
        """
        Initialize configuration manager.

        Args:
            root_path: Root path for configuration files
            auto_repair: Whether to automatically repair configuration issues
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.auto_repair = auto_repair

        self.logger = StructuredLogger("config_manager")
        self.validator = EnhancedConfigValidator(enable_auto_repair=auto_repair)
        self.base_config_manager = ConfigManager(self.root_path / "config")
        self.storage_manager = StorageManager(self.root_path)

        # Configuration discovery and health tracking
        self._config_health: dict[str, bool] = {}
        self._last_validation_results: dict[str, ConfigRepairResult] = {}

    def ensure_config_health(
        self,
        config_paths: list[str | Path] | None = None,
        repair_strategy: ConfigRepairStrategy = ConfigRepairStrategy.MODERATE,
    ) -> dict[str, ConfigRepairResult]:
        """
        Ensure all configuration files are healthy with automatic discovery and repair.

        Args:
            config_paths: Specific config paths to check (auto-discover if None)
            repair_strategy: Strategy for repairing configuration issues

        Returns:
            Dictionary of configuration paths to repair results
        """
        with operation_context("ensure_config_health", "config_manager", repair_strategy=repair_strategy.value):
            if config_paths is None:
                config_paths = self._discover_config_files()

            results = {}
            healthy_count = 0
            total_count = len(config_paths)

            for config_path in config_paths:
                config_path = Path(config_path)

                try:
                    result = self.validator.validate_with_repair(
                        config_path=config_path,
                        repair_strategy=repair_strategy,
                        create_backup=True,
                        enable_fallback=True,
                    )

                    results[str(config_path)] = result
                    self._config_health[str(config_path)] = result.success
                    self._last_validation_results[str(config_path)] = result

                    if result.success:
                        healthy_count += 1
                        self.logger.info(f"Configuration healthy: {config_path}")
                    else:
                        self.logger.warning(
                            f"Configuration requires attention: {config_path}",
                            repair_actions=len(result.repair_actions),
                        )

                except Exception as e:
                    error_context = ErrorContext(
                        category=ErrorCategory.CONFIGURATION,
                        operation="ensure_config_health",
                        metadata={"config_path": str(config_path), "error": str(e)},
                    )

                    self.logger.error(f"Failed to process configuration: {config_path}", error_context=error_context)

                    results[str(config_path)] = ConfigRepairResult(success=False, repaired_config={}, repair_actions=[])
                    self._config_health[str(config_path)] = False

            # Log overall health summary
            health_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0
            self.logger.info(
                "Configuration health check completed",
                healthy_configs=healthy_count,
                total_configs=total_count,
                health_percentage=f"{health_percentage:.1f}%",
            )

            return results

    def setup_minimal_working_config(
        self, force_overwrite: bool = False, include_examples: bool = True
    ) -> dict[str, bool]:
        """
        Set up minimal working configuration for all supported tools.

        Args:
            force_overwrite: Whether to overwrite existing configurations
            include_examples: Whether to include example configurations

        Returns:
            Dictionary of tool names to success status
        """
        with operation_context("setup_minimal_config", "config_manager", force_overwrite=force_overwrite):
            results = {}
            tools_config = {
                "aichat": {
                    "path": self.root_path / "config" / "aichat" / "config.yaml",
                    "generator": self.validator._generate_aichat_fallback,
                },
                "codex": {
                    "path": self.root_path / "config" / "codex" / "config.toml",
                    "generator": self.validator._generate_codex_fallback,
                },
                "mods": {
                    "path": self.root_path / "config" / "mods" / "config.yml",
                    "generator": self.validator._generate_mods_fallback,
                },
            }

            for tool_name, config_info in tools_config.items():
                try:
                    config_path = config_info["path"]

                    # Check if config exists and we're not forcing overwrite
                    if config_path.exists() and not force_overwrite:
                        # Validate existing config
                        result = self.validator.validate_with_repair(
                            config_path=config_path, repair_strategy=ConfigRepairStrategy.MODERATE, create_backup=True
                        )
                        results[tool_name] = result.success

                        if result.success:
                            self.logger.info(f"Existing {tool_name} configuration is healthy: {config_path}")
                        else:
                            self.logger.warning(f"Repaired {tool_name} configuration: {config_path}")

                    else:
                        # Generate new configuration
                        config_path.parent.mkdir(parents=True, exist_ok=True)

                        fallback_config = config_info["generator"]()

                        # Add examples if requested
                        if include_examples:
                            fallback_config = self._add_config_examples(fallback_config, tool_name)

                        # Write configuration
                        if config_path.suffix in [".yaml", ".yml"]:
                            content = yaml.dump(fallback_config, default_flow_style=False, allow_unicode=True)
                        elif config_path.suffix == ".toml":
                            content = toml.dumps(fallback_config)
                        else:
                            content = json.dumps(fallback_config, indent=2)

                        config_path.write_text(content, encoding="utf-8")
                        results[tool_name] = True

                        self.logger.info(f"Created {tool_name} configuration: {config_path}")

                except Exception as e:
                    error_context = ErrorContext(
                        category=ErrorCategory.CONFIGURATION,
                        operation="setup_minimal_config",
                        metadata={"tool": tool_name, "error": str(e)},
                    )

                    self.logger.error(f"Failed to setup {tool_name} configuration", error_context=error_context)
                    results[tool_name] = False

            # Log setup summary
            successful_tools = sum(1 for success in results.values() if success)
            total_tools = len(results)

            self.logger.info(
                "Configuration setup completed",
                successful_tools=successful_tools,
                total_tools=total_tools,
                success_rate=f"{(successful_tools / total_tools) * 100:.1f}%",
            )

            return results

    def validate_environment_setup(self) -> dict[str, Any]:
        """
        Validate the complete environment setup including configurations and dependencies.

        Returns:
            Comprehensive environment validation report
        """
        with operation_context("validate_environment", "config_manager"):
            report = {
                "overall_status": "unknown",
                "config_health": {},
                "environment_variables": {},
                "dependencies": {},
                "recommendations": [],
            }

            # Check configuration health
            config_results = self.ensure_config_health()
            report["config_health"] = {path: result.success for path, result in config_results.items()}

            # Check environment variables
            required_env_vars = [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GROQ_API_KEY",
                "DEEPINFRA_API_KEY",
                "TOGETHER_API_KEY",
            ]

            for env_var in required_env_vars:
                is_set = env_var in os.environ and bool(os.environ[env_var])
                report["environment_variables"][env_var] = {
                    "set": is_set,
                    "value_length": len(os.environ.get(env_var, "")) if is_set else 0,
                }

                if not is_set:
                    report["recommendations"].append(
                        f"Set {env_var} environment variable for {env_var.split('_')[0].lower()} provider support"
                    )

            # Check Python dependencies
            required_modules = ["yaml", "toml", "httpx", "loguru", "rich", "fire"]
            for module in required_modules:
                try:
                    __import__(module)
                    report["dependencies"][module] = {"available": True, "version": None}
                except ImportError:
                    report["dependencies"][module] = {"available": False, "version": None}
                    report["recommendations"].append(f"Install missing Python module: {module}")

            # Determine overall status
            config_healthy = all(report["config_health"].values()) if report["config_health"] else False
            required_env_set = any(
                report["environment_variables"][var]["set"] for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
            )
            deps_available = all(dep["available"] for dep in report["dependencies"].values())

            if config_healthy and required_env_set and deps_available:
                report["overall_status"] = "healthy"
            elif required_env_set and deps_available:
                report["overall_status"] = "functional"
            else:
                report["overall_status"] = "requires_attention"

            # Add specific recommendations based on status
            if report["overall_status"] == "requires_attention":
                if not required_env_set:
                    report["recommendations"].insert(
                        0, "Set at least one API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) to use the model catalog"
                    )

                if not config_healthy:
                    report["recommendations"].insert(
                        0, "Run configuration health check and repair to fix configuration issues"
                    )

            self.logger.info(
                "Environment validation completed",
                overall_status=report["overall_status"],
                config_issues=len([h for h in report["config_health"].values() if not h]),
                missing_env_vars=len([v for v in report["environment_variables"].values() if not v["set"]]),
                missing_deps=len([d for d in report["dependencies"].values() if not d["available"]]),
            )

            return report

    def get_config_diagnostics(self) -> dict[str, Any]:
        """
        Get comprehensive configuration diagnostics for troubleshooting.

        Returns:
            Detailed diagnostics information
        """
        diagnostics = {
            "timestamp": self.logger._get_structured_record.__defaults__[0](),  # Get current timestamp
            "config_health": dict(self._config_health),
            "last_validation_summary": {},
            "discovered_configs": [],
            "environment_status": {},
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "working_directory": str(Path.cwd()),
                "config_root": str(self.root_path),
            },
        }

        # Summarize last validation results
        for path, result in self._last_validation_results.items():
            diagnostics["last_validation_summary"][path] = {
                "success": result.success,
                "repair_actions_count": len(result.repair_actions),
                "backup_created": result.backup_created,
                "validation_issues": len(result.validation_result.issues) if result.validation_result else 0,
            }

        # Discover all configuration files
        diagnostics["discovered_configs"] = [str(p) for p in self._discover_config_files()]

        # Environment variable status
        env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"]
        for var in env_vars:
            diagnostics["environment_status"][var] = {"set": var in os.environ, "length": len(os.environ.get(var, ""))}

        return diagnostics

    def _discover_config_files(self) -> list[Path]:
        """Discover configuration files in the project."""
        config_patterns = [
            "config/**/*.yaml",
            "config/**/*.yml",
            "config/**/*.toml",
            "config/**/*.json",
            "*.yaml",
            "*.yml",
            "*.toml",
        ]

        discovered_files = []

        for pattern in config_patterns:
            discovered_files.extend(self.root_path.glob(pattern))

        # Filter to only include files that look like configuration
        config_files = []
        for file_path in discovered_files:
            # Skip backup files, temporary files, and hidden files
            if (
                not file_path.name.startswith(".")
                and not file_path.name.endswith(".tmp")
                and "backup" not in file_path.name.lower()
                and file_path.is_file()
            ):
                config_files.append(file_path)

        return sorted(set(config_files))

    def _add_config_examples(self, config: dict[str, Any], tool_name: str) -> dict[str, Any]:
        """Add example configurations and comments."""
        if tool_name == "aichat":
            config["_examples"] = {
                "usage": "Set OPENAI_API_KEY environment variable and run: aichat 'Hello, world!'",
                "models": ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"],
                "temperature_guide": "0.0 = deterministic, 1.0 = creative",
            }

        elif tool_name == "codex":
            config["_examples"] = {
                "usage": "Set OPENAI_API_KEY and run: codex chat",
                "profile_switching": "Use --profile gpt35 for faster responses",
                "available_models": ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"],
            }

        elif tool_name == "mods":
            config["_examples"] = {
                "usage": "Set OPENAI_API_KEY and run: echo 'Explain this code' | mods",
                "model_selection": "Use -m gpt-4 for better quality responses",
                "pipe_integration": "Works great with shell pipes and command substitution",
            }

        return config


def create_production_config_manager(root_path: str | Path | None = None) -> ConfigurationManager:
    """
    Create a production-ready configuration manager with optimal settings.

    Args:
        root_path: Root path for configuration files

    Returns:
        Configured ConfigurationManager instance
    """
    return ConfigurationManager(root_path=root_path, auto_repair=True)


def quick_config_health_check(root_path: str | Path | None = None) -> bool:
    """
    Quick health check for configuration files.

    Args:
        root_path: Root path to check (defaults to current directory)

    Returns:
        True if all configurations are healthy, False otherwise
    """
    manager = ConfigurationManager(root_path=root_path, auto_repair=True)
    results = manager.ensure_config_health()
    return all(result.success for result in results.values())


def setup_project_configs(root_path: str | Path | None = None, force_overwrite: bool = False) -> bool:
    """
    Set up all project configurations with minimal viable defaults.

    Args:
        root_path: Root path for the project
        force_overwrite: Whether to overwrite existing configurations

    Returns:
        True if setup was successful, False otherwise
    """
    manager = ConfigurationManager(root_path=root_path, auto_repair=True)
    results = manager.setup_minimal_working_config(force_overwrite=force_overwrite)
    return all(results.values())


# Default instance for easy use
default_config_manager = ConfigurationManager()
