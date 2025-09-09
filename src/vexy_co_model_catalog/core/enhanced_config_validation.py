"""
this_file: src/vexy_co_model_catalog/core/enhanced_config_validation.py

Enhanced configuration validation with comprehensive checks and graceful fallback mechanisms.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    pass
else:
    pass
import tomli_w as toml_write
import yaml

from vexy_co_model_catalog.core.config_validator import (
    ConfigFormat,
    ConfigValidationResult,
    ConfigValidator,
    ValidationIssue,
    ValidationSeverity,
)
from vexy_co_model_catalog.core.enhanced_logging import ErrorCategory, ErrorContext, StructuredLogger, operation_context

if TYPE_CHECKING:
    from collections.abc import Callable


class ConfigRepairStrategy(Enum):
    """Configuration repair strategies."""

    CONSERVATIVE = "conservative"  # Only fix obvious syntax errors
    MODERATE = "moderate"  # Fix syntax and add missing required fields
    AGGRESSIVE = "aggressive"  # Full repair with defaults and restructuring


class FallbackSource(Enum):
    """Sources for fallback configuration values."""

    DEFAULT_VALUES = "default_values"  # Built-in default values
    TEMPLATE_CONFIG = "template_config"  # Template configuration files
    ENVIRONMENT_VARS = "environment_vars"  # Environment variables
    USER_INPUT = "user_input"  # Interactive user prompts
    EXISTING_CONFIG = "existing_config"  # Values from existing valid configs


# Configuration repair confidence thresholds
CONFIG_REPAIR_MODERATE_CONFIDENCE = 0.7   # Confidence threshold for moderate repair strategy
CONFIG_REPAIR_CONSERVATIVE_CONFIDENCE = 0.9  # Confidence threshold for conservative repair strategy


@dataclass
class ConfigRepairAction:
    """Individual configuration repair action."""

    action_type: str  # Type of repair action
    field_path: str  # Path to field being repaired
    old_value: Any  # Original value (if any)
    new_value: Any  # New/repaired value
    reason: str  # Reason for the repair
    confidence: float  # Confidence level (0.0-1.0)
    fallback_source: FallbackSource  # Source of the fallback value


@dataclass
class ConfigRepairResult:
    """Result of configuration repair operation."""

    success: bool
    repaired_config: dict[str, Any]
    repair_actions: list[ConfigRepairAction] = field(default_factory=list)
    backup_created: bool = False
    backup_path: str | None = None
    validation_result: ConfigValidationResult | None = None


@dataclass
class EnhancedValidationRule:
    """Enhanced validation rule with repair capabilities."""

    name: str
    description: str
    validator: Callable[[dict[str, Any]], list[ValidationIssue]]
    repairer: Callable[[dict[str, Any]], list[ConfigRepairAction]] | None = None
    severity: ValidationSeverity = ValidationSeverity.ERROR
    category: str = "general"


class EnhancedConfigValidator:
    """Enhanced configuration validator with repair and fallback capabilities."""

    def __init__(self, enable_auto_repair: bool = True) -> None:
        """Initialize enhanced config validator."""
        self.base_validator = ConfigValidator()
        self.logger = StructuredLogger("config_validator")
        self.enable_auto_repair = enable_auto_repair

        # Enhanced validation rules
        self.validation_rules = self._initialize_validation_rules()

        # Default configuration templates
        self.config_templates = self._load_config_templates()

        # Environment variable mappings
        self.env_var_mappings = self._get_env_var_mappings()

    def validate_with_repair(
        self,
        config_path: str | Path,
        repair_strategy: ConfigRepairStrategy = ConfigRepairStrategy.MODERATE,
        create_backup: bool = True,
        enable_fallback: bool = True,
    ) -> ConfigRepairResult:
        """
        Validate configuration with automatic repair and fallback mechanisms.

        Args:
            config_path: Path to configuration file
            repair_strategy: Strategy for repairing configuration issues
            create_backup: Whether to create backup before repairs
            enable_fallback: Whether to enable fallback mechanisms

        Returns:
            ConfigRepairResult with repair details and repaired configuration
        """
        with operation_context(
            "validate_with_repair",
            "config_validator",
            config_path=str(config_path),
            repair_strategy=repair_strategy.value,
        ):
            config_path = Path(config_path)

            # Check if file exists
            if not config_path.exists():
                return self._handle_missing_config(config_path, enable_fallback)

            # Create backup if requested
            backup_path = None
            if create_backup:
                backup_path = self._create_backup(config_path)

            try:
                # Perform initial validation
                validation_result = self.base_validator.validate_file(config_path)

                if validation_result.is_valid:
                    self.logger.info(f"Configuration is valid: {config_path}")
                    return ConfigRepairResult(
                        success=True,
                        repaired_config=validation_result.parsed_content or {},
                        backup_created=backup_path is not None,
                        backup_path=str(backup_path) if backup_path else None,
                        validation_result=validation_result,
                    )

                # Configuration has issues, attempt repair
                self.logger.warning(
                    f"Configuration has {len(validation_result.issues)} issues, attempting repair",
                    error_count=len([i for i in validation_result.issues if i.severity == ValidationSeverity.ERROR]),
                    warning_count=len(
                        [i for i in validation_result.issues if i.severity == ValidationSeverity.WARNING]
                    ),
                )

                return self._repair_configuration(
                    config_path, validation_result, repair_strategy, enable_fallback, backup_path
                )

            except Exception as e:
                error_context = ErrorContext(
                    category=ErrorCategory.CONFIGURATION,
                    operation="validate_with_repair",
                    metadata={"config_path": str(config_path), "error": str(e)},
                )

                self.logger.error(
                    f"Failed to validate/repair configuration: {config_path}", error_context=error_context
                )

                if enable_fallback:
                    return self._generate_fallback_config(config_path)

                return ConfigRepairResult(
                    success=False,
                    repaired_config={},
                    backup_created=backup_path is not None,
                    backup_path=str(backup_path) if backup_path else None,
                )

    def validate_environment_dependencies(self, config_data: dict[str, Any]) -> list[ValidationIssue]:
        """Validate that required environment variables are available."""
        issues = []

        # Check for API key requirements
        api_key_patterns = [r".*api[_-]?key.*", r".*token.*", r".*secret.*", r".*auth.*"]

        def find_env_refs(obj: Any, path: str = "") -> list[str]:
            """Recursively find environment variable references."""
            env_refs = []

            if isinstance(obj, str):
                # Look for ${VAR} or $VAR patterns
                env_pattern = r"\$\{?([A-Z_][A-Z0-9_]*)\}?"
                matches = re.findall(env_pattern, obj)
                env_refs.extend(matches)

                # Check if the value looks like an env var name
                if any(re.match(pattern, path.lower()) for pattern in api_key_patterns):
                    if obj.isupper() and "_" in obj:
                        env_refs.append(obj)

            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    env_refs.extend(find_env_refs(value, new_path))

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    env_refs.extend(find_env_refs(item, new_path))

            return env_refs

        # Find all environment variable references
        env_refs = find_env_refs(config_data)

        # Check if environment variables exist
        for env_var in set(env_refs):
            if env_var not in os.environ:
                suggestion = f"Set environment variable: export {env_var}=your_value"

                # Provide specific suggestions for common variables
                if "api_key" in env_var.lower():
                    suggestion += " (API key for service configuration)"
                elif "url" in env_var.lower():
                    suggestion += " (Service endpoint URL)"
                elif "token" in env_var.lower():
                    suggestion += " (Authentication token)"

                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Required environment variable not set: {env_var}",
                        field_path=f"env.{env_var}",
                        suggestion=suggestion,
                    )
                )

        return issues

    def validate_provider_configs(self, config_data: dict[str, Any]) -> list[ValidationIssue]:
        """Validate provider-specific configuration requirements."""
        issues = []

        # Check for provider configurations
        providers_section = config_data.get("providers", config_data.get("models", {}))

        if not providers_section:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="No provider configurations found",
                    field_path="providers",
                    suggestion="Add at least one provider configuration",
                )
            )
            return issues

        # Validate individual provider configs
        for provider_name, provider_config in providers_section.items():
            if isinstance(provider_config, dict):
                # Check for required fields
                required_fields = ["base_url", "models_path"]
                for field in required_fields:
                    if field not in provider_config:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                message=f"Missing required field '{field}' for provider '{provider_name}'",
                                field_path=f"providers.{provider_name}.{field}",
                                suggestion=f"Add {field} configuration for provider {provider_name}",
                            )
                        )

                # Validate URL format
                base_url = provider_config.get("base_url", "")
                if base_url and not re.match(r"^https?://", base_url):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Provider '{provider_name}' base_url should use https:// protocol",
                            field_path=f"providers.{provider_name}.base_url",
                            suggestion="Use https:// for secure communication",
                        )
                    )

        return issues

    def _repair_configuration(
        self,
        config_path: Path,
        validation_result: ConfigValidationResult,
        repair_strategy: ConfigRepairStrategy,
        _enable_fallback: bool,
        backup_path: Path | None,
    ) -> ConfigRepairResult:
        """Repair configuration based on validation issues."""
        repair_actions = []
        repaired_config = validation_result.parsed_content.copy() if validation_result.parsed_content else {}

        # Apply validation rules with repair capabilities
        for rule in self.validation_rules:
            if rule.repairer:
                try:
                    actions = rule.repairer(repaired_config)
                    repair_actions.extend(actions)

                    # Apply repair actions based on strategy
                    for action in actions:
                        if self._should_apply_repair(action, repair_strategy):
                            self._apply_repair_action(repaired_config, action)
                            self.logger.debug(
                                f"Applied repair: {action.action_type}",
                                field_path=action.field_path,
                                confidence=action.confidence,
                            )

                except Exception as e:
                    self.logger.warning(f"Failed to apply repair rule {rule.name}: {e}")

        # Add missing required fields based on templates
        if repair_strategy in [ConfigRepairStrategy.MODERATE, ConfigRepairStrategy.AGGRESSIVE]:
            template_actions = self._add_missing_template_fields(repaired_config, config_path)
            repair_actions.extend(template_actions)

            for action in template_actions:
                self._apply_repair_action(repaired_config, action)

        # Validate repaired configuration
        try:
            format_type = self.base_validator._detect_format_from_path(config_path)

            if format_type == ConfigFormat.YAML:
                content = yaml.dump(repaired_config, default_flow_style=False, allow_unicode=True)
            elif format_type == ConfigFormat.TOML:
                content = toml_write.dumps(repaired_config)
            else:
                content = json.dumps(repaired_config, indent=2)

            final_validation = self.base_validator.validate_content(content, format_type=format_type)

            # Write repaired configuration back to file
            if final_validation.is_valid or repair_strategy == ConfigRepairStrategy.AGGRESSIVE:
                config_path.write_text(content, encoding="utf-8")
                self.logger.info(
                    f"Successfully repaired and saved configuration: {config_path}", repair_count=len(repair_actions)
                )

                return ConfigRepairResult(
                    success=True,
                    repaired_config=repaired_config,
                    repair_actions=repair_actions,
                    backup_created=backup_path is not None,
                    backup_path=str(backup_path) if backup_path else None,
                    validation_result=final_validation,
                )
            self.logger.error(f"Repair failed - configuration still invalid: {config_path}")
            return ConfigRepairResult(
                success=False,
                repaired_config=repaired_config,
                repair_actions=repair_actions,
                backup_created=backup_path is not None,
                backup_path=str(backup_path) if backup_path else None,
                validation_result=final_validation,
            )

        except Exception as e:
            self.logger.error(f"Failed to save repaired configuration: {e}")
            return ConfigRepairResult(
                success=False,
                repaired_config=repaired_config,
                repair_actions=repair_actions,
                backup_created=backup_path is not None,
                backup_path=str(backup_path) if backup_path else None,
            )

    def _handle_missing_config(self, config_path: Path, enable_fallback: bool) -> ConfigRepairResult:
        """Handle missing configuration file with fallback generation."""
        if not enable_fallback:
            return ConfigRepairResult(
                success=False,
                repaired_config={},
                repair_actions=[
                    ConfigRepairAction(
                        action_type="file_missing",
                        field_path=str(config_path),
                        old_value=None,
                        new_value=None,
                        reason="Configuration file does not exist",
                        confidence=1.0,
                        fallback_source=FallbackSource.DEFAULT_VALUES,
                    )
                ],
            )

        self.logger.warning(f"Configuration file missing, generating fallback: {config_path}")
        return self._generate_fallback_config(config_path)

    def _generate_fallback_config(self, config_path: Path) -> ConfigRepairResult:
        """Generate fallback configuration from templates and defaults."""
        # Determine config type from path and name
        config_name = config_path.stem
        format_type = self.base_validator._detect_format_from_path(config_path)

        # Generate appropriate fallback config
        fallback_config = {}
        repair_actions = []

        # Use templates based on file name and location
        if "aichat" in str(config_path) or config_name == "config":
            fallback_config = self._generate_aichat_fallback()
            repair_actions.append(
                ConfigRepairAction(
                    action_type="generate_fallback",
                    field_path="root",
                    old_value=None,
                    new_value=fallback_config,
                    reason="Generated aichat fallback configuration",
                    confidence=0.8,
                    fallback_source=FallbackSource.TEMPLATE_CONFIG,
                )
            )

        elif "codex" in str(config_path):
            fallback_config = self._generate_codex_fallback()
            repair_actions.append(
                ConfigRepairAction(
                    action_type="generate_fallback",
                    field_path="root",
                    old_value=None,
                    new_value=fallback_config,
                    reason="Generated codex fallback configuration",
                    confidence=0.8,
                    fallback_source=FallbackSource.TEMPLATE_CONFIG,
                )
            )

        elif "mods" in str(config_path):
            fallback_config = self._generate_mods_fallback()
            repair_actions.append(
                ConfigRepairAction(
                    action_type="generate_fallback",
                    field_path="root",
                    old_value=None,
                    new_value=fallback_config,
                    reason="Generated mods fallback configuration",
                    confidence=0.8,
                    fallback_source=FallbackSource.TEMPLATE_CONFIG,
                )
            )

        else:
            # Generic fallback
            fallback_config = self._generate_generic_fallback()
            repair_actions.append(
                ConfigRepairAction(
                    action_type="generate_fallback",
                    field_path="root",
                    old_value=None,
                    new_value=fallback_config,
                    reason="Generated generic fallback configuration",
                    confidence=0.6,
                    fallback_source=FallbackSource.DEFAULT_VALUES,
                )
            )

        # Write fallback config to file
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type == ConfigFormat.YAML:
                content = yaml.dump(fallback_config, default_flow_style=False, allow_unicode=True)
            elif format_type == ConfigFormat.TOML:
                content = toml_write.dumps(fallback_config)
            else:
                content = json.dumps(fallback_config, indent=2)

            config_path.write_text(content, encoding="utf-8")

            self.logger.info(f"Generated fallback configuration: {config_path}")

            return ConfigRepairResult(
                success=True,
                repaired_config=fallback_config,
                repair_actions=repair_actions,
                backup_created=False,
                backup_path=None,
            )

        except Exception as e:
            self.logger.error(f"Failed to write fallback configuration: {e}")
            return ConfigRepairResult(success=False, repaired_config=fallback_config, repair_actions=repair_actions)

    def _generate_aichat_fallback(self) -> dict[str, Any]:
        """Generate fallback aichat configuration."""
        return {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "top_p": 1.0,
            "clients": [{"type": "openai", "api_key": "${OPENAI_API_KEY}", "api_base": "https://api.openai.com/v1"}],
        }

    def _generate_codex_fallback(self) -> dict[str, Any]:
        """Generate fallback codex configuration."""
        return {
            "default_profile": "gpt4",
            "profiles": {
                "gpt4": {"model": "gpt-4", "provider": "openai", "temperature": 0.1, "max_tokens": 4096},
                "gpt35": {"model": "gpt-3.5-turbo", "provider": "openai", "temperature": 0.1, "max_tokens": 4096},
            },
        }

    def _generate_mods_fallback(self) -> dict[str, Any]:
        """Generate fallback mods configuration."""
        return {
            "default_model": "gpt-3.5-turbo",
            "models": ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"],
            "api": {"openai": {"api_key": "${OPENAI_API_KEY}", "base_url": "https://api.openai.com/v1"}},
        }

    def _generate_generic_fallback(self) -> dict[str, Any]:
        """Generate generic fallback configuration."""
        return {
            "version": "1.0",
            "created_by": "vexy-co-model-catalog",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "providers": {},
            "models": [],
        }

    def _create_backup(self, config_path: Path) -> Path | None:
        """Create backup of configuration file."""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = config_path.with_suffix(f".backup_{timestamp}{config_path.suffix}")
            shutil.copy2(config_path, backup_path)
            self.logger.info(f"Created configuration backup: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
            return None

    def _should_apply_repair(self, action: ConfigRepairAction, strategy: ConfigRepairStrategy) -> bool:
        """Determine if repair action should be applied based on strategy."""
        if strategy == ConfigRepairStrategy.AGGRESSIVE:
            return True
        if strategy == ConfigRepairStrategy.MODERATE:
            return action.confidence >= CONFIG_REPAIR_MODERATE_CONFIDENCE
        # CONSERVATIVE
        return action.confidence >= CONFIG_REPAIR_CONSERVATIVE_CONFIDENCE

    def _apply_repair_action(self, config: dict[str, Any], action: ConfigRepairAction) -> None:
        """Apply a repair action to the configuration."""
        path_parts = action.field_path.split(".")

        # Navigate to the parent of the target field
        current = config
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the repaired value
        if path_parts:
            current[path_parts[-1]] = action.new_value

    def _add_missing_template_fields(self, config: dict[str, Any], _config_path: Path) -> list[ConfigRepairAction]:
        """Add missing fields from configuration templates."""
        actions = []

        # Add basic required fields that are commonly missing
        required_fields = {"version": "1.0", "providers": {}, "models": []}

        for field, default_value in required_fields.items():
            if field not in config:
                actions.append(
                    ConfigRepairAction(
                        action_type="add_missing_field",
                        field_path=field,
                        old_value=None,
                        new_value=default_value,
                        reason=f"Added missing required field: {field}",
                        confidence=0.8,
                        fallback_source=FallbackSource.TEMPLATE_CONFIG,
                    )
                )

        return actions

    def _initialize_validation_rules(self) -> list[EnhancedValidationRule]:
        """Initialize enhanced validation rules with repair capabilities."""
        rules = []

        # Rule: Environment variable validation
        def validate_env_vars(config: dict[str, Any]) -> list[ValidationIssue]:
            return self.validate_environment_dependencies(config)

        rules.append(
            EnhancedValidationRule(
                name="environment_variables",
                description="Validate required environment variables",
                validator=validate_env_vars,
                severity=ValidationSeverity.ERROR,
                category="environment",
            )
        )

        # Rule: Provider configuration validation
        def validate_providers(config: dict[str, Any]) -> list[ValidationIssue]:
            return self.validate_provider_configs(config)

        rules.append(
            EnhancedValidationRule(
                name="provider_configs",
                description="Validate provider configurations",
                validator=validate_providers,
                severity=ValidationSeverity.ERROR,
                category="providers",
            )
        )

        return rules

    def _load_config_templates(self) -> dict[str, dict[str, Any]]:
        """Load configuration templates for fallback generation."""
        return {
            "aichat": self._generate_aichat_fallback(),
            "codex": self._generate_codex_fallback(),
            "mods": self._generate_mods_fallback(),
            "generic": self._generate_generic_fallback(),
        }

    def _get_env_var_mappings(self) -> dict[str, str]:
        """Get environment variable mappings for configuration fields."""
        return {
            "openai.api_key": "OPENAI_API_KEY",
            "anthropic.api_key": "ANTHROPIC_API_KEY",
            "groq.api_key": "GROQ_API_KEY",
            "deepinfra.api_key": "DEEPINFRA_API_KEY",
            "together.api_key": "TOGETHER_API_KEY",
        }


# Default instance for easy use
enhanced_config_validator = EnhancedConfigValidator()
