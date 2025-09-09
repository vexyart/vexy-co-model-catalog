"""
this_file: src/vexy_co_model_catalog/cli.py

Comprehensive CLI for AI model catalog management with provider fetching and config integration.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fire
from loguru import logger
from rich.console import Console
from rich.table import Table

from vexy_co_model_catalog import __version__
from vexy_co_model_catalog.core.analytics import initialize_analytics, finalize_analytics
from vexy_co_model_catalog.core.caching import (
    get_api_response_cache,
    get_model_cache,
    get_validation_cache,
    invalidate_provider_cache,
)
from vexy_co_model_catalog.core.cli_optimization_integration import (
    apply_cli_optimizations,
    optimize_cli_class,
    provider_command_cache,
    quick_command_cache,
    stats_command_cache,
)
from vexy_co_model_catalog.core.completion import get_completion_generator
from vexy_co_model_catalog.core.config import ConfigManager, ConfigTemplate
from vexy_co_model_catalog.core.config_validator import (
    ConfigFormat,
    ConfigValidator,
    ValidationSeverity,
)
from vexy_co_model_catalog.core.enhanced_user_experience import (
    get_configuration_guidance,
    get_enhanced_error_handler,
    get_enhanced_help_system,
)
from vexy_co_model_catalog.core.monitoring import get_monitoring_system
from vexy_co_model_catalog.core.failure_tracker import FailureTracker
from vexy_co_model_catalog.core.fetcher import ModelFetcher
from vexy_co_model_catalog.core.graceful_degradation import GracefulDegradationManager
from vexy_co_model_catalog.core.health_check import ProviderHealthChecker
from vexy_co_model_catalog.core.health_monitor import HealthStatus, get_health_monitor
from vexy_co_model_catalog.core.integrity import get_integrity_manager, verify_all_files
from vexy_co_model_catalog.core.model_validator import ModelDataValidator, ModelValidationSeverity
from vexy_co_model_catalog.core.performance import (
    initialize_performance_monitoring,
)
from vexy_co_model_catalog.core.production_deployment import (
    check_production_readiness,
    get_production_manager,
    get_production_metrics,
    initialize_production_mode,
)
from vexy_co_model_catalog.core.production_error_handling import get_error_summary
from vexy_co_model_catalog.core.production_graceful_degradation import get_system_status
from vexy_co_model_catalog.core.production_reliability import (
    ReliabilityLevel,
    get_production_reliability_hardening,
)
from vexy_co_model_catalog.core.provider import Model, ProviderConfig, get_all_providers, get_provider_by_name
from vexy_co_model_catalog.core.rate_limiter import get_rate_limiter
from vexy_co_model_catalog.core.storage import StorageManager
from vexy_co_model_catalog.core.validator import ProviderValidator

if TYPE_CHECKING:
    from vexy_co_model_catalog.core.monitoring import MonitoringSystem

# Constants for HTTP status codes and performance thresholds
HTTP_OK_START = 200
HTTP_OK_END = 300
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_TOO_MANY_REQUESTS = 429

# Performance and usage thresholds
PARTS_COUNT_FOR_MODELS_FILE = 2
MAX_RECOMMENDATIONS_DISPLAY = 5
CACHE_HIT_RATE_EXCELLENT = 80  # Green status threshold
CACHE_HIT_RATE_GOOD = 50       # Yellow status threshold
MEMORY_USAGE_HIGH = 80         # MB threshold for high memory warning
CPU_USAGE_CRITICAL = 90        # % threshold for critical CPU usage
CPU_USAGE_WARNING = 70         # % threshold for CPU warning
MEMORY_USAGE_CRITICAL = 90     # % threshold for critical memory usage
DISK_USAGE_CRITICAL = 95       # % threshold for critical disk usage (red status)
DISK_USAGE_WARNING = 85        # % threshold for warning disk usage (yellow status)
RESPONSE_TIME_CRITICAL = 10000 # ms threshold for critical response time (red status)
RESPONSE_TIME_WARNING = 3000   # ms threshold for warning response time (yellow status)

# Performance and display thresholds
MIN_RELIABILITY_THRESHOLD = 2
MAX_HEALTH_CHECK_ISSUES = 5
STATS_TABLE_WIDTH = 80
NAME_COLUMN_WIDTH = 50
PERFORMANCE_EXCELLENT_THRESHOLD = 90
PERFORMANCE_GOOD_THRESHOLD = 70

# Display and pagination constants
MAX_ALIASES_SHORT_DISPLAY = 3          # Maximum short aliases to display
MIN_COLUMN_WIDTH_COMMAND = 15          # Minimum width for command column
MIN_COLUMN_WIDTH_ALIASES = 30          # Minimum width for aliases column
MIN_COLUMN_WIDTH_USAGE = 35            # Minimum width for usage column
MAX_URL_DISPLAY_WIDTH = 40             # Maximum width for URL display
DEFAULT_MAX_CONCURRENCY = 8            # Default max concurrent operations
MAX_MODEL_LIMIT_CONFIG = 5             # Maximum models for config generation
MAX_INPUT_TOKENS_DEFAULT = 8192        # Default max input tokens for models
MAX_OUTPUT_TOKENS_DEFAULT = 4096       # Default max output tokens for models

# Failure and error thresholds
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Failures needed to open circuit breaker
DEFAULT_CLEANUP_DAYS = 30              # Default days for cleanup operations
SIGNIFICANT_PERFORMANCE_CHANGE = 20    # % change considered significant
MODERATE_PERFORMANCE_CHANGE = 10       # % change considered moderate
MINOR_PERFORMANCE_CHANGE = 5           # % change considered minor
MONITORING_INTERVAL_SECONDS = 300      # Default monitoring interval (5 minutes)
RULE_EVALUATION_INTERVAL = 30          # Alert rules evaluation interval

# Display limits for lists and results
MAX_SAMPLE_MODELS_DISPLAY = 3          # Maximum sample models to show
MAX_RECENT_EVENTS_DISPLAY = 10         # Maximum recent events to show
MAX_COMMANDS_HISTORY_DISPLAY = 20      # Maximum command history to show
MAX_ALERTS_DISPLAY = 10                # Maximum alerts to display
MAX_CRITICAL_RESULTS_DISPLAY = 3       # Maximum critical results to show
MAX_RECOMMENDATIONS_SHORT = 2          # Maximum short recommendations to show
ALERT_THRESHOLD_PERFORMANCE_CHANGE = 15 # % change threshold for alerts
ALERT_TABLE_COLUMN_WIDTH = 10          # Width for alert table columns
MIN_METRICS_FOR_INSIGHTS = 5           # Minimum metrics needed for performance insights

console = Console()


@optimize_cli_class
class CLI:
    """AI Model Catalog Manager - fetch, normalize, and manage model catalogs from 40+ providers."""

    def __init__(self) -> None:
        """Initialize CLI with storage and configuration managers."""
        self.storage = StorageManager()
        self.failure_tracker = FailureTracker(self.storage)
        self.config_manager = ConfigManager(self.storage.config_dir)

        # Initialize analytics collection (opt-in by environment variable)
        analytics_enabled = os.environ.get("VEXY_ANALYTICS_ENABLED", "true").lower() in ["true", "1", "yes"]
        analytics_dir = self.storage.root / "analytics"
        self.analytics = initialize_analytics(analytics_dir, enabled=analytics_enabled)

        if analytics_enabled:
            logger.debug("CLI analytics collection enabled")
        else:
            logger.debug("CLI analytics collection disabled")

        # Initialize performance monitoring
        performance_enabled = os.environ.get("VEXY_PERFORMANCE_ENABLED", "true").lower() in ["true", "1", "yes"]

        # Initialize production error handling if in production mode
        production_mode = os.environ.get("VMC_PRODUCTION_MODE", "false").lower() in ["true", "1", "yes"]
        if production_mode:
            self._initialize_production_mode()
            logger.info("Production mode enabled with enhanced error handling")
        if performance_enabled:
            performance_dir = self.storage.root / "performance"
            self.performance_monitor = initialize_performance_monitoring(performance_dir)
            logger.debug("CLI performance monitoring enabled")
        else:
            self.performance_monitor = None
            logger.debug("CLI performance monitoring disabled")

        # Command aliases for improved UX
        self.command_aliases = {
            # Short forms
            "ls": "providers",
            "list": "providers",
            "get": "fetch",
            "dl": "fetch",
            "download": "fetch",
            "sync": "fetch",
            "st": "stats",
            "status": "stats",
            "check": "validate",
            "test": "validate",
            "verify": "validate",
            "rm": "clean",
            "remove": "clean",
            "delete": "clean",
            "clear": "clean",
            "ln": "link",
            "connect": "link",
            "import": "migrate",
            "move": "migrate",
            "analytics": "show_analytics",
            "metrics": "show_analytics",
            "limits": "rate_limits",
            "throttle": "rate_limits",
            "health-check": "health",
            "healthcheck": "health",
            "diag": "health",
            "diagnostics": "health",
            "files": "integrity",
            "backup": "integrity",
            "restore": "integrity",
            # Setup and guidance
            "setup": "setup_wizard",
            "wizard": "setup_wizard",
            "configure": "setup_wizard",
            "init": "setup_wizard",
            # Production reliability
            "diagnose": "production_diagnostics",
            "diagnostic": "production_diagnostics",
            "prod-check": "production_diagnostics",
            "reliability": "production_health_advanced",
            "health-monitor": "production_health_advanced",
            # Common typos
            "provider": "providers",
            "fech": "fetch",
            "stat": "stats",
            "validat": "validate",
            "cach": "cache",
        }

        # Apply global CLI optimizations for better performance
        apply_cli_optimizations()

        # Initialize enhanced user experience components
        self.error_handler = get_enhanced_error_handler()
        self.help_system = get_enhanced_help_system()
        self.config_guidance = get_configuration_guidance()

    def _track_command(self, command_name: str, **kwargs) -> contextlib.AbstractContextManager[None]:
        """Helper method to safely track commands with analytics."""
        if self.analytics and self.analytics.enabled:
            return self.analytics.track_command(command_name, **kwargs)
        return contextlib.nullcontext()

    def _track_performance(
        self, command_name: str, metadata: dict | None = None
    ) -> contextlib.AbstractContextManager[None]:
        """Helper method to track command performance."""

        @contextlib.contextmanager
        def performance_context() -> None:
            """Context manager for performance monitoring."""
            if self.performance_monitor:
                self.performance_monitor.start_monitoring(command_name, metadata)
                try:
                    yield
                    self.performance_monitor.stop_monitoring(success=True)
                except Exception as e:
                    self.performance_monitor.stop_monitoring(success=False, error_message=str(e))
                    raise
            else:
                yield

        return performance_context()

    def __getattr__(self, name: str) -> Any:
        """Handle command aliases by redirecting to actual command methods."""
        if name in self.command_aliases:
            actual_command = self.command_aliases[name]
            if hasattr(self, actual_command):
                console.print(f"[dim]→ Using alias '{name}' for command '{actual_command}'[/dim]")
                return getattr(self, actual_command)
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def _parse_provider_list(self, providers: str) -> list[ProviderConfig]:
        """Parse provider string into list of ProviderConfig objects."""
        if providers == "all" or not providers:
            return get_all_providers()

        provider_names = [name.strip() for name in providers.split(",")]
        provider_configs = []
        for name in provider_names:
            provider = get_provider_by_name(name)
            if provider:
                provider_configs.append(provider)
            else:
                console.print(f"[red]Warning: Provider '{name}' not found[/red]")
        return provider_configs

    def _get_aliases_for_command(self, command: str) -> str:
        """Get formatted aliases for a command."""
        aliases = [alias for alias, cmd in self.command_aliases.items() if cmd == command]
        if aliases:
            # Show most common/useful aliases (limit to 3)
            short_aliases = sorted(aliases, key=len)[:MAX_ALIASES_SHORT_DISPLAY]
            return ", ".join(short_aliases)
        return "-"

    @quick_command_cache()
    def version(self) -> str:
        """Print and return the package version."""
        with self._track_performance("version", {"action": "show_version"}):
            console.print(f"[bold green]vexy-co-model-catalog[/bold green] v{__version__}")
            return __version__

    @quick_command_cache()
    def help(self) -> None:
        """Show comprehensive help with examples and usage patterns."""
        with self._track_performance("help", {"action": "show_help"}):
            self.help_system.show_comprehensive_help()

    def setup_wizard(self) -> None:
        """Run interactive setup wizard for first-time users."""
        with self._track_performance("setup_wizard", {"action": "run_wizard"}):
            with self._track_command("setup_wizard"):
                self.config_guidance.show_setup_wizard()

    def aliases(self) -> None:
        """Show all available command aliases for improved user experience."""
        console.print(f"\n[bold green]vexy-co-model-catalog[/bold green] v{__version__}")
        console.print("[dim]Available Command Aliases[/dim]\n")

        console.print("[bold cyan]🔗 Command Aliases:[/bold cyan]")

        # Group aliases by target command for better organization
        aliases_by_command = {}
        for alias, command in self.command_aliases.items():
            if command not in aliases_by_command:
                aliases_by_command[command] = []
            aliases_by_command[command].append(alias)

        alias_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
        alias_table.add_column("Command", style="cyan", no_wrap=True, min_width=MIN_COLUMN_WIDTH_COMMAND)
        alias_table.add_column("Available Aliases", style="yellow", min_width=MIN_COLUMN_WIDTH_ALIASES)
        alias_table.add_column("Usage Example", style="dim", min_width=MIN_COLUMN_WIDTH_USAGE)

        for command, aliases in sorted(aliases_by_command.items()):
            aliases_str = ", ".join(sorted(aliases, key=len))
            example_alias = sorted(aliases, key=len)[0]  # Use shortest alias
            alias_table.add_row(command, aliases_str, f"python -m vexy_co_model_catalog {example_alias}")

        console.print(alias_table)

        console.print("\n[bold yellow]💡 Examples:[/bold yellow]")
        console.print("  [green]python -m vexy_co_model_catalog ls[/green]           # Same as 'providers'")
        console.print("  [green]python -m vexy_co_model_catalog get --providers=openai[/green]  # Same as 'fetch'")
        console.print("  [green]python -m vexy_co_model_catalog st[/green]            # Same as 'stats'")
        console.print("  [green]python -m vexy_co_model_catalog check[/green]         # Same as 'validate'")
        console.print("  [green]python -m vexy_co_model_catalog diag[/green]          # Same as 'health'")
        console.print("  [green]python -m vexy_co_model_catalog rm --temp[/green]     # Same as 'clean'")
        console.print()

    def completion(self, shell: str = "bash", *, install: bool = False) -> None:
        """Generate shell completion scripts for enhanced CLI user experience.

        Args:
            shell: Target shell (bash, zsh, fish)
            install: Show installation instructions instead of script
        """
        with self._track_performance("completion", {"shell": shell, "install": install}):
            with self._track_command("completion", shell=shell):
                completion_gen = get_completion_generator()

                if install:
                    # Show installation instructions
                    instructions = completion_gen.install_completion_scripts(shell)
                    console.print(f"[bold cyan]Shell Completion Installation for {shell.title()}[/bold cyan]")
                    console.print(instructions)
                    console.print("\n[bold green]✨ Once installed, you'll have:[/bold green]")
                    console.print("  • Tab completion for all commands")
                    console.print("  • Automatic provider name completion")
                    console.print("  • Context-aware option suggestions")
                    console.print("  • File path completion for config files")
                else:
                    # Generate and output the completion script
                    try:
                        script = completion_gen.generate_completion_script(shell)
                        console.print(f"[dim]# {shell.title()} completion script for vexy-co-model-catalog[/dim]")
                        console.print(script)
                        console.print(f"\n[dim]# To install: vexy completion {shell} --install[/dim]")
                    except ValueError as e:
                        console.print(f"[red]Error: {e}[/red]")
                        console.print("[yellow]Supported shells: bash, zsh, fish[/yellow]")

    # Command aliases - create actual methods for Fire CLI compatibility
    def ls(self, **kwargs) -> None:
        """Alias for 'providers' command. List all AI providers with status and configuration."""
        console.print("[dim]→ Using alias 'ls' for command 'providers'[/dim]")
        return self.providers(**kwargs)

    def list(self, **kwargs) -> None:
        """Alias for 'providers' command. List all AI providers with status and configuration."""
        console.print("[dim]→ Using alias 'list' for command 'providers'[/dim]")
        return self.providers(**kwargs)

    def get(self, **kwargs) -> None:
        """Alias for 'fetch' command. Fetch model catalogs from providers."""
        console.print("[dim]→ Using alias 'get' for command 'fetch'[/dim]")
        return self.fetch(**kwargs)

    def dl(self, **kwargs) -> None:
        """Alias for 'fetch' command. Download model catalogs from providers."""
        console.print("[dim]→ Using alias 'dl' for command 'fetch'[/dim]")
        return self.fetch(**kwargs)

    def download(self, **kwargs) -> None:
        """Alias for 'fetch' command. Download model catalogs from providers."""
        console.print("[dim]→ Using alias 'download' for command 'fetch'[/dim]")
        return self.fetch(**kwargs)

    def sync(self, **kwargs) -> None:
        """Alias for 'fetch' command. Sync model catalogs from providers."""
        console.print("[dim]→ Using alias 'sync' for command 'fetch'[/dim]")
        return self.fetch(**kwargs)

    def st(self, **kwargs) -> None:
        """Alias for 'stats' command. Show system statistics."""
        console.print("[dim]→ Using alias 'st' for command 'stats'[/dim]")
        return self.stats(**kwargs)

    def status(self, **kwargs) -> None:
        """Alias for 'stats' command. Show system status and statistics."""
        console.print("[dim]→ Using alias 'status' for command 'stats'[/dim]")
        return self.stats(**kwargs)

    def check(self, **kwargs) -> None:
        """Alias for 'validate' command. Check provider configurations."""
        console.print("[dim]→ Using alias 'check' for command 'validate'[/dim]")
        return self.validate(**kwargs)

    def test(self, **kwargs) -> None:
        """Alias for 'validate' command. Test provider configurations."""
        console.print("[dim]→ Using alias 'test' for command 'validate'[/dim]")
        return self.validate(**kwargs)

    def verify(self, **kwargs) -> None:
        """Alias for 'validate' command. Verify provider configurations."""
        console.print("[dim]→ Using alias 'verify' for command 'validate'[/dim]")
        return self.validate(**kwargs)

    def rm(self, **kwargs) -> None:
        """Alias for 'clean' command. Remove/clean temporary files and configs."""
        console.print("[dim]→ Using alias 'rm' for command 'clean'[/dim]")
        return self.clean(**kwargs)

    def clear(self, **kwargs) -> None:
        """Alias for 'clean' command. Clear temporary files and configs."""
        console.print("[dim]→ Using alias 'clear' for command 'clean'[/dim]")
        return self.clean(**kwargs)

    def remove(self, **kwargs) -> None:
        """Alias for 'clean' command. Remove temporary files and configs."""
        console.print("[dim]→ Using alias 'remove' for command 'clean'[/dim]")
        return self.clean(**kwargs)

    def diag(self, **kwargs) -> None:
        """Alias for 'health' command. Run system diagnostics."""
        console.print("[dim]→ Using alias 'diag' for command 'health'[/dim]")
        return self.health(**kwargs)

    def diagnostics(self, **kwargs) -> None:
        """Alias for 'health' command. Run system diagnostics."""
        console.print("[dim]→ Using alias 'diagnostics' for command 'health'[/dim]")
        return self.health(**kwargs)

    def healthcheck(self, **kwargs) -> None:
        """Alias for 'health' command. Run health checks."""
        console.print("[dim]→ Using alias 'healthcheck' for command 'health'[/dim]")
        return self.health(**kwargs)

    @provider_command_cache()
    def providers(self, action: str = "list", name: str = "", kind: str = "openai", **kwargs) -> None:
        """Manage providers: list, add, remove, or show provider details."""
        if action == "list":
            self._list_providers()
        elif action == "show" and name:
            self._show_provider(name)
        elif action == "add" and name:
            self._add_provider(name, kind, **kwargs)
        elif action == "remove" and name:
            self._remove_provider(name)
        else:
            console.print("[red]Usage: providers [list|show|add|remove] [--name=<name>] [options][/red]")

    def _list_providers(self) -> None:
        """List all registered providers with their status."""
        providers = get_all_providers()

        table = Table(title="Registered AI Providers", show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Status", justify="center")
        table.add_column("API Key Env", style="dim")
        table.add_column("Base URL", style="blue", max_width=MAX_URL_DISPLAY_WIDTH)

        for provider in providers:
            # Check if provider has failed recently
            is_failed = self.failure_tracker.is_provider_failed(provider.name)
            status = "[red]Failed[/red]" if is_failed else "[green]Ready[/green]"

            # Check if API key is available
            if provider.api_key_env and os.environ.get(provider.api_key_env):
                status += " 🔑"

            table.add_row(
                provider.name, provider.kind.value, status, provider.api_key_env or "", provider.get_base_url() or ""
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(providers)} providers configured[/dim]")

    def _show_provider(self, name: str) -> None:
        """Show detailed information about a specific provider."""
        provider = get_provider_by_name(name)
        if not provider:
            console.print(f"[red]Provider '{name}' not found[/red]")
            return

        failure_info = self.failure_tracker.get_provider_failure_info(name)

        console.print(f"\n[bold cyan]Provider: {provider.name}[/bold cyan]")
        console.print(f"Type: {provider.kind.value}")
        console.print(f"API Key Env: {provider.api_key_env}")
        console.print(f"Base URL: {provider.get_base_url()}")

        if failure_info.get("failure_count", 0) > 0:
            console.print("\n[yellow]Reliability Stats:[/yellow]")
            console.print(f"  Success Count: {failure_info.get('success_count', 0)}")
            console.print(f"  Failure Count: {failure_info.get('failure_count', 0)}")
            console.print(f"  Currently Failed: {failure_info.get('failed', False)}")

            if failure_info.get("last_failure"):
                console.print(f"  Last Failure: {failure_info['last_failure']}")

    def _add_provider(self, _name: str, _kind: str, **_kwargs) -> None:
        """Add a new provider (not implemented - providers are configured in code)."""
        console.print("[yellow]Provider management is currently read-only[/yellow]")
        console.print("Providers are configured in the PROVIDER_CONFIG constant")

    def _remove_provider(self, _name: str) -> None:
        """Remove a provider (not implemented - providers are configured in code)."""
        console.print("[yellow]Provider management is currently read-only[/yellow]")
        console.print("Use failure tracking to disable problematic providers")

    def fetch(
        self,
        providers: str = "",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        *,
        force: bool = False,
        output_json: bool = True,
        output_txt: bool = True,
        output_config: bool = False,
        only_successful: bool = False,
        legacy_output: bool = False,
    ) -> None:
        """
        Fetch model catalogs from AI providers.

        Args:
            providers: Comma-separated provider names, or 'all' for all providers
            force: Force refetch even if files already exist
            max_concurrency: Maximum concurrent requests
            output_json: Generate JSON files (default: True)
            output_txt: Generate TXT model lists (default: True)
            output_config: Generate tool config files (default: False)
            only_successful: Only process providers that aren't marked as failed
            legacy_output: Use legacy models/ directory structure (compatible with dump_models.py)
        """
        # Count provider parameters for analytics
        provider_list = [p.strip() for p in providers.split(",") if p.strip()] if providers != "all" else []
        provider_count = len(provider_list) if provider_list else len(get_all_providers())

        # Performance metadata for tracking
        perf_metadata = {
            "providers": providers if providers != "all" else f"all ({provider_count})",
            "provider_count": provider_count,
            "max_concurrency": max_concurrency,
            "force": force,
            "output_formats": [
                f for f, enabled in [("json", output_json), ("txt", output_txt), ("config", output_config)] if enabled
            ],
            "only_successful": only_successful,
            "legacy_output": legacy_output,
        }

        # Track command usage with analytics and performance
        with (
            self._track_command(
                "fetch",
                parameter_count=sum(
                    [
                        1
                        for p in [
                            providers,
                            max_concurrency,
                            output_json,
                            output_txt,
                            output_config,
                            only_successful,
                            legacy_output,
                        ]
                        if p != ""
                    ]
                ),
                has_force_flag=force,
                has_config_flags=(output_json or output_txt or output_config),
                provider_count=provider_count,
            ),
            self._track_performance("fetch", perf_metadata),
        ):
            asyncio.run(
                self._fetch_async(
                    providers,
                    max_concurrency,
                    force=force,
                    output_json=output_json,
                    output_txt=output_txt,
                    output_config=output_config,
                    only_successful=only_successful,
                    legacy_output=legacy_output,
                )
            )

    async def _fetch_async(
        self,
        providers: str,
        max_concurrency: int,
        *,
        force: bool,
        output_json: bool,
        output_txt: bool,
        output_config: bool,
        only_successful: bool,
        legacy_output: bool = False,
    ) -> None:
        """Async implementation of fetch command."""
        # Parse provider list
        provider_configs = self._parse_provider_list(providers)

        if not provider_configs:
            console.print("[red]No valid providers specified[/red]")
            return

        # Filter out failed providers if requested
        if only_successful:
            original_count = len(provider_configs)
            provider_configs = [p for p in provider_configs if not self.failure_tracker.is_provider_failed(p.name)]
            if len(provider_configs) < original_count:
                skipped = original_count - len(provider_configs)
                console.print(f"[yellow]Skipped {skipped} failed providers[/yellow]")

        console.print(f"[blue]Fetching from {len(provider_configs)} providers...[/blue]")
        if legacy_output:
            console.print("[dim]Using legacy models/ directory structure (compatible with dump_models.py)[/dim]")
        else:
            console.print("[dim]Using config/ directory structure[/dim]")

        # Validate provider configurations before fetching
        console.print("[dim]Validating provider configurations...[/dim]")
        validator = ProviderValidator()
        validation_summary = validator.validate_providers(provider_configs)

        if validation_summary.invalid_providers > 0:
            console.print(
                f"[yellow]⚠ Found {validation_summary.invalid_providers} providers with configuration issues:[/yellow]"
            )

            for result in validation_summary.results:
                if not result.is_valid:
                    console.print(f"\n[red]✗ {result.provider_name}:[/red]")
                    for issue in result.issues:
                        console.print(f"  • {issue}")
                    if result.guidance:
                        console.print("  [cyan]💡 Guidance:[/cyan]")
                        for guide in result.guidance:
                            console.print(f"    - {guide}")
                elif result.warnings:
                    console.print(f"\n[yellow]⚠ {result.provider_name}:[/yellow]")
                    for warning in result.warnings:
                        console.print(f"  • {warning}")

            console.print(
                f"\n[cyan]Validation Summary:[/cyan] "
                f"{validation_summary.valid_providers}/{validation_summary.total_providers} providers ready "
                f"({validation_summary.success_rate:.1f}%)"
            )

            # Remove invalid providers from the list
            provider_configs = [p for p in provider_configs if validator.validate_provider(p).is_valid]
            if not provider_configs:
                console.print(
                    "[red]No valid providers remaining after validation. Please fix configuration issues.[/red]"
                )
                return
        else:
            console.print("[green]✓ All provider configurations valid[/green]")

        # Initialize graceful degradation manager
        degradation_manager = GracefulDegradationManager(max_retries=2, retry_delay_base=1.0)

        # Create provider operation wrapper
        async def fetch_provider_operation(provider: ProviderConfig) -> None:
            """Wrapper function for fetching a single provider with graceful degradation."""
            # Check if files exist and force not set
            if not force and self._files_exist(provider.name, legacy_output=legacy_output):
                console.print(f"[dim]Skipping {provider.name} (files exist, use --force to overwrite)[/dim]")
                return None  # Signal skip

            # Check for API key if needed
            if provider.kind.value != "url" and provider.api_key_env:
                if not os.environ.get(provider.api_key_env):
                    console.print(f"[yellow]Skipping {provider.name} (missing {provider.api_key_env})[/yellow]")
                    return None  # Signal skip

            # Fetch provider data
            async with ModelFetcher(max_concurrency=max_concurrency, failure_tracker=self.failure_tracker) as fetcher:
                data = await fetcher.fetch_provider_models(provider)
                sorted_data = fetcher.sort_json_data(data)
                model_ids = fetcher.extract_model_ids(sorted_data)

                # Output files based on flags and legacy mode
                if output_json:
                    if legacy_output:
                        # Use legacy models/ directory structure (dump_models.py compatibility)
                        self.storage.write_json(f"models_{provider.name}", sorted_data)
                    else:
                        # Use new config/ directory structure
                        self.storage.write_config_json(f"models_{provider.name}", sorted_data)

                if output_txt:
                    if legacy_output:
                        # Use legacy models/ directory structure (dump_models.py compatibility)
                        self.storage.write_text(f"models_{provider.name}", model_ids)
                    else:
                        # Use new config/ directory structure
                        self.storage.write_config_txt(f"models_{provider.name}", model_ids)

                if output_config:
                    await self._generate_provider_configs(provider, model_ids)

                return {"model_count": len(model_ids)}

        # Progress callback for real-time updates
        def progress_callback(message: str, result: Any) -> None:
            """Display progress messages with appropriate styling."""
            if result.result.value == "retrying":
                console.print(f"[yellow]{message}[/yellow]")
            else:
                console.print(f"[cyan]{message}[/cyan]")

        # Execute batch operation with graceful degradation
        batch_result = await degradation_manager.execute_batch_operation(
            provider_configs, fetch_provider_operation, "provider_fetch", progress_callback
        )

        # Print real-time results
        for result in batch_result.provider_results:
            if result.is_success:
                console.print(f"[green]✓ {result.provider_name}: {result.model_count} models[/green]")
            elif result.result.value == "skipped":
                # Skip message already printed in operation
                pass
            else:
                error_type = f" ({result.failure_type.value})" if result.failure_type else ""
                retry_info = f" after {result.retry_count} retries" if result.retry_count > 0 else ""
                console.print(f"[red]✗ {result.provider_name}: {result.error_message}{error_type}{retry_info}[/red]")

        # Enhanced summary with graceful degradation insights
        console.print(
            f"\n[bold]Fetch Complete:[/bold] {batch_result.successful} successful, "
            f"{batch_result.failed_permanent + batch_result.failed_temporary} failed, "
            f"{batch_result.skipped} skipped"
        )

        if batch_result.retries_attempted > 0:
            console.print(f"[dim]Retries attempted: {batch_result.retries_attempted}[/dim]")

        if batch_result.has_partial_success:
            console.print(
                f"[green]✓ Partial success achieved: {batch_result.success_rate:.1f}% completion rate[/green]"
            )

        # Show fallback strategies if there were failures
        if batch_result.failed_temporary > 0 or batch_result.failed_permanent > 0:
            strategies = degradation_manager.generate_fallback_strategies(batch_result)
            if strategies:
                console.print("\n[bold cyan]💡 Suggested Actions:[/bold cyan]")
                for strategy in strategies:
                    console.print(f"  {strategy}")

        # Legacy failure tracker update for compatibility
        if batch_result.failed_permanent + batch_result.failed_temporary > 0:
            failure_summary = self.failure_tracker.get_failure_summary()
            console.print(
                f"\n[dim]Provider failure tracking: {failure_summary['currently_failed']} currently failed[/dim]"
            )

    def _files_exist(self, provider_name: str, *, legacy_output: bool = False) -> bool:
        """Check if output files already exist for a provider."""
        if legacy_output:
            # Check legacy models/ directory structure
            json_files = self.storage.list_files("json", f"models_{provider_name}.json")
            txt_files = self.storage.list_files("text", f"models_{provider_name}.txt")
        else:
            # Check new config/ directory structure
            json_files = self.storage.list_files("config_json", f"models_{provider_name}.json")
            txt_files = self.storage.list_files("config_txt", f"models_{provider_name}.txt")
        return len(json_files) > 0 and len(txt_files) > 0

    async def _generate_provider_configs(self, provider: ProviderConfig, model_ids: list[str]) -> None:
        """Generate tool-specific config files for a provider."""

        # Convert model IDs to Model objects (simplified)
        models = [
            Model(
                id=model_id,
                provider=provider.name,
                max_input_tokens=MAX_INPUT_TOKENS_DEFAULT,
                max_output_tokens=MAX_OUTPUT_TOKENS_DEFAULT,
            )
            for model_id in model_ids[:MAX_MODEL_LIMIT_CONFIG]  # Limit to first models for config generation
        ]

        template = ConfigTemplate(tool_name="", provider_id=provider.name, models=models, provider_config=provider)

        # Generate configs for each supported tool
        tools = ["aichat", "codex", "mods"]
        validator = ConfigValidator()

        for tool in tools:
            try:
                parser = self.config_manager.get_parser(tool)
                config_data = parser.generate(template)

                # Validate generated config before writing
                format_type = ConfigFormat.TOML if tool == "codex" else ConfigFormat.YAML
                validation_result = validator.validate_generated_config(config_data, tool, format_type)

                if validation_result.is_valid:
                    # Write per-provider config file
                    if tool == "codex":
                        self.storage.write_config_toml(f"models_{provider.name}", config_data, directory="codex")
                    else:
                        self.storage.write_yaml(f"models_{provider.name}", config_data, directory=tool)

                    logger.debug(f"Generated valid {tool} config for {provider.name}")
                else:
                    # Log validation issues but don't fail completely
                    error_count = sum(
                        1 for issue in validation_result.issues if issue.severity == ValidationSeverity.ERROR
                    )
                    warning_count = sum(
                        1 for issue in validation_result.issues if issue.severity == ValidationSeverity.WARNING
                    )

                    if error_count > 0:
                        logger.warning(
                            f"Generated {tool} config for {provider.name} has {error_count} errors - skipping write"
                        )
                        for issue in validation_result.issues:
                            if issue.severity == ValidationSeverity.ERROR:
                                logger.warning(f"  Config error: {issue.message}")
                    else:
                        # Only warnings, write anyway but log them
                        if tool == "codex":
                            self.storage.write_config_toml(f"models_{provider.name}", config_data, directory="codex")
                        else:
                            self.storage.write_yaml(f"models_{provider.name}", config_data, directory=tool)

                        logger.debug(f"Generated {tool} config for {provider.name} with {warning_count} warnings")
                        for issue in validation_result.issues:
                            if issue.severity == ValidationSeverity.WARNING:
                                logger.debug(f"  Config warning: {issue.message}")

            except Exception as e:
                {
                    'tool': tool,
                    'provider': provider.name,
                    'error_type': type(e).__name__,
                    'error_msg': str(e)
                }

                # Provide specific recovery guidance based on error type
                if isinstance(e, PermissionError):
                    recovery_msg = (
                        f"Permission denied while generating {tool} config for {provider.name}. "
                        f"Try: 1) Check write permissions for config directory, "
                        f"2) Run 'chmod 755 config/{tool}' to fix permissions, "
                        f"3) Verify you can create files in the target location."
                    )
                elif isinstance(e, FileNotFoundError):
                    recovery_msg = (
                        f"Config directory missing for {tool} while processing {provider.name}. "
                        f"Try: 1) Run 'mkdir -p config/{tool}' to create directory, "
                        f"2) Use --ensure-dirs flag to auto-create directories, "
                        f"3) Verify config root path is correct."
                    )
                elif isinstance(e, OSError | IOError):
                    recovery_msg = (
                        f"I/O error generating {tool} config for {provider.name}: {e}. "
                        f"Try: 1) Check available disk space, 2) Verify target path exists, "
                        f"3) Check for filesystem corruption, 4) Retry with --force flag."
                    )
                elif "json" in str(e).lower() or "yaml" in str(e).lower() or "toml" in str(e).lower():
                    recovery_msg = (
                        f"Configuration format error for {tool}/{provider.name}: {e}. "
                        f"Try: 1) Verify provider data structure, 2) Check for special characters in model names, "
                        f"3) Run validate command on provider data, 4) Report issue if provider data is malformed."
                    )
                else:
                    recovery_msg = (
                        f"Unexpected error generating {tool} config for {provider.name}: {e}. "
                        f"Try: 1) Run with --verbose for detailed logs, 2) Retry with --force flag, "
                        f"3) Check provider {provider.name} data integrity, 4) Report issue if persistent."
                    )

                logger.warning(recovery_msg)

    def validate(self, providers: str = "all") -> None:
        """
        Validate provider configurations and environment variables.

        Args:
            providers: Comma-separated provider names, or 'all' for all providers
        """
        # Count providers for analytics
        provider_list = [p.strip() for p in providers.split(",") if p.strip()] if providers != "all" else []
        provider_count = len(provider_list) if provider_list else len(get_all_providers())

        # Track command usage with analytics
        with self._track_command(
            "validate", provider_count=provider_count, parameter_count=1 if providers != "all" else 0
        ):
            # Parse provider list
            if providers == "all" or not providers:
                provider_configs = get_all_providers()
            else:
                provider_names = [name.strip() for name in providers.split(",")]
                provider_configs = []
                for name in provider_names:
                    provider = get_provider_by_name(name)
                    if provider:
                        provider_configs.append(provider)
                    else:
                        console.print(f"[red]Warning: Provider '{name}' not found[/red]")

            if not provider_configs:
                console.print("[red]No valid providers specified[/red]")
                return

            console.print(f"[blue]Validating {len(provider_configs)} provider configurations...[/blue]")

            validator = ProviderValidator()
            validation_summary = validator.validate_providers(provider_configs)

            # Display detailed results
            for result in validation_summary.results:
                if not result.is_valid:
                    console.print(f"\n[red]✗ {result.provider_name} - INVALID[/red]")
                    for issue in result.issues:
                        console.print(f"  🔴 {issue}")
                elif result.warnings:
                    console.print(f"\n[yellow]⚠ {result.provider_name} - Valid with warnings[/yellow]")
                    for warning in result.warnings:
                        console.print(f"  🟡 {warning}")
                else:
                    console.print(f"\n[green]✓ {result.provider_name} - Valid[/green]")

                # Show guidance if available
                if result.guidance:
                    console.print("  [cyan]💡 Guidance:[/cyan]")
                    for guide in result.guidance:
                        console.print(f"    - {guide}")

            # Summary
            console.print("\n[bold cyan]Validation Summary:[/bold cyan]")
            console.print(f"  Total providers: {validation_summary.total_providers}")
            console.print(f"  Valid providers: {validation_summary.valid_providers}")
            console.print(f"  Invalid providers: {validation_summary.invalid_providers}")
            console.print(f"  Success rate: {validation_summary.success_rate:.1f}%")

            if validation_summary.invalid_providers > 0:
                console.print(
                    "\n[yellow]💡 Run validation again after fixing issues to verify your configuration.[/yellow]"
                )

    def provider_health(self, providers: str = "all", timeout: float = 10.0) -> None:
        """
        Check provider endpoint health and connectivity status.

        Args:
            providers: Comma-separated provider names, or 'all' for all providers
            timeout: Request timeout in seconds for health checks
        """
        # Parse provider list
        provider_configs = self._parse_provider_list(providers)

        if not provider_configs:
            console.print("[red]No valid providers specified[/red]")
            return

        console.print(f"[blue]Checking health of {len(provider_configs)} providers...[/blue]")

        # Run health checks asynchronously
        asyncio.run(self._health_check_async(provider_configs, timeout))

    async def _health_check_async(self, provider_configs: list[ProviderConfig], timeout: float) -> None:
        """Async implementation of health check command."""
        async with ProviderHealthChecker(timeout=timeout) as health_checker:
            health_summary = await health_checker.check_multiple_providers(provider_configs)

            # Display detailed results
            for result in health_summary.results:
                status_emoji = {
                    HealthStatus.HEALTHY: "✅",
                    HealthStatus.DEGRADED: "⚠️",
                    HealthStatus.UNHEALTHY: "❌",
                    HealthStatus.UNKNOWN: "❓",
                }.get(result.status, "❓")

                status_color = {
                    HealthStatus.HEALTHY: "green",
                    HealthStatus.DEGRADED: "yellow",
                    HealthStatus.UNHEALTHY: "red",
                    HealthStatus.UNKNOWN: "dim",
                }.get(result.status, "dim")

                status_text = f"{status_emoji} [{status_color}]{result.provider_name}[/{status_color}]"
                status_message = f"{status_text} - {result.status.value.title()}"
                console.print(status_message)

                if result.response_time_ms > 0:
                    console.print(f"   Response time: {result.response_time_ms}ms")

                if result.endpoint_tested:
                    console.print(f"   Endpoint: [dim]{result.endpoint_tested}[/dim]")

                if result.status_code:
                    color = (
                        "green"
                        if HTTP_OK_START <= result.status_code < HTTP_OK_END
                        else "yellow"
                        if result.status_code in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN, HTTP_TOO_MANY_REQUESTS)
                        else "red"
                    )
                    console.print(f"   HTTP status: [{color}]{result.status_code}[/{color}]")

                if result.error_message:
                    console.print(f"   Error: [red]{result.error_message}[/red]")

                console.print()

            # Summary
            console.print("[bold cyan]Health Summary:[/bold cyan]")
            console.print(f"  Total providers: {health_summary.total_providers}")
            console.print(f"  ✅ Healthy: {health_summary.healthy_providers}")
            console.print(f"  ⚠️ Degraded: {health_summary.degraded_providers}")
            console.print(f"  ❌ Unhealthy: {health_summary.unhealthy_providers}")
            console.print(f"  ❓ Unknown: {health_summary.unknown_providers}")
            console.print(f"  Overall health: {health_summary.health_percentage:.1f}%")

            # Recommendations
            if health_summary.unhealthy_providers > 0 or health_summary.degraded_providers > 0:
                console.print("\n[yellow]💡 Recommendations:[/yellow]")

                if health_summary.degraded_providers > 0:
                    console.print("  • Check API keys for degraded providers (401/403 errors)")
                    console.print("  • Consider rate limiting for providers returning 429 errors")

                if health_summary.unhealthy_providers > 0:
                    console.print("  • Verify network connectivity for unhealthy providers")
                    console.print("  • Check if provider endpoints have changed")
                    console.print("  • Use 'validate' command to check configuration")

    def validate_config(self, config_path: str, tool_name: str = "") -> None:
        """
        Validate configuration file syntax and schema.

        Args:
            config_path: Path to configuration file to validate
            tool_name: Tool name for schema validation (aichat, codex, mods)
        """
        config_file = Path(config_path)

        if not config_file.exists():
            console.print(f"[red]Configuration file not found: {config_path}[/red]")
            return

        # Detect tool name from path if not provided
        if not tool_name:
            tool_name = self._detect_tool_from_path(config_file)

        console.print(f"[blue]Validating configuration file: {config_path}[/blue]")
        if tool_name:
            console.print(f"[dim]Tool: {tool_name}[/dim]")

        # Validate the configuration
        validator = ConfigValidator()
        result = validator.validate_file(config_path)

        # Display results
        if result.is_valid:
            console.print(f"[green]✅ Configuration is valid ({result.format_type.value.upper()})[/green]")
        else:
            console.print(f"[red]❌ Configuration has issues ({result.format_type.value.upper()})[/red]")

        # Display issues
        if result.issues:
            console.print("\n[bold]Issues Found:[/bold]")

            for issue in result.issues:
                severity_emoji = {
                    ValidationSeverity.ERROR: "🔴",
                    ValidationSeverity.WARNING: "🟡",
                    ValidationSeverity.INFO: "ℹ️",
                }.get(issue.severity, "ℹ️")

                severity_color = {
                    ValidationSeverity.ERROR: "red",
                    ValidationSeverity.WARNING: "yellow",
                    ValidationSeverity.INFO: "blue",
                }.get(issue.severity, "blue")

                severity_text = f"{severity_emoji} [{severity_color}]{issue.severity.value.upper()}[/{severity_color}]"
                console.print(f"{severity_text}: {issue.message}")

                if issue.line_number:
                    console.print(f"   Line: {issue.line_number}")

                if issue.field_path:
                    console.print(f"   Field: [dim]{issue.field_path}[/dim]")

                if issue.suggestion:
                    console.print(f"   [cyan]💡 Suggestion: {issue.suggestion}[/cyan]")

                console.print()

        # Summary statistics
        if result.issues:
            error_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.ERROR)
            warning_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.WARNING)
            info_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.INFO)

            console.print("[bold cyan]Summary:[/bold cyan]")
            console.print(f"  🔴 Errors: {error_count}")
            console.print(f"  🟡 Warnings: {warning_count}")
            console.print(f"  ℹ️ Info: {info_count}")

            if error_count > 0:
                console.print("\n[red]💡 Fix errors before using this configuration file.[/red]")
            elif warning_count > 0:
                console.print("\n[yellow]💡 Review warnings to ensure configuration works as expected.[/yellow]")
            else:
                console.print("\n[green]💡 Configuration looks good![/green]")

    def _detect_tool_from_path(self, config_file: Path) -> str:
        """Detect tool name from configuration file path."""
        path_str = str(config_file).lower()

        if "aichat" in path_str:
            return "aichat"
        if "codex" in path_str:
            return "codex"
        if "mods" in path_str:
            return "mods"
        return ""

    def validate_models(self, file_path: str, provider_name: str = "") -> None:
        """
        Validate and normalize model metadata from provider response data.

        Args:
            file_path: Path to JSON file containing provider response data
            provider_name: Provider name for validation context
        """
        data_file = Path(file_path)

        if not data_file.exists():
            console.print(f"[red]Data file not found: {file_path}[/red]")
            return

        # Detect provider name from path if not provided
        if not provider_name:
            provider_name = self._detect_provider_from_path(data_file)

        if not provider_name:
            console.print("[yellow]Could not detect provider from path. Use --provider-name to specify.[/yellow]")
            return

        # Get provider configuration
        provider_config = get_provider_by_name(provider_name)
        if not provider_config:
            console.print(f"[red]Provider '{provider_name}' not found[/red]")
            return

        console.print(f"[blue]Validating model data: {file_path}[/blue]")
        console.print(f"[dim]Provider: {provider_name}[/dim]")

        # Load and validate the data
        try:
            with data_file.open("r", encoding="utf-8") as f:
                response_data = json.load(f)
        except FileNotFoundError:
            console.print(f"[red]JSON file not found: {file_path}[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print(f"  1) Verify the file path: {file_path}")
            console.print("  2) Run 'fetch' command first to generate provider data")
            console.print("  3) Check if file was moved or deleted")
            return
        except PermissionError:
            console.print(f"[red]Permission denied reading: {file_path}[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print(f"  1) Check file permissions: ls -la {file_path}")
            console.print(f"  2) Fix permissions: chmod 644 {file_path}")
            console.print("  3) Run command with appropriate user privileges")
            return
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON format in {file_path}[/red]")
            console.print(f"[red]Error at line {e.lineno}, column {e.colno}: {e.msg}[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print("  1) Validate JSON syntax at jsonlint.com")
            console.print("  2) Re-fetch provider data to regenerate file")
            console.print("  3) Check for file corruption or manual edits")
            console.print(f"  4) Backup and regenerate: mv {file_path} {file_path}.backup")
            return
        except Exception as e:
            console.print(f"[red]Failed to load JSON file {file_path}: {e}[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print("  1) Verify file isn't corrupted or locked by another process")
            console.print("  2) Check available memory (file might be too large)")
            console.print("  3) Re-fetch provider data to regenerate file")
            console.print("  4) Run with --verbose for detailed error information")
            return

        # Validate the model data
        validator = ModelDataValidator()
        result = validator.validate_and_normalize(response_data, provider_config)

        # Display results
        console.print("\n[bold cyan]Model Validation Results:[/bold cyan]")
        console.print(f"  Original models: {result.original_count}")
        console.print(f"  Normalized models: {result.normalized_count}")
        console.print(f"  Success rate: {result.success_rate:.1f}%")

        if result.is_valid:
            console.print("[green]✅ Model data validation passed[/green]")
        else:
            console.print("[red]❌ Model data validation failed[/red]")

        # Display sample normalized models
        if result.models:
            console.print("\n[bold]Sample Normalized Models (showing first 3):[/bold]")
            for i, model in enumerate(result.models[:MAX_SAMPLE_MODELS_DISPLAY]):
                console.print(f"\n[cyan]{i + 1}. {model.id}[/cyan]")
                if model.name and model.name != model.id:
                    console.print(f"   Name: {model.name}")
                if model.max_input_tokens:
                    console.print(f"   Max input tokens: {model.max_input_tokens:,}")
                if model.max_output_tokens:
                    console.print(f"   Max output tokens: {model.max_output_tokens:,}")
                if model.context_length:
                    console.print(f"   Context length: {model.context_length:,}")
                if model.input_price:
                    console.print(f"   Input price: ${model.input_price:.6f}")
                if model.output_price:
                    console.print(f"   Output price: ${model.output_price:.6f}")

                capabilities = []
                if model.supports_functions:
                    capabilities.append("functions")
                if model.supports_vision:
                    capabilities.append("vision")
                if model.supports_streaming:
                    capabilities.append("streaming")

                if capabilities:
                    console.print(f"   Capabilities: {', '.join(capabilities)}")

        # Display validation issues
        if result.issues:
            console.print("\n[bold]Validation Issues:[/bold]")

            for issue in result.issues:
                severity_emoji = {
                    ModelValidationSeverity.ERROR: "🔴",
                    ModelValidationSeverity.WARNING: "🟡",
                    ModelValidationSeverity.INFO: "ℹ️",
                }.get(issue.severity, "ℹ️")

                severity_color = {
                    ModelValidationSeverity.ERROR: "red",
                    ModelValidationSeverity.WARNING: "yellow",
                    ModelValidationSeverity.INFO: "blue",
                }.get(issue.severity, "blue")

                severity_text = f"{severity_emoji} [{severity_color}]{issue.severity.value.upper()}[/{severity_color}]"
                console.print(f"{severity_text}: {issue.message}")

                if issue.model_id:
                    console.print(f"   Model: [dim]{issue.model_id}[/dim]")

                if issue.field_name:
                    console.print(f"   Field: [dim]{issue.field_name}[/dim]")

                if issue.original_value is not None and issue.normalized_value is not None:
                    console.print(f"   [dim]{issue.original_value} → {issue.normalized_value}[/dim]")

                if issue.suggestion:
                    console.print(f"   [cyan]💡 {issue.suggestion}[/cyan]")

                console.print()

        # Summary statistics
        error_count = sum(1 for issue in result.issues if issue.severity == ModelValidationSeverity.ERROR)
        warning_count = sum(1 for issue in result.issues if issue.severity == ModelValidationSeverity.WARNING)
        info_count = sum(1 for issue in result.issues if issue.severity == ModelValidationSeverity.INFO)

        console.print("[bold cyan]Issue Summary:[/bold cyan]")
        console.print(f"  🔴 Errors: {error_count}")
        console.print(f"  🟡 Warnings: {warning_count}")
        console.print(f"  ℹ️ Info: {info_count}")

        if error_count > 0:
            console.print("\n[red]💡 Errors indicate data quality issues that may affect functionality.[/red]")
        elif warning_count > 0:
            console.print("\n[yellow]💡 Warnings suggest potential improvements to data quality.[/yellow]")
        else:
            console.print("\n[green]💡 Model data looks excellent![/green]")

    def _detect_provider_from_path(self, data_file: Path) -> str:
        """Detect provider name from file path."""
        path_str = str(data_file).lower()

        # Look for models_<provider>.json pattern
        match = re.search(r"models_([^./]+)\.json", path_str)
        if match:
            return match.group(1)

        # Look for provider names in path
        known_providers = [p.name for p in get_all_providers()]
        for provider in known_providers:
            if provider in path_str:
                return provider

        return ""

    @stats_command_cache()
    def stats(self) -> None:
        """Show system statistics including provider status and file counts."""
        file_stats = self.storage.get_file_stats()
        failure_stats = self.failure_tracker.get_failure_summary()

        console.print("[bold cyan]System Statistics[/bold cyan]")

        # File counts
        console.print("\n[yellow]Storage:[/yellow]")
        console.print(f"  Config JSON files: {file_stats.get('config_json_files', 0)}")
        console.print(f"  Config TXT files: {file_stats.get('config_txt_files', 0)}")
        console.print(f"  Aichat configs: {file_stats.get('config_aichat_files', 0)}")
        console.print(f"  Codex configs: {file_stats.get('config_codex_files', 0)}")
        console.print(f"  Mods configs: {file_stats.get('config_mods_files', 0)}")

        # Provider reliability
        console.print("\n[yellow]Provider Reliability:[/yellow]")
        console.print(f"  Total tracked: {failure_stats['total_providers_tracked']}")
        console.print(f"  Currently working: {failure_stats['currently_working']}")
        console.print(f"  Currently failed: {failure_stats['currently_failed']}")
        console.print(f"  Overall success rate: {failure_stats['overall_success_rate']:.1%}")

    def clean(self, *, temp: bool = False, configs: bool = False, failed: bool = False) -> None:
        """Clean up temporary files, generated configs, or failure tracking data."""
        if temp:
            self.storage.cleanup_temp_files()
            console.print("[green]Cleaned up temporary files[/green]")

        if configs:
            # Remove generated config files
            for directory in ["config_aichat", "config_codex", "config_mods"]:
                files = self.storage.list_files(directory)
                for file_path in files:
                    if file_path.name.startswith("models_"):
                        file_path.unlink()
            console.print("[green]Cleaned up generated config files[/green]")

        if failed:
            self.failure_tracker.reset_all_failures()
            console.print("[green]Reset all provider failure tracking[/green]")

        if not any([temp, configs, failed]):
            console.print("[yellow]Specify what to clean: --temp, --configs, or --failed[/yellow]")

    def link(
        self, target_dir: str = "models", source_dir: str = "config", *, copy_files: bool = False, providers: str = ""
    ) -> None:
        """
        Create symlinks or copies for existing tooling integration.

        Args:
            target_dir: Target directory for links/copies (default: models)
            source_dir: Source directory to link from (default: config)
            copy_files: Copy files instead of creating symlinks
            providers: Comma-separated provider names, or 'all' for all providers
        """
        # Calculate source subdirectories based on source_dir parameter
        json_subdir = f"{source_dir}_json" if source_dir != "config" else "config_json"
        txt_subdir = f"{source_dir}_txt" if source_dir != "config" else "config_txt"
        
        # Parse provider list
        if providers == "all" or not providers:
            # Find all existing model files in the specified source directory structure
            json_files = self.storage.list_files(json_subdir, "models_*.json")
            txt_files = self.storage.list_files(txt_subdir, "models_*.txt")
            provider_names = set()

            for file_path in json_files + txt_files:
                # Extract provider name from models_PROVIDER.ext
                name_parts = file_path.stem.split("_", 1)
                if len(name_parts) == PARTS_COUNT_FOR_MODELS_FILE and name_parts[0] == "models":
                    provider_names.add(name_parts[1])

            provider_list = list(provider_names)
        else:
            provider_list = [name.strip() for name in providers.split(",")]

        if not provider_list:
            console.print("[yellow]No providers found or specified[/yellow]")
            return

        # Create target directory structure
        target_path = Path(target_dir)
        target_path.mkdir(exist_ok=True)

        linked_count = 0
        failed_count = 0
        operation = "Copying" if copy_files else "Linking"

        console.print(f"[blue]{operation} files for {len(provider_list)} providers to {target_dir}/[/blue]")

        for provider_name in provider_list:
            try:
                # Process JSON files
                json_files = self.storage.list_files(json_subdir, f"models_{provider_name}.json")
                for source_file in json_files:
                    target_file = target_path / source_file.name

                    # Remove existing link/file
                    if target_file.exists() or target_file.is_symlink():
                        target_file.unlink()

                    if copy_files:
                        shutil.copy2(source_file, target_file)
                    else:
                        target_file.symlink_to(source_file.resolve())

                    linked_count += 1

                # Process TXT files
                txt_files = self.storage.list_files(txt_subdir, f"models_{provider_name}.txt")
                for source_file in txt_files:
                    target_file = target_path / source_file.name

                    # Remove existing link/file
                    if target_file.exists() or target_file.is_symlink():
                        target_file.unlink()

                    if copy_files:
                        shutil.copy2(source_file, target_file)
                    else:
                        target_file.symlink_to(source_file.resolve())

                    linked_count += 1

                console.print(f"[green]✓ {provider_name}[/green]")

            except Exception as e:
                error_type = type(e).__name__
                if isinstance(e, KeyError):
                    recovery_msg = (
                        f"[red]✗ {provider_name}: Provider not found in cache. "
                        f"Try: 1) Run 'fetch' command first, 2) Verify provider name spelling, "
                        f"3) Check if provider is supported.[/red]"
                    )
                elif isinstance(e, PermissionError):
                    recovery_msg = (
                        f"[red]✗ {provider_name}: Permission denied accessing cache. "
                        f"Try: 1) Check cache directory permissions, 2) Run as appropriate user, "
                        f"3) Clear cache with 'cache --clear'.[/red]"
                    )
                elif isinstance(e, FileNotFoundError | OSError):
                    recovery_msg = (
                        f"[red]✗ {provider_name}: Cache file issues ({error_type}). "
                        f"Try: 1) Clear and rebuild cache with 'cache --clear --rebuild', "
                        f"2) Run 'fetch' to regenerate data, 3) Check disk space.[/red]"
                    )
                else:
                    recovery_msg = (
                        f"[red]✗ {provider_name}: {e}. "
                        f"Try: 1) Clear cache with 'cache --clear', 2) Re-fetch provider data, "
                        f"3) Run with --verbose for details.[/red]"
                    )

                console.print(recovery_msg)
                failed_count += 1

        # Summary
        operation_past = "copied" if copy_files else "linked"
        success_count = len(provider_list) - failed_count
        operation_message = f"\n[bold]{operation_past.title()} {linked_count} files for {success_count} providers[/bold]"
        console.print(operation_message)

        if failed_count > 0:
            console.print(f"[yellow]{failed_count} providers failed[/yellow]")

    def migrate(self, search_paths: str = "") -> None:
        """
        Run migration assistant to help transition from dump_models.py.

        Args:
            search_paths: Comma-separated paths to search for existing dump_models.py output
        """
        # Find the migration script
        script_path = Path(__file__).parent.parent.parent / "scripts" / "migrate_from_dump_models.py"

        if not script_path.exists():
            console.print("[red]Migration script not found[/red]")
            console.print("Please ensure the migration script is available in scripts/migrate_from_dump_models.py")
            return

        try:
            # Run the migration script
            console.print("[blue]Starting migration assistant...[/blue]")
            env = os.environ.copy()
            if search_paths:
                env["SEARCH_PATHS"] = search_paths

            result = subprocess.run([sys.executable, str(script_path)], env=env, check=False)

            if result.returncode != 0:
                console.print("[yellow]Migration assistant encountered an issue[/yellow]")

        except FileNotFoundError:
            console.print(f"[red]Migration script not found: {script_path}[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print("  1) Verify package installation: pip show vexy-co-model-catalog")
            console.print("  2) Reinstall package: pip install --upgrade vexy-co-model-catalog")
            console.print("  3) Check if script exists in package data")
        except PermissionError:
            console.print(f"[red]Permission denied running migration script: {script_path}[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print(f"  1) Make script executable: chmod +x {script_path}")
            console.print("  2) Run with python: python {script_path}")
            console.print("  3) Check directory permissions")
        except ImportError as e:
            console.print(f"[red]Missing dependencies for migration: {e}[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print("  1) Install missing dependencies: pip install -r requirements.txt")
            console.print("  2) Update package: pip install --upgrade vexy-co-model-catalog")
            console.print("  3) Check Python environment compatibility")
        except Exception as e:
            console.print(f"[red]Failed to run migration assistant: {type(e).__name__}: {e}[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print(f"  1) Run manually: python {script_path}")
            console.print("  2) Check script permissions and Python path")
            console.print("  3) Run with --verbose for detailed error information")
            console.print("  4) Verify current working directory and file paths")

    def show_analytics(self, *, show: bool = True, disable: bool = False) -> None:
        """
        Show CLI usage analytics and performance metrics.

        Args:
            show: Display analytics summary (default: True)
            disable: Disable analytics collection
        """
        with self._track_command("analytics", parameter_count=int(not show) + int(disable)):
            if disable:
                self.analytics.set_enabled(False)
                console.print("[yellow]📊 Analytics collection disabled[/yellow]")
                console.print("Analytics can be re-enabled by setting VEXY_ANALYTICS_ENABLED=true")
                return

            if not show:
                return

            # Get analytics summary
            stats = self.analytics.get_summary_stats()

            if not stats.get("analytics_enabled", False):
                console.print("[yellow]📊 Analytics collection is disabled[/yellow]")
                console.print("Enable with: export VEXY_ANALYTICS_ENABLED=true")
                return

            # Display analytics table
            analytics_table = Table(title="📊 CLI Usage Analytics", show_header=True, header_style="bold magenta")
            analytics_table.add_column("Metric", style="cyan", no_wrap=True)
            analytics_table.add_column("Value", style="white", justify="right")

            analytics_table.add_row("Analytics Enabled", "✓")
            analytics_table.add_row("Total Sessions", str(stats.get("total_sessions", 0)))
            analytics_table.add_row("Total Commands", str(stats.get("total_commands", 0)))
            analytics_table.add_row("Average Success Rate", f"{stats.get('average_success_rate', 0):.1f}%")
            analytics_table.add_row("Most Used Command", stats.get("most_used_command", "none"))

            console.print(analytics_table)

            # Show command usage breakdown
            command_usage = stats.get("command_usage", {})
            if command_usage:
                console.print("\n[bold cyan]Command Usage Breakdown:[/bold cyan]")
                for cmd, count in sorted(command_usage.items(), key=lambda x: x[1], reverse=True):
                    console.print(f"  {cmd}: {count} times")

            # Privacy note
            console.print("\n[dim]📋 Analytics are stored locally and not transmitted anywhere.[/dim]")
            console.print("[dim]To disable: vexy show-analytics --disable[/dim]")

    def rate_limits(self, provider: str = "all") -> None:
        """
        Show intelligent rate limiting statistics and current throttling status.

        Args:
            provider: Show stats for specific provider, or 'all' for all providers
        """
        with self._track_command("rate_limits", parameter_count=1 if provider != "all" else 0):
            rate_limiter = get_rate_limiter()
            all_stats = rate_limiter.get_all_stats()

            if not all_stats:
                console.print("[yellow]No rate limiting data available yet.[/yellow]")
                console.print("[dim]Rate limiting statistics are collected after making requests to providers.[/dim]")
                return

            # Filter stats if specific provider requested
            if provider != "all" and provider in all_stats:
                stats_to_show = {provider: all_stats[provider]}
            elif provider != "all":
                console.print(f"[red]Provider '{provider}' not found in rate limiting data.[/red]")
                return
            else:
                stats_to_show = all_stats

            # Create rate limiting table
            rate_table = Table(title="🚦 Rate Limiting Status", show_header=True, header_style="bold magenta")
            rate_table.add_column("Provider", style="cyan", no_wrap=True)
            rate_table.add_column("Strategy", style="blue", no_wrap=True)
            rate_table.add_column("Req/Sec Limit", style="white", justify="right")
            rate_table.add_column("Current Delay", style="yellow", justify="right")
            rate_table.add_column("Success Streak", style="green", justify="right")
            rate_table.add_column("Failure Streak", style="red", justify="right")
            rate_table.add_column("Burst Tokens", style="magenta", justify="right")
            rate_table.add_column("Req This Min", style="dim", justify="right")

            for provider_name, stats in stats_to_show.items():
                # Format values for display
                strategy = stats.get("strategy", "unknown")
                rps_limit = (
                    f"{stats.get('requests_per_second_limit', 0):.1f}"
                    if stats.get("requests_per_second_limit")
                    else "N/A"
                )
                current_delay = f"{stats.get('current_delay', 0):.2f}s" if stats.get("current_delay", 0) > 0 else "None"
                success_streak = str(stats.get("consecutive_successes", 0))
                failure_streak = (
                    str(stats.get("consecutive_failures", 0)) if stats.get("consecutive_failures", 0) > 0 else "-"
                )
                burst_tokens = str(int(stats.get("burst_tokens", 0)))
                req_min = str(stats.get("requests_this_minute", 0))

                # Color code the row based on status
                if stats.get("consecutive_failures", 0) > 0 or stats.get("current_delay", 0) > 1.0:
                    pass
                else:
                    pass

                rate_table.add_row(
                    provider_name,
                    strategy,
                    rps_limit,
                    current_delay,
                    success_streak,
                    failure_streak,
                    burst_tokens,
                    req_min,
                )

            console.print(rate_table)

            # Show summary information
            total_providers = len(stats_to_show)
            throttled_providers = len([s for s in stats_to_show.values() if s.get("current_delay", 0) > 0])
            failing_providers = len([s for s in stats_to_show.values() if s.get("consecutive_failures", 0) > 0])

            console.print("\n[bold cyan]Rate Limiting Summary:[/bold cyan]")
            console.print(f"  Total providers tracked: {total_providers}")
            console.print(f"  Currently throttled: {throttled_providers}")
            console.print(f"  Currently failing: {failing_providers}")

            if throttled_providers > 0:
                console.print(
                    "\n[yellow]💡 Throttled providers are automatically managed by intelligent rate limiting.[/yellow]"
                )

            if failing_providers > 0:
                console.print("[red]⚠️ Failing providers may indicate API issues or rate limit violations.[/red]")

            console.print(
                "\n[dim]Rate limiting adapts automatically based on provider responses and API headers.[/dim]"
            )

    def health(self, *, detailed: bool = False, fix: bool = False) -> None:
        """
        Run comprehensive system health check and diagnostics.

        Args:
            detailed: Show detailed health information and metrics
            fix: Attempt automatic fixes for detected issues (where safe)
        """
        asyncio.run(self._health_async(detailed=detailed, fix=fix))

    async def _health_async(self, *, detailed: bool, fix: bool) -> None:
        """Async implementation of health check."""
        with self._track_command("health", parameter_count=int(detailed) + int(fix)):
            console.print("[cyan]🔍 Running comprehensive system health check...[/cyan]")

            # Get health monitor and run check
            health_monitor = get_health_monitor(self.storage.root)
            report = await health_monitor.run_comprehensive_health_check()

            # Display overall status
            status_color = {"healthy": "green", "warning": "yellow", "critical": "red", "unknown": "dim"}.get(
                report["overall_status"], "dim"
            )

            status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "🚨", "unknown": "❓"}.get(
                report["overall_status"], "❓"
            )

            console.print(
                f"\n[bold {status_color}]{status_icon} Overall Health: {report['overall_status'].upper()}[/bold {status_color}]"
            )

            # Show summary statistics
            console.print(f"[dim]Health check completed in {report['execution_time_ms']:.1f}ms[/dim]")

            if report["total_issues"] > 0:
                console.print(
                    f"[yellow]Found {report['total_issues']} issues: "
                    f"{report['critical_issues']} critical, {report['warning_issues']} warnings[/yellow]"
                )
            else:
                console.print("[green]No issues detected - system is healthy![/green]")

            # Display system metrics if detailed
            if detailed:
                self._display_system_metrics(report["system_metrics"])

            # Display issues by component
            if report["total_issues"] > 0:
                self._display_health_issues(report["issues_by_component"], detailed)

            # Display recommendations
            if report["recommendations"]:
                console.print("\n[bold cyan]💡 Recommended Actions:[/bold cyan]")
                for i, rec in enumerate(report["recommendations"][:MAX_RECOMMENDATIONS_DISPLAY], 1):
                    console.print(f"  {i}. {rec}")

                if len(report["recommendations"]) > MAX_RECOMMENDATIONS_DISPLAY:
                    remaining_count = len(report['recommendations']) - MAX_RECOMMENDATIONS_DISPLAY
                    console.print(f"[dim]  ... and {remaining_count} more recommendations[/dim]")

            # Display quick fixes
            if report["quick_fixes"]:
                console.print("\n[bold green]⚡ Quick Fixes:[/bold green]")
                for fix_cmd in report["quick_fixes"]:
                    console.print(f"  • {fix_cmd}")

            # Attempt automatic fixes if requested
            if fix and report["quick_fixes"]:
                console.print("\n[cyan]🔧 Attempting automatic fixes...[/cyan]")
                await self._attempt_automatic_fixes(report["quick_fixes"])

            # Health monitoring tips
            if report["overall_status"] in ["warning", "critical"]:
                console.print("\n[dim]💡 Tip: Run 'vexy health --detailed' for more information[/dim]")
                console.print("[dim]💡 Tip: Run 'vexy health --fix' to attempt automatic fixes[/dim]")

    def cache(self, action: str = "stats", provider: str = "", *, clear_all: bool = False) -> None:
        """
        Manage intelligent caching system for performance optimization.

        Args:
            action: Action to perform - 'stats', 'clear', 'invalidate'
            provider: Specific provider to target for invalidation
            clear_all: Clear all caches (use with caution)
        """
        with self._track_command("cache", parameter_count=1):
            if action == "stats":
                self._show_cache_stats()
            elif action == "clear" and clear_all:
                self._clear_all_caches()
            elif action == "invalidate" and provider:
                self._invalidate_provider_cache(provider)
            elif action == "clear" and provider:
                self._clear_provider_cache(provider)
            else:
                console.print("[red]Invalid cache action or missing parameters[/red]")
                console.print("\n[cyan]Available cache commands:[/cyan]")
                console.print("  [yellow]vexy cache stats[/yellow] - Show cache statistics")
                console.print(
                    "  [yellow]vexy cache invalidate --provider=openai[/yellow] - Invalidate specific provider"
                )
                console.print("  [yellow]vexy cache clear --provider=openai[/yellow] - Clear specific provider cache")
                console.print("  [yellow]vexy cache clear --clear-all[/yellow] - Clear all caches")

    def _show_cache_stats(self) -> None:
        """Display comprehensive cache statistics."""
        console.print("[bold cyan]📊 Intelligent Cache Statistics[/bold cyan]")

        caches = {
            "Model Metadata": get_model_cache(),
            "Validation Results": get_validation_cache(),
            "API Responses": get_api_response_cache(),
        }

        # Create statistics table
        stats_table = Table(show_header=True, header_style="bold blue")
        stats_table.add_column("Cache Type", style="cyan", no_wrap=True)
        stats_table.add_column("Entries", style="white", justify="right")
        stats_table.add_column("Memory (MB)", style="white", justify="right")
        stats_table.add_column("Hit Rate", style="white", justify="right")
        stats_table.add_column("Hits/Misses", style="white", justify="center")
        stats_table.add_column("Status", style="white", justify="center")

        total_memory = 0
        total_entries = 0

        for cache_name, cache in caches.items():
            stats = cache.get_stats()

            memory_mb = stats["memory_usage_mb"]
            total_memory += memory_mb
            total_entries += stats["entry_count"]

            hit_rate = stats["hit_rate_percent"]
            hit_status = "🟢" if hit_rate > CACHE_HIT_RATE_EXCELLENT else "🟡" if hit_rate > CACHE_HIT_RATE_GOOD else "🔴"

            stats_table.add_row(
                cache_name,
                str(stats["entry_count"]),
                f"{memory_mb:.1f}",
                f"{hit_rate:.1f}%",
                f"{stats['hits']}/{stats['misses']}",
                hit_status,
            )

        console.print(stats_table)

        # Summary
        console.print(f"\n[cyan]Total Memory Usage:[/cyan] {total_memory:.1f} MB")
        console.print(f"[cyan]Total Cache Entries:[/cyan] {total_entries}")

        # Performance recommendations
        if total_memory > MEMORY_USAGE_HIGH:
            console.print("\n[yellow]⚠️ High memory usage detected. Consider clearing some caches.[/yellow]")
        elif any(cache.get_stats()["hit_rate_percent"] < CACHE_HIT_RATE_GOOD for cache in caches.values()):
            console.print("\n[yellow]⚠️ Low cache hit rates detected. Cache may need tuning.[/yellow]")
        else:
            console.print("\n[green]✅ Cache performance is optimal.[/green]")

    def _clear_all_caches(self) -> None:
        """Clear all intelligent caches."""
        caches = [get_model_cache(), get_validation_cache(), get_api_response_cache()]

        total_cleared = 0
        for cache in caches:
            stats = cache.get_stats()
            total_cleared += stats["entry_count"]
            cache.clear()

        console.print(f"[green]✅ Cleared all caches ({total_cleared} entries removed)[/green]")
        console.print("[dim]Cache will rebuild automatically on next access[/dim]")

    def _invalidate_provider_cache(self, provider_name: str) -> None:
        """Invalidate all cache entries for a specific provider."""
        invalidate_provider_cache(provider_name)
        console.print(f"[green]✅ Invalidated all cache entries for provider '{provider_name}'[/green]")

    def _clear_provider_cache(self, provider_name: str) -> None:
        """Clear cache entries for a specific provider."""
        caches = [get_model_cache(), get_validation_cache(), get_api_response_cache()]

        total_cleared = 0
        for cache in caches:
            cleared = cache.invalidate_by_pattern(f".*{provider_name}.*")
            total_cleared += cleared

        console.print(f"[green]✅ Cleared {total_cleared} cache entries for provider '{provider_name}'[/green]")

    def integrity(
        self, 
        action: str = "report", 
        *, 
        verify_all: bool = False, 
        repair: bool = False, 
        cleanup_days: int = DEFAULT_CLEANUP_DAYS
    ) -> None:
        """
        Manage data integrity with checksums, corruption detection, and automatic repair.

        Args:
            action: Action to perform - 'report', 'verify', 'cleanup'
            verify_all: Verify integrity of all tracked files
            repair: Enable automatic repair during verification
            cleanup_days: Days to retain backup files (for cleanup action)
        """
        with self._track_command("integrity", parameter_count=1):
            if action == "report":
                self._show_integrity_report()
            elif action == "verify":
                self._verify_integrity(verify_all, repair)
            elif action == "cleanup":
                self._cleanup_integrity_backups(cleanup_days)
            else:
                console.print("[red]Invalid integrity action[/red]")
                console.print("\n[cyan]Available integrity commands:[/cyan]")
                console.print("  [yellow]vexy integrity report[/yellow] - Show comprehensive integrity report")
                console.print(
                    "  [yellow]vexy integrity verify --verify-all --repair[/yellow] - Verify all files with auto-repair"
                )
                console.print("  [yellow]vexy integrity cleanup --cleanup-days=30[/yellow] - Clean up old backup files")

    def _show_integrity_report(self) -> None:
        """Display comprehensive integrity report."""
        console.print("[bold cyan]🔒 Data Integrity Report[/bold cyan]")

        try:
            integrity_manager = get_integrity_manager(self.storage.root)
            report = integrity_manager.get_integrity_report()

            # Summary statistics
            summary = report["summary"]
            console.print("\n[cyan]Summary:[/cyan]")
            console.print(f"  Total tracked files: {summary['total_tracked_files']}")
            console.print(f"  Currently corrupted: {summary['currently_corrupted']}")
            console.print(f"  Total corruption events: {summary['total_corruption_events']}")
            console.print(f"  Recent events (24h): {summary['recent_events_24h']}")

            # Integrity levels breakdown
            console.print("\n[cyan]Files by Integrity Level:[/cyan]")
            for level, count in report["integrity_levels"].items():
                if count > 0:
                    level_color = {
                        "critical": "red",
                        "important": "yellow",
                        "standard": "white",
                        "monitoring": "dim",
                    }.get(level, "white")
                    console.print(f"  [{level_color}]{level.title()}:[/{level_color}] {count} files")

            # Recent corruption events
            if report["recent_events"]:
                console.print("\n[bold red]🚨 Recent Corruption Events:[/bold red]")

                events_table = Table(show_header=True, header_style="bold red")
                events_table.add_column("File", style="cyan", no_wrap=False)
                events_table.add_column("Severity", style="white", justify="center")
                events_table.add_column("Time", style="white", justify="center")
                events_table.add_column("Repair", style="white", justify="center")

                for event in report["recent_events"][-MAX_RECENT_EVENTS_DISPLAY:]:  # Last recent events
                    file_path = Path(event["file_path"]).name  # Just filename for display
                    severity = event["severity"].upper()

                    # Format time
                    event_time = datetime.datetime.fromtimestamp(event["detected_at"])
                    time_str = event_time.strftime("%H:%M:%S")

                    # Repair status
                    if event.get("repair_attempted"):
                        repair_status = "✅" if event.get("repair_successful") else "❌"
                    else:
                        repair_status = "⏸️"

                    severity_color = {
                        "CATASTROPHIC": "red",
                        "SEVERE": "red",
                        "MODERATE": "yellow",
                        "MINOR": "white",
                    }.get(severity, "white")

                    events_table.add_row(
                        file_path, f"[{severity_color}]{severity}[/{severity_color}]", time_str, repair_status
                    )

                console.print(events_table)
            else:
                console.print("\n[green]✅ No recent corruption events detected[/green]")

            # Health recommendations
            if summary["currently_corrupted"] > 0:
                console.print("\n[bold yellow]⚠️ Recommendations:[/bold yellow]")
                console.print("  • Run 'vexy integrity verify --verify-all --repair' to attempt repairs")
                console.print("  • Check system stability and storage health")
                console.print("  • Consider increasing backup frequency for critical files")
            else:
                console.print("\n[green]✅ All tracked files are intact and verified[/green]")

        except PermissionError:
            console.print("[red]Permission denied generating integrity report[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print("  1) Check write permissions for current directory")
            console.print("  2) Run with appropriate user privileges")
            console.print("  3) Specify different output directory with --output")
        except OSError as e:
            console.print(f"[red]I/O error generating integrity report: {e}[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print("  1) Check available disk space")
            console.print("  2) Verify target directory exists and is writable")
            console.print("  3) Check filesystem health")
        except Exception as e:
            console.print(f"[red]Failed to generate integrity report: {type(e).__name__}: {e}[/red]")
            console.print("[yellow]Try:[/yellow]")
            console.print("  1) Run with --verbose for detailed error information")
            console.print("  2) Check system resources (memory, disk space)")
            console.print("  3) Verify all tracked files are accessible")
            console.print("  4) Clear corrupted integrity data and regenerate")

    def _verify_integrity(self, *, verify_all: bool, repair: bool) -> None:
        """Verify file integrity with optional repair."""
        if verify_all:
            console.print("[cyan]🔍 Verifying integrity of all tracked files...[/cyan]")

            try:
                results = verify_all_files(auto_repair=repair)

                # Count results
                total_files = len(results)
                intact_files = sum(1 for success in results.values() if success)
                corrupted_files = total_files - intact_files

                # Display results
                if corrupted_files == 0:
                    console.print(f"[green]✅ All {total_files} files verified successfully[/green]")
                else:
                    console.print(f"[red]❌ {corrupted_files} of {total_files} files failed verification[/red]")

                    # Show failed files
                    console.print("\n[red]Failed files:[/red]")
                    for file_path, success in results.items():
                        if not success:
                            console.print(f"  • {Path(file_path).name}")

                    if repair:
                        console.print("\n[yellow]Repair attempts were made automatically[/yellow]")
                    else:
                        console.print("\n[dim]💡 Tip: Add --repair flag to attempt automatic fixes[/dim]")

            except PermissionError:
                console.print("[red]Permission denied during file verification[/red]")
                console.print("[yellow]Try:[/yellow]")
                console.print("  1) Check read permissions for tracked files")
                console.print("  2) Run with appropriate user privileges")
                console.print("  3) Skip restricted files with --skip-permission-errors")
            except FileNotFoundError as e:
                console.print(f"[red]Tracked file missing during verification: {e}[/red]")
                console.print("[yellow]Try:[/yellow]")
                console.print("  1) Re-fetch missing provider data files")
                console.print("  2) Update integrity tracking with --update-tracking")
                console.print("  3) Remove missing files from tracking with --clean-tracking")
            except Exception as e:
                console.print(f"[red]Verification failed: {type(e).__name__}: {e}[/red]")
                console.print("[yellow]Try:[/yellow]")
                console.print("  1) Run with --verbose for detailed error information")
                console.print("  2) Check filesystem integrity and available memory")
                console.print("  3) Clear and rebuild integrity tracking")
                console.print("  4) Verify system resources and file accessibility")
        else:
            console.print("[yellow]Use --verify-all flag to verify all tracked files[/yellow]")

    def _cleanup_integrity_backups(self, retention_days: int) -> None:
        """Clean up old backup files."""
        console.print(f"[cyan]🧹 Cleaning up backup files older than {retention_days} days...[/cyan]")

        try:
            integrity_manager = get_integrity_manager(self.storage.root)
            removed_count = integrity_manager.cleanup_old_backups(retention_days)

            console.print(f"[green]✅ Cleaned up {removed_count} old backup files[/green]")

            if removed_count == 0:
                console.print("[dim]No old backup files found to clean up[/dim]")

        except Exception as e:
            console.print(f"[red]Backup cleanup failed: {e}[/red]")

    def _display_system_metrics(self, metrics: dict[str, Any]) -> None:
        """Display detailed system metrics."""
        console.print("\n[bold cyan]📊 System Metrics:[/bold cyan]")

        # Create metrics table
        metrics_table = Table(show_header=True, header_style="bold blue")
        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("Value", style="white", justify="right")
        metrics_table.add_column("Status", style="white", justify="center")

        # System resources
        cpu_status = (
            "🔴" if metrics.get("cpu_percent", 0) > CPU_USAGE_CRITICAL else "🟡" if metrics.get("cpu_percent", 0) > CPU_USAGE_WARNING else "🟢"
        )
        metrics_table.add_row("CPU Usage", f"{metrics.get('cpu_percent', 0):.1f}%", cpu_status)

        memory_status = (
            "🔴" if metrics.get("memory_percent", 0) > MEMORY_USAGE_CRITICAL else "🟡" if metrics.get("memory_percent", 0) > MEMORY_USAGE_HIGH else "🟢"
        )
        metrics_table.add_row("Memory Usage", f"{metrics.get('memory_percent', 0):.1f}%", memory_status)

        disk_status = (
            "🔴"
            if metrics.get("disk_usage_percent", 0) > DISK_USAGE_CRITICAL
            else "🟡"
            if metrics.get("disk_usage_percent", 0) > DISK_USAGE_WARNING
            else "🟢"
        )
        metrics_table.add_row("Disk Usage", f"{metrics.get('disk_usage_percent', 0):.1f}%", disk_status)

        metrics_table.add_row("Free Memory", f"{metrics.get('available_memory_mb', 0):.0f} MB", "")
        metrics_table.add_row("Free Disk", f"{metrics.get('disk_free_gb', 0):.1f} GB", "")

        # Network metrics
        network_status = "🟢" if metrics.get("network_connectivity") else "🔴"
        metrics_table.add_row(
            "Network", "Connected" if metrics.get("network_connectivity") else "Disconnected", network_status
        )

        if metrics.get("avg_response_time_ms", 0) > 0:
            response_status = (
                "🔴"
                if metrics.get("avg_response_time_ms", 0) > RESPONSE_TIME_CRITICAL
                else "🟡"
                if metrics.get("avg_response_time_ms", 0) > RESPONSE_TIME_WARNING
                else "🟢"
            )
            metrics_table.add_row(
                "Avg Response Time", f"{metrics.get('avg_response_time_ms', 0):.0f} ms", response_status
            )

        # Provider metrics
        if metrics.get("total_providers", 0) > 0:
            provider_status = "🟢" if metrics.get("healthy_providers", 0) == metrics.get("total_providers", 0) else "🟡"
            metrics_table.add_row(
                "Healthy Providers",
                f"{metrics.get('healthy_providers', 0)}/{metrics.get('total_providers', 0)}",
                provider_status,
            )

        console.print(metrics_table)

    def _display_health_issues(self, issues_by_component: dict[str, list[dict[str, Any]]], *, detailed: bool) -> None:
        """Display health issues organized by component."""
        console.print("\n[bold red]🚨 Detected Issues:[/bold red]")

        component_icons = {
            "system": "💻",
            "storage": "💾",
            "network": "🌐",
            "provider": "🔌",
            "configuration": "⚙️",
            "performance": "⚡",
        }

        for component_type, issues in issues_by_component.items():
            if not issues:
                continue

            icon = component_icons.get(component_type, "❓")
            console.print(f"\n[bold]{icon} {component_type.title()} Issues:[/bold]")

            for issue in issues:
                status_color = {"critical": "red", "warning": "yellow", "healthy": "green"}.get(issue["status"], "dim")

                status_symbol = "🚨" if issue["status"] == "critical" else "⚠️"
                console.print(f"  {status_symbol} [{status_color}]{issue['message']}[/{status_color}]")

                if detailed and issue.get("details"):
                    console.print(f"     [dim]{issue['details']}[/dim]")

                if detailed and issue.get("recommendations"):
                    for rec in issue["recommendations"][:MAX_RECOMMENDATIONS_SHORT]:  # Show top recommendations
                        console.print(f"     [cyan]• {rec}[/cyan]")

    async def _attempt_automatic_fixes(self, quick_fixes: list[str]) -> None:
        """Attempt to run automatic fixes where safe."""
        fixed_count = 0

        for fix in quick_fixes:
            try:
                if fix.startswith("vexy clean"):
                    console.print(f"[yellow]Running: {fix}[/yellow]")
                    # Parse and run clean command
                    if "--temp" in fix:
                        # Safe to run temp cleanup
                        self.clean(temp=True)
                        console.print("[green]✓ Cleaned temporary files[/green]")
                        fixed_count += 1
                    elif "--configs" in fix:
                        console.print("[yellow]⚠️ Config cleanup requires manual approval[/yellow]")

                elif "export" in fix and "API_KEY" in fix:
                    console.print(f"[yellow]Manual action required: {fix}[/yellow]")

                else:
                    console.print(f"[dim]Manual action required: {fix}[/dim]")

            except Exception as e:
                console.print(f"[red]Failed to apply fix '{fix}': {e}[/red]")

        if fixed_count > 0:
            console.print(f"\n[green]✅ Applied {fixed_count} automatic fixes[/green]")
        else:
            console.print("\n[yellow]No automatic fixes applied - manual intervention required[/yellow]")

    def performance(self, action: str = "stats", *, save: bool = False, clear: bool = False) -> None:
        """Manage performance monitoring and display metrics.

        Args:
            action: Action to perform (stats, history, save, clear)
            save: Save current metrics to file
            clear: Clear all stored metrics
        """
        if not self.performance_monitor:
            console.print("[yellow]⚠️ Performance monitoring is disabled[/yellow]")
            console.print("Enable with: export VEXY_PERFORMANCE_ENABLED=true")
            return

        with self._track_performance("performance", {"action": action}):
            if action == "stats":
                self._show_performance_stats()
            elif action == "history":
                self._show_performance_history()
            elif action == "save" or save:
                self._save_performance_metrics()
            elif action == "clear" or clear:
                self._clear_performance_metrics()
            else:
                console.print(f"[red]Unknown performance action: {action}[/red]")
                console.print("Available actions: stats, history, report, bottlenecks, optimize, profile, save, clear")
                console.print("Examples:")
                console.print("  [yellow]vexy performance stats[/yellow] - Basic performance statistics")
                console.print("  [yellow]vexy performance report[/yellow] - Comprehensive performance report")
                console.print("  [yellow]vexy performance bottlenecks[/yellow] - Identify performance bottlenecks")
                console.print("  [yellow]vexy performance optimize[/yellow] - Optimize memory usage")
                console.print("  [yellow]vexy performance profile --enable-memory[/yellow] - Memory profiling report")

    def _show_performance_stats(self) -> None:
        """Display performance statistics summary."""
        stats = self.performance_monitor.get_statistics()

        console.print("\n[bold green]📊 Performance Statistics[/bold green]")
        console.print("[dim]Monitoring session stats[/dim]\n")

        # Summary stats table
        summary_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="white", no_wrap=True)
        summary_table.add_column("Details", style="dim")

        summary_table.add_row("Total Commands", str(stats["total_commands"]), "Commands executed")
        summary_table.add_row(
            "Successful", str(stats["successful_commands"]), f"{stats['successful_commands']}/{stats['total_commands']}"
        )
        summary_table.add_row(
            "Failed", str(stats["failed_commands"]), f"{stats['failed_commands']}/{stats['total_commands']}"
        )

        if stats["total_commands"] > 0:
            summary_table.add_row("Total Time", f"{stats['total_execution_time']:.2f}s", "Sum of all command times")
            summary_table.add_row("Average Time", f"{stats['average_execution_time']:.3f}s", "Mean execution time")
            summary_table.add_row(
                "Fastest", f"{stats['min_execution_time']:.3f}s", f"{stats['fastest_command']['command']}"
            )
            summary_table.add_row(
                "Slowest", f"{stats['max_execution_time']:.3f}s", f"{stats['slowest_command']['command']}"
            )

        console.print(summary_table)

        # Command breakdown
        if stats.get("commands_by_type"):
            console.print("\n[bold cyan]📋 Command Breakdown:[/bold cyan]")

            cmd_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
            cmd_table.add_column("Command", style="cyan")
            cmd_table.add_column("Count", style="white")
            cmd_table.add_column("Success Rate", style="green")
            cmd_table.add_column("Avg Time", style="yellow")
            cmd_table.add_column("Total Time", style="dim")

            for command, cmd_stats in sorted(stats["commands_by_type"].items()):
                success_rate = (cmd_stats["successful"] / cmd_stats["count"]) * 100 if cmd_stats["count"] > 0 else 0
                cmd_table.add_row(
                    command,
                    str(cmd_stats["count"]),
                    f"{success_rate:.1f}%",
                    f"{cmd_stats['avg_time']:.3f}s",
                    f"{cmd_stats['total_time']:.2f}s",
                )

            console.print(cmd_table)

        # Resource usage
        if stats.get("resource_usage"):
            resource_stats = stats["resource_usage"]
            console.print("\n[bold cyan]💾 Resource Usage:[/bold cyan]")

            resource_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
            resource_table.add_column("Resource", style="cyan")
            resource_table.add_column("Average Change", style="white")
            resource_table.add_column("Peak Usage", style="yellow")

            if "cpu_usage" in resource_stats:
                cpu = resource_stats["cpu_usage"]
                resource_table.add_row("CPU", f"{cpu['avg_change']:.1f}%", f"Max: {cpu['max_change']:.1f}%")

            if "memory_usage" in resource_stats:
                memory = resource_stats["memory_usage"]
                resource_table.add_row(
                    "Memory", f"{memory['avg_change_mb']:.1f} MB", f"Peak: {memory['max_peak_mb']:.1f} MB"
                )

            console.print(resource_table)

    def _show_performance_history(self) -> None:
        """Display recent performance history."""
        if not self.performance_monitor.metrics:
            console.print("[yellow]No performance history available[/yellow]")
            return

        console.print("\n[bold green]📈 Performance History[/bold green]")
        console.print("[dim]Recent command executions[/dim]\n")

        history_table = Table(show_header=True, header_style="bold magenta", border_style="dim")
        history_table.add_column("Time", style="dim", no_wrap=True)
        history_table.add_column("Command", style="cyan", max_width=25)
        history_table.add_column("Duration", style="yellow")
        history_table.add_column("Status", style="white", max_width=30)
        history_table.add_column("Memory Peak", style="blue")
        history_table.add_column("Disk I/O", style="magenta")
        history_table.add_column("Network", style="cyan")

        # Show last 20 commands
        recent_metrics = sorted(self.performance_monitor.metrics, key=lambda m: m.start_time, reverse=True)[:MAX_COMMANDS_HISTORY_DISPLAY]

        for metric in recent_metrics:
            status = "✅" if metric.success else "❌"
            if metric.memory_leaks_detected:
                status += " 💧"  # Memory leak indicator

            status_text = "Success" if metric.success else f"Failed: {metric.error_message or 'Unknown error'}"

            # Enhanced memory info with growth rate
            memory_info = f"{metric.memory_peak_mb:.1f} MB"
            if metric.memory_growth_rate_mb_per_sec > 0:
                memory_info += f" (+{metric.memory_growth_rate_mb_per_sec:.1f}/s)"

            # Disk I/O info
            disk_io = f"{metric.disk_io_total_mb:.1f} MB" if metric.disk_io_total_mb > 0 else "-"

            # Network info
            network_io = f"{metric.network_total_bytes:,} B" if metric.network_total_bytes > 0 else "-"

            history_table.add_row(
                metric.start_time.strftime("%H:%M:%S"),
                metric.command,
                f"{metric.duration_seconds:.3f}s",
                f"{status} {status_text}" if not metric.success else status,
                memory_info,
                disk_io,
                network_io,
            )

        console.print(history_table)

        # Performance insights
        if len(recent_metrics) > MIN_METRICS_FOR_INSIGHTS:
            avg_duration = sum(m.duration_seconds for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_peak_mb for m in recent_metrics) / len(recent_metrics)
            memory_leaks = sum(1 for m in recent_metrics if m.memory_leaks_detected)

            console.print("\n[dim]📈 Recent Performance Insights:[/dim]")
            console.print(f"[dim]  Average duration: {avg_duration:.3f}s[/dim]")
            console.print(f"[dim]  Average peak memory: {avg_memory:.1f} MB[/dim]")
            if memory_leaks > 0:
                console.print(f"[dim]  ⚠️ Commands with potential memory leaks: {memory_leaks}[/dim]")

    def _save_performance_metrics(self) -> None:
        """Save performance metrics to file."""
        try:
            filepath = self.performance_monitor.save_metrics()
            console.print(f"[green]✅ Performance metrics saved to {filepath}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save metrics: {e}[/red]")

    def _clear_performance_metrics(self) -> None:
        """Clear all performance metrics."""
        count = len(self.performance_monitor.metrics)
        self.performance_monitor.clear_metrics()
        console.print(f"[green]✅ Cleared {count} performance metric(s)[/green]")

    def _initialize_production_mode(self) -> None:
        """Initialize production mode with enhanced error handling."""
        try:
            success = initialize_production_mode()
            if success:
                logger.info("Production mode initialized successfully")
            else:
                logger.error("Failed to initialize production mode")
        except Exception as e:
            logger.error(f"Error initializing production mode: {e}")

    def production_status(self) -> None:
        """Show production deployment status and health metrics."""
        console.print("[bold]🏭 Production Status[/bold]")

        manager = get_production_manager()
        health_status = manager.get_health_status()

        # Create status table
        status_table = Table(title="System Health")
        status_table.add_column("Metric", style="cyan")
        status_table.add_column("Status", style="green" if health_status["healthy"] else "red")
        status_table.add_column("Details")

        status_table.add_row(
            "Overall Health",
            "✅ Healthy" if health_status["healthy"] else "⚠️ Degraded",
            health_status["status"]
        )

        status_table.add_row(
            "Initialization",
            "✅ Ready" if health_status["initialized"] else "❌ Not Ready",
            "Production environment initialized" if health_status["initialized"] else "Call production_init first"
        )

        # Error summary
        error_summary = health_status.get("error_summary", {})
        status_table.add_row(
            "Error Count",
            f"🔄 {error_summary.get('total_errors', 0)}",
            f"Critical: {error_summary.get('critical_errors', 0)}"
        )

        # System status
        system_status = health_status.get("system_status", {})
        degraded_services = system_status.get("degraded_services", [])
        status_table.add_row(
            "Service Health",
            "✅ All Operational" if not degraded_services else f"⚠️ {len(degraded_services)} Degraded",
            ", ".join(degraded_services) if degraded_services else "All services running normally"
        )

        status_table.add_row(
            "Log Directory",
            "📁 Available",
            health_status.get("log_directory", "Unknown")
        )

        console.print(status_table)

        # Show production metrics
        try:
            metrics = get_production_metrics()
            console.print("\n[bold]📊 Production Metrics[/bold]")

            metrics_table = Table()
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="yellow")

            # Memory usage
            memory = metrics.get("memory_usage", {})
            if "rss_mb" in memory:
                metrics_table.add_row("Memory Usage", f"{memory['rss_mb']:.1f} MB ({memory['percent']:.1f}%)")

            # Disk usage
            disk = metrics.get("disk_usage", {})
            if "percent_used" in disk:
                metrics_table.add_row("Log Disk Usage", f"{disk['percent_used']:.1f}% ({disk['free_gb']:.1f} GB free)")

            # Log files
            log_files = metrics.get("log_files", [])
            metrics_table.add_row("Log Files", str(len(log_files)))

            console.print(metrics_table)

        except Exception as e:
            console.print(f"[red]❌ Error getting production metrics: {e}[/red]")

    def production_init(self) -> None:
        """Initialize production mode with enhanced error handling and logging."""
        console.print("[bold]🚀 Initializing Production Mode[/bold]")

        try:
            success = initialize_production_mode()
            if success:
                console.print("[green]✅ Production mode initialized successfully[/green]")
                console.print("Enhanced error handling and logging are now active.")
            else:
                console.print("[red]❌ Failed to initialize production mode[/red]")
                console.print("Check log files for detailed error information.")
        except Exception as e:
            console.print(f"[red]❌ Error during production initialization: {e}[/red]")

    def production_readiness(self) -> None:
        """Check production deployment readiness."""
        console.print("[bold]🔍 Production Readiness Check[/bold]")

        try:
            readiness = check_production_readiness()

            # Create readiness table
            readiness_table = Table(title="Readiness Assessment")
            readiness_table.add_column("Check", style="cyan")
            readiness_table.add_column("Status", style="green" if readiness["ready"] else "red")
            readiness_table.add_column("Description")

            for check_name, passed in readiness["checks"].items():
                status = "✅ Pass" if passed else "❌ Fail"
                description = check_name.replace("_", " ").title()
                readiness_table.add_row(check_name, status, description)

            console.print(readiness_table)

            # Overall readiness
            overall_status = "✅ Ready for Production" if readiness["ready"] else "⚠️ Not Ready for Production"
            score = f"Score: {readiness['score']:.1%}"
            console.print(f"\n[bold]{overall_status}[/bold] ({score})")

            # Show recommendations if any
            if readiness["recommendations"]:
                console.print("\n[bold]📝 Recommendations:[/bold]")
                for i, recommendation in enumerate(readiness["recommendations"], 1):
                    console.print(f"{i}. {recommendation}")

        except Exception as e:
            console.print(f"[red]❌ Error checking production readiness: {e}[/red]")

    def production_errors(self) -> None:
        """Show production error summary and statistics."""
        console.print("[bold]🚨 Production Error Summary[/bold]")

        try:
            error_summary = get_error_summary()
            system_status = get_system_status()

            # Error statistics
            console.print(f"Total Errors: {error_summary['total_errors']}")
            console.print(f"Critical Errors: {error_summary['critical_errors']}")

            if error_summary["error_counts"]:
                # Create error table
                error_table = Table(title="Error Breakdown")
                error_table.add_column("Error Type", style="red")
                error_table.add_column("Count", style="yellow")

                for error_type, count in error_summary["error_counts"].items():
                    error_table.add_row(error_type, str(count))

                console.print(error_table)
            else:
                console.print("[green]✅ No errors recorded[/green]")

            # Circuit breakers
            circuit_breakers = system_status.get("circuit_breakers", {})
            if circuit_breakers:
                console.print("\n[bold]⚡ Circuit Breaker Status[/bold]")
                for service, status in circuit_breakers.items():
                    failures = status.get("failures", 0)
                    state = "OPEN" if failures >= CIRCUIT_BREAKER_FAILURE_THRESHOLD else "CLOSED"
                    console.print(f"{service}: {state} ({failures} failures)")

        except Exception as e:
            console.print(f"[red]❌ Error getting error summary: {e}[/red]")

    def production_diagnostics(self, level: str = "standard", *, auto_fix: bool = False) -> None:
        """Run comprehensive production reliability diagnostics.

        Args:
            level: Diagnostic level (basic, standard, enterprise, critical)
            auto_fix: Apply automatic fixes where available
        """
        async def run_diagnostics() -> None:
            reliability_hardening = get_production_reliability_hardening()

            try:
                reliability_level = ReliabilityLevel(level.lower())
            except ValueError:
                console.print(f"[red]Invalid level '{level}'. Use: basic, standard, enterprise, critical[/red]")
                return

            with self._track_command("production_diagnostics", level=level, auto_fix=auto_fix):
                # Run comprehensive diagnostics
                summary = await reliability_hardening.run_comprehensive_diagnostics(reliability_level)

                # Display results
                reliability_hardening.display_diagnostic_report(summary)

                # Apply auto-fixes if requested
                if auto_fix and summary.get("auto_fixes_available", 0) > 0:
                    console.print("\n[cyan]🔧 Applying automatic fixes...[/cyan]")
                    fix_results = await reliability_hardening.apply_automatic_fixes()

                    if fix_results["fixes_applied"]:
                        console.print(f"[green]✅ Applied {len(fix_results['fixes_applied'])} automatic fixes[/green]")
                        for fix in fix_results["fixes_applied"]:
                            console.print(f"  • {fix['test']}: {fix['action']}")

                    if fix_results["fixes_failed"]:
                        console.print(f"[red]❌ {len(fix_results['fixes_failed'])} fixes failed[/red]")
                        for fix in fix_results["fixes_failed"]:
                            console.print(f"  • {fix['test']}: {fix['error']}")

                # Recommendations
                critical_issues = summary.get("critical_tests", 0)
                if critical_issues > 0:
                    console.print(f"\n[bold red]⚠️  {critical_issues} critical issues must be resolved before production[/bold red]")
                else:
                    console.print("\n[bold green]✅ System appears ready for production deployment[/bold green]")

        asyncio.run(run_diagnostics())

    def production_health_advanced(
        self, *, continuous: bool = False, interval: int = MONITORING_INTERVAL_SECONDS
    ) -> None:
        """Advanced production health monitoring with continuous checks.

        Args:
            continuous: Run continuous health monitoring
            interval: Check interval in seconds for continuous mode
        """
        async def run_health_monitoring() -> None:
            reliability_hardening = get_production_reliability_hardening()

            with self._track_command("production_health_advanced", continuous=continuous):
                if continuous:
                    console.print(f"[cyan]🔄 Starting continuous health monitoring (every {interval}s)[/cyan]")
                    console.print("[dim]Press Ctrl+C to stop[/dim]")

                    try:
                        while True:
                            console.print(f"\n[bold cyan]🏥 Health Check - {time.strftime('%H:%M:%S')}[/bold cyan]")

                            # Run basic diagnostics
                            summary = await reliability_hardening.run_comprehensive_diagnostics(ReliabilityLevel.BASIC)

                            # Show summary status
                            health_score = summary.get("health_score", 0)
                            overall_status = summary.get("overall_status")

                            status_colors = {"healthy": "green", "warning": "yellow", "critical": "red"}
                            color = status_colors.get(overall_status.value if overall_status else "unknown", "white")

                            console.print(f"[{color}]Health Score: {health_score:.1f}% ({overall_status.value if overall_status else 'unknown'})[/{color}]")

                            # Show critical issues only in continuous mode
                            critical_results = [r for r in summary.get("results", []) if r.status.value == "critical"]
                            if critical_results:
                                console.print(f"[red]❌ {len(critical_results)} critical issues detected[/red]")
                                for result in critical_results[:MAX_CRITICAL_RESULTS_DISPLAY]:  # Show first critical results
                                    console.print(f"  • {result.component}: {result.message}")

                            await asyncio.sleep(interval)

                    except KeyboardInterrupt:
                        console.print("\n[yellow]🛑 Continuous monitoring stopped[/yellow]")
                else:
                    # Single comprehensive check
                    summary = await reliability_hardening.run_comprehensive_diagnostics(ReliabilityLevel.ENTERPRISE)
                    reliability_hardening.display_diagnostic_report(summary)

        asyncio.run(run_health_monitoring())

    def monitor(
        self, action: str = "dashboard", *, export: bool = False, start: bool = False, stop: bool = False
    ) -> None:
        """Advanced monitoring with real-time dashboards, metrics collection, and operational intelligence.

        Args:
            action: Action to perform (dashboard, metrics, alerts, trends, export)
            export: Export metrics to JSON file
            start: Start background monitoring
            stop: Stop background monitoring
        """
        with self._track_command("monitor", parameter_count=1):
            monitoring_system = get_monitoring_system(self.storage.root)

            if start:
                monitoring_system.start_monitoring()
                console.print("[green]✓ Background monitoring started[/green]")
                return

            if stop:
                monitoring_system.stop_monitoring()
                console.print("[green]✓ Background monitoring stopped[/green]")
                return

            if action == "dashboard":
                self._show_monitoring_dashboard(monitoring_system)
            elif action == "metrics":
                self._show_monitoring_metrics(monitoring_system)
            elif action == "alerts":
                self._show_monitoring_alerts(monitoring_system)
            elif action == "trends":
                self._show_monitoring_trends(monitoring_system)
            elif action == "export" or export:
                self._export_monitoring_data(monitoring_system)
            else:
                console.print(f"[red]Unknown monitoring action: {action}[/red]")
                console.print("Available actions: dashboard, metrics, alerts, trends, export")
                console.print("Control actions: --start, --stop")

    def _show_monitoring_dashboard(self, monitoring_system: MonitoringSystem) -> None:
        """Display the main monitoring dashboard."""
        console.print("[cyan]🔄 Generating monitoring dashboard...[/cyan]")

        # Get text dashboard
        dashboard_text = monitoring_system.get_text_dashboard()
        console.print(dashboard_text)

        # Show interactive options
        console.print("\n[dim]Interactive Commands:[/dim]")
        console.print("[dim]  vexy monitor metrics    - Detailed metrics view[/dim]")
        console.print("[dim]  vexy monitor alerts     - Active alerts and history[/dim]")
        console.print("[dim]  vexy monitor trends     - Performance trend analysis[/dim]")
        console.print("[dim]  vexy monitor --start    - Start background monitoring[/dim]")

    def _show_monitoring_metrics(self, monitoring_system: MonitoringSystem) -> None:
        """Display detailed metrics information."""
        console.print("\n[bold blue]📊 Detailed Metrics Report[/bold blue]")

        dashboard_data = monitoring_system.get_dashboard_data()
        metrics = dashboard_data.get("metrics", {})

        if not metrics:
            console.print("[yellow]No metrics data available yet.[/yellow]")
            console.print("[dim]Start using the tool to collect metrics, or run 'vexy monitor --start' for background collection.[/dim]")
            return

        # Create metrics table
        metrics_table = Table(title="System Metrics Overview")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Current", style="white", justify="right")
        metrics_table.add_column("Type", style="blue")
        metrics_table.add_column("30min Stats", style="yellow")
        metrics_table.add_column("Trend", style="white", justify="center")

        for metric_key, metric_data in metrics.items():
            current_val = metric_data.get("current_value")
            metric_type = metric_data.get("metric_type", "unknown")
            stats = metric_data.get("statistics", {})

            # Format current value
            if current_val is not None:
                if metric_type == "timer":
                    current_str = f"{current_val:.3f}s"
                elif metric_type == "gauge" and "memory" in metric_key:
                    current_str = f"{current_val:.1f} MB"
                elif metric_type == "gauge" and "percent" in metric_key:
                    current_str = f"{current_val:.1f}%"
                else:
                    current_str = f"{current_val:.2f}"
            else:
                current_str = "N/A"

            # Format statistics
            if stats:
                stats_str = f"μ={stats.get('mean', 0):.2f} σ={stats.get('stddev', 0):.2f} p95={stats.get('p95', 0):.2f}"
            else:
                stats_str = "No data"

            # Simple trend indicator (would be enhanced with actual trend analysis)
            trend_icon = "🔄"  # Placeholder

            metrics_table.add_row(
                metric_key,
                current_str,
                metric_type,
                stats_str,
                trend_icon
            )

        console.print(metrics_table)

        # Show collection info
        console.print(f"\n[dim]Metrics collected at: {dashboard_data['timestamp']}[/dim]")
        console.print("[dim]Use 'vexy monitor trends' to see performance trend analysis[/dim]")

    def _show_monitoring_alerts(self, monitoring_system: MonitoringSystem) -> None:
        """Display monitoring alerts and alert management."""
        console.print("\n[bold red]🚨 Alert Management System[/bold red]")

        alert_summary = monitoring_system.alert_manager.get_alert_summary()

        # Alert summary
        console.print("\n[cyan]Alert Summary:[/cyan]")
        console.print(f"  Total Alerts: {alert_summary['total_alerts']}")
        console.print(f"  Active Alerts: {alert_summary['active_alerts']}")
        console.print(f"  Resolved Alerts: {alert_summary['resolved_alerts']}")

        # Alerts by level
        console.print("\n[cyan]Alerts by Level:[/cyan]")
        level_colors = {
            "emergency": "red",
            "critical": "red",
            "warning": "yellow",
            "info": "blue"
        }

        for level, count in alert_summary["by_level"].items():
            color = level_colors.get(level, "white")
            icon = {"emergency": "🚨", "critical": "🔴", "warning": "🟡", "info": "🔵"}.get(level, "⚪")
            console.print(f"  [{color}]{icon} {level.title()}: {count}[/{color}]")

        # Recent alerts
        if alert_summary["recent_alerts"]:
            console.print("\n[cyan]Recent Alerts:[/cyan]")

            alerts_table = Table()
            alerts_table.add_column("Time", style="dim", width=ALERT_TABLE_COLUMN_WIDTH)
            alerts_table.add_column("Level", style="white", width=ALERT_TABLE_COLUMN_WIDTH)
            alerts_table.add_column("Title", style="cyan")
            alerts_table.add_column("Status", style="white", justify="center")
            alerts_table.add_column("Source", style="blue")

            for alert in alert_summary["recent_alerts"][:MAX_ALERTS_DISPLAY]:
                time_str = alert["timestamp"][:16].replace("T", " ")
                level_color = level_colors.get(alert["level"], "white")
                level_str = f"[{level_color}]{alert['level'].upper()}[/{level_color}]"
                status = "✅ Resolved" if alert["resolved"] else "🔄 Active"

                alerts_table.add_row(
                    time_str,
                    level_str,
                    alert["title"],
                    status,
                    alert["source"]
                )

            console.print(alerts_table)
        else:
            console.print("\n[green]✓ No alerts in system[/green]")

        # Alert rules info
        console.print("\n[dim]Alert Rules Active:[/dim]")
        rule_count = len(monitoring_system.alert_manager.alert_rules)
        console.print(f"[dim]  {rule_count} monitoring rules configured[/dim]")
        console.print(f"[dim]  Rules evaluate every {RULE_EVALUATION_INTERVAL} seconds in background mode[/dim]")

    def _show_monitoring_trends(self, monitoring_system: MonitoringSystem) -> None:
        """Display performance trend analysis."""
        console.print("\n[bold green]📈 Performance Trend Analysis[/bold green]")

        dashboard_data = monitoring_system.get_dashboard_data()
        trends = dashboard_data.get("trends", {})

        if not trends:
            console.print("[yellow]No trend data available yet.[/yellow]")
            console.print("[dim]Trends are calculated after collecting metrics over time.[/dim]")
            return

        # Create trends table
        trends_table = Table(title="Performance Trends (Last 30 vs 60 minutes)")
        trends_table.add_column("Metric", style="cyan")
        trends_table.add_column("Trend", style="white", justify="center")
        trends_table.add_column("Change", style="yellow", justify="right")
        trends_table.add_column("Current", style="white", justify="right")
        trends_table.add_column("Previous", style="dim", justify="right")
        trends_table.add_column("Impact", style="white", justify="center")

        for metric_name, trend_data in trends.items():
            trend = trend_data["trend"]
            change_pct = trend_data["change_percent"]
            current_val = trend_data["current_value"]
            previous_val = trend_data["previous_value"]

            # Trend icon and color
            trend_display = {
                "increasing": "📈 Increasing",
                "decreasing": "📉 Decreasing",
                "stable": "➡️ Stable"
            }.get(trend, "❔ Unknown")

            # Change display
            change_str = f"{change_pct:+.1f}%" if change_pct != 0 else "0%"

            # Impact assessment
            if abs(change_pct) > SIGNIFICANT_PERFORMANCE_CHANGE:
                impact = "🔴 High"
            elif abs(change_pct) > MODERATE_PERFORMANCE_CHANGE:
                impact = "🟡 Medium"
            elif abs(change_pct) > MINOR_PERFORMANCE_CHANGE:
                impact = "🟢 Low"
            else:
                impact = "⚫ None"

            trends_table.add_row(
                metric_name.replace("_", " ").title(),
                trend_display,
                change_str,
                f"{current_val:.3f}",
                f"{previous_val:.3f}",
                impact
            )

        console.print(trends_table)

        # Trend insights
        console.print("\n[bold cyan]Trend Insights:[/bold cyan]")

        concerning_trends = [name for name, data in trends.items()
                           if abs(data["change_percent"]) > ALERT_THRESHOLD_PERFORMANCE_CHANGE and data["trend"] != "stable"]

        if concerning_trends:
            console.print("[yellow]⚠️ Metrics with significant changes:[/yellow]")
            for metric in concerning_trends:
                console.print(f"  • {metric.replace('_', ' ').title()}")
        else:
            console.print("[green]✓ All metrics show stable or acceptable trends[/green]")

        console.print("\n[dim]Trends help identify performance degradation and optimization opportunities[/dim]")

    def _export_monitoring_data(self, monitoring_system: MonitoringSystem) -> None:
        """Export monitoring data to file."""
        console.print("[cyan]📋 Exporting monitoring data...[/cyan]")

        try:
            filepath = monitoring_system.export_metrics()
            console.print(f"[green]✓ Monitoring data exported to: {filepath}[/green]")

            # Show export summary
            with open(filepath) as f:
                export_data = json.load(f)

            metric_count = len(export_data.get("metrics", {}))
            alert_count = export_data.get("alerts", {}).get("total_alerts", 0)

            console.print(f"[dim]  Metrics exported: {metric_count}[/dim]")
            console.print(f"[dim]  Alerts exported: {alert_count}[/dim]")
            console.print(f"[dim]  Export timestamp: {export_data.get('timestamp', 'Unknown')}[/dim]")

        except Exception as e:
            console.print(f"[red]❌ Export failed: {e}[/red]")

    def monitoring_status(self) -> None:
        """Show monitoring system status and configuration."""
        console.print("[bold]🔍 Monitoring System Status[/bold]")

        monitoring_system = get_monitoring_system(self.storage.root)

        # System status
        is_running = (monitoring_system._monitoring_thread and
                     monitoring_system._monitoring_thread.is_alive())

        status_table = Table(title="Monitoring System Configuration")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green" if is_running else "yellow")
        status_table.add_column("Details", style="white")

        status_table.add_row(
            "Background Monitoring",
            "✅ Running" if is_running else "⏸️ Stopped",
            "Collects system metrics every 30s" if is_running else "Use 'vexy monitor --start' to begin"
        )

        metric_count = len(monitoring_system.metric_collector.get_metric_names())
        status_table.add_row(
            "Metric Collection",
            "✅ Active" if metric_count > 0 else "🟡 No Data",
            f"{metric_count} metric types being tracked"
        )

        alert_count = len(monitoring_system.alert_manager.get_active_alerts())
        status_table.add_row(
            "Alert System",
            "✅ Operational",
            f"{alert_count} active alerts, {len(monitoring_system.alert_manager.alert_rules)} rules configured"
        )

        dashboard_available = True  # Dashboard is always available
        status_table.add_row(
            "Dashboard",
            "✅ Available" if dashboard_available else "❌ Unavailable",
            "Real-time system monitoring dashboard"
        )

        storage_path = monitoring_system.storage_dir
        status_table.add_row(
            "Data Storage",
            "✅ Ready",
            f"Storing data in: {storage_path}"
        )

        console.print(status_table)

        # Quick start guide
        console.print("\n[bold cyan]Quick Start Guide:[/bold cyan]")
        console.print("[yellow]1.[/yellow] Start background monitoring: [green]vexy monitor --start[/green]")
        console.print("[yellow]2.[/yellow] View dashboard: [green]vexy monitor dashboard[/green]")
        console.print("[yellow]3.[/yellow] Check alerts: [green]vexy monitor alerts[/green]")
        console.print("[yellow]4.[/yellow] Analyze trends: [green]vexy monitor trends[/green]")
        console.print("[yellow]5.[/yellow] Export data: [green]vexy monitor export[/green]")

        # Show current system health
        try:
            dashboard_data = monitoring_system.get_dashboard_data()
            health = dashboard_data["system_health"]["status"]
            health_color = {"healthy": "green", "warning": "yellow", "critical": "red"}.get(health, "white")
            console.print(f"\n[{health_color}]Current System Health: {health.upper()}[/{health_color}]")
        except Exception as e:
            console.print(f"\n[dim]Could not determine current health status: {e}[/dim]")


def main() -> None:
    """Main CLI entry point."""
    # Configure loguru for clean CLI output
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<level>{level}</level>: {message}",
        colorize=True,
    )

    try:
        fire.Fire(CLI)
    finally:
        # Save analytics session when CLI exits
        finalize_analytics()


if __name__ == "__main__":
    main()
