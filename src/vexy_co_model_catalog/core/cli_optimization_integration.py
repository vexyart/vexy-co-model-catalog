"""
this_file: src/vexy_co_model_catalog/core/cli_optimization_integration.py

CLI optimization integration for improved startup time and response caching.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from vexy_co_model_catalog.core.cli_optimization import (
    cached_command,
    get_cached_help_text,
    get_cached_provider_list,
    get_cli_optimizer,
    get_cli_performance_report,
    invalidate_cli_cache,
    lazy_import,
    optimize_cli_startup,
    persist_cli_cache,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def optimize_cli_class(cli_class: type) -> type:
    """
    Class decorator to optimize CLI performance with caching and lazy loading.

    Args:
        cli_class: CLI class to optimize

    Returns:
        Optimized CLI class
    """
    # Store original methods
    original_init = cli_class.__init__
    original_providers = getattr(cli_class, "providers", None)
    original_stats = getattr(cli_class, "stats", None)
    original_help = getattr(cli_class, "help", None)
    original_version = getattr(cli_class, "version", None)

    def optimized_init(self, *args: Any, **kwargs: Any) -> None:
        """Optimized initialization with performance tracking."""
        start_time = time.time()

        # Initialize CLI optimizer
        self._cli_optimizer = get_cli_optimizer()

        # Call original initialization
        original_init(self, *args, **kwargs)

        # Apply startup optimizations
        optimization_results = optimize_cli_startup()

        init_time = time.time() - start_time

        # Store initialization metrics
        self._init_time = init_time
        self._optimization_results = optimization_results

        # Enable performance tracking if available
        if hasattr(self, "performance_monitor") and self.performance_monitor:
            if hasattr(self.performance_monitor, "record_cli_startup"):
                self.performance_monitor.record_cli_startup(init_time, optimization_results)

    @cached_command(ttl_seconds=60)  # Cache provider list for 1 minute
    def optimized_providers(self, *args, **kwargs) -> None:
        """Optimized providers command with caching."""
        if original_providers:
            return original_providers(self, *args, **kwargs)
        # Fallback implementation
        console = Console()
        console.print("Providers command not available")
        return None

    @cached_command(ttl_seconds=30)  # Cache stats for 30 seconds
    def optimized_stats(self, *args, **kwargs) -> None:
        """Optimized stats command with caching."""
        if original_stats:
            return original_stats(self, *args, **kwargs)
        # Fallback implementation
        console = Console()
        console.print("Stats command not available")
        return None

    def optimized_help(self, command: str | None = None, *args, **kwargs) -> None:
        """Optimized help command with caching."""
        if command:
            # Use cached help text for specific commands
            help_text = get_cached_help_text(command)
            console = Console()
            console.print(f"[bold blue]{command}[/bold blue]: {help_text}")
        elif original_help:
            return original_help(self, *args, **kwargs)
        else:
            # Fallback help implementation
            console = Console()
            table = Table(title="Available Commands")
            table.add_column("Command", style="bold blue")
            table.add_column("Aliases", style="green")
            table.add_column("Description")

            commands = [
                ("providers", "ls, list", "Manage AI model providers"),
                ("fetch", "get, dl, download", "Fetch model catalogs"),
                ("stats", "st, status", "Show system statistics"),
                ("clean", "rm, remove", "Clean up files"),
                ("validate", "check, verify", "Validate configurations"),
                ("health", "diag", "System health check"),
                ("performance", "perf", "Performance metrics"),
            ]

            for cmd, aliases, desc in commands:
                table.add_row(cmd, aliases, desc)

            console.print(table)
        return None

    def optimized_version(self, *_args, **_kwargs) -> None:
        """Optimized version command."""
        # Use lazy import for version info
        __version__ = lazy_import("vexy_co_model_catalog", "__version__")

        console = Console()
        console.print(f"vexy-co-model-catalog version {__version__}")

        # Show optimization info
        if hasattr(self, "_init_time"):
            console.print(f"CLI startup time: {self._init_time:.3f}s")

            if hasattr(self, "_optimization_results"):
                opts = self._optimization_results
                console.print(
                    f"Optimizations: {opts['cache_cleanup']} cache entries cleaned, {opts['lazy_imports']} lazy imports"
                )

    def cli_performance(_self, action: str = "stats") -> None:
        """
        CLI performance management and optimization.

        Args:
            action: Action to perform (stats, optimize, cache_clear, cache_persist)
        """
        console = Console()

        if action == "stats":
            # Show performance report
            report = get_cli_performance_report()

            console.print("[bold blue]CLI Performance Report[/bold blue]")
            console.print(f"Uptime: {report['uptime']}")

            # Cache statistics
            cache_stats = report["cache_stats"]
            cache_table = Table(title="Cache Statistics")
            cache_table.add_column("Metric", style="bold")
            cache_table.add_column("Value", style="green")

            for key, value in cache_stats.items():
                cache_table.add_row(key.replace("_", " ").title(), str(value))

            console.print(cache_table)

            # Command statistics
            if report["command_stats"]:
                cmd_table = Table(title="Command Performance")
                cmd_table.add_column("Command", style="bold blue")
                cmd_table.add_column("Executions", style="cyan")
                cmd_table.add_column("Avg Time", style="green")
                cmd_table.add_column("Total Time", style="yellow")

                for cmd, stats in report["command_stats"].items():
                    cmd_table.add_row(cmd, str(stats["executions"]), stats["avg_time"], stats["total_time"])

                console.print(cmd_table)

            # Recommendations
            if report["optimization_recommendations"]:
                console.print("\n[bold yellow]Optimization Recommendations:[/bold yellow]")
                for rec in report["optimization_recommendations"]:
                    console.print(f"• {rec}")

        elif action == "optimize":
            # Run optimization
            results = optimize_cli_startup()
            console.print("[green]Optimization complete![/green]")
            console.print(f"Cleaned {results['cache_cleanup']} expired cache entries")
            console.print(f"Using {results['lazy_imports']} lazy imports")

        elif action == "cache_clear":
            # Clear cache
            cleared = invalidate_cli_cache()
            console.print(f"[green]Cleared {cleared} cache entries[/green]")

        elif action == "cache_persist":
            # Persist cache
            success = persist_cli_cache()
            if success:
                console.print("[green]Cache persisted successfully[/green]")
            else:
                console.print("[red]Failed to persist cache[/red]")

        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("Available actions: stats, optimize, cache_clear, cache_persist")

    # Add new method for CLI optimization info
    def optimization_info(self) -> None:
        """Show CLI optimization information."""
        console = Console()
        optimizer = get_cli_optimizer()

        console.print("[bold blue]CLI Optimization Status[/bold blue]")

        # Startup info
        if hasattr(self, "_init_time"):
            console.print(f"Startup time: {self._init_time:.3f}s")

        # Cache info
        cache_stats = optimizer.response_cache.get_stats()
        console.print(f"Cache entries: {cache_stats['entries']}")
        console.print(f"Cache hit rate: {cache_stats['hit_rate']}")

        # Import info
        import_stats = optimizer.lazy_import_manager.get_import_stats()
        console.print(f"Lazy imports: {import_stats['imported_modules']}")
        console.print(f"Import time: {import_stats['total_import_time']}")

    # Replace methods with optimized versions
    cli_class.__init__ = optimized_init

    if original_providers:
        cli_class.providers = optimized_providers
    if original_stats:
        cli_class.stats = optimized_stats
    if original_help:
        cli_class.help = optimized_help
    if original_version:
        cli_class.version = optimized_version

    # Add new optimization methods
    cli_class.cli_performance = cli_performance
    cli_class.optimization_info = optimization_info

    # Add optimization aliases
    if hasattr(cli_class, "command_aliases"):
        cli_class.command_aliases.update(
            {"perf": "cli_performance", "opt": "optimization_info", "optimize": "cli_performance optimize"}
        )

    return cli_class


def add_performance_tracking(method_name: str, ttl_seconds: int = 300) -> Callable:
    """
    Decorator to add performance tracking to CLI methods.

    Args:
        method_name: Name of the method for tracking
        ttl_seconds: Cache TTL in seconds
    """

    def decorator(func: Callable) -> Callable:
        @cached_command(ttl_seconds=ttl_seconds)
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                execution_time = time.time() - start_time
                optimizer = get_cli_optimizer()
                optimizer._track_command_performance(method_name, execution_time)

        return wrapper

    return decorator


def create_optimized_provider_cache_key(provider_name: str | None = None, *_args, **_kwargs) -> str:
    """Create cache key for provider-related commands."""
    if provider_name:
        return f"providers_{provider_name}"
    return "providers_list"


def create_optimized_stats_cache_key(*_args, **kwargs) -> str:
    """Create cache key for stats command."""
    # Include any relevant parameters that affect stats
    params = []
    if "detailed" in kwargs:
        params.append(f"detailed_{kwargs['detailed']}")
    if "format" in kwargs:
        params.append(f"format_{kwargs['format']}")

    if params:
        return f"stats_{'_'.join(params)}"
    return "stats_default"


# Pre-configured decorators for common CLI optimizations
def provider_command_cache() -> Callable:
    return cached_command(ttl_seconds=60, cache_key_func=create_optimized_provider_cache_key)


def stats_command_cache() -> Callable:
    return cached_command(ttl_seconds=30, cache_key_func=create_optimized_stats_cache_key)


def quick_command_cache() -> Callable:
    return cached_command(ttl_seconds=10)


def heavy_command_cache() -> Callable:
    return cached_command(ttl_seconds=300)


def apply_cli_optimizations() -> None:
    """Apply global CLI optimizations."""
    # This function can be called to apply optimizations globally
    optimizer = get_cli_optimizer()

    # Pre-warm frequently used imports
    lazy_import("vexy_co_model_catalog.core.provider", "get_all_providers")
    lazy_import("vexy_co_model_catalog.core.storage", "StorageManager")
    lazy_import("vexy_co_model_catalog.core.config", "ConfigManager")

    # Pre-populate cache with commonly accessed data
    try:
        provider_list = get_cached_provider_list()
        if provider_list:
            optimizer.response_cache.set("providers_list", provider_list, ttl_seconds=300)
    except Exception:
        pass  # Silently ignore if providers can't be loaded

    return optimizer.get_performance_report()
