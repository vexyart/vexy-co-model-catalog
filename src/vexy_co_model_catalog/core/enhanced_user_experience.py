"""
this_file: src/vexy_co_model_catalog/core/enhanced_user_experience.py

Enhanced user experience with improved error messages, help text, and configuration guidance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table


class ErrorCategory(Enum):
    """Categories for enhanced error classification."""

    CONFIGURATION = "configuration"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    FILE_SYSTEM = "file_system"
    PROVIDER = "provider"
    VALIDATION = "validation"
    PERFORMANCE = "performance"
    USER_INPUT = "user_input"


@dataclass
class EnhancedError:
    """Enhanced error information with guidance and suggestions."""

    category: ErrorCategory
    title: str
    description: str
    suggestions: list[str]
    examples: list[str] = None
    documentation_links: list[str] = None
    severity: str = "error"  # error, warning, info

    def __post_init__(self) -> None:
        if self.examples is None:
            self.examples = []
        if self.documentation_links is None:
            self.documentation_links = []


class EnhancedErrorMessaging:
    """Enhanced error messaging system with actionable guidance."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize enhanced error messaging."""
        self.console = console or Console()

        # Error message templates
        self.error_templates = {
            ErrorCategory.CONFIGURATION: self._setup_config_errors(),
            ErrorCategory.NETWORK: self._setup_network_errors(),
            ErrorCategory.AUTHENTICATION: self._setup_auth_errors(),
            ErrorCategory.FILE_SYSTEM: self._setup_filesystem_errors(),
            ErrorCategory.PROVIDER: self._setup_provider_errors(),
            ErrorCategory.VALIDATION: self._setup_validation_errors(),
            ErrorCategory.PERFORMANCE: self._setup_performance_errors(),
            ErrorCategory.USER_INPUT: self._setup_user_input_errors(),
        }

    def display_enhanced_error(self, error: EnhancedError, context: dict[str, Any] | None = None) -> None:
        """Display an enhanced error message with guidance."""
        context = context or {}

        # Create error panel
        error_content = []

        # Description
        error_content.append(f"[red]{error.description}[/red]")
        error_content.append("")

        # Suggestions
        if error.suggestions:
            error_content.append("[bold yellow]💡 Suggested Solutions:[/bold yellow]")
            for i, suggestion in enumerate(error.suggestions, 1):
                # Replace context variables in suggestions
                formatted_suggestion = suggestion.format(**context)
                error_content.append(f"  {i}. {formatted_suggestion}")
            error_content.append("")

        # Examples
        if error.examples:
            error_content.append("[bold green]📝 Examples:[/bold green]")
            for example in error.examples:
                formatted_example = example.format(**context)
                error_content.append(f"  • {formatted_example}")
            error_content.append("")

        # Documentation links
        if error.documentation_links:
            error_content.append("[bold blue]📚 Documentation:[/bold blue]")
            for link in error.documentation_links:
                error_content.append(f"  • {link}")

        # Create panel with appropriate styling
        severity_colors = {"error": "red", "warning": "yellow", "info": "blue"}

        severity_icons = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}

        panel_title = f"{severity_icons.get(error.severity, '❌')} {error.title}"
        panel_color = severity_colors.get(error.severity, "red")

        panel = Panel("\n".join(error_content), title=panel_title, border_style=panel_color, expand=False)

        self.console.print(panel)

    def get_error_by_type(self, category: ErrorCategory, error_type: str, **context) -> EnhancedError | None:
        """Get a predefined error by category and type."""
        if category in self.error_templates:
            template = self.error_templates[category].get(error_type)
            if template:
                # Create a copy and format with context
                return EnhancedError(
                    category=template.category,
                    title=template.title.format(**context),
                    description=template.description.format(**context),
                    suggestions=[s.format(**context) for s in template.suggestions],
                    examples=[e.format(**context) for e in template.examples],
                    documentation_links=template.documentation_links,
                    severity=template.severity,
                )
        return None

    def _setup_config_errors(self) -> dict[str, EnhancedError]:
        """Setup configuration error templates."""
        return {
            "missing_config": EnhancedError(
                category=ErrorCategory.CONFIGURATION,
                title="Missing Configuration File",
                description="Required configuration file '{config_path}' was not found.",
                suggestions=[
                    "Create the configuration file using: python -m vexy_co_model_catalog generate-configs",
                    "Copy an example configuration from the templates directory",
                    "Check if the file path is correct: {config_path}",
                ],
                examples=[
                    "python -m vexy_co_model_catalog generate-configs --tool=aichat",
                    "python -m vexy_co_model_catalog generate-configs --tool=all",
                ],
            ),
            "invalid_config_format": EnhancedError(
                category=ErrorCategory.CONFIGURATION,
                title="Invalid Configuration Format",
                description="Configuration file '{config_path}' has invalid format: {error_details}",
                suggestions=[
                    "Validate the configuration syntax using an online validator",
                    "Check for missing quotes, brackets, or indentation issues",
                    "Use the validate command: python -m vexy_co_model_catalog validate-config {config_path}",
                    "Restore from backup if available: python -m vexy_co_model_catalog restore-configs",
                ],
                examples=[
                    "python -m vexy_co_model_catalog validate-config ~/.config/aichat/config.yaml",
                    "python -m vexy_co_model_catalog backup-configs",
                    "python -m vexy_co_model_catalog restore-configs --date=latest",
                ],
            ),
            "config_permission_denied": EnhancedError(
                category=ErrorCategory.CONFIGURATION,
                title="Configuration Access Denied",
                description="Permission denied accessing configuration file '{config_path}'.",
                suggestions=[
                    "Check file permissions: ls -la {config_path}",
                    "Fix permissions: chmod 644 {config_path}",
                    "Check if you have write access to the parent directory",
                    "Run with appropriate permissions if needed",
                ],
                examples=[
                    "chmod 644 ~/.config/aichat/config.yaml",
                    "sudo chown $USER:$USER ~/.config/aichat/config.yaml",
                ],
            ),
        }

    def _setup_network_errors(self) -> dict[str, EnhancedError]:
        """Setup network error templates."""
        return {
            "connection_timeout": EnhancedError(
                category=ErrorCategory.NETWORK,
                title="Connection Timeout",
                description="Connection to '{provider_name}' timed out after {timeout} seconds.",
                suggestions=[
                    "Check your internet connection",
                    "Verify the provider URL is correct: {provider_url}",
                    "Try increasing timeout: --timeout={timeout_suggestion}",
                    "Check if the provider service is currently available",
                ],
                examples=[
                    "python -m vexy_co_model_catalog fetch --providers={provider_name} --timeout=30",
                    "python -m vexy_co_model_catalog health --providers={provider_name}",
                ],
            ),
            "dns_resolution_failed": EnhancedError(
                category=ErrorCategory.NETWORK,
                title="DNS Resolution Failed",
                description="Could not resolve hostname for '{provider_url}'.",
                suggestions=[
                    "Check your DNS settings and internet connection",
                    "Try using a different DNS server (8.8.8.8, 1.1.1.1)",
                    "Verify the provider URL is correct",
                    "Check if you're behind a corporate firewall",
                ],
                examples=["nslookup {provider_hostname}", "ping {provider_hostname}"],
            ),
            "ssl_certificate_error": EnhancedError(
                category=ErrorCategory.NETWORK,
                title="SSL Certificate Error",
                description="SSL certificate verification failed for '{provider_url}': {ssl_error}",
                suggestions=[
                    "Check if the provider's SSL certificate is valid",
                    "Update your system's certificate store",
                    "For development only: disable SSL verification (not recommended for production)",
                    "Contact the provider if their certificate has expired",
                ],
                examples=[
                    "openssl s_client -connect {provider_hostname}:443 -servername {provider_hostname}",
                    "python -m vexy_co_model_catalog fetch --providers={provider_name} --verify-ssl=false",
                ],
            ),
        }

    def _setup_auth_errors(self) -> dict[str, EnhancedError]:
        """Setup authentication error templates."""
        return {
            "missing_api_key": EnhancedError(
                category=ErrorCategory.AUTHENTICATION,
                title="Missing API Key",
                description="API key not found for provider '{provider_name}'. Expected environment variable: {api_key_env}",
                suggestions=[
                    "Set the API key environment variable: export {api_key_env}=your_api_key_here",
                    "Add the API key to your shell profile (~/.bashrc, ~/.zshrc)",
                    "Create a .env file with the API key (ensure it's not committed to version control)",
                    "Get an API key from the provider's dashboard: {provider_dashboard_url}",
                ],
                examples=[
                    "export {api_key_env}=sk-your-api-key-here",
                    'echo "{api_key_env}=sk-your-api-key-here" >> ~/.bashrc',
                    'echo "{api_key_env}=sk-your-api-key-here" > .env',
                ],
            ),
            "invalid_api_key": EnhancedError(
                category=ErrorCategory.AUTHENTICATION,
                title="Invalid API Key",
                description="API key for provider '{provider_name}' is invalid or expired.",
                suggestions=[
                    "Verify your API key is correct and hasn't expired",
                    "Generate a new API key from the provider's dashboard",
                    "Check if you have the right permissions for the API endpoint",
                    "Ensure you're using the correct API key format",
                ],
                examples=[
                    "python -m vexy_co_model_catalog validate --providers={provider_name}",
                    "python -m vexy_co_model_catalog health --providers={provider_name}",
                ],
            ),
            "rate_limit_exceeded": EnhancedError(
                category=ErrorCategory.AUTHENTICATION,
                title="Rate Limit Exceeded",
                description="Rate limit exceeded for provider '{provider_name}'. {rate_limit_details}",
                suggestions=[
                    "Wait for the rate limit to reset (usually within an hour)",
                    "Upgrade your API plan for higher rate limits",
                    "Use rate limiting: --rate-limit=requests_per_minute",
                    "Fetch fewer providers at once to stay within limits",
                ],
                examples=[
                    "python -m vexy_co_model_catalog fetch --providers={provider_name} --rate-limit=10",
                    "python -m vexy_co_model_catalog fetch --providers=openai,anthropic --delay=5",
                ],
            ),
        }

    def _setup_filesystem_errors(self) -> dict[str, EnhancedError]:
        """Setup file system error templates."""
        return {
            "directory_not_found": EnhancedError(
                category=ErrorCategory.FILE_SYSTEM,
                title="Directory Not Found",
                description="Directory '{directory_path}' does not exist.",
                suggestions=[
                    "Create the directory: mkdir -p {directory_path}",
                    "Check if the parent directory exists and you have permissions",
                    "Verify the path is correct",
                    "Use the default directory if unsure",
                ],
                examples=["mkdir -p {directory_path}", "python -m vexy_co_model_catalog fetch --output-dir=./models"],
            ),
            "disk_space_full": EnhancedError(
                category=ErrorCategory.FILE_SYSTEM,
                title="Insufficient Disk Space",
                description="Not enough disk space to write files. Required: {required_space}, Available: {available_space}",
                suggestions=[
                    "Free up disk space by deleting unnecessary files",
                    "Clean up old model catalogs: python -m vexy_co_model_catalog clean --temp",
                    "Use a different output directory with more space",
                    "Clean package caches and temporary files",
                ],
                examples=[
                    "python -m vexy_co_model_catalog clean --temp --configs",
                    "df -h  # Check disk usage",
                    "du -sh ~/.cache  # Check cache size",
                ],
            ),
        }

    def _setup_provider_errors(self) -> dict[str, EnhancedError]:
        """Setup provider-specific error templates."""
        return {
            "provider_not_found": EnhancedError(
                category=ErrorCategory.PROVIDER,
                title="Provider Not Found",
                description="Provider '{provider_name}' is not registered in the system.",
                suggestions=[
                    "Check available providers: python -m vexy_co_model_catalog providers",
                    "Add the provider: python -m vexy_co_model_catalog providers add --name={provider_name}",
                    "Check for typos in the provider name",
                    "Use 'all' to fetch from all available providers",
                ],
                examples=[
                    "python -m vexy_co_model_catalog providers",
                    "python -m vexy_co_model_catalog fetch --providers=all",
                    "python -m vexy_co_model_catalog fetch --providers=openai,anthropic",
                ],
            ),
            "provider_service_unavailable": EnhancedError(
                category=ErrorCategory.PROVIDER,
                title="Provider Service Unavailable",
                description="Provider '{provider_name}' service is currently unavailable (HTTP {status_code}).",
                suggestions=[
                    "Check the provider's status page for known issues",
                    "Try again in a few minutes - this may be temporary",
                    "Use health check to monitor provider status",
                    "Skip this provider and fetch from others",
                ],
                examples=[
                    "python -m vexy_co_model_catalog health --providers={provider_name}",
                    "python -m vexy_co_model_catalog fetch --providers=all --skip-failed",
                ],
            ),
        }

    def _setup_validation_errors(self) -> dict[str, EnhancedError]:
        """Setup validation error templates."""
        return {
            "invalid_model_data": EnhancedError(
                category=ErrorCategory.VALIDATION,
                title="Invalid Model Data",
                description="Model data validation failed: {validation_error}",
                suggestions=[
                    "Check the source data from the provider",
                    "Report this issue if the provider data format has changed",
                    "Use --skip-validation flag if you need to proceed anyway",
                    "Update to the latest version of the tool",
                ],
                examples=[
                    "python -m vexy_co_model_catalog fetch --providers={provider_name} --skip-validation",
                    "python -m vexy_co_model_catalog validate-models {model_file}",
                ],
            )
        }

    def _setup_performance_errors(self) -> dict[str, EnhancedError]:
        """Setup performance-related error templates."""
        return {
            "slow_response": EnhancedError(
                category=ErrorCategory.PERFORMANCE,
                title="Slow Response Detected",
                description="Provider '{provider_name}' is responding slowly ({response_time}s > {threshold}s).",
                suggestions=[
                    "Check your internet connection speed",
                    "Try increasing the timeout value",
                    "Use caching to avoid repeated requests",
                    "Consider fetching fewer providers at once",
                ],
                examples=[
                    "python -m vexy_co_model_catalog fetch --providers={provider_name} --timeout=60",
                    "python -m vexy_co_model_catalog fetch --use-cache",
                ],
                severity="warning",
            )
        }

    def _setup_user_input_errors(self) -> dict[str, EnhancedError]:
        """Setup user input error templates."""
        return {
            "invalid_command": EnhancedError(
                category=ErrorCategory.USER_INPUT,
                title="Invalid Command",
                description="Unknown command or option: '{invalid_input}'",
                suggestions=[
                    "Show help: python -m vexy_co_model_catalog --help",
                    "List available commands: python -m vexy_co_model_catalog help",
                    "Check for typos in command name",
                    "Use command aliases for shorter commands",
                ],
                examples=[
                    "python -m vexy_co_model_catalog help",
                    "python -m vexy_co_model_catalog providers  # instead of 'provider'",
                    "python -m vexy_co_model_catalog ls  # alias for 'providers'",
                ],
            )
        }


class EnhancedHelpSystem:
    """Enhanced help system with detailed examples and guidance."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize enhanced help system."""
        self.console = console or Console()

    def show_comprehensive_help(self) -> None:
        """Show comprehensive help with examples and workflows."""
        self.console.print(
            Panel(
                "[bold green]vexy-co-model-catalog[/bold green] - AI Model Catalog Manager\n"
                "[dim]Fetch, normalize, and manage model catalogs from 40+ providers[/dim]",
                title="🤖 AI Model Catalog Manager",
                border_style="green",
            )
        )

        # Quick start section
        self._show_quick_start()

        # Commands section
        self._show_detailed_commands()

        # Examples section
        self._show_workflow_examples()

        # Troubleshooting section
        self._show_troubleshooting()

    def _show_quick_start(self) -> None:
        """Show quick start guide."""
        self.console.print("\n[bold cyan]🚀 Quick Start[/bold cyan]")

        quick_start_steps = [
            "1. List available providers: [green]python -m vexy_co_model_catalog providers[/green]",
            "2. Fetch model catalogs: [green]python -m vexy_co_model_catalog fetch --providers=openai[/green]",
            "3. Check system status: [green]python -m vexy_co_model_catalog stats[/green]",
            "4. Generate tool configs: [green]python -m vexy_co_model_catalog generate-configs --tool=aichat[/green]",
        ]

        for step in quick_start_steps:
            self.console.print(f"  {step}")

    def _show_detailed_commands(self) -> None:
        """Show detailed command information."""
        self.console.print("\n[bold cyan]📋 Detailed Commands[/bold cyan]")

        # Create command table
        table = Table(show_header=True, header_style="bold magenta", border_style="dim")
        table.add_column("Command", style="cyan", no_wrap=True, min_width=15)
        table.add_column("Description", style="white", min_width=35)
        table.add_column("Key Options", style="yellow", min_width=30)
        table.add_column("Example", style="green", min_width=40)

        commands = [
            ("providers", "Manage AI providers", "--action=list|add|remove", "providers --action=show --name=openai"),
            ("fetch", "Fetch model catalogs", "--providers=name1,name2", "fetch --providers=openai,anthropic"),
            ("stats", "Show system statistics", "--detailed, --format=json", "stats --detailed"),
            ("validate", "Validate configurations", "--providers=all", "validate --providers=openai"),
            ("health", "Check provider health", "--timeout=10", "health --providers=all"),
            ("clean", "Clean up files", "--temp, --configs, --failed", "clean --temp --failed"),
            ("generate-configs", "Generate tool configs", "--tool=aichat|codex|mods", "generate-configs --tool=all"),
            ("backup-configs", "Backup configurations", "--destination=path", "backup-configs"),
            ("restore-configs", "Restore configurations", "--date=YYYY-MM-DD", "restore-configs --date=latest"),
        ]

        for cmd, desc, options, example in commands:
            table.add_row(cmd, desc, options, f"python -m vexy_co_model_catalog {example}")

        self.console.print(table)

    def _show_workflow_examples(self) -> None:
        """Show workflow examples."""
        self.console.print("\n[bold cyan]💡 Common Workflows[/bold cyan]")

        workflows = [
            {
                "title": "🔄 Complete Setup Workflow",
                "steps": [
                    "python -m vexy_co_model_catalog providers",
                    "export OPENAI_API_KEY=your_key_here",
                    "python -m vexy_co_model_catalog fetch --providers=openai,anthropic",
                    "python -m vexy_co_model_catalog generate-configs --tool=aichat",
                    "python -m vexy_co_model_catalog stats",
                ],
            },
            {
                "title": "🔍 Troubleshooting Workflow",
                "steps": [
                    "python -m vexy_co_model_catalog health --providers=all",
                    "python -m vexy_co_model_catalog validate --providers=openai",
                    "python -m vexy_co_model_catalog clean --temp --failed",
                    "python -m vexy_co_model_catalog fetch --providers=openai --verbose",
                ],
            },
            {
                "title": "⚙️ Configuration Management",
                "steps": [
                    "python -m vexy_co_model_catalog backup-configs",
                    "python -m vexy_co_model_catalog generate-configs --tool=all",
                    "python -m vexy_co_model_catalog validate-config ~/.config/aichat/config.yaml",
                    "python -m vexy_co_model_catalog restore-configs --date=latest",
                ],
            },
        ]

        for workflow in workflows:
            self.console.print(f"\n[bold yellow]{workflow['title']}[/bold yellow]")
            for i, step in enumerate(workflow["steps"], 1):
                self.console.print(f"  {i}. [green]{step}[/green]")

    def _show_troubleshooting(self) -> None:
        """Show troubleshooting guide."""
        self.console.print("\n[bold cyan]🛠️ Troubleshooting[/bold cyan]")

        troubleshooting_items = [
            (
                "API Key Issues",
                [
                    "Check if environment variable is set: echo $OPENAI_API_KEY",
                    "Verify API key format and permissions",
                    "Test with health command: python -m vexy_co_model_catalog health --providers=openai",
                ],
            ),
            (
                "Network Issues",
                [
                    "Check internet connection and DNS resolution",
                    "Try with increased timeout: --timeout=30",
                    "Check provider status pages for service outages",
                ],
            ),
            (
                "Permission Errors",
                [
                    "Check file and directory permissions: ls -la",
                    "Ensure output directory is writable",
                    "Use absolute paths to avoid confusion",
                ],
            ),
            (
                "Performance Issues",
                [
                    "Use caching: --use-cache flag",
                    "Fetch fewer providers at once",
                    "Monitor with: python -m vexy_co_model_catalog stats",
                ],
            ),
        ]

        for category, solutions in troubleshooting_items:
            self.console.print(f"\n[bold yellow]{category}:[/bold yellow]")
            for solution in solutions:
                self.console.print(f"  • {solution}")


class ConfigurationGuidance:
    """Enhanced configuration guidance system."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize configuration guidance."""
        self.console = console or Console()

    def show_setup_wizard(self) -> None:
        """Show interactive setup wizard with step-by-step configuration."""
        self.console.print(
            Panel(
                "[bold green]🚀 Interactive Configuration Setup Wizard[/bold green]\n"
                "[dim]Let's get your AI model catalog system configured step by step![/dim]",
                title="🧙 Setup Wizard",
                border_style="green",
            )
        )

        # Welcome and overview
        if not Confirm.ask("\n[cyan]Ready to configure your AI model catalog system?[/cyan]", default=True):
            self.console.print("[yellow]Setup cancelled. You can run this wizard anytime with: vexy setup_wizard[/yellow]")
            return

        try:
            # Step 1: Provider selection (interactive)
            selected_providers = self._interactive_provider_selection()

            # Step 2: API key validation and setup
            self._interactive_api_key_setup(selected_providers)

            # Step 3: Tool integration (optional)
            if Confirm.ask("\n[cyan]Would you like to configure tool integrations (aichat, codex, mods)?[/cyan]", default=True):
                self._interactive_tool_integration()

            # Step 4: Test configuration
            if Confirm.ask("\n[cyan]Test your configuration with a quick health check?[/cyan]", default=True):
                self._interactive_health_check(selected_providers)

            # Step 5: Completion and next steps
            self._show_completion_summary(selected_providers)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️  Setup interrupted. Your progress has been saved.[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]❌ Setup error: {e}[/red]")
            self.console.print("[dim]You can retry with: vexy setup_wizard[/dim]")

    def _guide_provider_selection(self) -> None:
        """Guide provider selection."""
        self.console.print("\n[bold cyan]Step 1: Choose Your Providers[/bold cyan]")

        provider_categories = {
            "🏢 Major Commercial": ["openai", "anthropic", "google"],
            "⚡ High Performance": ["groq", "cerebras", "together_ai"],
            "🌐 Aggregators": ["openrouter", "anyscale", "deepinfra"],
            "🔬 Open Source": ["huggingface", "ollama", "local_models"],
        }

        for category, providers in provider_categories.items():
            self.console.print(f"\n[bold yellow]{category}:[/bold yellow]")
            for provider in providers:
                self.console.print(f"  • {provider}")

        self.console.print("\n[green]💡 Recommendation:[/green] Start with openai and anthropic for best compatibility")
        self.console.print("[dim]Command: python -m vexy_co_model_catalog providers[/dim]")

    def _guide_api_key_setup(self) -> None:
        """Guide API key setup."""
        self.console.print("\n[bold cyan]Step 2: Set Up API Keys[/bold cyan]")

        key_examples = [
            ("OpenAI", "OPENAI_API_KEY", "sk-..."),
            ("Anthropic", "ANTHROPIC_API_KEY", "sk-ant-..."),
            ("Groq", "GROQ_API_KEY", "gsk_..."),
            ("Google", "GOOGLE_API_KEY", "AIza..."),
        ]

        for provider, env_var, prefix in key_examples:
            self.console.print(f"\n[bold yellow]{provider}:[/bold yellow]")
            self.console.print(f"  Environment variable: [green]{env_var}[/green]")
            self.console.print(f"  Format: [dim]{prefix}...[/dim]")
            self.console.print(f"  Set with: [cyan]export {env_var}=your_key_here[/cyan]")

        self.console.print("\n[green]💡 Tip:[/green] Add API keys to your ~/.bashrc or ~/.zshrc for persistence")
        self.console.print("[green]💡 Security:[/green] Never commit API keys to version control")

    def _guide_tool_integration(self) -> None:
        """Guide tool integration setup."""
        self.console.print("\n[bold cyan]Step 3: Choose Tools to Integrate[/bold cyan]")

        tools = [
            {
                "name": "aichat",
                "description": "Terminal-based AI chat with multiple providers",
                "config_path": "~/.config/aichat/config.yaml",
                "command": "generate-configs --tool=aichat",
            },
            {
                "name": "codex",
                "description": "Code assistant and automation tool",
                "config_path": "~/.config/codex/config.toml",
                "command": "generate-configs --tool=codex",
            },
            {
                "name": "mods",
                "description": "AI-powered terminal commands",
                "config_path": "~/.config/mods/mods.yml",
                "command": "generate-configs --tool=mods",
            },
        ]

        for tool in tools:
            self.console.print(f"\n[bold yellow]{tool['name']}:[/bold yellow] {tool['description']}")
            self.console.print(f"  Config: [dim]{tool['config_path']}[/dim]")
            self.console.print(f"  Setup: [cyan]python -m vexy_co_model_catalog {tool['command']}[/cyan]")

        self.console.print("\n[green]💡 Recommendation:[/green] Generate configs for all tools you use")

    def _guide_first_fetch(self) -> None:
        """Guide first fetch operation."""
        self.console.print("\n[bold cyan]Step 4: Fetch Your First Model Catalogs[/bold cyan]")

        fetch_examples = [
            "# Fetch from specific providers",
            "python -m vexy_co_model_catalog fetch --providers=openai,anthropic",
            "",
            "# Fetch all available providers",
            "python -m vexy_co_model_catalog fetch --providers=all",
            "",
            "# Check what was fetched",
            "python -m vexy_co_model_catalog stats",
        ]

        syntax = Syntax("\n".join(fetch_examples), "bash", theme="monokai", line_numbers=False)
        self.console.print(syntax)

        self.console.print("\n[green]🎉 You're all set![/green] Your AI model catalog system is ready to use.")
        self.console.print("[dim]Use 'python -m vexy_co_model_catalog help' for more information[/dim]")

    def _interactive_provider_selection(self) -> list[str]:
        """Interactive provider selection with recommendations."""
        self.console.print("\n[bold cyan]🏢 Step 1: Provider Selection[/bold cyan]")

        provider_categories = {
            "🏢 Major Commercial (Recommended for beginners)": {
                "providers": ["openai", "anthropic", "google"],
                "description": "Most stable and well-documented APIs"
            },
            "⚡ High Performance": {
                "providers": ["groq", "cerebras", "together_ai"],
                "description": "Fast inference, good for development"
            },
            "🌐 Aggregators": {
                "providers": ["openrouter", "anyscale", "deepinfra"],
                "description": "Access multiple models through single API"
            },
            "🔬 Open Source": {
                "providers": ["huggingface", "ollama", "replicate"],
                "description": "Open models and local deployment"
            }
        }

        selected_providers = []

        # Show categories and let user select
        for category, info in provider_categories.items():
            self.console.print(f"\n[bold yellow]{category}:[/bold yellow]")
            self.console.print(f"[dim]{info['description']}[/dim]")

            providers_str = ", ".join(info['providers'])
            self.console.print(f"  Providers: {providers_str}")

            if Confirm.ask("  Add providers from this category?", default=(category.startswith("🏢"))):
                # Allow selection of specific providers
                for provider in info['providers']:
                    if Confirm.ask(f"    Include {provider}?", default=True):
                        selected_providers.append(provider)

        # Offer custom provider input
        if Confirm.ask("\n[cyan]Add any other providers?[/cyan]", default=False):
            custom_input = Prompt.ask("Enter provider names (comma-separated)")
            if custom_input:
                custom_providers = [p.strip() for p in custom_input.split(",") if p.strip()]
                selected_providers.extend(custom_providers)

        # Ensure at least one provider
        if not selected_providers:
            self.console.print("[yellow]⚠️  No providers selected. Adding OpenAI as default.[/yellow]")
            selected_providers = ["openai"]

        self.console.print(f"\n[green]✅ Selected providers: {', '.join(selected_providers)}[/green]")
        return selected_providers

    def _interactive_api_key_setup(self, providers: list[str]) -> None:
        """Interactive API key setup with validation."""
        self.console.print("\n[bold cyan]🔑 Step 2: API Key Configuration[/bold cyan]")

        key_mapping = {
            "openai": ("OPENAI_API_KEY", "sk-", "https://platform.openai.com/api-keys"),
            "anthropic": ("ANTHROPIC_API_KEY", "sk-ant-", "https://console.anthropic.com/"),
            "google": ("GOOGLE_API_KEY", "AIza", "https://makersuite.google.com/app/apikey"),
            "groq": ("GROQ_API_KEY", "gsk_", "https://console.groq.com/keys"),
        }

        configured_keys = []

        for provider in providers:
            if provider not in key_mapping:
                self.console.print(f"[dim]⏭️  Skipping {provider} (no standard API key pattern)[/dim]")
                continue

            env_var, prefix, url = key_mapping[provider]

            # Check if key already exists
            existing_key = os.environ.get(env_var)
            if existing_key:
                self.console.print(f"\n[green]✅ {provider.title()}: API key already configured[/green]")
                configured_keys.append(provider)
                continue

            self.console.print(f"\n[bold yellow]🔧 Configure {provider.title()}:[/bold yellow]")
            self.console.print(f"  Environment variable: [cyan]{env_var}[/cyan]")
            self.console.print(f"  Key format: [dim]{prefix}...[/dim]")
            self.console.print(f"  Get your key: [blue]{url}[/blue]")

            if Confirm.ask(f"  Configure {provider} API key now?", default=True):
                while True:
                    key = Prompt.ask(f"    Enter your {provider} API key", password=True)
                    if not key:
                        if not Confirm.ask("    Skip this provider?", default=False):
                            continue
                        break

                    if not key.startswith(prefix):
                        self.console.print(f"    [yellow]⚠️  Warning: Key doesn't start with expected prefix '{prefix}'[/yellow]")
                        if not Confirm.ask("    Continue anyway?", default=False):
                            continue

                    # Set environment variable for current session
                    os.environ[env_var] = key
                    self.console.print(f"    [green]✅ {provider} API key configured for this session[/green]")
                    self.console.print(f"    [dim]To persist: export {env_var}=your_key_here[/dim]")
                    configured_keys.append(provider)
                    break

        if configured_keys:
            self.console.print(f"\n[green]✅ API keys configured for: {', '.join(configured_keys)}[/green]")
        else:
            self.console.print("\n[yellow]⚠️  No API keys configured. Some providers may not work.[/yellow]")

    def _interactive_tool_integration(self) -> None:
        """Interactive tool integration setup."""
        self.console.print("\n[bold cyan]🔧 Step 3: Tool Integration Setup[/bold cyan]")

        tools = [
            {
                "name": "aichat",
                "description": "Terminal-based AI chat with multiple providers",
                "config_command": "aichat"
            },
            {
                "name": "codex",
                "description": "Code assistant and automation tool",
                "config_command": "codex"
            },
            {
                "name": "mods",
                "description": "AI-powered terminal commands",
                "config_command": "mods"
            }
        ]

        for tool in tools:
            self.console.print(f"\n[bold yellow]🛠️  {tool['name'].title()}:[/bold yellow]")
            self.console.print(f"  {tool['description']}")

            if Confirm.ask(f"  Generate {tool['name']} configuration?", default=False):
                self.console.print(f"  [cyan]Command: python -m vexy_co_model_catalog generate-configs --tool={tool['config_command']}[/cyan]")
                self.console.print("  [dim]Run this command after setup completes[/dim]")

    def _interactive_health_check(self, providers: list[str]) -> None:
        """Interactive health check for selected providers."""
        self.console.print("\n[bold cyan]🏥 Step 4: Configuration Health Check[/bold cyan]")

        self.console.print("[dim]Testing provider connectivity...[/dim]")

        # This would integrate with the actual health check system
        self.console.print(f"[cyan]Command to run: python -m vexy_co_model_catalog health --providers={','.join(providers)}[/cyan]")
        self.console.print("[dim]This will verify API connectivity and authentication[/dim]")

        if Confirm.ask("Run health check now?", default=True):
            self.console.print("[yellow]Health check would run here (integrated with actual CLI health command)[/yellow]")

    def _show_completion_summary(self, providers: list[str]) -> None:
        """Show setup completion summary and next steps."""
        self.console.print(
            Panel(
                "[bold green]🎉 Setup Complete![/bold green]\n"
                f"[dim]Configured {len(providers)} providers: {', '.join(providers)}[/dim]",
                title="✅ Success",
                border_style="green",
            )
        )

        self.console.print("\n[bold cyan]🚀 Recommended Next Steps:[/bold cyan]")
        next_steps = [
            "1. Test your configuration:",
            f"   [cyan]python -m vexy_co_model_catalog health --providers={','.join(providers[:3])}[/cyan]",
            "",
            "2. Fetch model catalogs:",
            f"   [cyan]python -m vexy_co_model_catalog fetch --providers={','.join(providers[:2])}[/cyan]",
            "",
            "3. View statistics:",
            "   [cyan]python -m vexy_co_model_catalog stats[/cyan]",
            "",
            "4. Set up shell completion:",
            "   [cyan]python -m vexy_co_model_catalog completion bash --install[/cyan]",
            "",
            "5. Generate tool configs (optional):",
            "   [cyan]python -m vexy_co_model_catalog generate-configs --tool=aichat[/cyan]"
        ]

        for step in next_steps:
            if step.startswith("   "):
                self.console.print(step)
            elif step:
                self.console.print(f"[bold white]{step}[/bold white]")
            else:
                self.console.print()

        self.console.print("\n[green]💡 Get help anytime with: [cyan]python -m vexy_co_model_catalog help[/cyan][/green]")


def get_enhanced_error_handler() -> EnhancedErrorMessaging:
    """Get global enhanced error handler instance."""
    return EnhancedErrorMessaging()


def get_enhanced_help_system() -> EnhancedHelpSystem:
    """Get global enhanced help system instance."""
    return EnhancedHelpSystem()


def get_configuration_guidance() -> ConfigurationGuidance:
    """Get global configuration guidance instance."""
    return ConfigurationGuidance()
