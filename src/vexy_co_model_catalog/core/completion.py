"""
this_file: src/vexy_co_model_catalog/core/completion.py

Shell completion support for vexy-co-model-catalog CLI commands.
Provides bash, zsh, and fish completion scripts and integration.
"""

from __future__ import annotations

# Removed unused imports: os, Path, Any
from vexy_co_model_catalog.core.provider import get_all_providers


class CompletionGenerator:
    """Generate shell completion scripts for the vexy CLI."""

    def __init__(self) -> None:
        """Initialize completion generator."""
        self.main_commands = [
            "fetch", "providers", "validate", "health", "validate_config",
            "validate_models", "stats", "clean", "cache", "performance",
            "production_status", "production_init", "production_readiness",
            "production_errors", "monitor", "monitoring_status", "setup_wizard",
            "version", "help", "aliases"
        ]

        self.aliases = [
            "ls", "list", "get", "dl", "download", "sync", "st", "status",
            "check", "test", "verify", "rm", "clear", "remove", "diag",
            "diagnostics", "healthcheck", "analytics", "metrics", "limits",
            "throttle", "health-check", "files", "backup", "restore",
            "setup", "wizard", "configure", "init"
        ]

        self.provider_actions = ["list", "show", "add", "remove"]
        self.provider_kinds = ["openai", "anthropic", "url"]
        self.cache_actions = ["stats", "clear"]
        self.performance_actions = ["stats", "history", "save", "clear"]
        self.monitor_actions = ["dashboard", "metrics", "alerts", "trends", "export", "status"]

    def get_provider_names(self) -> list[str]:
        """Get list of available provider names."""
        try:
            providers = get_all_providers()
            return [provider.name for provider in providers]
        except Exception:
            # Fallback to common provider names if registry fails
            return [
                "openai", "anthropic", "groq", "together", "perplexity",
                "huggingface", "cohere", "replicate", "anyscale", "fireworks"
            ]

    def generate_bash_completion(self) -> str:
        """Generate bash completion script."""
        provider_names = " ".join(self.get_provider_names())

        return f"""#!/bin/bash
# Bash completion script for vexy-co-model-catalog
# Install by: source this file or place in /etc/bash_completion.d/

_vexy_completion() {{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"

    # Main commands and aliases
    local main_commands="{' '.join(self.main_commands)}"
    local aliases="{' '.join(self.aliases)}"
    local all_commands="$main_commands $aliases"

    # Provider names
    local providers="{provider_names}"

    case "${{prev}}" in
        vexy)
            COMPREPLY=( $(compgen -W "$all_commands" -- "$cur") )
            return 0
            ;;
        providers)
            COMPREPLY=( $(compgen -W "{' '.join(self.provider_actions)}" -- "$cur") )
            return 0
            ;;
        --name)
            COMPREPLY=( $(compgen -W "$providers" -- "$cur") )
            return 0
            ;;
        --kind)
            COMPREPLY=( $(compgen -W "{' '.join(self.provider_kinds)}" -- "$cur") )
            return 0
            ;;
        --provider)
            COMPREPLY=( $(compgen -W "$providers all" -- "$cur") )
            return 0
            ;;
        cache)
            COMPREPLY=( $(compgen -W "{' '.join(self.cache_actions)}" -- "$cur") )
            return 0
            ;;
        performance)
            COMPREPLY=( $(compgen -W "{' '.join(self.performance_actions)}" -- "$cur") )
            return 0
            ;;
        monitor)
            COMPREPLY=( $(compgen -W "{' '.join(self.monitor_actions)}" -- "$cur") )
            return 0
            ;;
        --config-path|--file-path)
            # Complete file paths
            COMPREPLY=( $(compgen -f -- "$cur") )
            return 0
            ;;
    esac

    # Handle options with equals sign
    case "$cur" in
        --name=*)
            local value=${{cur#*=}}
            COMPREPLY=( $(compgen -W "$providers" -- "$value") )
            COMPREPLY=( "${{COMPREPLY[@]/#/--name=}}" )
            return 0
            ;;
        --kind=*)
            local value=${{cur#*=}}
            COMPREPLY=( $(compgen -W "{' '.join(self.provider_kinds)}" -- "$value") )
            COMPREPLY=( "${{COMPREPLY[@]/#/--kind=}}" )
            return 0
            ;;
        --provider=*)
            local value=${{cur#*=}}
            COMPREPLY=( $(compgen -W "$providers all" -- "$value") )
            COMPREPLY=( "${{COMPREPLY[@]/#/--provider=}}" )
            return 0
            ;;
    esac

    # Complete options
    if [[ "$cur" == --* ]]; then
        local opts="--help --verbose --name --kind --provider --timeout"
        opts="$opts --detailed --fix --save --clear --export --start --stop"
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
        return 0
    fi

    # Default to commands
    COMPREPLY=( $(compgen -W "$all_commands" -- "$cur") )
    return 0
}}

complete -F _vexy_completion vexy
"""

    def generate_zsh_completion(self) -> str:
        """Generate zsh completion script."""
        provider_names = " ".join(self.get_provider_names())

        return f"""#compdef vexy
# Zsh completion script for vexy-co-model-catalog
# Install by: place in your fpath (e.g., ~/.zsh/completions/)

_vexy() {{
    local context state line
    typeset -A opt_args

    local providers=({provider_names})

    _arguments -C \\
        '--help[Show help message]' \\
        '--verbose[Enable verbose output]' \\
        '*::command:->commands' \\
        && return 0

    case $state in
        commands)
            local commands=(
                {' '.join([f'"{cmd}:Main command"' for cmd in self.main_commands])}
                {' '.join([f'"{alias}:Alias command"' for alias in self.aliases])}
            )

            _describe 'commands' commands
            ;;
    esac
}}

_vexy "$@"
"""

    def generate_fish_completion(self) -> str:
        """Generate fish completion script."""
        provider_names = self.get_provider_names()

        completions = []

        # Main commands
        for cmd in self.main_commands:
            completions.append(f"complete -c vexy -f -a '{cmd}' -d 'Main command'")

        # Aliases
        for alias in self.aliases:
            completions.append(f"complete -c vexy -f -a '{alias}' -d 'Alias command'")

        # Provider actions
        for action in self.provider_actions:
            completions.append(
                f"complete -c vexy -f -n '__fish_seen_subcommand_from providers' "
                f"-a '{action}' -d 'Provider action'"
            )

        # Provider names for --name and --provider options
        for provider in provider_names:
            completions.append(f"complete -c vexy -f -l name -a '{provider}' -d 'Provider name'")
            completions.append(f"complete -c vexy -f -l provider -a '{provider}' -d 'Provider name'")

        # Provider kinds
        for kind in self.provider_kinds:
            completions.append(f"complete -c vexy -f -l kind -a '{kind}' -d 'Provider type'")

        # Cache actions
        for action in self.cache_actions:
            completions.append(
                f"complete -c vexy -f -n '__fish_seen_subcommand_from cache' "
                f"-a '{action}' -d 'Cache action'"
            )

        # Performance actions
        for action in self.performance_actions:
            completions.append(
                f"complete -c vexy -f -n '__fish_seen_subcommand_from performance' "
                f"-a '{action}' -d 'Performance action'"
            )

        # Monitor actions
        for action in self.monitor_actions:
            completions.append(
                f"complete -c vexy -f -n '__fish_seen_subcommand_from monitor' "
                f"-a '{action}' -d 'Monitor action'"
            )

        # Common options
        completions.extend([
            "complete -c vexy -f -l help -d 'Show help message'",
            "complete -c vexy -f -l verbose -d 'Enable verbose output'",
            "complete -c vexy -f -l timeout -d 'Request timeout'",
            "complete -c vexy -f -l detailed -d 'Show detailed information'",
            "complete -c vexy -f -l fix -d 'Attempt automatic fixes'",
            "complete -c vexy -f -l save -d 'Save results'",
            "complete -c vexy -f -l clear -d 'Clear data'",
            "complete -c vexy -f -l export -d 'Export data'",
            "complete -c vexy -f -l start -d 'Start service'",
            "complete -c vexy -f -l stop -d 'Stop service'",
        ])

        return "\\n".join(completions)

    def install_completion_scripts(self, shell: str = "auto") -> str:
        """Generate installation instructions for completion scripts."""
        instructions = []

        if shell in ("bash", "auto"):
            instructions.append("""
## Bash Completion Installation

### Method 1: System-wide installation
```bash
# Generate and install completion script
vexy completion bash > /tmp/vexy-completion.bash
sudo cp /tmp/vexy-completion.bash /etc/bash_completion.d/vexy
```

### Method 2: User installation
```bash
# Add to your ~/.bashrc
vexy completion bash >> ~/.bashrc
source ~/.bashrc
```
""")

        if shell in ("zsh", "auto"):
            instructions.append("""
## Zsh Completion Installation

### Method 1: Oh My Zsh
```bash
# Create completions directory if it doesn't exist
mkdir -p ~/.oh-my-zsh/custom/plugins/vexy
vexy completion zsh > ~/.oh-my-zsh/custom/plugins/vexy/_vexy
# Add 'vexy' to plugins in ~/.zshrc
```

### Method 2: Manual installation
```bash
# Create completions directory
mkdir -p ~/.zsh/completions
vexy completion zsh > ~/.zsh/completions/_vexy
# Add to ~/.zshrc:
# fpath=(~/.zsh/completions $fpath)
# autoload -U compinit && compinit
```
""")

        if shell in ("fish", "auto"):
            instructions.append("""
## Fish Completion Installation

```bash
# Install completion script
vexy completion fish > ~/.config/fish/completions/vexy.fish
```
""")

        return "\\n".join(instructions)

    def generate_completion_script(self, shell: str) -> str:
        """Generate completion script for specified shell."""
        if shell == "bash":
            return self.generate_bash_completion()
        if shell == "zsh":
            return self.generate_zsh_completion()
        if shell == "fish":
            return self.generate_fish_completion()
        msg = f"Unsupported shell: {shell}. Supported: bash, zsh, fish"
        raise ValueError(msg)


def get_completion_generator() -> CompletionGenerator:
    """Get completion generator instance."""
    return CompletionGenerator()
