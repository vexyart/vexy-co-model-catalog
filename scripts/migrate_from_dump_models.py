#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["rich", "pathlib", "json", "os", "sys"]
# ///
# this_file: scripts/migrate_from_dump_models.py

"""
Migration script to help users transition from dump_models.py to vexy-co-model-catalog.

This script analyzes existing dump_models.py output and helps users migrate to the new system.
"""

import json
import os
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress
    from rich.table import Table
except ImportError:
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "rich"], check=True)
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress
    from rich.table import Table

console = Console()


def find_dump_models_output(search_paths: list[str] | None = None) -> dict[str, list[Path]]:
    """Find existing dump_models.py output files."""
    if search_paths is None:
        search_paths = [".", "models", "../models", "~/models"]

    found_files = {"json": [], "txt": [], "toml": []}

    for search_path in search_paths:
        path = Path(search_path).expanduser()
        if not path.exists():
            continue

        # Look for models_*.json, models_*.txt, models_*.toml files
        for pattern in ["models_*.json", "models_*.txt", "models_*.toml"]:
            for file_path in path.glob(pattern):
                extension = file_path.suffix[1:]  # Remove the dot
                if extension in found_files:
                    found_files[extension].append(file_path)

    return found_files


def analyze_existing_files(files: dict[str, list[Path]]) -> dict[str, any]:
    """Analyze existing dump_models.py output files."""
    analysis = {
        "providers": set(),
        "total_models": 0,
        "file_count": {ext: len(files[ext]) for ext in files},
        "provider_models": {},
        "missing_files": [],
        "environment_vars": set(),
    }

    # Extract provider names from filenames
    all_files = []
    for file_list in files.values():
        all_files.extend(file_list)

    for file_path in all_files:
        # Extract provider name from models_PROVIDER.ext
        name_parts = file_path.stem.split("_", 1)
        if len(name_parts) == 2 and name_parts[0] == "models":
            provider_name = name_parts[1]
            analysis["providers"].add(provider_name)

    # Analyze JSON files for model counts and required env vars
    for json_file in files["json"]:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            provider_name = json_file.stem.split("_", 1)[1]

            # Count models
            model_count = 0
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    model_count = len(data["data"])
                elif isinstance(data, dict):
                    model_count = len([k for k in data if k != "sample_spec"])
            elif isinstance(data, list):
                model_count = len(data)

            analysis["provider_models"][provider_name] = model_count
            analysis["total_models"] += model_count

            # Infer required environment variables
            provider_upper = provider_name.upper()
            possible_env_vars = [
                f"{provider_upper}_API_KEY",
                f"{provider_upper}_API_OPENAI",
                f"{provider_name.upper()}_KEY",
                f"{provider_name.upper()}_TOKEN",
            ]
            analysis["environment_vars"].update(possible_env_vars)

        except Exception as e:
            console.print(f"[yellow]Warning: Could not analyze {json_file}: {e}[/yellow]")

    return analysis


def check_environment_compatibility() -> dict[str, bool]:
    """Check which environment variables are set."""
    common_env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "FIREWORKS_API_KEY",
        "TOGETHERAI_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPINFRA_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
        "CEREBRAS_API_KEY",
    ]

    env_status = {}
    for var in common_env_vars:
        env_status[var] = var in os.environ and bool(os.environ[var])

    return env_status


def generate_migration_plan(analysis: dict, env_status: dict[str, bool]) -> dict[str, any]:
    """Generate a migration plan based on the analysis."""
    plan = {"ready_providers": [], "missing_env_vars": [], "new_commands": [], "compatibility_notes": []}

    # Check which providers are ready to migrate
    provider_env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "togetherai": "TOGETHERAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "deepinfra": "DEEPINFRA_API_KEY",
        "huggingface": "HUGGINGFACEHUB_API_TOKEN",
        "cerebras": "CEREBRAS_API_KEY",
    }

    for provider in analysis["providers"]:
        env_var = provider_env_map.get(provider)
        if env_var and env_status.get(env_var, False):
            plan["ready_providers"].append(provider)
        elif env_var:
            plan["missing_env_vars"].append(env_var)

    # Generate command suggestions
    if plan["ready_providers"]:
        provider_list = ",".join(plan["ready_providers"])
        plan["new_commands"] = [
            f"python -m vexy_co_model_catalog fetch --providers={provider_list}",
            "python -m vexy_co_model_catalog fetch --providers=all --legacy-output",
            "python -m vexy_co_model_catalog link --target-dir=models",
            "python -m vexy_co_model_catalog providers list",
        ]

    # Add compatibility notes
    plan["compatibility_notes"] = [
        "Use --legacy-output flag to maintain dump_models.py directory structure",
        "New system outputs to config/ directory by default for better organization",
        "Use 'link' command to create symlinks for existing tooling",
        "All environment variables work the same way as dump_models.py",
    ]

    return plan


def main():
    """Main migration script."""
    console.print(
        Panel.fit(
            "[bold blue]vexy-co-model-catalog Migration Tool[/bold blue]\nHelping you transition from dump_models.py",
            title="🚀 Migration Assistant",
        )
    )

    # Step 1: Find existing files
    console.print("\n[bold]Step 1: Searching for existing dump_models.py output...[/bold]")

    with Progress() as progress:
        task = progress.add_task("Scanning directories...", total=100)
        files = find_dump_models_output()
        progress.update(task, completed=100)

    total_files = sum(len(file_list) for file_list in files.values())

    if total_files == 0:
        console.print("[yellow]No existing dump_models.py output files found.[/yellow]")
        console.print("You can proceed directly with the new system:")
        console.print("  python -m vexy_co_model_catalog fetch --providers=all")
        return

    console.print(f"[green]Found {total_files} output files[/green]")

    # Step 2: Analyze files
    console.print("\n[bold]Step 2: Analyzing existing configuration...[/bold]")
    analysis = analyze_existing_files(files)

    # Display analysis results
    table = Table(title="Current dump_models.py Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Providers Found", str(len(analysis["providers"])))
    table.add_row("Total Models", str(analysis["total_models"]))
    table.add_row("JSON Files", str(analysis["file_count"]["json"]))
    table.add_row("TXT Files", str(analysis["file_count"]["txt"]))
    table.add_row("TOML Files", str(analysis["file_count"]["toml"]))

    console.print(table)

    # Step 3: Check environment
    console.print("\n[bold]Step 3: Checking environment compatibility...[/bold]")
    env_status = check_environment_compatibility()

    env_table = Table(title="Environment Variables")
    env_table.add_column("Variable", style="cyan")
    env_table.add_column("Status", style="green")

    for var, is_set in env_status.items():
        status = "✓ Set" if is_set else "✗ Missing"
        color = "green" if is_set else "red"
        env_table.add_row(var, f"[{color}]{status}[/{color}]")

    console.print(env_table)

    # Step 4: Generate migration plan
    console.print("\n[bold]Step 4: Generating migration plan...[/bold]")
    plan = generate_migration_plan(analysis, env_status)

    if plan["ready_providers"]:
        console.print(f"\n[green]✓ Ready to migrate {len(plan['ready_providers'])} providers:[/green]")
        for provider in plan["ready_providers"]:
            model_count = analysis["provider_models"].get(provider, "unknown")
            console.print(f"  • {provider} ({model_count} models)")

        console.print("\n[bold]Recommended migration commands:[/bold]")
        for i, command in enumerate(plan["new_commands"], 1):
            console.print(f"  {i}. [yellow]{command}[/yellow]")

    if plan["missing_env_vars"]:
        console.print("\n[yellow]⚠ Missing environment variables:[/yellow]")
        for var in plan["missing_env_vars"]:
            console.print(f"  • {var}")

    # Step 5: Compatibility notes
    console.print("\n[bold]Step 5: Migration notes:[/bold]")
    for note in plan["compatibility_notes"]:
        console.print(f"  • {note}")

    # Step 6: Next steps
    console.print(
        "\n"
        + Panel.fit(
            "[bold]Next Steps:[/bold]\n\n"
            "1. Install: [yellow]pip install vexy-co-model-catalog[/yellow]\n"
            "2. Test: [yellow]python -m vexy_co_model_catalog providers list[/yellow]\n"
            "3. Fetch: [yellow]python -m vexy_co_model_catalog fetch --providers=all --legacy-output[/yellow]\n"
            "4. Link: [yellow]python -m vexy_co_model_catalog link[/yellow]\n"
            "5. Verify: Compare new output with existing files\n"
            "6. Update your tools to use new output locations",
            title="🎯 Action Plan",
        )
    )


if __name__ == "__main__":
    main()
