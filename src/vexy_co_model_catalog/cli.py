"""
this_file: src/vexy_co_model_catalog/cli.py

Simple Fire-based CLI. Minimal commands provided to avoid importing heavy
submodules at package import time.
"""

from __future__ import annotations

import sys
from typing import Iterable

import fire
from rich.console import Console
from rich.table import Table

from . import __version__

console = Console()


class CLI:
    """Model Catalog Manager CLI (minimal)."""

    def version(self) -> str:
        """Print and return the package version."""
        console.print(f"vexy_co_model_catalog v{__version__}")
        return __version__

    def list_providers(self) -> list[str]:
        """List registered providers (placeholder)."""
        # Placeholder until provider registry is wired
        table = Table(title="Registered Providers")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        console.print(table)
        return []


def main() -> None:
    """Main CLI entry point."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
