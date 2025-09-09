#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["loguru", "rich"]
# ///
# this_file: scripts/test_security.py

"""
Quick test script to verify the security module is working correctly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console

from vexy_co_model_catalog.core.security import get_protector, mask_text_for_logs, sanitize_headers

console = Console()


def test_api_key_masking():
    """Test API key masking functionality."""
    console.print("[bold cyan]Testing API Key Masking[/bold cyan]")

    test_keys = [
        "sk-1234567890abcdef1234567890abcdef",
        "ant-1234567890abcdef1234567890abcdef1234567890",
        "gsk_1234567890abcdef1234567890abcdef1234567890abcdef",
        "some-random-long-token-that-should-be-masked-1234567890",
    ]

    protector = get_protector()

    for key in test_keys:
        masked = protector.mask_api_key(key)
        console.print(f"Original: [red]{key}[/red]")
        console.print(f"Masked:   [green]{masked}[/green]")
        console.print()


def test_header_sanitization():
    """Test HTTP header sanitization."""
    console.print("[bold cyan]Testing Header Sanitization[/bold cyan]")

    test_headers = {
        "Authorization": "Bearer sk-1234567890abcdef1234567890abcdef",
        "X-API-Key": "ant-1234567890abcdef1234567890abcdef1234567890",
        "Content-Type": "application/json",
        "User-Agent": "ModelDumper/1.0",
        "x-anthropic-api-key": "ant-secretkey123456789",
        "X-RateLimit-Remaining": "100",
    }

    sanitized = sanitize_headers(test_headers)

    for key, value in test_headers.items():
        safe_value = sanitized.get(key, value)
        "red" if key.lower() in ["authorization", "x-api-key", "x-anthropic-api-key"] else "white"
        safe_color = "green" if safe_value != value else "white"

        console.print(f"{key}: [color]{value}[/color] → [{safe_color}]{safe_value}[/{safe_color}]")
    console.print()


def test_text_masking():
    """Test text masking for logs."""
    console.print("[bold cyan]Testing Text Masking[/bold cyan]")

    test_texts = [
        "API call failed: sk-1234567890abcdef1234567890abcdef returned 401",
        "export OPENAI_API_KEY='sk-secret123'",
        "Error response: {'error': 'Invalid API key: ant-12345...'}",
        "Making request to https://api.openai.com/v1/models",
        "Bearer token expired: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.1234567890abcdef",
    ]

    for text in test_texts:
        masked = mask_text_for_logs(text)
        console.print(f"Original: [red]{text}[/red]")
        console.print(f"Masked:   [green]{masked}[/green]")
        console.print()


def test_environment_masking():
    """Test environment variable masking."""
    console.print("[bold cyan]Testing Environment Variable Masking[/bold cyan]")

    test_env = {
        "OPENAI_API_KEY": "sk-1234567890abcdef1234567890abcdef",
        "ANTHROPIC_API_KEY": "ant-1234567890abcdef1234567890abcdef1234567890",
        "HOME": "/Users/testuser",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "SECRET_TOKEN": "super-secret-value-123",
        "DATABASE_PASSWORD": "db-password-123",
    }

    protector = get_protector()
    masked_env = protector.mask_environment_vars(test_env)

    for key, value in test_env.items():
        safe_value = masked_env.get(key, value)
        is_sensitive = value != safe_value
        color = "red" if is_sensitive else "white"
        safe_color = "green" if is_sensitive else "white"

        console.print(f"{key}: [{color}]{value}[/{color}] → [{safe_color}]{safe_value}[/{safe_color}]")
    console.print()


if __name__ == "__main__":
    console.print("[bold green]Security Module Test Suite[/bold green]")
    console.print()

    test_api_key_masking()
    test_header_sanitization()
    test_text_masking()
    test_environment_masking()

    console.print("[bold green]✅ Security testing complete![/bold green]")
