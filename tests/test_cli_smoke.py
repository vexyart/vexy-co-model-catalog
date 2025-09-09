"""
this_file: tests/test_cli_smoke.py

Smoke tests for CLI commands to ensure core functionality works without crashing.
"""

import subprocess
import sys


class TestCLISmoke:
    """Smoke tests for CLI commands."""

    @staticmethod
    def run_cli_command(command_args: list[str]) -> tuple[int, str, str]:
        """Run a CLI command and return exit code, stdout, stderr."""
        cmd = [sys.executable, "-m", "vexy_co_model_catalog", *command_args]
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout for safety
        )
        return result.returncode, result.stdout, result.stderr

    def test_version_command(self):
        """Test that version command works and returns version info."""
        exit_code, stdout, stderr = self.run_cli_command(["version"])

        assert exit_code == 0, f"Version command failed with stderr: {stderr}"
        assert "vexy-co-model-catalog" in stdout, "Version output missing package name"
        assert "v" in stdout, "Version output missing version number"

    def test_help_command(self):
        """Test that help command works and shows comprehensive help."""
        exit_code, stdout, stderr = self.run_cli_command(["help"])

        assert exit_code == 0, f"Help command failed with stderr: {stderr}"
        assert "Available Commands" in stdout, "Help missing commands section"
        assert "providers" in stdout, "Help missing providers command"
        assert "fetch" in stdout, "Help missing fetch command"
        assert "Quick Start Examples" in stdout, "Help missing examples section"

    def test_providers_list_command(self):
        """Test that providers list command works and shows providers."""
        exit_code, stdout, stderr = self.run_cli_command(["providers"])

        assert exit_code == 0, f"Providers command failed with stderr: {stderr}"
        assert "Registered AI Providers" in stdout, "Providers output missing table header"
        assert "Total:" in stdout, "Providers output missing total count"
        assert "openai" in stdout, "Providers output missing OpenAI provider"

    def test_providers_show_command(self):
        """Test that providers show command works for a known provider."""
        exit_code, stdout, stderr = self.run_cli_command(["providers", "--action=show", "--name=openai"])

        assert exit_code == 0, f"Providers show command failed with stderr: {stderr}"
        assert "Provider: openai" in stdout, "Provider show missing provider name"
        assert "Type: openai" in stdout, "Provider show missing type info"

    def test_stats_command(self):
        """Test that stats command works and shows system information."""
        exit_code, stdout, stderr = self.run_cli_command(["stats"])

        assert exit_code == 0, f"Stats command failed with stderr: {stderr}"
        assert "System Statistics" in stdout, "Stats output missing header"
        assert "Storage:" in stdout, "Stats output missing storage section"
        assert "Provider Reliability:" in stdout, "Stats output missing reliability section"

    def test_clean_help_message(self):
        """Test that clean command without parameters shows help message."""
        exit_code, stdout, stderr = self.run_cli_command(["clean"])

        assert exit_code == 0, f"Clean command failed with stderr: {stderr}"
        assert "Specify what to clean" in stdout, "Clean output missing help message"
        assert "--temp" in stdout, "Clean help missing temp option"
        assert "--configs" in stdout, "Clean help missing configs option"

    def test_invalid_command_handling(self):
        """Test that invalid commands are handled gracefully."""
        exit_code, stdout, stderr = self.run_cli_command(["nonexistent_command"])

        # Should fail but not crash catastrophically
        assert exit_code != 0, "Invalid command should return non-zero exit code"

    def test_providers_show_nonexistent(self):
        """Test provider show command with non-existent provider."""
        exit_code, stdout, stderr = self.run_cli_command(["providers", "--action=show", "--name=nonexistent_provider"])

        assert exit_code == 0, "Provider show should handle missing providers gracefully"
        assert "not found" in stdout, "Should show not found message"

    def test_validate_command(self):
        """Test that validate command works and shows validation results."""
        exit_code, stdout, stderr = self.run_cli_command(["validate", "--providers=openai"])

        assert exit_code == 0, f"Validate command failed with stderr: {stderr}"
        assert "Validating" in stdout, "Validate output missing validation header"
        assert "openai" in stdout, "Validate output missing provider name"
        assert any(word in stdout for word in ["Valid", "INVALID", "warnings"]), "Validate output missing status"
        assert "Validation Summary" in stdout, "Validate output missing summary"

    def test_validate_all_providers_command(self):
        """Test that validate command works with all providers."""
        exit_code, stdout, stderr = self.run_cli_command(["validate", "--providers=all"])

        assert exit_code == 0, f"Validate all command failed with stderr: {stderr}"
        assert "Validating" in stdout, "Validate output missing validation header"
        assert "Validation Summary" in stdout, "Validate output missing summary"
        assert "Total providers:" in stdout, "Validate output missing total count"
        assert "Success rate:" in stdout, "Validate output missing success rate"


class TestCLIIntegration:
    """Integration tests for CLI workflow scenarios."""

    @staticmethod
    def run_cli_command(command_args: list[str]) -> tuple[int, str, str]:
        """Run a CLI command and return exit code, stdout, stderr."""
        cmd = [sys.executable, "-m", "vexy_co_model_catalog", *command_args]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr

    def test_command_chaining_workflow(self):
        """Test a typical workflow: version -> providers -> stats."""
        # Test version
        exit_code, stdout, stderr = self.run_cli_command(["version"])
        assert exit_code == 0, "Version command should work"

        # Test providers
        exit_code, stdout, stderr = self.run_cli_command(["providers"])
        assert exit_code == 0, "Providers command should work"

        # Test stats
        exit_code, stdout, stderr = self.run_cli_command(["stats"])
        assert exit_code == 0, "Stats command should work"

    def test_help_then_specific_command(self):
        """Test help command followed by using the suggested commands."""
        # Get help
        exit_code, stdout, stderr = self.run_cli_command(["help"])
        assert exit_code == 0, "Help command should work"
        assert "python -m vexy_co_model_catalog providers" in stdout

        # Use suggested command
        exit_code, stdout, stderr = self.run_cli_command(["providers"])
        assert exit_code == 0, "Suggested providers command should work"


def test_cli_module_importable():
    """Test that the CLI module can be imported without errors."""
    try:
        from vexy_co_model_catalog.cli import CLI

        assert CLI is not None, "CLI class should be importable"
    except ImportError as e:
        msg = f"CLI module import failed: {e}"
        raise AssertionError(msg)


def test_cli_fire_integration():
    """Test that Fire integration works correctly."""
    try:
        from vexy_co_model_catalog.cli import main

        assert callable(main), "Main function should be callable"
    except ImportError as e:
        msg = f"Main function import failed: {e}"
        raise AssertionError(msg)
