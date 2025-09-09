#!/usr/bin/env python3
# this_file: scripts/test_package_install.py

"""
Package Installation Integrity Test Script

Tests the built wheel package to ensure:
1. Clean installation in isolated environment
2. All CLI entry points work correctly
3. Core functionality is accessible
4. Import paths are correct
5. Dependencies are properly resolved
"""

import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run command and return exit code, stdout, stderr."""
    result = subprocess.run(
        cmd,
        check=False, cwd=cwd,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def test_package_installation():
    """Test package installation in clean virtual environment."""

    project_root = Path(__file__).parent.parent

    # Find the built wheel
    dist_dir = project_root / "dist"
    wheel_files = list(dist_dir.glob("*.whl"))

    if not wheel_files:
        return False

    wheel_file = max(wheel_files, key=lambda p: p.stat().st_mtime)

    with tempfile.TemporaryDirectory(prefix="vmc_test_") as temp_dir:
        temp_path = Path(temp_dir)
        venv_path = temp_path / "test_venv"

        # Create virtual environment
        venv.create(venv_path, with_pip=True)

        # Get python and pip paths
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"

        # Install the wheel
        exit_code, stdout, stderr = run_command([
            str(pip_exe), "install", str(wheel_file)
        ])

        if exit_code != 0:
            return False


        # Test CLI entry points

        # Test main entry point
        exit_code, stdout, stderr = run_command([
            str(python_exe), "-m", "vexy_co_model_catalog", "version"
        ])

        if exit_code == 0:
            pass
        else:
            return False

        # Test script entry points
        if sys.platform != "win32":
            script_path = venv_path / "bin" / "vexy-model-catalog"
        else:
            script_path = venv_path / "Scripts" / "vexy-model-catalog.exe"

        if script_path.exists():
            exit_code, stdout, stderr = run_command([str(script_path), "version"])
            if exit_code == 0:
                pass
            else:
                return False

        # Test short alias
        if sys.platform != "win32":
            alias_path = venv_path / "bin" / "vmc"
        else:
            alias_path = venv_path / "Scripts" / "vmc.exe"

        if alias_path.exists():
            exit_code, stdout, stderr = run_command([str(alias_path), "version"])
            if exit_code == 0:
                pass
            else:
                return False

        # Test core imports

        import_tests = [
            "import vexy_co_model_catalog",
            "from vexy_co_model_catalog import ModelCatalog",
            "from vexy_co_model_catalog.core.provider import ProviderConfig",
            "from vexy_co_model_catalog.core.fetcher import ModelFetcher",
            "from vexy_co_model_catalog.core.storage import StorageManager",
        ]

        for test in import_tests:
            exit_code, stdout, stderr = run_command([
                str(python_exe), "-c", test
            ])

            if exit_code == 0:
                pass
            else:
                return False

        # Test basic functionality

        functionality_test = """
import sys
from vexy_co_model_catalog import ModelCatalog
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as temp_dir:
    catalog = ModelCatalog(storage_root=Path(temp_dir))
    providers = catalog.list_providers()
    print(f"Found {len(providers)} providers")
    if len(providers) >= 40:
        print("✅ Provider count looks correct")
        sys.exit(0)
    else:
        print("❌ Provider count too low")
        sys.exit(1)
"""

        exit_code, stdout, stderr = run_command([
            str(python_exe), "-c", functionality_test
        ])

        if exit_code == 0:
            pass
        else:
            return False

    return True


if __name__ == "__main__":
    success = test_package_installation()
    sys.exit(0 if success else 1)
