#!/usr/bin/env python3
# this_file: scripts/test_cross_platform_compatibility.py

"""
Cross-Platform Compatibility Test Script

Tests for compatibility across different platforms and Python versions:
1. Platform-specific code detection
2. Python version compatibility checks
3. Path handling compatibility
4. Import compatibility across platforms
5. Platform-specific dependency issues
"""

import ast
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run command and return exit code, stdout, stderr."""
    result = subprocess.run(
        cmd,
        check=False, capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def test_python_version_compatibility():
    """Test Python version compatibility."""

    current_version = sys.version_info

    # Check minimum version requirement
    min_version = (3, 10)
    if current_version >= min_version:
        pass
    else:
        return False

    return True


def check_syntax_compatibility():
    """Check for Python syntax compatibility across versions."""

    project_root = Path(__file__).parent.parent
    src_files = list((project_root / "src").glob("**/*.py"))

    issues = []

    for file_path in src_files:
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # Try to parse the AST
            ast.parse(content, filename=str(file_path))

            # Check for problematic patterns
            if "match " in content and "case " in content:
                # Basic check for match/case (Python 3.10+)
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if line.strip().startswith('match ') and any('case ' in next_line for next_line in lines[i:i+10]):
                        pass

        except SyntaxError as e:
            issues.append(f"❌ Syntax error in {file_path.relative_to(project_root)}: {e}")

    if issues:
        for _issue in issues:
            pass
        return False
    return True


def test_platform_detection():
    """Test platform detection and compatibility."""

    platform.system()

    # Test path handling cross-platform compatibility
    test_paths = [
        "simple_file.txt",
        "dir/subdir/file.txt",
        "file with spaces.txt",
        "very/deep/nested/path/structure/file.txt"
    ]

    for test_path_str in test_paths:
        try:
            # Test pathlib Path creation and manipulation
            Path(test_path_str)


        except Exception:
            return False

    return True


def test_import_compatibility():
    """Test critical imports work on current platform."""

    critical_imports = [
        ("fire", "CLI framework"),
        ("httpx", "HTTP client"),
        ("loguru", "Logging"),
        ("psutil", "System utilities"),
        ("yaml", "YAML processing (PyYAML)"),
        ("rich", "Terminal formatting"),
        ("tomli", "TOML reading (3.10 compat)"),
        ("tomli_w", "TOML writing"),
    ]

    # Test standard library imports
    stdlib_imports = [
        ("pathlib", "Path handling"),
        ("json", "JSON processing"),
        ("sys", "System interface"),
        ("os", "Operating system interface"),
        ("tempfile", "Temporary files"),
        ("subprocess", "Process management"),
    ]

    all_good = True

    for module_name, _description in critical_imports:
        try:
            __import__(module_name)
        except ImportError:
            all_good = False

    for module_name, _description in stdlib_imports:
        try:
            __import__(module_name)
        except ImportError:
            all_good = False

    return all_good


def check_platform_specific_code():
    """Check for platform-specific code that might cause issues."""

    project_root = Path(__file__).parent.parent
    src_files = list((project_root / "src").glob("**/*.py"))

    # Patterns that might indicate platform-specific code
    problematic_patterns = [
        ("os.system(", "Direct shell execution"),
        ("subprocess.call(", "Direct process call"),
        ("\\\\", "Windows-style path separators"),
        ("/tmp/", "Unix-specific temp directory"),
        ("C:\\", "Windows-specific drive letters"),
        ("$HOME", "Unix-specific environment variable"),
        ("%USERPROFILE%", "Windows-specific environment variable"),
    ]

    issues_found = []

    for file_path in src_files:
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            for pattern, description in problematic_patterns:
                if pattern in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if pattern in line:
                            relative_path = file_path.relative_to(project_root)
                            issues_found.append(f"⚠️  {relative_path}:{i} - {description}: {line.strip()}")

        except Exception:
            pass

    if issues_found:
        for _issue in issues_found:
            pass
    else:
        pass

    return True  # Return True since these are warnings, not errors


def test_python_version_specific_features():
    """Test for Python version-specific features and their compatibility."""

    features_tested = []

    # Test Python 3.10+ features

    # Test union type syntax
    try:
        code = "x: str | int = 'test'"
        ast.parse(code)
        features_tested.append("union_types")
    except SyntaxError:
        pass

    # Test match/case statement
    try:
        code = """
match value:
    case 1:
        pass
    case _:
        pass
"""
        ast.parse(code)
        features_tested.append("match_case")
    except SyntaxError:
        pass

    # Test TOML library compatibility
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli
        features_tested.append("toml_reading")
    except ImportError:
        pass

    return len(features_tested) >= 2  # At least 2 features should work


def main():
    """Run all cross-platform compatibility tests."""

    tests = [
        ("Python Version Compatibility", test_python_version_compatibility),
        ("Syntax Compatibility", check_syntax_compatibility),
        ("Platform Detection", test_platform_detection),
        ("Import Compatibility", test_import_compatibility),
        ("Platform-Specific Code Check", check_platform_specific_code),
        ("Python Version Features", test_python_version_specific_features),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception:
            results.append((test_name, False))

    # Summary

    passed = 0
    total = len(results)

    for test_name, result in results:
        if result:
            passed += 1


    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
