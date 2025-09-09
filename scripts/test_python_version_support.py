#!/usr/bin/env python3
# this_file: scripts/test_python_version_support.py

"""
Python Version Support Verification Script

Verifies that the package works correctly across different Python versions:
1. Version-specific import handling (tomllib vs tomli)
2. Type hint compatibility (union types, generics)
3. Feature compatibility across Python 3.10-3.12
4. Version-specific code paths
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


def test_version_specific_imports():
    """Test that version-specific imports work correctly."""

    # Test TOML reading compatibility

    try:
        if sys.version_info >= (3, 11):
            import tomllib as toml_read
        else:
            import tomli as toml_read

        # Test basic TOML parsing
        test_toml = 'key = "value"'
        parsed = toml_read.loads(test_toml)
        if parsed == {"key": "value"}:
            pass
        else:
            return False

    except ImportError:
        return False

    # Test TOML writing
    try:
        import tomli_w as toml_write
        test_data = {"test": "value"}
        result = toml_write.dumps(test_data)
        if 'test = "value"' in result:
            pass
        else:
            return False
    except ImportError:
        return False

    return True


def test_type_hints_compatibility():
    """Test type hints compatibility across Python versions."""

    # Test union types (Python 3.10+)
    try:
        # This should work on Python 3.10+
        def test_union_types(value: str | int) -> str:
            return str(value)

        result = test_union_types("test")
        if result == "test":
            pass
        else:
            return False

    except SyntaxError:
        return False
    except Exception:
        return False

    # Test generic types
    try:
        def test_generics(items: list[str]) -> dict[str, Any]:
            return {item: len(item) for item in items}

        result = test_generics(["test", "hello"])
        if result == {"test": 4, "hello": 5}:
            pass
        else:
            return False

    except Exception:
        return False

    # Test Optional and Union from typing module (backward compatibility)
    try:
        def test_typing_union(value: str | int | None) -> str:
            return str(value) if value is not None else "None"

        result = test_typing_union(42)
        if result == "42":
            pass
        else:
            return False

    except Exception:
        return False

    return True


def test_pathlib_compatibility():
    """Test pathlib usage across platforms and Python versions."""

    try:
        # Test basic path operations
        test_path = Path("test") / "subdir" / "file.txt"

        # Test path properties


        # Test path resolution
        test_path.resolve()

        return True

    except Exception:
        return False


def test_async_compatibility():
    """Test async/await compatibility if used in the codebase."""

    try:
        async def test_async_function():
            await asyncio.sleep(0.001)  # Very short sleep
            return "async_result"

        # Test running async code
        result = asyncio.run(test_async_function())
        if result == "async_result":
            pass
        else:
            return False

    except Exception:
        return False

    return True


def test_dataclass_compatibility():
    """Test dataclass compatibility across Python versions."""

    try:
        @dataclass
        class TestConfig:
            name: str
            value: int = 0
            options: dict[str, Any] = field(default_factory=dict)

        # Test dataclass creation and usage
        config1 = TestConfig("test")
        config2 = TestConfig("test2", 42, {"key": "value"})

        if (config1.name == "test" and config1.value == 0 and
            config2.name == "test2" and config2.value == 42):
            pass
        else:
            return False

    except Exception:
        return False

    return True


def test_json_compatibility():
    """Test JSON processing compatibility."""

    try:
        test_data = {
            "string": "test",
            "number": 42,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "object": {"nested": "value"}
        }

        # Test serialization
        json_str = json.dumps(test_data, indent=2)

        # Test deserialization
        parsed_data = json.loads(json_str)

        if parsed_data == test_data:
            pass
        else:
            return False

    except Exception:
        return False

    return True


def test_logging_compatibility():
    """Test logging compatibility with loguru."""

    try:
        # Capture log output
        log_stream = StringIO()

        # Add a handler that writes to our string buffer
        handler_id = logger.add(log_stream, format="{message}")

        # Test logging
        logger.info("Test log message")
        logger.debug("Debug message")
        logger.warning("Warning message")

        # Remove the handler
        logger.remove(handler_id)

        # Check output
        log_output = log_stream.getvalue()
        if "Test log message" in log_output:
            pass
        else:
            return False

    except Exception:
        return False

    return True


def main():
    """Run all Python version support tests."""

    tests = [
        ("Version-Specific Imports", test_version_specific_imports),
        ("Type Hints Compatibility", test_type_hints_compatibility),
        ("Pathlib Compatibility", test_pathlib_compatibility),
        ("Async/Await Compatibility", test_async_compatibility),
        ("Dataclass Compatibility", test_dataclass_compatibility),
        ("JSON Compatibility", test_json_compatibility),
        ("Logging Compatibility", test_logging_compatibility),
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
