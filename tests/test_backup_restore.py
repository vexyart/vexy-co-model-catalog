"""
this_file: tests/test_backup_restore.py

Comprehensive tests for config backup and restore functionality.
Tests backup creation, restoration, conflict handling, and data integrity.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from vexy_co_model_catalog.core.config import ConfigManager
from vexy_co_model_catalog.core.storage import StorageManager


class TestConfigBackupRestore:
    """Test backup and restore functionality for config files."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory with tool subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create tool subdirectories
            for tool in ["aichat", "codex", "mods"]:
                (config_dir / tool).mkdir()

            # Create backup directory
            (config_dir / "backups").mkdir()

            yield config_dir

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create ConfigManager with temporary directory."""
        return ConfigManager(temp_config_dir)

    @pytest.fixture
    def sample_aichat_config(self):
        """Create sample aichat config data."""
        return {
            "default_model": "gpt-4",
            "clients": [
                {"openai": {"type": "openai", "api_base": "https://api.openai.com/v1", "api_key": "${OPENAI_API_KEY}"}},
                {"anthropic": {"type": "claude", "api_key": "${ANTHROPIC_API_KEY}"}},
            ],
        }

    @pytest.fixture
    def sample_codex_config(self):
        """Create sample codex config data."""
        return {
            "profiles": {
                "openai-gpt-4": {
                    "name": "openai-gpt-4",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4",
                    "max_tokens": 4096,
                    "context_window": 8192,
                },
                "anthropic-claude": {
                    "name": "anthropic-claude",
                    "base_url": "https://api.anthropic.com/v1",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 4096,
                    "context_window": 200000,
                },
            }
        }

    def test_aichat_config_backup_creation(self, config_manager, temp_config_dir, sample_aichat_config):
        """Test creating backup of aichat config file."""
        # Create initial config file
        aichat_config_path = temp_config_dir / "aichat" / "config.yaml"
        with aichat_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(sample_aichat_config, f, default_flow_style=False, indent=2)

        # Create backup
        backup_path = config_manager.backup_config("aichat")

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.parent == temp_config_dir / "backups"
        assert "aichat" in backup_path.name
        assert backup_path.name.endswith(".yaml")

        # Verify backup contains original data
        with backup_path.open("r", encoding="utf-8") as f:
            backup_data = yaml.safe_load(f)

        assert backup_data == sample_aichat_config
        assert backup_data["default_model"] == "gpt-4"
        assert len(backup_data["clients"]) == 2

    def test_codex_config_backup_creation(self, config_manager, temp_config_dir, sample_codex_config):
        """Test creating backup of codex config file."""
        # Create initial config file
        import tomli_w

        codex_config_path = temp_config_dir / "codex" / "config.toml"
        with codex_config_path.open("wb") as f:
            tomli_w.dump(sample_codex_config, f)

        # Create backup
        backup_path = config_manager.backup_config("codex")

        assert backup_path is not None
        assert backup_path.exists()
        assert "codex" in backup_path.name
        assert backup_path.name.endswith(".toml")

        # Verify backup contains original data
        try:
            import tomllib

            toml_loads = tomllib.loads
        except ImportError:
            import tomli

            toml_loads = tomli.loads

        with backup_path.open("r", encoding="utf-8") as f:
            backup_data = toml_loads(f.read())

        assert backup_data == sample_codex_config
        assert "profiles" in backup_data
        assert len(backup_data["profiles"]) == 2

    def test_backup_nonexistent_config(self, config_manager, temp_config_dir):
        """Test backup of non-existent config file."""
        # Try to backup non-existent file
        backup_path = config_manager.backup_config("aichat")

        # Should return None for non-existent files
        assert backup_path is None

    def test_backup_filename_uniqueness(self, config_manager, temp_config_dir, sample_aichat_config):
        """Test that backup filenames are unique and timestamped."""
        # Create initial config file
        aichat_config_path = temp_config_dir / "aichat" / "config.yaml"
        with aichat_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(sample_aichat_config, f)

        # Create first backup
        backup1_path = config_manager.backup_config("aichat")

        # Wait a moment to ensure different timestamps
        time.sleep(0.1)

        # Create second backup
        backup2_path = config_manager.backup_config("aichat")

        assert backup1_path != backup2_path
        assert backup1_path.exists()
        assert backup2_path.exists()

        # Both should contain the same data
        with backup1_path.open("r", encoding="utf-8") as f:
            data1 = yaml.safe_load(f)
        with backup2_path.open("r", encoding="utf-8") as f:
            data2 = yaml.safe_load(f)

        assert data1 == data2 == sample_aichat_config

    def test_backup_directory_creation(self, temp_config_dir):
        """Test that backup directory is created if it doesn't exist."""
        # Remove backup directory
        backup_dir = temp_config_dir / "backups"
        if backup_dir.exists():
            import shutil

            shutil.rmtree(backup_dir)

        config_manager = ConfigManager(temp_config_dir)

        # Create config file
        sample_config = {"test": "data"}
        aichat_config_path = temp_config_dir / "aichat" / "config.yaml"
        with aichat_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(sample_config, f)

        # Create backup (should create backup directory)
        backup_path = config_manager.backup_config("aichat")

        assert backup_dir.exists()
        assert backup_path is not None
        assert backup_path.exists()


class TestConfigRestoreValidation:
    """Test config restore functionality and data integrity validation."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            for tool in ["aichat", "codex", "mods"]:
                (config_dir / tool).mkdir()
            (config_dir / "backups").mkdir()
            yield config_dir

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create ConfigManager with temporary directory."""
        return ConfigManager(temp_config_dir)

    def test_config_data_integrity_after_backup_restore_cycle(self, config_manager, temp_config_dir):
        """Test complete backup and restore cycle maintains data integrity."""
        original_config = {
            "default_model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4096,
            "clients": [
                {
                    "openai": {
                        "type": "openai",
                        "api_base": "https://api.openai.com/v1",
                        "api_key": "${OPENAI_API_KEY}",
                        "models": ["gpt-4", "gpt-3.5-turbo"],
                    }
                }
            ],
            "custom_settings": {"advanced": True, "retry_count": 3, "timeout": 30.0},
        }

        # Create initial config
        aichat_config_path = temp_config_dir / "aichat" / "config.yaml"
        with aichat_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(original_config, f, default_flow_style=False, indent=2)

        # Create backup
        backup_path = config_manager.backup_config("aichat")
        assert backup_path is not None

        # Modify original config
        import copy
        modified_config = copy.deepcopy(original_config)
        modified_config["default_model"] = "gpt-3.5-turbo"
        modified_config["temperature"] = 0.9
        modified_config["clients"][0]["openai"]["models"].append("gpt-4-turbo")

        with aichat_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(modified_config, f, default_flow_style=False, indent=2)

        # Verify config was modified
        with aichat_config_path.open("r", encoding="utf-8") as f:
            current_config = yaml.safe_load(f)

        assert current_config["default_model"] == "gpt-3.5-turbo"
        assert current_config["temperature"] == 0.9

        # Restore from backup by copying backup over current config
        import shutil

        shutil.copy2(backup_path, aichat_config_path)

        # Verify restoration
        with aichat_config_path.open("r", encoding="utf-8") as f:
            restored_config = yaml.safe_load(f)

        assert restored_config == original_config
        assert restored_config["default_model"] == "gpt-4"
        assert restored_config["temperature"] == 0.7
        assert len(restored_config["clients"][0]["openai"]["models"]) == 2

    def test_backup_with_complex_nested_structures(self, config_manager, temp_config_dir):
        """Test backup handles complex nested config structures correctly."""
        complex_config = {
            "global_settings": {
                "debug": True,
                "log_level": "info",
                "cache": {"enabled": True, "ttl": 3600, "backends": ["memory", "disk"]},
            },
            "providers": {
                "openai": {
                    "credentials": {"api_key": "${OPENAI_API_KEY}", "organization": "${OPENAI_ORG}"},
                    "models": {
                        "gpt-4": {
                            "max_tokens": 4096,
                            "context_window": 8192,
                            "pricing": {"input_per_1k": 0.03, "output_per_1k": 0.06},
                            "capabilities": ["text", "function_calling"],
                        },
                        "gpt-3.5-turbo": {
                            "max_tokens": 4096,
                            "context_window": 4096,
                            "pricing": {"input_per_1k": 0.001, "output_per_1k": 0.002},
                            "capabilities": ["text", "function_calling"],
                        },
                    },
                }
            },
        }

        # Create config file
        config_path = temp_config_dir / "aichat" / "config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(complex_config, f, default_flow_style=False, indent=2)

        # Create backup
        backup_path = config_manager.backup_config("aichat")
        assert backup_path is not None

        # Verify backup preserves complex structure
        with backup_path.open("r", encoding="utf-8") as f:
            backup_data = yaml.safe_load(f)

        assert backup_data == complex_config

        # Test deep nested access
        assert backup_data["global_settings"]["cache"]["backends"] == ["memory", "disk"]
        assert backup_data["providers"]["openai"]["models"]["gpt-4"]["pricing"]["input_per_1k"] == 0.03
        assert "function_calling" in backup_data["providers"]["openai"]["models"]["gpt-3.5-turbo"]["capabilities"]

    def test_backup_handles_special_yaml_types(self, config_manager, temp_config_dir):
        """Test backup correctly handles special YAML data types."""
        special_config = {
            "string_value": "test string",
            "integer_value": 42,
            "float_value": 3.14159,
            "boolean_true": True,
            "boolean_false": False,
            "null_value": None,
            "list_value": [1, 2, 3, "four", 5.0],
            "nested_dict": {"inner_list": ["a", "b", {"nested": "value"}], "mixed_types": [1, True, None, "string"]},
            "multiline_string": "This is a\nmultiline string\nwith line breaks",
            "special_chars": "String with special chars: !@#$%^&*(){}[]|\\:;\"'<>,.?/~`",
        }

        # Create config file
        config_path = temp_config_dir / "aichat" / "config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(special_config, f, default_flow_style=False, indent=2)

        # Create backup
        backup_path = config_manager.backup_config("aichat")
        assert backup_path is not None

        # Verify backup preserves all data types
        with backup_path.open("r", encoding="utf-8") as f:
            backup_data = yaml.safe_load(f)

        assert backup_data == special_config
        assert isinstance(backup_data["integer_value"], int)
        assert isinstance(backup_data["float_value"], float)
        assert isinstance(backup_data["boolean_true"], bool)
        assert backup_data["boolean_true"] is True
        assert backup_data["null_value"] is None
        assert "\n" in backup_data["multiline_string"]
        assert backup_data["special_chars"].endswith("~`")


class TestBackupErrorHandling:
    """Test backup functionality error handling and edge cases."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            for tool in ["aichat", "codex", "mods"]:
                (config_dir / tool).mkdir()
            yield config_dir

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create ConfigManager with temporary directory."""
        return ConfigManager(temp_config_dir)

    def test_backup_invalid_tool_name(self, config_manager):
        """Test backup with invalid tool name."""
        with pytest.raises(ValueError, match="Unsupported tool: invalid_tool"):
            config_manager.backup_config("invalid_tool")

    def test_backup_permission_error_handling(self, config_manager, temp_config_dir):
        """Test backup handles permission errors gracefully."""
        # Create config file
        sample_config = {"test": "data"}
        config_path = temp_config_dir / "aichat" / "config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(sample_config, f)

        # Mock permission error during backup
        with patch("shutil.copy2") as mock_copy:
            mock_copy.side_effect = PermissionError("Permission denied")

            backup_path = config_manager.backup_config("aichat")

            # Should handle permission error gracefully
            assert backup_path is None

    def test_backup_with_corrupted_config_file(self, config_manager, temp_config_dir):
        """Test backup with corrupted/invalid YAML config."""
        # Create corrupted config file
        config_path = temp_config_dir / "aichat" / "config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            f.write("invalid: yaml: content: [\nunclosed bracket")

        # Backup should still work (copy file as-is)
        backup_path = config_manager.backup_config("aichat")

        assert backup_path is not None
        assert backup_path.exists()

        # Backup should contain the corrupted content
        with backup_path.open("r", encoding="utf-8") as f:
            backup_content = f.read()

        assert "unclosed bracket" in backup_content

    def test_backup_large_config_file(self, config_manager, temp_config_dir):
        """Test backup with large config file."""
        # Create large config with many entries
        large_config = {"models": {}}

        # Add 1000 model entries
        for i in range(1000):
            model_name = f"model-{i:04d}"
            large_config["models"][model_name] = {
                "max_tokens": 4096 + i,
                "context_window": 8192 + i,
                "description": f"Test model number {i} with description",
                "capabilities": ["text", "function_calling", "streaming"],
                "metadata": {
                    "created": f"2024-01-{(i % 30) + 1:02d}",
                    "provider": f"provider-{i % 10}",
                    "version": f"1.{i % 100}.0",
                },
            }

        # Write large config
        config_path = temp_config_dir / "aichat" / "config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(large_config, f, default_flow_style=False, indent=2)

        # Create backup
        backup_path = config_manager.backup_config("aichat")

        assert backup_path is not None
        assert backup_path.exists()

        # Verify backup file size is reasonable
        backup_size = backup_path.stat().st_size
        original_size = config_path.stat().st_size

        # Backup should be similar size to original
        assert abs(backup_size - original_size) < original_size * 0.1  # Within 10%

        # Verify backup contains correct data (sample check)
        with backup_path.open("r", encoding="utf-8") as f:
            backup_data = yaml.safe_load(f)

        assert len(backup_data["models"]) == 1000
        assert backup_data["models"]["model-0500"]["max_tokens"] == 4596


if __name__ == "__main__":
    pytest.main([__file__])
