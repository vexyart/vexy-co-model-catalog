#!/usr/bin/env python3
"""
this_file: tests/test_storage_unit.py

Comprehensive unit tests for the StorageManager class ensuring file operation reliability.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import tomli_w
import yaml

from vexy_co_model_catalog.core.integrity import IntegrityLevel
from vexy_co_model_catalog.core.storage import StorageError, StorageManager


class TestStorageManager(unittest.TestCase):
    """Comprehensive unit tests for StorageManager file operations."""

    def setUp(self):
        """Set up test fixtures with temporary directory."""
        self.test_dir = tempfile.mkdtemp()
        self.storage = StorageManager(self.test_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_storage_manager_initialization(self):
        """Test StorageManager initialization with different root paths."""
        # Test with explicit path
        storage = StorageManager(self.test_dir)
        assert storage.root == Path(self.test_dir).resolve()

        # Test with default path (current working directory)
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path("/test/cwd")
            storage_default = StorageManager()
            assert storage_default.root == Path("/test/cwd")

        # Test directory structure setup
        expected_dirs = [
            "json_dir",
            "text_dir",
            "extra_dir",
            "config_dir",
            "config_json_dir",
            "config_txt_dir",
            "config_aichat_dir",
            "config_codex_dir",
            "config_mods_dir",
        ]
        for dir_attr in expected_dirs:
            assert hasattr(storage, dir_attr)
            assert isinstance(getattr(storage, dir_attr), Path)

    def test_ensure_directory(self):
        """Test directory creation functionality."""
        test_dir = Path(self.test_dir) / "test_subdir" / "nested"
        assert not test_dir.exists()

        self.storage._ensure_directory(test_dir)
        assert test_dir.exists()
        assert test_dir.is_dir()

        # Test idempotent behavior
        self.storage._ensure_directory(test_dir)  # Should not raise
        assert test_dir.exists()

    def test_write_json(self):
        """Test JSON file writing with various options."""
        test_data = {
            "models": [{"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}, {"id": "gpt-4", "name": "GPT-4"}],
            "count": 2,
        }

        # Test basic JSON writing
        self.storage.write_json("test_models", test_data)

        json_file = self.storage.json_dir / "test_models.json"
        assert json_file.exists()

        # Verify content
        with open(json_file, encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

        # Test with different formatting options
        self.storage.write_json("test_compact", test_data, indent=None, sort_keys=True)
        compact_file = self.storage.json_dir / "test_compact.json"
        assert compact_file.exists()

        # Test with ensure_ascii=True
        unicode_data = {"message": "Hello, 世界! 🌍"}
        self.storage.write_json("test_unicode", unicode_data, ensure_ascii=True)
        unicode_file = self.storage.json_dir / "test_unicode.json"
        assert unicode_file.exists()

    def test_write_config_json(self):
        """Test config JSON file writing."""
        test_data = {"providers": ["openai", "anthropic"], "version": "1.0"}

        self.storage.write_config_json("test_config", test_data)

        config_file = self.storage.config_json_dir / "test_config.json"
        assert config_file.exists()
        assert self.storage.config_json_dir.exists()

        # Verify content
        with open(config_file, encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    def test_write_text(self):
        """Test text file writing."""
        test_lines = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"]

        self.storage.write_text("test_models", test_lines)

        text_file = self.storage.text_dir / "test_models.txt"
        assert text_file.exists()

        # Verify content
        content = text_file.read_text(encoding="utf-8")
        expected = "gpt-3.5-turbo\ngpt-4\nclaude-3-sonnet\n"
        assert content == expected

        # Test empty lines
        self.storage.write_text("test_empty", [])
        empty_file = self.storage.text_dir / "test_empty.txt"
        assert empty_file.exists()
        assert empty_file.read_text(encoding="utf-8") == "\n"

    def test_write_config_txt(self):
        """Test config text file writing."""
        test_lines = ["model-1", "model-2", "model-3"]

        self.storage.write_config_txt("test_config_models", test_lines)

        config_file = self.storage.config_txt_dir / "test_config_models.txt"
        assert config_file.exists()
        assert self.storage.config_txt_dir.exists()

        content = config_file.read_text(encoding="utf-8")
        expected = "model-1\nmodel-2\nmodel-3\n"
        assert content == expected

    def test_write_yaml(self):
        """Test YAML file writing to different directories."""
        test_data = {"models": {"openai": ["gpt-3.5-turbo", "gpt-4"]}, "config": {"api_key": "test-key"}}

        # Test config directory
        self.storage.write_yaml("test_config", test_data, directory="config")
        config_file = self.storage.config_dir / "test_config.yaml"
        assert config_file.exists()

        # Test aichat directory
        self.storage.write_yaml("aichat_config", test_data, directory="aichat")
        aichat_file = self.storage.config_aichat_dir / "aichat_config.yaml"
        assert aichat_file.exists()

        # Test mods directory (note: .yml extension)
        self.storage.write_yaml("mods_config", test_data, directory="mods")
        mods_file = self.storage.config_mods_dir / "mods_config.yml"
        assert mods_file.exists()

        # Test invalid directory
        with pytest.raises(ValueError, match="Invalid directory for YAML: invalid"):
            self.storage.write_yaml("test", test_data, directory="invalid")

        # Verify YAML content
        with open(config_file, encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data == test_data

    def test_write_toml(self):
        """Test TOML file writing."""
        test_data = {
            "provider": "openai",
            "models": ["gpt-3.5-turbo", "gpt-4"],
            "config": {"max_tokens": 4096, "temperature": 0.7},
        }

        self.storage.write_toml("test_config", test_data)

        toml_file = self.storage.json_dir / "test_config.toml"  # Note: writes to json_dir
        assert toml_file.exists()

        # Verify content by parsing
        content = toml_file.read_text(encoding="utf-8")
        assert 'provider = "openai"' in content
        assert "models = [" in content

    def test_write_config_toml(self):
        """Test config TOML file writing."""
        test_data = {"model": "gpt-4", "temperature": 0.7, "profiles": {"default": {"model": "gpt-3.5-turbo"}}}

        # Test codex directory
        self.storage.write_config_toml("test_codex", test_data, directory="codex")
        codex_file = self.storage.config_codex_dir / "test_codex.toml"
        assert codex_file.exists()
        assert self.storage.config_codex_dir.exists()

        # Test invalid directory
        with pytest.raises(ValueError, match="Invalid directory for TOML: invalid"):
            self.storage.write_config_toml("test", test_data, directory="invalid")

    def test_write_extra(self):
        """Test extra file writing."""
        test_data = {"stats": {"total_models": 50, "failed_providers": 2}}

        self.storage.write_extra("test_stats", test_data)

        extra_file = self.storage.extra_dir / "test_stats.json"
        assert extra_file.exists()
        assert self.storage.extra_dir.exists()

        # Verify content
        with open(extra_file, encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    def test_read_json(self):
        """Test JSON file reading from different directories."""
        test_data = {"test": "data", "numbers": [1, 2, 3]}

        # Write test files
        self.storage.write_json("test_read", test_data)
        self.storage.write_config_json("config_read", test_data)
        self.storage.write_extra("extra_read", test_data)

        # Test reading from json directory
        read_data = self.storage.read_json("test_read", directory="json")
        assert read_data == test_data

        # Test reading from config_json directory
        config_data = self.storage.read_json("config_read", directory="config_json")
        assert config_data == test_data

        # Test reading from extra directory
        extra_data = self.storage.read_json("extra_read", directory="extra")
        assert extra_data == test_data

        # Test reading non-existent file
        assert self.storage.read_json("nonexistent", directory="json") is None

        # Test invalid directory
        with pytest.raises(ValueError, match="Invalid directory: invalid"):
            self.storage.read_json("test", directory="invalid")

    def test_read_yaml(self):
        """Test YAML file reading from different directories."""
        test_data = {"config": {"key": "value"}, "list": ["a", "b", "c"]}

        # Write test files
        self.storage.write_yaml("test_config", test_data, directory="config")
        self.storage.write_yaml("aichat_test", test_data, directory="aichat")
        self.storage.write_yaml("mods_test", test_data, directory="mods")

        # Test reading from different directories
        config_data = self.storage.read_yaml("test_config", directory="config")
        assert config_data == test_data

        aichat_data = self.storage.read_yaml("aichat_test", directory="aichat")
        assert aichat_data == test_data

        mods_data = self.storage.read_yaml("mods_test", directory="mods")
        assert mods_data == test_data

        # Test reading non-existent file
        assert self.storage.read_yaml("nonexistent", directory="config") is None

        # Test invalid directory
        with pytest.raises(ValueError, match="Invalid directory for YAML: invalid"):
            self.storage.read_yaml("test", directory="invalid")

    def test_read_toml(self):
        """Test TOML file reading."""
        test_data = {"model": "gpt-4", "config": {"temperature": 0.7, "max_tokens": 4096}}

        # Write test file
        self.storage.write_config_toml("test_toml", test_data, directory="codex")

        # Test reading
        read_data = self.storage.read_toml("test_toml", directory="codex")
        assert read_data == test_data

        # Test reading non-existent file
        assert self.storage.read_toml("nonexistent", directory="codex") is None

        # Test invalid directory
        with pytest.raises(ValueError, match="Invalid directory for TOML: invalid"):
            self.storage.read_toml("test", directory="invalid")

    def test_list_files(self):
        """Test file listing functionality."""
        # Create test files
        self.storage.write_json("model1", {"test": "data"})
        self.storage.write_json("model2", {"test": "data"})
        self.storage.write_text("list1", ["line1", "line2"])
        self.storage.write_config_json("config1", {"config": "value"})

        # Test listing JSON files
        json_files = self.storage.list_files("json")
        assert len(json_files) == 2
        json_names = [f.name for f in json_files]
        assert "model1.json" in json_names
        assert "model2.json" in json_names

        # Test listing text files
        text_files = self.storage.list_files("text")
        assert len(text_files) == 1
        assert text_files[0].name == "list1.txt"

        # Test listing config JSON files
        config_files = self.storage.list_files("config_json")
        assert len(config_files) == 1
        assert config_files[0].name == "config1.json"

        # Test pattern matching
        json_files_pattern = self.storage.list_files("json", "model*.json")
        assert len(json_files_pattern) == 2

        # Test empty directory
        extra_files = self.storage.list_files("extra")
        assert len(extra_files) == 0

        # Test invalid directory
        with pytest.raises(ValueError, match="Invalid directory: invalid"):
            self.storage.list_files("invalid")

    def test_get_file_stats(self):
        """Test file statistics collection."""
        # Initially no files
        stats = self.storage.get_file_stats()
        for key in stats:
            assert stats[key] == 0

        # Create some files
        self.storage.write_json("test1", {"data": "test"})
        self.storage.write_json("test2", {"data": "test"})
        self.storage.write_text("test1", ["line1"])
        self.storage.write_config_json("config1", {"config": "test"})
        self.storage.write_extra("extra1", {"extra": "test"})

        # Check updated stats
        updated_stats = self.storage.get_file_stats()
        assert updated_stats["json_files"] == 2
        assert updated_stats["text_files"] == 1
        assert updated_stats["config_json_files"] == 1
        assert updated_stats["extra_files"] == 1
        assert updated_stats["config_txt_files"] == 0  # No config txt files created

    def test_cleanup_temp_files(self):
        """Test temporary file cleanup."""
        # Create some temporary files manually
        temp_files = [
            self.storage.json_dir / ".test1.json.tmp",
            self.storage.config_json_dir / ".config1.json.tmp",
            self.storage.text_dir / ".text1.txt.tmp",
        ]

        # Ensure directories exist
        for temp_file in temp_files:
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text("temp content")
            assert temp_file.exists()

        # Cleanup
        self.storage.cleanup_temp_files()

        # Verify cleanup
        for temp_file in temp_files:
            assert not temp_file.exists()

    @patch("vexy_co_model_catalog.core.storage.get_integrity_manager")
    def test_write_critical_json(self, mock_get_integrity):
        """Test critical JSON writing with integrity protection."""
        mock_integrity = Mock()
        mock_get_integrity.return_value = mock_integrity

        test_data = {"critical": "data", "important": True}

        self.storage.write_critical_json("critical_config", test_data)

        # Verify file was written
        critical_file = self.storage.json_dir / "critical_config.json"
        assert critical_file.exists()

        # Verify integrity tracking was called
        mock_integrity.add_file_tracking.assert_called_once()

    @patch("vexy_co_model_catalog.core.storage.get_integrity_manager")
    def test_write_important_config(self, mock_get_integrity):
        """Test important config writing with integrity protection."""
        mock_integrity = Mock()
        mock_get_integrity.return_value = mock_integrity

        test_data = {"important": "config", "version": "1.0"}

        # Test JSON config
        self.storage.write_important_config("important_json", test_data, config_type="json")
        json_file = self.storage.config_json_dir / "important_json.json"
        assert json_file.exists()

        # Test YAML config
        self.storage.write_important_config("important_yaml", test_data, config_type="yaml")
        yaml_file = self.storage.config_dir / "important_yaml.yaml"
        assert yaml_file.exists()

        # Test TOML config
        self.storage.write_important_config("important_toml", test_data, config_type="toml")
        toml_file = self.storage.config_codex_dir / "important_toml.toml"
        assert toml_file.exists()

        # Test invalid config type
        with pytest.raises(ValueError, match="Unsupported config type: invalid"):
            self.storage.write_important_config("test", test_data, config_type="invalid")

    @patch("vexy_co_model_catalog.core.storage.get_integrity_manager")
    def test_verify_file_integrity(self, mock_get_integrity):
        """Test file integrity verification."""
        mock_integrity = Mock()
        mock_integrity.verify_file_integrity.return_value = True
        mock_get_integrity.return_value = mock_integrity

        # Create test file
        self.storage.write_json("test_verify", {"test": "data"})

        # Test verification
        result = self.storage.verify_file_integrity("test_verify", directory="json")
        assert result

        mock_integrity.verify_file_integrity.assert_called_once()

    @patch("vexy_co_model_catalog.core.storage.get_integrity_manager")
    def test_get_integrity_report(self, mock_get_integrity):
        """Test integrity report generation."""
        mock_integrity = Mock()
        mock_report = {"total_files": 5, "verified": 4, "corrupted": 1}
        mock_integrity.get_integrity_report.return_value = mock_report
        mock_get_integrity.return_value = mock_integrity

        report = self.storage.get_integrity_report()
        assert report == mock_report

        mock_integrity.get_integrity_report.assert_called_once()

    @patch("vexy_co_model_catalog.core.storage.get_integrity_manager")
    def test_cleanup_old_backups(self, mock_get_integrity):
        """Test old backup cleanup."""
        mock_integrity = Mock()
        mock_integrity.cleanup_old_backups.return_value = 3
        mock_get_integrity.return_value = mock_integrity

        result = self.storage.cleanup_old_backups(retention_days=7)
        assert result == 3

        mock_integrity.cleanup_old_backups.assert_called_once_with(7)

    def test_atomic_write_error_handling(self):
        """Test atomic write error handling."""
        # Test write to invalid location
        invalid_storage = StorageManager("/nonexistent/path/that/cannot/be/created")

        with patch("tempfile.mkstemp", side_effect=OSError("Permission denied")), pytest.raises(StorageError):
            invalid_storage.write_json("test", {"data": "test"})

    def test_read_file_error_handling(self):
        """Test file reading error handling."""
        # Create a file and then make it unreadable
        self.storage.write_json("test_error", {"test": "data"})
        self.storage.json_dir / "test_error.json"

        # Mock file read error
        with patch("pathlib.Path.read_text", side_effect=OSError("Permission denied")):
            with pytest.raises(StorageError):
                self.storage.read_json("test_error", directory="json")

    def test_repr(self):
        """Test string representation of StorageManager."""
        # Create some files
        self.storage.write_json("test1", {"data": "test"})
        self.storage.write_config_json("config1", {"config": "test"})

        repr_str = repr(self.storage)
        assert "StorageManager" in repr_str
        assert f"root={self.storage.root}" in repr_str
        assert "legacy_files=" in repr_str
        assert "config_files=" in repr_str

    def test_directory_path_consistency(self):
        """Test that directory paths are consistent and correct."""
        # Test legacy directories
        assert self.storage.json_dir == self.storage.root / "models" / "json"
        assert self.storage.text_dir == self.storage.root / "models" / "text"
        assert self.storage.extra_dir == self.storage.root / "models" / "extra"

        # Test config directories
        assert self.storage.config_dir == self.storage.root / "config"
        assert self.storage.config_json_dir == self.storage.root / "config" / "json"
        assert self.storage.config_txt_dir == self.storage.root / "config" / "txt"
        assert self.storage.config_aichat_dir == self.storage.root / "config" / "aichat"
        assert self.storage.config_codex_dir == self.storage.root / "config" / "codex"
        assert self.storage.config_mods_dir == self.storage.root / "config" / "mods"


if __name__ == "__main__":
    unittest.main()
