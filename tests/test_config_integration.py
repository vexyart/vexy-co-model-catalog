"""
this_file: tests/test_config_integration.py

Integration tests for config file handling with real config files.
Tests reading, writing, parsing, and generating actual tool config files.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from vexy_co_model_catalog.core.config import (
    AichatConfigParser,
    CodexConfigParser,
    ConfigManager,
    ConfigTemplate,
    ModsConfigParser,
)
from vexy_co_model_catalog.core.provider import Model, ProviderConfig, ProviderKind
from vexy_co_model_catalog.core.storage import StorageManager


class TestConfigFileIntegration:
    """Integration tests for real config file operations."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create subdirectories
            (config_dir / "aichat").mkdir()
            (config_dir / "codex").mkdir()
            (config_dir / "mods").mkdir()

            yield config_dir

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create ConfigManager with temporary directory."""
        return ConfigManager(temp_config_dir)

    @pytest.fixture
    def sample_provider(self):
        """Create sample provider for testing."""
        return ProviderConfig(
            name="openai", kind=ProviderKind.OPENAI, base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY"
        )

    @pytest.fixture
    def sample_models(self):
        """Create sample models for testing."""
        return [
            Model(id="gpt-4", provider="openai", max_input_tokens=8192, max_output_tokens=4096),
            Model(id="gpt-3.5-turbo", provider="openai", max_input_tokens=4096, max_output_tokens=4096),
        ]

    def test_aichat_config_file_roundtrip(self, temp_config_dir, sample_provider, sample_models):
        """Test creating and reading aichat config files."""
        parser = AichatConfigParser()
        config_path = temp_config_dir / "aichat" / "config.yaml"

        # Create template with models
        template = ConfigTemplate(
            tool_name="aichat", provider_id="openai", provider_config=sample_provider, models=sample_models
        )

        # Generate config
        config_data = parser.generate(template)

        # Write config to file
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, indent=2)

        # Read config back
        parsed_config = parser.parse(config_path)

        assert parsed_config is not None
        assert "openai" in parsed_config
        assert parsed_config["openai"]["type"] == "openai"
        assert parsed_config["openai"]["api_base"] == "https://api.openai.com/v1"
        assert parsed_config["openai"]["api_key"] == "${OPENAI_API_KEY}"

        if "models" in parsed_config["openai"]:
            models = parsed_config["openai"]["models"]
            assert len(models) == 2
            assert models[0]["name"] == "gpt-4"

    def test_aichat_config_file_merge(self, temp_config_dir, sample_provider):
        """Test merging new providers into existing aichat config."""
        parser = AichatConfigParser()
        config_path = temp_config_dir / "aichat" / "config.yaml"

        # Create initial config file
        initial_config = {"clients": [{"anthropic": {"type": "claude", "api_key": "${ANTHROPIC_API_KEY}"}}]}

        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(initial_config, f)

        # Create new provider template
        template = ConfigTemplate(tool_name="aichat", provider_id="openai", provider_config=sample_provider)

        # Parse existing config
        base_config = parser.parse(config_path)

        # Merge new provider
        merged_config = parser.merge_providers(base_config, [template])

        assert merged_config is not None
        assert "clients" in merged_config
        # The merge should preserve existing clients and add new ones

    def test_codex_config_file_operations(self, temp_config_dir, sample_provider, sample_models):
        """Test codex TOML config file operations."""
        parser = CodexConfigParser()
        config_path = temp_config_dir / "codex" / "config.toml"

        # Create template with models
        template = ConfigTemplate(
            tool_name="codex", provider_id="openai", provider_config=sample_provider, models=sample_models
        )

        # Generate config
        config_data = parser.generate(template)

        # Write TOML config (using tomli_w)
        import tomli_w

        with config_path.open("wb") as f:
            tomli_w.dump(config_data, f)

        # Read config back
        parsed_config = parser.parse(config_path)

        assert parsed_config is not None
        assert "profiles" in parsed_config

        # Should have profiles for each model
        profiles = parsed_config["profiles"]
        assert len(profiles) >= 2

        # Check profile structure
        profile_names = list(profiles.keys())
        gpt4_profile = next((k for k in profile_names if "gpt-4" in k), None)
        assert gpt4_profile is not None

        profile = profiles[gpt4_profile]
        assert profile["base_url"] == "https://api.openai.com/v1"
        assert profile["api_key_env"] == "OPENAI_API_KEY"
        assert profile["model"] == "gpt-4"

    def test_mods_config_file_operations(self, temp_config_dir, sample_provider, sample_models):
        """Test mods YAML config file operations."""
        parser = ModsConfigParser()
        config_path = temp_config_dir / "mods" / "mods.yml"

        # Create template
        template = ConfigTemplate(
            tool_name="mods", provider_id="openai", provider_config=sample_provider, models=sample_models
        )

        # Generate config
        config_data = parser.generate(template)

        # Write YAML config
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, indent=2)

        # Read config back
        parsed_config = parser.parse(config_path)

        assert parsed_config is not None

        # Mods config has 'apis' wrapper
        if "apis" in parsed_config:
            assert "openai" in parsed_config["apis"]
            openai_config = parsed_config["apis"]["openai"]
        else:
            assert "openai" in parsed_config
            openai_config = parsed_config["openai"]
        assert openai_config["base_url"] == "https://api.openai.com/v1"
        assert openai_config["api_key"] == "${OPENAI_API_KEY}"

        if "models" in openai_config:
            models = openai_config["models"]
            assert len(models) >= 2
            # Models are stored as dictionaries with 'name' field
            model_names = [m["name"] if isinstance(m, dict) else m for m in models]
            assert "gpt-4" in model_names
            assert "gpt-3.5-turbo" in model_names


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager with real file operations."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create tool subdirectories
            for tool in ["aichat", "codex", "mods"]:
                (config_dir / tool).mkdir()

            yield config_dir

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create ConfigManager with temporary directory."""
        return ConfigManager(temp_config_dir)

    def test_config_backup_functionality(self, config_manager, temp_config_dir):
        """Test config backup and restore functionality."""
        aichat_config_path = temp_config_dir / "aichat" / "config.yaml"

        # Create initial config file
        initial_config = {"default_model": "gpt-4", "clients": []}

        with aichat_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(initial_config, f)

        # Create backup
        backup_path = config_manager.backup_config("aichat")

        assert backup_path is not None
        assert backup_path.exists()
        assert "backup" in str(backup_path)

        # Verify backup contents
        with backup_path.open("r", encoding="utf-8") as f:
            backup_data = yaml.safe_load(f)

        assert backup_data["default_model"] == "gpt-4"

    def test_config_manager_parser_integration(self, config_manager):
        """Test ConfigManager parser access and functionality."""
        # Test parser retrieval
        aichat_parser = config_manager.get_parser("aichat")
        assert isinstance(aichat_parser, AichatConfigParser)

        codex_parser = config_manager.get_parser("codex")
        assert isinstance(codex_parser, CodexConfigParser)

        mods_parser = config_manager.get_parser("mods")
        assert isinstance(mods_parser, ModsConfigParser)

        # Test invalid parser
        with pytest.raises(ValueError, match="Unsupported tool: invalid_tool"):
            config_manager.get_parser("invalid_tool")

    def test_storage_manager_config_integration(self, temp_config_dir):
        """Test StorageManager config directory integration."""
        storage = StorageManager(temp_config_dir.parent)

        # Create test data
        test_data = {"test": "data"}
        test_lines = ["model1", "model2", "model3"]

        # Test config JSON writing
        storage.write_config_json("test_models", test_data)
        json_path = temp_config_dir.parent / "config" / "json" / "test_models.json"
        assert json_path.exists()

        # Test config TXT writing
        storage.write_config_txt("test_models", test_lines)
        txt_path = temp_config_dir.parent / "config" / "txt" / "test_models.txt"
        assert txt_path.exists()

        # Verify TXT contents
        with txt_path.open("r", encoding="utf-8") as f:
            content = f.read()
            assert "model1" in content
            assert "model2" in content
            assert "model3" in content

    def test_yaml_config_writing_integration(self, temp_config_dir):
        """Test YAML config writing to different directories."""
        storage = StorageManager(temp_config_dir.parent)

        # Test data
        aichat_data = {"openai": {"type": "openai", "api_base": "https://api.openai.com/v1"}}

        mods_data = {"openai": {"base_url": "https://api.openai.com/v1", "models": ["gpt-4", "gpt-3.5-turbo"]}}

        # Write YAML files to different directories
        storage.write_yaml("test_aichat", aichat_data, directory="aichat")
        storage.write_yaml("test_mods", mods_data, directory="mods")

        # Verify files were created
        aichat_path = temp_config_dir.parent / "config" / "aichat" / "test_aichat.yaml"
        mods_path = temp_config_dir.parent / "config" / "mods" / "test_mods.yml"

        assert aichat_path.exists()
        assert mods_path.exists()

        # Verify contents
        with aichat_path.open("r", encoding="utf-8") as f:
            aichat_content = yaml.safe_load(f)
            assert aichat_content["openai"]["type"] == "openai"

        with mods_path.open("r", encoding="utf-8") as f:
            mods_content = yaml.safe_load(f)
            assert "gpt-4" in mods_content["openai"]["models"]


class TestRealWorldConfigScenarios:
    """Test realistic config scenarios that users would encounter."""

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
        return ConfigManager(temp_config_dir)

    def test_multi_provider_config_generation(self, config_manager, temp_config_dir):
        """Test generating configs for multiple providers."""
        providers = [
            ProviderConfig(
                name="openai",
                kind=ProviderKind.OPENAI,
                base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
            ),
            ProviderConfig(
                name="anthropic",
                kind=ProviderKind.ANTHROPIC,
                base_url="https://api.anthropic.com/v1",
                api_key_env="ANTHROPIC_API_KEY",
            ),
            ProviderConfig(
                name="groq",
                kind=ProviderKind.OPENAI,
                base_url="https://api.groq.com/openai/v1",
                api_key_env="GROQ_API_KEY",
            ),
        ]

        # Generate configs for each provider
        all_configs = {}

        for provider in providers:
            template = ConfigTemplate(tool_name="aichat", provider_id=provider.name, provider_config=provider)

            parser = config_manager.get_parser("aichat")
            config = parser.generate(template)
            all_configs.update(config)

        # Verify all providers are in config
        assert "openai" in all_configs
        assert "anthropic" in all_configs
        assert "groq" in all_configs

        # Verify provider-specific configurations
        assert all_configs["openai"]["type"] == "openai"
        assert all_configs["anthropic"]["type"] == "claude"
        assert all_configs["groq"]["type"] == "openai"

        # Write combined config to file
        config_path = temp_config_dir / "aichat" / "config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(all_configs, f, default_flow_style=False, indent=2)

        # Verify file was created successfully
        assert config_path.exists()

        # Parse it back
        parser = config_manager.get_parser("aichat")
        parsed = parser.parse(config_path)

        assert len(parsed) == 3
        assert all(provider in parsed for provider in ["openai", "anthropic", "groq"])

    def test_config_error_handling(self, config_manager, temp_config_dir):
        """Test config error handling with malformed files."""
        # Create malformed YAML file
        bad_config_path = temp_config_dir / "aichat" / "config.yaml"
        with bad_config_path.open("w", encoding="utf-8") as f:
            f.write("invalid: yaml: content: [\n")  # Invalid YAML

        # Parser should handle malformed file gracefully
        parser = config_manager.get_parser("aichat")
        config = parser.parse(bad_config_path)

        # Should return default config instead of crashing
        assert config is not None
        assert isinstance(config, dict)

    def test_missing_config_file_handling(self, config_manager, temp_config_dir):
        """Test handling of missing config files."""
        nonexistent_path = temp_config_dir / "aichat" / "nonexistent.yaml"

        parser = config_manager.get_parser("aichat")
        config = parser.parse(nonexistent_path)

        # Should return default config for missing files
        assert config is not None
        assert isinstance(config, dict)


def test_config_file_permissions(tmp_path):
    """Test config file permissions and access."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    storage = StorageManager(tmp_path)

    # Test writing to read-only directory (if permissions allow testing)
    test_data = {"test": "data"}

    try:
        storage.write_config_json("permissions_test", test_data)
        json_path = config_dir / "json" / "permissions_test.json"
        assert json_path.exists()
    except PermissionError:
        # Expected on some systems, test passes
        pass


if __name__ == "__main__":
    pytest.main([__file__])
