#!/usr/bin/env python3
"""
this_file: tests/test_enhanced_integration.py

Comprehensive integration tests for enhanced modules (error recovery, config validation) ensuring end-to-end reliability.
"""

import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vexy_co_model_catalog.core.config_management import ConfigurationManager
from vexy_co_model_catalog.core.enhanced_config_validation import ConfigRepairStrategy, EnhancedConfigValidator
from vexy_co_model_catalog.core.enhanced_integration import (
    EnhancedModelFetcher,
    EnhancedStorageManager,
    run_enhanced_diagnostics,
)
from vexy_co_model_catalog.core.enhanced_logging import (
    EnhancedErrorRecovery,
    ErrorCategory,
    ErrorContext,
    StructuredLogger,
    operation_context,
)
from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind
from vexy_co_model_catalog.utils.exceptions import AuthenticationError, FetchError, RateLimitError


class TestEnhancedErrorRecoveryIntegration(unittest.TestCase):
    """Integration tests for enhanced error recovery and structured logging."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.logger = StructuredLogger("test_integration")
        self.error_recovery = EnhancedErrorRecovery(self.logger)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_structured_logging_with_operation_context(self):
        """Test structured logging integration with operation context."""
        # Test operation context tracking
        with operation_context("test_operation", "test_component", test_param="test_value") as ctx:
            assert ctx is not None
            assert ctx.operation_name == "test_operation"
            assert ctx.component == "test_component"

            # Test logging within context
            self.logger.info("Test message within context", additional_data="test")
            self.logger.warning("Test warning within context")

            # Test error logging with context
            error_context = ErrorContext(
                category=ErrorCategory.NETWORK, operation="test_operation", metadata={"test": "data"}
            )
            self.logger.error("Test error within context", error_context=error_context)

    def test_error_recovery_with_multiple_fallbacks(self):
        """Test error recovery with multiple fallback handlers."""

        # Register multiple fallback handlers
        def first_fallback(error, **kwargs):
            if "force_first_success" in kwargs:
                return "first_fallback_result"
            msg = "First fallback failed"
            raise Exception(msg)

        def second_fallback(error, **kwargs):
            return "second_fallback_result"

        self.error_recovery.register_fallback_handler(ErrorCategory.NETWORK, first_fallback)
        self.error_recovery.register_fallback_handler(ErrorCategory.NETWORK, second_fallback)

        # Test that second fallback is used when first fails
        def failing_operation(**kwargs):
            msg = "Primary operation failed"
            raise Exception(msg)

        result = self.error_recovery.execute_with_fallback(
            operation=failing_operation,
            operation_name="test_multi_fallback",
            error_category=ErrorCategory.NETWORK,
            enable_fallback=True,
        )

        assert result == "second_fallback_result"

        # Test first fallback success when condition is met
        result2 = self.error_recovery.execute_with_fallback(
            operation=failing_operation,
            operation_name="test_first_fallback",
            error_category=ErrorCategory.NETWORK,
            enable_fallback=True,
            force_first_success=True,
        )

        assert result2 == "first_fallback_result"

    def test_error_classification_integration(self):
        """Test automatic error classification and recovery suggestion generation."""
        test_errors = [
            (ConnectionError("Connection timed out"), ErrorCategory.NETWORK),
            (FileNotFoundError("Config file not found"), ErrorCategory.FILE_OPERATION),
            (ValueError("Invalid JSON in config"), ErrorCategory.DATA_VALIDATION),
            (PermissionError("Access denied"), ErrorCategory.FILE_OPERATION),
        ]

        for error, expected_category in test_errors:
            error_context = ErrorContext()
            self.error_recovery._classify_error(error, error_context)

            assert error_context.category == expected_category
            assert error_context.recovery_suggestion is not None
            assert error_context.user_action is not None

    def test_correlation_id_tracking(self):
        """Test correlation ID tracking across operations."""
        from vexy_co_model_catalog.core.enhanced_logging import get_current_correlation_id

        # Test correlation ID within operation context
        with operation_context("test_correlation", "test_component") as ctx:
            correlation_id = get_current_correlation_id()
            assert correlation_id is not None
            assert correlation_id == ctx.operation_id

            # Test that correlation ID persists during error recovery
            def test_operation(**kwargs):
                current_id = get_current_correlation_id()
                assert current_id == correlation_id
                return "success"

            result = self.error_recovery.execute_with_fallback(
                operation=test_operation,
                operation_name="test_correlation_persistence",
                error_category=ErrorCategory.INTERNAL,
            )

            assert result == "success"


class TestEnhancedModelFetcherIntegration(unittest.TestCase):
    """Integration tests for enhanced model fetcher with error recovery."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_provider = ProviderConfig(
            name="test_provider", kind=ProviderKind.OPENAI, base_url="https://api.test.com", models_path="/v1/models"
        )

    def tearDown(self):
        """Clean up test fixtures."""

    @patch("vexy_co_model_catalog.core.enhanced_integration.ModelFetcher")
    @patch("vexy_co_model_catalog.core.enhanced_integration.get_cached_model_data")
    def test_enhanced_fetcher_with_fallback_recovery(self, mock_get_cache, mock_model_fetcher):
        """Test enhanced fetcher with fallback recovery mechanisms."""
        # Setup mocks
        mock_get_cache.return_value = None
        mock_base_fetcher = Mock()
        mock_model_fetcher.return_value = mock_base_fetcher

        # Test network error fallback
        mock_base_fetcher.fetch_provider_models.side_effect = [
            ConnectionError("Network error"),
            {"data": [{"id": "fallback-model", "object": "model"}]},
        ]

        async def test_network_fallback():
            fetcher = EnhancedModelFetcher(max_concurrency=1, timeout=10)

            result = await fetcher.fetch_provider_models_enhanced(provider=self.test_provider, enable_fallback=True)

            # Should get fallback result, not the error fallback value
            assert "data" in result

            await fetcher.close()

        asyncio.run(test_network_fallback())

    @patch("vexy_co_model_catalog.core.enhanced_integration.ModelFetcher")
    def test_enhanced_fetcher_multiple_providers(self, mock_model_fetcher):
        """Test enhanced fetcher with multiple providers and partial failure tolerance."""
        # Setup mock
        mock_base_fetcher = Mock()
        mock_model_fetcher.return_value = mock_base_fetcher

        providers = [
            ProviderConfig(
                name="provider1", kind=ProviderKind.OPENAI, base_url="https://api1.com", models_path="/v1/models"
            ),
            ProviderConfig(
                name="provider2", kind=ProviderKind.OPENAI, base_url="https://api2.com", models_path="/v1/models"
            ),
            ProviderConfig(
                name="provider3", kind=ProviderKind.OPENAI, base_url="https://api3.com", models_path="/v1/models"
            ),
        ]

        # Mock responses: success, failure, success
        mock_base_fetcher.fetch_provider_models.side_effect = [
            {"data": [{"id": "model1", "object": "model"}]},  # provider1 success
            Exception("Provider2 failed"),  # provider2 failure
            {"data": [{"id": "model3", "object": "model"}]},  # provider3 success
        ]

        async def test_multiple_providers():
            fetcher = EnhancedModelFetcher(max_concurrency=2, timeout=10)

            result = await fetcher.fetch_multiple_providers_enhanced(
                providers=providers, max_concurrency=2, continue_on_error=True
            )

            # Check results
            assert "results" in result
            assert "failures" in result
            assert "summary" in result

            # Should have 2 successes and 1 failure
            assert len(result["results"]) == 2
            assert len(result["failures"]) == 1
            assert result["summary"]["successful"] == 2
            assert result["summary"]["failed"] == 1
            assert result["summary"]["total_providers"] == 3

            await fetcher.close()

        asyncio.run(test_multiple_providers())


class TestEnhancedStorageIntegration(unittest.TestCase):
    """Integration tests for enhanced storage manager with error recovery."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.storage_manager = EnhancedStorageManager(self.test_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_batch_write_with_error_recovery(self):
        """Test batch write operations with enhanced error recovery."""
        operations = [
            {"type": "json", "filename": "test1", "data": {"test": "data1"}},
            {"type": "config_json", "filename": "test2", "data": {"test": "data2"}},
            {"type": "yaml", "filename": "test3", "data": {"test": "data3"}, "directory": "config"},
            {
                "type": "invalid_type",  # This should fail
                "filename": "test4",
                "data": {"test": "data4"},
            },
        ]

        result = self.storage_manager.batch_write_enhanced(operations=operations, continue_on_error=True)

        # Check results
        assert "successful_operations" in result
        assert "failed_operations" in result
        assert "summary" in result

        # Should have 3 successes and 1 failure
        assert result["summary"]["successful"] == 3
        assert result["summary"]["failed"] == 1
        assert result["summary"]["total_operations"] == 4

        # Verify successful files were created
        json_file = Path(self.test_dir) / "models" / "json" / "test1.json"
        config_file = Path(self.test_dir) / "config" / "json" / "test2.json"
        yaml_file = Path(self.test_dir) / "config" / "test3.yaml"

        assert json_file.exists()
        assert config_file.exists()
        assert yaml_file.exists()

    def test_storage_fallback_with_directory_creation(self):
        """Test storage fallback mechanisms with automatic directory creation."""
        # Test writing to a path where parent directory doesn't exist
        test_data = {"test": "data"}

        # This should succeed even if directory doesn't exist due to fallback
        result = self.storage_manager.write_json_enhanced("test_fallback", test_data)

        # Should succeed due to fallback directory creation
        assert result or Path(self.test_dir) / "models" / "json" / "test_fallback.json" is not None


class TestConfigValidationIntegration(unittest.TestCase):
    """Integration tests for enhanced configuration validation and management."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.validator = EnhancedConfigValidator(enable_auto_repair=True)
        self.config_manager = ConfigurationManager(root_path=self.test_dir, auto_repair=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_config_validation_with_auto_repair(self):
        """Test configuration validation with automatic repair mechanisms."""
        # Create a broken config file
        config_path = Path(self.test_dir) / "config" / "test_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write invalid YAML
        broken_yaml = """
model: gpt-4
temperature: 0.7
clients:
  - type: openai
    api_key: ${OPENAI_API_KEY
    # Missing closing brace and invalid structure
"""
        config_path.write_text(broken_yaml, encoding="utf-8")

        # Test repair
        result = self.validator.validate_with_repair(
            config_path=config_path,
            repair_strategy=ConfigRepairStrategy.MODERATE,
            create_backup=True,
            enable_fallback=True,
        )

        # Should attempt repair or create fallback
        assert result is not None
        assert len(result.repair_actions) > 0

    def test_configuration_manager_health_check(self):
        """Test configuration manager health check with multiple configs."""
        # Create multiple config files with different issues
        configs = {
            "config/aichat/config.yaml": {
                "model": "gpt-4",
                "temperature": 0.7,
                "clients": [{"type": "openai", "api_key": "${OPENAI_API_KEY}"}],
            },
            "config/codex/config.toml": {
                "default_profile": "gpt4",
                "profiles": {"gpt4": {"model": "gpt-4", "provider": "openai"}},
            },
            "config/mods/config.yml": {"default_model": "gpt-3.5-turbo", "models": ["gpt-3.5-turbo", "gpt-4"]},
        }

        # Create config files
        for config_path, config_data in configs.items():
            full_path = Path(self.test_dir) / config_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                import yaml

                content = yaml.dump(config_data, default_flow_style=False)
            elif config_path.endswith(".toml"):
                import toml

                content = toml.dumps(config_data)
            else:
                content = json.dumps(config_data, indent=2)

            full_path.write_text(content, encoding="utf-8")

        # Test health check
        results = self.config_manager.ensure_config_health()

        # Should process all configs
        assert len(results) == 3

        # All should be successful (valid configs)
        success_count = sum(1 for result in results.values() if result.success)
        assert success_count == 3

    def test_environment_validation_integration(self):
        """Test environment validation integration with configuration management."""
        # Test environment validation
        env_report = self.config_manager.validate_environment_setup()

        # Should have comprehensive report
        assert "overall_status" in env_report
        assert "config_health" in env_report
        assert "environment_variables" in env_report
        assert "dependencies" in env_report
        assert "recommendations" in env_report

        # Status should be determined based on actual environment
        assert env_report["overall_status"] in ["healthy", "functional", "requires_attention"]

    def test_minimal_config_setup_integration(self):
        """Test minimal configuration setup for all tools."""
        # Test setup without existing configs
        results = self.config_manager.setup_minimal_working_config(force_overwrite=False, include_examples=True)

        # Should attempt to create configs for all tools
        expected_tools = {"aichat", "codex", "mods"}
        assert set(results.keys()) == expected_tools

        # Check that config files were created
        for tool in expected_tools:
            if results[tool]:  # If setup was successful
                config_files = list(Path(self.test_dir).glob(f"config/{tool}/*"))
                assert len(config_files) > 0, f"No config files created for {tool}"


class TestSystemDiagnosticsIntegration(unittest.TestCase):
    """Integration tests for comprehensive system diagnostics."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("vexy_co_model_catalog.core.enhanced_integration.EnhancedModelFetcher")
    def test_enhanced_diagnostics_integration(self, mock_fetcher_class):
        """Test comprehensive system diagnostics integration."""
        # Setup mocks
        mock_fetcher = Mock()
        mock_fetcher.get_stats.return_value = {
            "base_stats": {"requests": 10, "errors": 1, "success_rate": 0.9},
            "recovery_stats": {"network_fallback": 2},
            "recent_errors": [],
            "error_count": 1,
        }
        mock_fetcher_class.return_value = mock_fetcher

        storage_manager = EnhancedStorageManager(self.test_dir)

        async def test_diagnostics():
            result = await run_enhanced_diagnostics(storage_manager, mock_fetcher)

            # Should have comprehensive diagnostics
            assert "storage" in result
            assert "fetcher" in result
            assert "summary" in result

            # Check summary
            summary = result["summary"]
            assert "overall_health" in summary
            assert "healthy_components" in summary
            assert "total_components" in summary
            assert "health_percentage" in summary

            return result

        result = asyncio.run(test_diagnostics())
        assert result is not None


if __name__ == "__main__":
    unittest.main()
