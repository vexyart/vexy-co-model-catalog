# this_file: tests/test_security_enhanced.py

"""
Comprehensive test suite for enhanced security features.
Tests the security hardening and robustness enhancements.
"""

import os
import tempfile
from pathlib import Path

import pytest

from vexy_co_model_catalog.core.security_enhanced import (
    EnhancedSecurityValidator,
    get_enhanced_security_validator,
    validate_environment_security,
    validate_file_permissions,
    validate_network_url,
)


class TestEnhancedSecurityValidator:
    """Test enhanced security validation functionality."""

    def test_validator_singleton(self):
        """Test that validator is a singleton."""
        validator1 = get_enhanced_security_validator()
        validator2 = get_enhanced_security_validator()
        assert validator1 is validator2

    def test_file_permissions_validation(self):
        """Test file permissions validation."""
        validator = EnhancedSecurityValidator()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Test basic file permissions
            tmp_path.chmod(0o644)  # Standard permissions
            assert validator.validate_file_permissions(tmp_path, secure_mode=False)

            # Test secure mode
            tmp_path.chmod(0o600)  # Secure permissions
            assert validator.validate_file_permissions(tmp_path, secure_mode=True)

            # Test insecure permissions
            tmp_path.chmod(0o666)  # World writable - should fail
            assert not validator.validate_file_permissions(tmp_path, secure_mode=False)

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_directory_permissions_validation(self):
        """Test directory permissions validation."""
        validator = EnhancedSecurityValidator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Test standard directory permissions
            tmp_path.chmod(0o755)
            assert validator.validate_file_permissions(tmp_path, secure_mode=False)

            # Test secure directory permissions
            tmp_path.chmod(0o700)
            assert validator.validate_file_permissions(tmp_path, secure_mode=True)

    def test_network_url_validation(self):
        """Test network URL security validation."""
        validator = EnhancedSecurityValidator()

        # Valid HTTPS URLs should pass
        assert validator.validate_network_url("https://api.openai.com/v1/models")
        assert validator.validate_network_url("https://api.anthropic.com/v1/models")

        # HTTP URLs should fail unless localhost
        assert not validator.validate_network_url("http://api.openai.com/v1/models")

        # Localhost HTTP should pass when allowed
        assert validator.validate_network_url("http://localhost:8080/models", allow_localhost=True)
        assert validator.validate_network_url("http://127.0.0.1:8080/models", allow_localhost=True)

        # Localhost HTTP should fail when not allowed
        assert not validator.validate_network_url("http://localhost:8080/models", allow_localhost=False)

        # Suspicious URLs should fail
        assert not validator.validate_network_url("https://169.254.169.254/metadata")
        assert not validator.validate_network_url("https://metadata.internal/config")

    def test_api_key_format_validation(self):
        """Test API key format validation."""
        validator = EnhancedSecurityValidator()

        # Valid OpenAI key format
        assert validator.validate_api_key_format("sk-1234567890abcdef1234567890abcdef", "openai")

        # Valid Anthropic key format
        assert validator.validate_api_key_format("ant-api-key-1234567890abcdef1234567890abcdefgh", "anthropic")

        # Valid Groq key format
        assert validator.validate_api_key_format("gsk_" + "a" * 50, "groq")

        # Invalid formats should fail
        assert not validator.validate_api_key_format("invalid-key", "openai")
        assert not validator.validate_api_key_format("too-short", "openai")
        assert not validator.validate_api_key_format("", "openai")

        # Placeholder keys should fail
        assert not validator.validate_api_key_format("your_api_key_here", "openai")
        assert not validator.validate_api_key_format("replace_me", "openai")

        # Keys with whitespace should fail
        assert not validator.validate_api_key_format(" sk-validkey ", "openai")

    def test_environment_security_validation(self):
        """Test environment security validation."""
        report = validate_environment_security()

        # Should return a valid report structure
        assert "overall_status" in report
        assert "issues" in report
        assert "warnings" in report
        assert "recommendations" in report

        assert report["overall_status"] in ["secure", "warning", "excellent"]
        assert isinstance(report["issues"], list)
        assert isinstance(report["warnings"], list)
        assert isinstance(report["recommendations"], list)

    def test_secure_file_write(self):
        """Test secure file writing."""
        validator = EnhancedSecurityValidator()

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test_secure.txt"
            content = "test content for security validation"

            # Test standard secure write
            assert validator.secure_file_write(test_file, content, secure_mode=False)
            assert test_file.exists()
            assert test_file.read_text() == content

            # Test secure mode write
            test_file_secure = Path(tmp_dir) / "test_very_secure.txt"
            assert validator.secure_file_write(test_file_secure, content, secure_mode=True)
            assert test_file_secure.exists()


class TestConvenienceFunctions:
    """Test convenience functions for security validation."""

    def test_validate_file_permissions_function(self):
        """Test standalone file permissions validation function."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            tmp_path.chmod(0o644)
            assert validate_file_permissions(tmp_path, secure_mode=False)

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_validate_network_url_function(self):
        """Test standalone network URL validation function."""
        assert validate_network_url("https://api.openai.com/v1/models")
        assert not validate_network_url("http://suspicious.internal/config")


class TestSecurityIntegration:
    """Test integration of security features with other modules."""

    def test_security_with_storage_module(self):
        """Test that storage module integrates with security validation."""
        # This tests that the import works correctly
        from vexy_co_model_catalog.core.storage import StorageManager

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Should be able to create StorageManager without issues
            storage = StorageManager(tmp_dir)
            assert storage is not None

    def test_security_with_fetcher_module(self):
        """Test that fetcher module integrates with security validation."""
        # This tests that the import works correctly
        from vexy_co_model_catalog.core.fetcher import ModelFetcher

        # Should be able to create ModelFetcher without issues
        fetcher = ModelFetcher()
        assert fetcher is not None


@pytest.mark.slow
class TestSecurityPerformance:
    """Test that security validation doesn't significantly impact performance."""

    def test_url_validation_performance(self):
        """Test URL validation performance on many URLs."""
        import time

        urls = [
            "https://api.openai.com/v1/models",
            "https://api.anthropic.com/v1/models",
            "https://api.groq.com/openai/v1/models",
        ] * 100  # Test with 300 URLs

        start_time = time.perf_counter()

        for url in urls:
            validate_network_url(url)

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Should validate 300 URLs in under 100ms
        assert duration_ms < 100, f"URL validation took {duration_ms:.1f}ms for {len(urls)} URLs"

    def test_file_validation_performance(self):
        """Test file validation performance on many files."""
        import time

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            test_files = []
            for i in range(50):
                test_file = Path(tmp_dir) / f"test_{i}.txt"
                test_file.write_text(f"content {i}")
                test_files.append(test_file)

            start_time = time.perf_counter()

            for test_file in test_files:
                validate_file_permissions(test_file)

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Should validate 50 files in under 50ms
            assert duration_ms < 50, f"File validation took {duration_ms:.1f}ms for {len(test_files)} files"
