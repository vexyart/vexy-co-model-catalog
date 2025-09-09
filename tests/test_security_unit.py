#!/usr/bin/env python3
# this_file: tests/test_security_unit.py

"""Comprehensive unit tests for the security module."""

import sys
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vexy_co_model_catalog.core.security import (
    MaskingRule,
    SensitiveDataProtector,
    SensitivityLevel,
    create_safe_log_context,
    get_protector,
    mask_api_key,
    mask_text_for_cli,
    mask_text_for_logs,
    sanitize_cli_headers,
    sanitize_headers,
)


class TestSensitivityLevel(unittest.TestCase):
    """Test SensitivityLevel enum."""

    def test_sensitivity_levels_exist(self):
        """Test that all expected sensitivity levels exist."""
        levels = [level.value for level in SensitivityLevel]
        expected = ["public", "internal", "confidential", "secret"]
        assert sorted(levels) == sorted(expected)

    def test_sensitivity_level_types(self):
        """Test that sensitivity levels are proper enum members."""
        assert isinstance(SensitivityLevel.PUBLIC, SensitivityLevel)
        assert isinstance(SensitivityLevel.SECRET, SensitivityLevel)
        assert SensitivityLevel.PUBLIC != SensitivityLevel.SECRET


class TestMaskingRule(unittest.TestCase):
    """Test MaskingRule dataclass."""

    def test_masking_rule_creation(self):
        """Test creating a masking rule."""
        rule = MaskingRule(
            pattern=r"api[_-]?key",
            sensitivity=SensitivityLevel.SECRET,
            replacement="***[API-KEY]",
            description="API key pattern",
        )
        assert rule.pattern == r"api[_-]?key"
        assert rule.sensitivity == SensitivityLevel.SECRET
        assert rule.replacement == "***[API-KEY]"
        assert rule.description == "API key pattern"

    def test_masking_rule_defaults(self):
        """Test masking rule with default values."""
        rule = MaskingRule(pattern=r"test")
        assert rule.sensitivity == SensitivityLevel.CONFIDENTIAL
        assert rule.replacement == "***[REDACTED]"
        assert rule.description is None


class TestSensitiveDataProtector(unittest.TestCase):
    """Test SensitiveDataProtector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.protector = SensitiveDataProtector()

    def test_protector_initialization(self):
        """Test protector initializes with default rules."""
        assert len(self.protector.rules) > 0
        # Check that some basic patterns are included
        patterns = [rule.pattern for rule in self.protector.rules]
        assert any("api" in pattern.lower() for pattern in patterns)

    def test_mask_api_key_basic(self):
        """Test basic API key masking."""
        api_key = "sk-1234567890abcdef1234567890abcdef"
        masked = self.protector.mask_api_key(api_key)
        assert "***[REDACTED]" in masked
        assert masked.startswith("sk-1")
        assert masked.endswith("cdef")

    def test_mask_api_key_short(self):
        """Test masking of short API keys."""
        short_key = "abc123"
        masked = self.protector.mask_api_key(short_key)
        assert masked == "***[REDACTED]"

    def test_mask_api_key_empty(self):
        """Test masking empty API key."""
        masked = self.protector.mask_api_key("")
        assert masked == "***[REDACTED]"

    def test_mask_api_key_none(self):
        """Test masking None API key."""
        masked = self.protector.mask_api_key(None)
        assert masked == "***[REDACTED]"

    def test_mask_api_key_custom_show_chars(self):
        """Test API key masking with custom show characters."""
        api_key = "sk-1234567890abcdef1234567890abcdef"
        masked = self.protector.mask_api_key(api_key, show_chars=6)
        assert masked.startswith("sk-123")
        assert masked.endswith("abcdef")

    def test_sanitize_headers_basic(self):
        """Test basic header sanitization."""
        headers = {
            "Authorization": "Bearer sk-1234567890abcdef",
            "X-API-Key": "api_key_12345",
            "Content-Type": "application/json",
            "User-Agent": "test-client/1.0",
        }
        sanitized = self.protector.sanitize_headers(headers)
        assert "***[REDACTED]" in sanitized["Authorization"]
        assert "***[REDACTED]" in sanitized["X-API-Key"]
        assert sanitized["Content-Type"] == "application/json"
        assert sanitized["User-Agent"] == "test-client/1.0"

    def test_sanitize_headers_empty(self):
        """Test sanitizing empty headers."""
        sanitized = self.protector.sanitize_headers({})
        assert sanitized == {}

    def test_sanitize_headers_none(self):
        """Test sanitizing None headers."""
        sanitized = self.protector.sanitize_headers(None)
        assert sanitized == {}

    def test_mask_text_basic(self):
        """Test basic text masking for logs."""
        text = "The API key is sk-1234567890abcdef and the password is secret123"
        masked = mask_text_for_logs(text)
        # Should contain some masking
        assert text != masked

    def test_mask_text_multiple_patterns(self):
        """Test masking text with multiple sensitive patterns."""
        text = "API_KEY=sk-123 and PASSWORD=abc and TOKEN=xyz"
        masked = mask_text_for_logs(text)
        # Should mask sensitive items
        assert text != masked

    def test_mask_text_no_sensitive_data(self):
        """Test masking text with no sensitive data."""
        text = "This is just normal text with no secrets"
        masked = mask_text_for_logs(text)
        # Should not change non-sensitive text
        assert text == masked

    def test_protector_initialization(self):
        """Test that protector initializes with masking rules."""
        assert self.protector.masking_rules is not None
        assert len(self.protector.masking_rules) > 0

    def test_create_safe_log_context(self):
        """Test creating safe log context."""
        context = create_safe_log_context(
            provider_name="test_provider",
            url="https://api.example.com/test?api_key=secret123",
            headers={"Authorization": "Bearer token123"},
        )
        assert "provider" in context
        assert context["provider"] == "test_provider"
        # URL and headers should be sanitized
        assert "secret123" not in str(context)
        assert "token123" not in str(context)


class TestModuleFunctions(unittest.TestCase):
    """Test module-level convenience functions."""

    def test_mask_api_key_function(self):
        """Test mask_api_key convenience function."""
        api_key = "sk-1234567890abcdef1234567890abcdef"
        masked = mask_api_key(api_key)
        # Should mask the API key
        assert api_key != masked
        assert masked.startswith("sk-1")

    def test_sanitize_headers_function(self):
        """Test sanitize_headers convenience function."""
        headers = {"Authorization": "Bearer token123", "Content-Type": "application/json"}
        sanitized = sanitize_headers(headers)
        # Authorization should be masked
        assert headers["Authorization"] != sanitized["Authorization"]
        # Content-Type should remain unchanged
        assert headers["Content-Type"] == sanitized["Content-Type"]

    def test_sanitize_cli_headers_function(self):
        """Test sanitize_cli_headers convenience function."""
        headers = {"Authorization": "Bearer token123", "User-Agent": "test"}
        sanitized = sanitize_cli_headers(headers)
        # Should be sanitized for CLI output
        assert headers["Authorization"] != sanitized["Authorization"]

    def test_mask_text_for_logs_function(self):
        """Test mask_text_for_logs convenience function."""
        text = "API key: sk-123456789"
        masked = mask_text_for_logs(text)
        # Should mask sensitive content for logs
        assert text != masked

    def test_mask_text_for_cli_function(self):
        """Test mask_text_for_cli convenience function."""
        text = "Token: abc123def456"
        masked = mask_text_for_cli(text)
        # Should mask sensitive content for CLI
        assert text != masked

    def test_get_protector_function(self):
        """Test get_protector convenience function."""
        protector = get_protector()
        assert isinstance(protector, SensitiveDataProtector)
        assert protector.masking_rules is not None


class TestSecurityIntegration(unittest.TestCase):
    """Test security module integration scenarios."""

    def test_protector_thread_safety(self):
        """Test that protector can be used safely across threads."""
        import threading
        import time

        protector = SensitiveDataProtector()
        results = []

        def mask_in_thread():
            for i in range(10):
                masked = protector.mask_api_key(f"sk-test{i}{'0' * 20}")
                results.append(masked)
                time.sleep(0.001)  # Small delay to increase chance of race conditions

        threads = [threading.Thread(target=mask_in_thread) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All results should be properly masked
        assert len(results) == 30
        for result in results:
            assert "***[REDACTED]" in result

    def test_large_text_performance(self):
        """Test that masking large text performs reasonably."""
        import time

        protector = SensitiveDataProtector()
        large_text = "Normal text " * 10000 + " API_KEY=sk-1234567890abcdef " + "More text " * 10000

        start_time = time.time()
        masked = protector.mask_text(large_text)
        elapsed = time.time() - start_time

        # Should complete in reasonable time (less than 1 second)
        assert elapsed < 1.0
        assert "***[REDACTED]" in masked

    def test_edge_case_patterns(self):
        """Test edge cases in pattern matching."""
        protector = SensitiveDataProtector()

        # Test with special characters
        text_with_special = "API-KEY: sk-abc123!@#$%^&*()_+{}|:<>?[]\\;'\",./"
        masked = protector.mask_text(text_with_special)
        assert "***[REDACTED]" in masked

        # Test with unicode characters
        text_with_unicode = "API_KEY=sk-测试123αβγ"
        masked = protector.mask_text(text_with_unicode)
        assert "***[REDACTED]" in masked


if __name__ == "__main__":
    unittest.main()
