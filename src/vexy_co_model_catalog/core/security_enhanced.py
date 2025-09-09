# this_file: src/vexy_co_model_catalog/core/security_enhanced.py

"""
Enhanced security utilities for production-grade robustness.
Extends the core security module with additional validation and checks.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from vexy_co_model_catalog.core.security import get_protector, mask_text_for_logs

# Security constants for file permission validation
SAFE_FILE_PERMISSIONS = 0o644  # rw-r--r--
SAFE_DIR_PERMISSIONS = 0o755   # rwxr-xr-x
SECURE_FILE_PERMISSIONS = 0o600  # rw-------
SECURE_DIR_PERMISSIONS = 0o700   # rwx------


class EnhancedSecurityValidator:
    """Enhanced security validation for production environments."""

    def __init__(self) -> None:
        """Initialize enhanced security validator."""
        self.protector = get_protector()

    def validate_file_permissions(self, file_path: Path, *, secure_mode: bool = False) -> bool:
        """
        Validate file permissions are secure and appropriate.

        Args:
            file_path: Path to validate
            secure_mode: If True, require stricter permissions for sensitive files

        Returns:
            bool: True if permissions are acceptable, False otherwise
        """
        try:
            if not file_path.exists():
                logger.warning(f"File does not exist for permission check: {file_path}")
                return False

            file_stat = file_path.stat()
            current_perms = stat.filemode(file_stat.st_mode)
            octal_perms = stat.S_IMODE(file_stat.st_mode)

            # Determine expected permissions based on file type and security mode
            if file_path.is_dir():
                expected_desc = "700 (rwx------)" if secure_mode else "755 (rwxr-xr-x)"
            else:
                expected_desc = "600 (rw-------)" if secure_mode else "644 (rw-r--r--)"

            # Check for overly permissive permissions
            if octal_perms & stat.S_IWOTH:  # World writable
                logger.error(f"SECURITY: File {file_path} is world-writable ({current_perms})")
                return False

            if secure_mode and (octal_perms & (stat.S_IRGRP | stat.S_IROTH)):  # Group/others readable
                logger.warning(
                    f"SECURITY: Sensitive file {file_path} has group/other read permissions "
                    f"({current_perms}), should be {expected_desc}"
                )
                return False

            # Check for executable files that shouldn't be
            if (file_path.suffix in {'.json', '.yaml', '.yml', '.toml', '.txt', '.md'}
                and octal_perms & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
                logger.warning(f"SECURITY: Config file {file_path} is executable ({current_perms})")
                return False

            logger.debug(f"File permissions OK: {file_path} ({current_perms})")
            return True

        except (OSError, PermissionError) as e:
            logger.error(f"Failed to check permissions for {file_path}: {mask_text_for_logs(str(e))}")
            return False

    def validate_network_url(self, url: str, *, allow_localhost: bool = False) -> bool:
        """
        Validate network URL for security issues.

        Args:
            url: URL to validate
            allow_localhost: Whether to allow localhost/127.0.0.1 URLs

        Returns:
            bool: True if URL is safe, False otherwise
        """
        try:
            parsed = urlparse(url)

            # Require HTTPS except for allowed localhost
            if parsed.scheme != 'https':
                if parsed.scheme == 'http' and allow_localhost and parsed.hostname in ('localhost', '127.0.0.1'):
                    logger.debug(f"Allowing HTTP localhost URL: {mask_text_for_logs(url)}")
                else:
                    logger.warning(f"SECURITY: Non-HTTPS URL detected: {mask_text_for_logs(url)}")
                    return False

            # Check for suspicious hostnames
            if parsed.hostname:
                hostname_lower = parsed.hostname.lower()
                suspicious_patterns = [
                    'metadata',  # Cloud metadata services
                    'internal',
                    'private',
                    '169.254',   # AWS metadata IP
                    '10.',       # Private IP ranges (basic check)
                    '192.168',   # Private IP ranges (basic check)
                ]

                for pattern in suspicious_patterns:
                    if pattern in hostname_lower:
                        if not (allow_localhost and pattern in ('127.0.0.1', 'localhost')):
                            logger.warning(f"SECURITY: Potentially unsafe hostname: {mask_text_for_logs(url)}")
                            return False

            # Check for suspicious ports
            if parsed.port and parsed.port in {22, 23, 25, 53, 139, 445, 3389}:  # Common service ports
                logger.warning(f"SECURITY: URL uses suspicious port {parsed.port}: {mask_text_for_logs(url)}")
                return False

            logger.debug(f"Network URL validation passed: {mask_text_for_logs(url)}")
            return True

        except Exception as e:
            logger.error(f"Failed to validate URL {mask_text_for_logs(url)}: {mask_text_for_logs(str(e))}")
            return False

    def validate_api_key_format(self, api_key: str, provider_name: str) -> bool:
        """
        Validate API key format for known patterns.

        Args:
            api_key: API key to validate
            provider_name: Provider name for pattern matching

        Returns:
            bool: True if format appears valid, False otherwise
        """
        if not api_key or len(api_key.strip()) == 0:
            return False

        api_key = api_key.strip()

        # Basic length validation
        if len(api_key) < 8:
            logger.warning(f"SECURITY: API key for {provider_name} appears too short")
            return False

        # Pattern validation for known providers
        provider_patterns = {
            'openai': r'^sk-[a-zA-Z0-9]{20,}$',
            'anthropic': r'^ant-[a-zA-Z0-9_-]{40,}$',
            'groq': r'^gsk_[a-zA-Z0-9]{50,}$',
        }

        if provider_name.lower() in provider_patterns:
            pattern = provider_patterns[provider_name.lower()]
            if not re.match(pattern, api_key):
                logger.warning(f"SECURITY: API key format invalid for {provider_name}")
                return False

        # Check for common issues
        if api_key.startswith((' ', '\t')) or api_key.endswith((' ', '\t')):
            logger.warning(f"SECURITY: API key for {provider_name} has leading/trailing whitespace")
            return False

        # Check for placeholder values
        placeholder_indicators = ['your_api_key', 'replace_me', 'placeholder', 'example']
        if any(indicator in api_key.lower() for indicator in placeholder_indicators):
            logger.warning(f"SECURITY: API key for {provider_name} appears to be a placeholder")
            return False

        logger.debug(f"API key format validation passed for {provider_name}")
        return True

    def secure_file_write(self, file_path: Path, content: str, *, secure_mode: bool = False) -> bool:
        """
        Securely write content to file with appropriate permissions.

        Args:
            file_path: Path to write to
            content: Content to write
            secure_mode: If True, use stricter permissions

        Returns:
            bool: True if write successful, False otherwise
        """
        try:
            # Ensure parent directory exists with secure permissions
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Set directory permissions
            dir_perms = SECURE_DIR_PERMISSIONS if secure_mode else SAFE_DIR_PERMISSIONS
            file_path.parent.chmod(dir_perms)

            # Write file content
            with file_path.open('w', encoding='utf-8') as f:
                f.write(content)

            # Set file permissions
            file_perms = SECURE_FILE_PERMISSIONS if secure_mode else SAFE_FILE_PERMISSIONS
            file_path.chmod(file_perms)

            # Validate the permissions were set correctly
            if not self.validate_file_permissions(file_path, secure_mode=secure_mode):
                logger.error(f"SECURITY: Failed to set secure permissions for {file_path}")
                return False

            logger.debug(f"Securely wrote file: {file_path}")
            return True

        except (OSError, PermissionError) as e:
            logger.error(f"Failed to securely write {file_path}: {mask_text_for_logs(str(e))}")
            return False

    def validate_environment_security(self) -> dict[str, Any]:
        """
        Validate current environment for security issues.

        Returns:
            dict: Security validation report
        """
        report = {
            "overall_status": "secure",
            "issues": [],
            "warnings": [],
            "recommendations": []
        }

        # Check file creation mask
        try:
            current_umask = os.umask(0o022)  # Set and immediately restore
            os.umask(current_umask)

            if current_umask != 0o022:
                report["warnings"].append(f"Unusual umask detected: {oct(current_umask)}")

        except Exception as e:
            report["warnings"].append(f"Could not check umask: {e!s}")

        # Check for common security environment variables
        sensitive_env_vars = []
        for key in os.environ:
            if self.protector.is_sensitive_key(key) and os.environ[key]:
                sensitive_env_vars.append(key)

        if sensitive_env_vars:
            report["warnings"].append(f"Found {len(sensitive_env_vars)} sensitive environment variables")
            logger.debug(f"Sensitive env vars found: {sensitive_env_vars}")

        # Check working directory permissions
        try:
            cwd = Path.cwd()
            if not self.validate_file_permissions(cwd):
                report["issues"].append(f"Working directory has insecure permissions: {cwd}")
                report["overall_status"] = "warning"

        except Exception as e:
            report["warnings"].append(f"Could not validate working directory: {e!s}")

        # Add recommendations based on findings
        if report["issues"]:
            report["recommendations"].append("Review and fix file permission issues")

        if report["warnings"]:
            report["recommendations"].append("Review security warnings and consider hardening")

        if not report["issues"] and not report["warnings"]:
            report["overall_status"] = "excellent"

        return report


# Global instance for easy access
_enhanced_validator: EnhancedSecurityValidator | None = None


def get_enhanced_security_validator() -> EnhancedSecurityValidator:
    """Get the global enhanced security validator instance."""
    global _enhanced_validator
    if _enhanced_validator is None:
        _enhanced_validator = EnhancedSecurityValidator()
    return _enhanced_validator


# Convenience functions
def validate_file_permissions(file_path: Path, *, secure_mode: bool = False) -> bool:
    """Validate file permissions are secure."""
    return get_enhanced_security_validator().validate_file_permissions(file_path, secure_mode=secure_mode)


def validate_network_url(url: str, *, allow_localhost: bool = False) -> bool:
    """Validate network URL for security issues."""
    return get_enhanced_security_validator().validate_network_url(url, allow_localhost=allow_localhost)


def secure_file_write(file_path: Path, content: str, *, secure_mode: bool = False) -> bool:
    """Securely write content to file with appropriate permissions."""
    return get_enhanced_security_validator().secure_file_write(file_path, content, secure_mode=secure_mode)


def validate_environment_security() -> dict[str, Any]:
    """Validate current environment for security issues."""
    return get_enhanced_security_validator().validate_environment_security()
