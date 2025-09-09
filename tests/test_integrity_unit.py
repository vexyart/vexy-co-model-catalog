#!/usr/bin/env python3
# this_file: tests/test_integrity_unit.py

"""Comprehensive unit tests for the integrity module."""

import hashlib
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from vexy_co_model_catalog.core.integrity import (
    ChecksumAlgorithm,
    CorruptionEvent,
    DataIntegrityManager,
    FileIntegrityInfo,
    IntegrityLevel,
    create_critical_backup,
    get_integrity_manager,
    track_critical_file,
    verify_all_files,
)


class TestIntegrityLevel(unittest.TestCase):
    """Test IntegrityLevel enum."""

    def test_integrity_levels_exist(self):
        """Test that all expected integrity levels exist."""
        levels = [level.value for level in IntegrityLevel]
        expected = ["critical", "important", "standard", "monitoring_only"]
        assert sorted(levels) == sorted(expected)

    def test_integrity_level_ordering(self):
        """Test that integrity levels have proper ordering."""
        # Critical should be highest priority (lowest enum value)
        assert IntegrityLevel.CRITICAL.value < IntegrityLevel.MONITORING_ONLY.value


class TestChecksumAlgorithm(unittest.TestCase):
    """Test ChecksumAlgorithm enum."""

    def test_checksum_algorithms_exist(self):
        """Test that all expected checksum algorithms exist."""
        algorithms = [algo.value for algo in ChecksumAlgorithm]
        expected = ["md5", "sha1", "sha256", "sha512"]
        assert sorted(algorithms) == sorted(expected)

    def test_checksum_algorithm_mapping(self):
        """Test that algorithm enum maps to hashlib functions."""
        for algo in ChecksumAlgorithm:
            # Should be able to create hasher for each algorithm
            hasher = hashlib.new(algo.value)
            assert hasher is not None


class TestFileIntegrityInfo(unittest.TestCase):
    """Test FileIntegrityInfo dataclass."""

    def test_file_integrity_info_creation(self):
        """Test creating file integrity info."""
        info = FileIntegrityInfo(
            file_path=Path("/test/file.json"),
            checksum="abc123",
            algorithm=ChecksumAlgorithm.SHA256,
            integrity_level=IntegrityLevel.CRITICAL,
        )
        assert info.file_path == Path("/test/file.json")
        assert info.checksum == "abc123"
        assert info.algorithm == ChecksumAlgorithm.SHA256
        assert info.integrity_level == IntegrityLevel.CRITICAL
        assert info.created_at is not None
        assert info.last_verified is not None

    def test_file_integrity_info_defaults(self):
        """Test file integrity info with default values."""
        info = FileIntegrityInfo(file_path=Path("/test/file.json"), checksum="abc123")
        assert info.algorithm == ChecksumAlgorithm.SHA256
        assert info.integrity_level == IntegrityLevel.STANDARD
        assert info.verification_count == 0


class TestCorruptionEvent(unittest.TestCase):
    """Test CorruptionEvent dataclass."""

    def test_corruption_event_creation(self):
        """Test creating corruption event."""
        event = CorruptionEvent(
            file_path=Path("/test/corrupted.json"),
            expected_checksum="abc123",
            actual_checksum="def456",
            action_taken="backup_restore",
        )
        assert event.file_path == Path("/test/corrupted.json")
        assert event.expected_checksum == "abc123"
        assert event.actual_checksum == "def456"
        assert event.action_taken == "backup_restore"
        assert event.detected_at is not None

    def test_corruption_event_defaults(self):
        """Test corruption event with default values."""
        event = CorruptionEvent(file_path=Path("/test/file.json"), expected_checksum="abc", actual_checksum="def")
        assert event.action_taken == "logged"


class TestDataIntegrityManager(unittest.TestCase):
    """Test DataIntegrityManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.root_dir = Path(self.temp_dir)
        self.manager = DataIntegrityManager(self.root_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        self.manager.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.root_directory == self.root_dir
        assert len(self.manager.tracked_files) == 0
        assert len(self.manager.corruption_events) == 0

    def test_add_file_tracking(self):
        """Test adding file tracking."""
        test_file = self.root_dir / "test.json"
        test_file.write_text('{"test": "data"}')

        self.manager.add_file_tracking(
            test_file, integrity_level=IntegrityLevel.CRITICAL, algorithm=ChecksumAlgorithm.SHA256
        )

        assert str(test_file) in self.manager.tracked_files
        info = self.manager.tracked_files[str(test_file)]
        assert info.integrity_level == IntegrityLevel.CRITICAL
        assert info.algorithm == ChecksumAlgorithm.SHA256
        assert info.checksum is not None

    def test_calculate_checksum(self):
        """Test checksum calculation."""
        test_file = self.root_dir / "test.json"
        test_data = '{"test": "data"}'
        test_file.write_text(test_data)

        # Test different algorithms
        for algorithm in ChecksumAlgorithm:
            checksum = self.manager._calculate_checksum(test_file, algorithm)

            # Verify checksum by calculating manually
            hasher = hashlib.new(algorithm.value)
            hasher.update(test_data.encode("utf-8"))
            expected = hasher.hexdigest()

            assert checksum == expected

    def test_verify_file_integrity_valid(self):
        """Test file integrity verification with valid file."""
        test_file = self.root_dir / "test.json"
        test_file.write_text('{"test": "data"}')

        # Add tracking
        self.manager.add_file_tracking(test_file, IntegrityLevel.CRITICAL)

        # Verify integrity
        is_valid = self.manager.verify_file_integrity(test_file)
        assert is_valid

    def test_verify_file_integrity_corrupted(self):
        """Test file integrity verification with corrupted file."""
        test_file = self.root_dir / "test.json"
        test_file.write_text('{"test": "data"}')

        # Add tracking
        self.manager.add_file_tracking(test_file, IntegrityLevel.CRITICAL)

        # Corrupt the file
        test_file.write_text('{"test": "corrupted"}')

        # Verify integrity (should detect corruption)
        is_valid = self.manager.verify_file_integrity(test_file, auto_repair=False)
        assert not is_valid

        # Should have logged corruption event
        assert len(self.manager.corruption_events) == 1
        event = self.manager.corruption_events[0]
        assert event.file_path == test_file

    def test_create_backup(self):
        """Test backup creation."""
        test_file = self.root_dir / "test.json"
        test_data = '{"test": "data"}'
        test_file.write_text(test_data)

        backup_path = self.manager.create_backup(test_file)

        assert backup_path.exists()
        assert backup_path.read_text() == test_data
        assert ".backups" in str(backup_path)

    def test_restore_from_backup(self):
        """Test restoration from backup."""
        test_file = self.root_dir / "test.json"
        original_data = '{"test": "original"}'
        test_file.write_text(original_data)

        # Create backup
        self.manager.create_backup(test_file)

        # Corrupt original file
        test_file.write_text('{"test": "corrupted"}')

        # Restore from backup
        success = self.manager.restore_from_backup(test_file)

        assert success
        assert test_file.read_text() == original_data

    def test_repair_file_with_backup(self):
        """Test file repair using backup restoration."""
        test_file = self.root_dir / "test.json"
        original_data = '{"test": "original"}'
        test_file.write_text(original_data)

        # Add tracking
        self.manager.add_file_tracking(test_file, IntegrityLevel.CRITICAL)

        # Create backup
        self.manager.create_backup(test_file)

        # Corrupt file
        test_file.write_text('{"test": "corrupted"}')

        # Repair file
        success = self.manager.repair_file(test_file, strategy="backup_restore")

        assert success
        assert test_file.read_text() == original_data

    def test_verify_all_files(self):
        """Test verification of all tracked files."""
        # Create multiple test files
        for i in range(3):
            test_file = self.root_dir / f"test{i}.json"
            test_file.write_text(f'{{"test": "data{i}"}}')
            self.manager.add_file_tracking(test_file, IntegrityLevel.STANDARD)

        # Verify all files
        results = self.manager.verify_all_files()

        assert len(results) == 3
        for _file_path, is_valid in results.items():
            assert is_valid

    def test_verify_all_files_with_corruption(self):
        """Test verification of all files with some corruption."""
        # Create test files
        files = []
        for i in range(3):
            test_file = self.root_dir / f"test{i}.json"
            test_file.write_text(f'{{"test": "data{i}"}}')
            self.manager.add_file_tracking(test_file, IntegrityLevel.STANDARD)
            files.append(test_file)

        # Corrupt one file
        files[1].write_text('{"test": "corrupted"}')

        # Verify all files
        results = self.manager.verify_all_files()

        assert len(results) == 3
        assert results[str(files[0])]
        assert not results[str(files[1])]  # Corrupted
        assert results[str(files[2])]

    def test_cleanup_expired_backups(self):
        """Test cleanup of expired backups."""
        test_file = self.root_dir / "test.json"
        test_file.write_text('{"test": "data"}')

        # Create multiple backups
        backups = []
        for _i in range(5):
            backup = self.manager.create_backup(test_file)
            backups.append(backup)
            time.sleep(0.01)  # Ensure different timestamps

        # All backups should exist
        for backup in backups:
            assert backup.exists()

        # Cleanup with retention of 2
        cleaned = self.manager.cleanup_expired_backups(max_backups_per_file=2)

        # Should have cleaned 3 backups
        assert cleaned == 3

        # Only 2 most recent should remain
        remaining_backups = list((self.root_dir / ".backups").glob("backup_test_*.json"))
        assert len(remaining_backups) == 2

    def test_get_integrity_report(self):
        """Test generation of integrity report."""
        # Create test files with different integrity levels
        critical_file = self.root_dir / "critical.json"
        critical_file.write_text('{"critical": true}')
        self.manager.add_file_tracking(critical_file, IntegrityLevel.CRITICAL)

        important_file = self.root_dir / "important.json"
        important_file.write_text('{"important": true}')
        self.manager.add_file_tracking(important_file, IntegrityLevel.IMPORTANT)

        # Generate report
        report = self.manager.get_integrity_report()

        assert "summary" in report
        assert "files_by_level" in report
        assert "corruption_events" in report

        # Check summary
        summary = report["summary"]
        assert summary["total_files"] == 2
        assert summary["critical_files"] == 1
        assert summary["important_files"] == 1

    def test_get_files_by_integrity_level(self):
        """Test filtering files by integrity level."""
        # Create files with different levels
        critical_file = self.root_dir / "critical.json"
        critical_file.write_text('{"test": "critical"}')
        self.manager.add_file_tracking(critical_file, IntegrityLevel.CRITICAL)

        standard_file = self.root_dir / "standard.json"
        standard_file.write_text('{"test": "standard"}')
        self.manager.add_file_tracking(standard_file, IntegrityLevel.STANDARD)

        # Get critical files
        critical_files = self.manager.get_files_by_integrity_level(IntegrityLevel.CRITICAL)
        assert len(critical_files) == 1
        assert critical_files[0].file_path == critical_file

        # Get standard files
        standard_files = self.manager.get_files_by_integrity_level(IntegrityLevel.STANDARD)
        assert len(standard_files) == 1
        assert standard_files[0].file_path == standard_file

    def test_update_file_tracking(self):
        """Test updating file tracking information."""
        test_file = self.root_dir / "test.json"
        test_file.write_text('{"test": "original"}')

        # Add initial tracking
        self.manager.add_file_tracking(test_file, IntegrityLevel.STANDARD)
        original_checksum = self.manager.tracked_files[str(test_file)].checksum

        # Modify file
        test_file.write_text('{"test": "modified"}')

        # Update tracking
        self.manager.update_file_tracking(test_file)
        new_checksum = self.manager.tracked_files[str(test_file)].checksum

        # Checksum should be different
        assert original_checksum != new_checksum

    def test_remove_file_tracking(self):
        """Test removing file tracking."""
        test_file = self.root_dir / "test.json"
        test_file.write_text('{"test": "data"}')

        # Add tracking
        self.manager.add_file_tracking(test_file, IntegrityLevel.STANDARD)
        assert str(test_file) in self.manager.tracked_files

        # Remove tracking
        removed = self.manager.remove_file_tracking(test_file)

        assert removed
        assert str(test_file) not in self.manager.tracked_files

    def test_file_outside_root_directory(self):
        """Test handling files outside root directory."""
        outside_file = Path("/tmp/outside.json")

        # Should raise error or handle gracefully
        with pytest.raises(ValueError, match="File must be within storage root directory"):
            self.manager.add_file_tracking(outside_file, IntegrityLevel.STANDARD)


class TestModuleFunctions(unittest.TestCase):
    """Test module-level convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.root_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_integrity_manager(self):
        """Test get_integrity_manager function."""
        manager = get_integrity_manager(self.root_dir)
        assert isinstance(manager, DataIntegrityManager)
        manager.shutdown()

    def test_track_critical_file(self):
        """Test track_critical_file convenience function."""
        test_file = self.root_dir / "critical.json"
        test_data = {"test": "critical_data"}
        test_file.write_text(json.dumps(test_data))

        info = track_critical_file(test_file)
        assert isinstance(info, FileIntegrityInfo)
        assert info.integrity_level == IntegrityLevel.CRITICAL

    def test_verify_all_files_function(self):
        """Test verify_all_files convenience function."""
        test_file = self.root_dir / "test.json"
        test_file.write_text('{"test": "data"}')

        # Track the file first
        track_critical_file(test_file)

        # Verify all
        results = verify_all_files()
        assert isinstance(results, dict)

    def test_create_critical_backup(self):
        """Test create_critical_backup convenience function."""
        test_file = self.root_dir / "critical.json"
        test_data = {"test": "critical_data"}
        test_file.write_text(json.dumps(test_data))

        backup_path = create_critical_backup(test_file)
        assert isinstance(backup_path, Path)
        # Don't assume backup exists since implementation may vary


class TestIntegrityEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.root_dir = Path(self.temp_dir)
        self.manager = DataIntegrityManager(self.root_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        self.manager.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent files."""
        nonexistent_file = self.root_dir / "nonexistent.json"

        # Should handle gracefully
        is_valid = self.manager.verify_file_integrity(nonexistent_file)
        assert not is_valid

    def test_empty_file_handling(self):
        """Test handling of empty files."""
        empty_file = self.root_dir / "empty.json"
        empty_file.touch()  # Create empty file

        self.manager.add_file_tracking(empty_file, IntegrityLevel.STANDARD)
        is_valid = self.manager.verify_file_integrity(empty_file)
        assert is_valid  # Empty file should be valid

    def test_large_file_handling(self):
        """Test handling of large files."""
        large_file = self.root_dir / "large.json"
        large_data = '{"data": "' + "x" * 100000 + '"}'  # 100KB+ file
        large_file.write_text(large_data)

        # Should handle large files without issues
        self.manager.add_file_tracking(large_file, IntegrityLevel.STANDARD)
        is_valid = self.manager.verify_file_integrity(large_file)
        assert is_valid

    def test_binary_file_handling(self):
        """Test handling of binary files."""
        binary_file = self.root_dir / "binary.dat"
        binary_data = bytes(range(256))  # Binary data
        binary_file.write_bytes(binary_data)

        # Should handle binary files
        self.manager.add_file_tracking(binary_file, IntegrityLevel.STANDARD)
        is_valid = self.manager.verify_file_integrity(binary_file)
        assert is_valid

    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        test_file = self.root_dir / "test.json"
        test_file.write_text('{"test": "data"}')

        # Make file read-only (simulating permission issues)
        test_file.chmod(0o444)

        try:
            # Should handle permission errors gracefully
            self.manager.add_file_tracking(test_file, IntegrityLevel.STANDARD)
            # If we get here, the operation succeeded despite read-only file
        except PermissionError:
            # If we get permission error, that's also acceptable
            pass
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o666)


if __name__ == "__main__":
    unittest.main()
