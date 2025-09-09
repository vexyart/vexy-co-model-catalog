#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["loguru", "rich"]
# ///
# this_file: scripts/test_integrity.py

"""
Test script to verify the data integrity system functionality.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console

from vexy_co_model_catalog.core.integrity import (
    ChecksumAlgorithm,
    DataIntegrityManager,
    IntegrityLevel,
    create_critical_backup,
    track_critical_file,
    verify_all_files,
)

console = Console()


def test_basic_integrity_tracking():
    """Test basic file tracking and checksum calculation."""
    console.print("[bold cyan]Testing Basic Integrity Tracking[/bold cyan]")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        manager = DataIntegrityManager(temp_path)

        # Create a test file
        test_file = temp_path / "test.json"
        test_data = {"message": "Hello, World!", "version": 1}

        with open(test_file, "w") as f:
            json.dump(test_data, f, indent=2)

        # Add to tracking
        info = manager.add_file_tracking(test_file, IntegrityLevel.CRITICAL, ["test"])

        # Verify initial state
        checksum_matches = info.checksum == manager.calculate_checksum(test_file)
        console.print(f"Initial checksum calculation: {'✅ PASS' if checksum_matches else '❌ FAIL'}")

        # Verify integrity
        integrity_ok = manager.verify_file_integrity(test_file)
        console.print(f"Integrity verification: {'✅ PASS' if integrity_ok else '❌ FAIL'}")

        manager.shutdown()

    console.print()


def test_corruption_detection():
    """Test corruption detection and handling."""
    console.print("[bold cyan]Testing Corruption Detection[/bold cyan]")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        manager = DataIntegrityManager(temp_path)

        # Create and track a test file
        test_file = temp_path / "corrupted.json"
        test_data = {"important": "data", "value": 42}

        with open(test_file, "w") as f:
            json.dump(test_data, f)

        manager.add_file_tracking(test_file, IntegrityLevel.IMPORTANT, ["test"])

        # Verify file is initially OK
        initial_ok = manager.verify_file_integrity(test_file)
        console.print(f"Initial integrity check: {'✅ PASS' if initial_ok else '❌ FAIL'}")

        # Corrupt the file
        with open(test_file, "w") as f:
            f.write("corrupted content")

        # Verify corruption is detected
        corrupted_detected = not manager.verify_file_integrity(test_file)
        console.print(f"Corruption detection: {'✅ PASS' if corrupted_detected else '❌ FAIL'}")

        # Check corruption events
        report = manager.get_integrity_report()
        corruption_events = report["summary"]["total_corruption_events"]
        console.print(f"Corruption events logged: {'✅ PASS' if corruption_events > 0 else '❌ FAIL'}")

        manager.shutdown()

    console.print()


def test_backup_and_restore():
    """Test backup creation and restoration."""
    console.print("[bold cyan]Testing Backup and Restore[/bold cyan]")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        manager = DataIntegrityManager(temp_path)

        # Create a test file
        test_file = temp_path / "backup_test.json"
        original_data = {"version": 1, "data": "original"}

        with open(test_file, "w") as f:
            json.dump(original_data, f, indent=2)

        # Create backup
        backup_path = manager.create_backup(test_file)
        backup_created = backup_path.exists()
        console.print(f"Backup creation: {'✅ PASS' if backup_created else '❌ FAIL'}")

        # Verify backup content
        if backup_created:
            with open(backup_path) as f:
                backup_data = json.load(f)
            backup_matches = backup_data == original_data
            console.print(f"Backup content verification: {'✅ PASS' if backup_matches else '❌ FAIL'}")

        manager.shutdown()

    console.print()


def test_integrity_levels():
    """Test different integrity levels and their behavior."""
    console.print("[bold cyan]Testing Integrity Levels[/bold cyan]")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        manager = DataIntegrityManager(temp_path)

        # Create files with different integrity levels
        levels_data = [
            (IntegrityLevel.CRITICAL, "critical.json", {"level": "critical"}),
            (IntegrityLevel.IMPORTANT, "important.json", {"level": "important"}),
            (IntegrityLevel.STANDARD, "standard.json", {"level": "standard"}),
            (IntegrityLevel.MONITORING_ONLY, "monitoring.json", {"level": "monitoring"}),
        ]

        for level, filename, data in levels_data:
            file_path = temp_path / filename
            with open(file_path, "w") as f:
                json.dump(data, f)

            manager.add_file_tracking(file_path, level, [level.value])

        # Get integrity report
        report = manager.get_integrity_report()

        # Check that all levels are represented
        integrity_levels = report["integrity_levels"]
        all_levels_tracked = all(integrity_levels.get(level.value, 0) == 1 for level, _, _ in levels_data)

        console.print(f"All integrity levels tracked: {'✅ PASS' if all_levels_tracked else '❌ FAIL'}")
        console.print(f"  Tracked files by level: {integrity_levels}")

        manager.shutdown()

    console.print()


def test_checksum_algorithms():
    """Test different checksum algorithms."""
    console.print("[bold cyan]Testing Checksum Algorithms[/bold cyan]")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        manager = DataIntegrityManager(temp_path)

        # Create a test file
        test_file = temp_path / "algorithm_test.txt"
        test_content = "This is test content for checksum algorithms."

        with open(test_file, "w") as f:
            f.write(test_content)

        # Test different algorithms
        algorithms = [ChecksumAlgorithm.MD5, ChecksumAlgorithm.SHA1, ChecksumAlgorithm.SHA256, ChecksumAlgorithm.SHA512]
        checksums = {}

        for algorithm in algorithms:
            checksum = manager.calculate_checksum(test_file, algorithm)
            checksums[algorithm.value] = checksum

            # Verify checksum length is reasonable for algorithm
            expected_lengths = {"md5": 32, "sha1": 40, "sha256": 64, "sha512": 128}
            expected_length = expected_lengths[algorithm.value]
            length_ok = len(checksum) == expected_length

            console.print(
                f"{algorithm.value.upper()} checksum: {'✅ PASS' if length_ok else '❌ FAIL'} ({len(checksum)} chars)"
            )

        # Verify all checksums are different (they should be for different algorithms)
        unique_checksums = len(set(checksums.values())) == len(checksums)
        console.print(f"Algorithm uniqueness: {'✅ PASS' if unique_checksums else '❌ FAIL'}")

        manager.shutdown()

    console.print()


def test_convenience_functions():
    """Test convenience functions for common operations."""
    console.print("[bold cyan]Testing Convenience Functions[/bold cyan]")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        critical_file = temp_path / "critical.json"
        with open(critical_file, "w") as f:
            json.dump({"critical": True}, f)

        # Test track_critical_file function
        try:
            info = track_critical_file(critical_file, ["convenience_test"])
            critical_tracking_ok = info.integrity_level == IntegrityLevel.CRITICAL
            console.print(f"Critical file tracking: {'✅ PASS' if critical_tracking_ok else '❌ FAIL'}")
        except Exception as e:
            console.print(f"Critical file tracking: ❌ FAIL - {e}")

        # Test create_critical_backup function
        try:
            backup_path = create_critical_backup(critical_file)
            backup_ok = backup_path.exists()
            console.print(f"Critical backup creation: {'✅ PASS' if backup_ok else '❌ FAIL'}")
        except Exception as e:
            console.print(f"Critical backup creation: ❌ FAIL - {e}")

        # Test verify_all_files function
        try:
            results = verify_all_files(auto_repair=False)
            verification_ok = len(results) > 0 and all(results.values())
            console.print(f"Verify all files: {'✅ PASS' if verification_ok else '❌ FAIL'}")
        except Exception as e:
            console.print(f"Verify all files: ❌ FAIL - {e}")

    console.print()


def test_report_generation():
    """Test integrity report generation."""
    console.print("[bold cyan]Testing Report Generation[/bold cyan]")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        manager = DataIntegrityManager(temp_path)

        # Create and track several files
        for i in range(3):
            test_file = temp_path / f"report_test_{i}.json"
            with open(test_file, "w") as f:
                json.dump({"file": i, "data": f"test_{i}"}, f)

            level = [IntegrityLevel.CRITICAL, IntegrityLevel.IMPORTANT, IntegrityLevel.STANDARD][i]
            manager.add_file_tracking(test_file, level, ["report_test"])

        # Generate report
        report = manager.get_integrity_report()

        # Verify report structure
        required_keys = ["summary", "integrity_levels", "recent_events"]
        report_structure_ok = all(key in report for key in required_keys)
        console.print(f"Report structure: {'✅ PASS' if report_structure_ok else '❌ FAIL'}")

        # Verify summary data
        summary = report["summary"]
        total_files_ok = summary["total_tracked_files"] == 3
        console.print(f"Summary data accuracy: {'✅ PASS' if total_files_ok else '❌ FAIL'}")

        # Verify integrity levels breakdown
        levels = report["integrity_levels"]
        levels_breakdown_ok = levels["critical"] == 1 and levels["important"] == 1 and levels["standard"] == 1
        console.print(f"Integrity levels breakdown: {'✅ PASS' if levels_breakdown_ok else '❌ FAIL'}")

        manager.shutdown()

    console.print()


if __name__ == "__main__":
    console.print("[bold green]Data Integrity System Test Suite[/bold green]")
    console.print()

    test_basic_integrity_tracking()
    test_corruption_detection()
    test_backup_and_restore()
    test_integrity_levels()
    test_checksum_algorithms()
    test_convenience_functions()
    test_report_generation()

    console.print("[bold green]✅ Data integrity testing complete![/bold green]")
