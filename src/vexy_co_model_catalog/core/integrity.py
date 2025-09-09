# this_file: src/vexy_co_model_catalog/core/integrity.py

"""
Data integrity management with checksums, corruption detection, and automatic repair.
Ensures reliability and consistency of critical data files across the model catalog system.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable


class IntegrityLevel(Enum):
    """Integrity protection levels for different file types."""

    CRITICAL = "critical"  # Essential files requiring immediate repair
    IMPORTANT = "important"  # Important files with backup strategies
    STANDARD = "standard"  # Regular files with basic checking
    MONITORING_ONLY = "monitoring"  # Check only, no auto-repair


class ChecksumAlgorithm(Enum):
    """Supported checksum algorithms."""

    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"


class CorruptionSeverity(Enum):
    """Severity levels for detected corruption."""

    MINOR = "minor"  # Non-critical data loss
    MODERATE = "moderate"  # Important data affected
    SEVERE = "severe"  # Critical system data corrupted
    CATASTROPHIC = "catastrophic"  # Complete data loss


@dataclass
class FileIntegrityInfo:
    """Information about a file's integrity status."""

    file_path: Path
    checksum: str
    algorithm: ChecksumAlgorithm
    file_size: int
    last_modified: float
    last_verified: float
    integrity_level: IntegrityLevel
    verification_count: int = 0
    corruption_count: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "file_path": str(self.file_path),
            "checksum": self.checksum,
            "algorithm": self.algorithm.value,
            "file_size": self.file_size,
            "last_modified": self.last_modified,
            "last_verified": self.last_verified,
            "integrity_level": self.integrity_level.value,
            "verification_count": self.verification_count,
            "corruption_count": self.corruption_count,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileIntegrityInfo:
        """Create from dictionary for persistence."""
        return cls(
            file_path=Path(data["file_path"]),
            checksum=data["checksum"],
            algorithm=ChecksumAlgorithm(data["algorithm"]),
            file_size=data["file_size"],
            last_modified=data["last_modified"],
            last_verified=data["last_verified"],
            integrity_level=IntegrityLevel(data["integrity_level"]),
            verification_count=data.get("verification_count", 0),
            corruption_count=data.get("corruption_count", 0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CorruptionEvent:
    """Record of a detected corruption event."""

    file_path: Path
    detected_at: float
    severity: CorruptionSeverity
    expected_checksum: str
    actual_checksum: str
    repair_attempted: bool = False
    repair_successful: bool = False
    repair_method: str | None = None
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "file_path": str(self.file_path),
            "detected_at": self.detected_at,
            "severity": self.severity.value,
            "expected_checksum": self.expected_checksum,
            "actual_checksum": self.actual_checksum,
            "repair_attempted": self.repair_attempted,
            "repair_successful": self.repair_successful,
            "repair_method": self.repair_method,
            "details": self.details,
        }


class DataIntegrityManager:
    """
    Comprehensive data integrity management system with checksums,
    corruption detection, and automatic repair capabilities.
    """

    def __init__(self, storage_root: Path, backup_root: Path | None = None) -> None:
        """Initialize the data integrity manager."""
        self.storage_root = Path(storage_root)
        self.backup_root = Path(backup_root) if backup_root else self.storage_root / ".backups"

        # Integrity tracking
        self.integrity_db_path = self.storage_root / ".integrity" / "integrity.json"
        self.corruption_log_path = self.storage_root / ".integrity" / "corruption_log.json"

        # Thread safety
        self._lock = threading.RLock()

        # File tracking
        self._tracked_files: dict[str, FileIntegrityInfo] = {}
        self._corruption_events: list[CorruptionEvent] = []

        # Configuration
        self.default_algorithm = ChecksumAlgorithm.SHA256
        self.verification_interval = 3600  # 1 hour
        self.backup_retention_days = 30

        # Background monitoring
        self._monitor_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Repair strategies
        self._repair_strategies: dict[str, Callable] = {
            "backup_restore": self._repair_from_backup,
            "regenerate": self._repair_by_regeneration,
            "validate_and_fix": self._repair_validate_and_fix,
        }

        # Setup directories and load existing data
        self._setup_directories()
        self._load_integrity_database()
        self._start_monitoring()

        logger.debug(f"Initialized data integrity manager for {storage_root}")

    def _setup_directories(self) -> None:
        """Setup required directories for integrity management."""
        self.integrity_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_root.mkdir(parents=True, exist_ok=True)

    def _load_integrity_database(self) -> None:
        """Load existing integrity database from disk."""
        if not self.integrity_db_path.exists():
            return

        try:
            with open(self.integrity_db_path) as f:
                data = json.load(f)

            # Load tracked files
            for file_data in data.get("tracked_files", []):
                info = FileIntegrityInfo.from_dict(file_data)
                self._tracked_files[str(info.file_path)] = info

            # Load corruption events
            for event_data in data.get("corruption_events", []):
                event = CorruptionEvent(
                    file_path=Path(event_data["file_path"]),
                    detected_at=event_data["detected_at"],
                    severity=CorruptionSeverity(event_data["severity"]),
                    expected_checksum=event_data["expected_checksum"],
                    actual_checksum=event_data["actual_checksum"],
                    repair_attempted=event_data.get("repair_attempted", False),
                    repair_successful=event_data.get("repair_successful", False),
                    repair_method=event_data.get("repair_method"),
                    details=event_data.get("details", ""),
                )
                self._corruption_events.append(event)

            logger.debug(
                f"Loaded {len(self._tracked_files)} tracked files and {len(self._corruption_events)} corruption events"
            )

        except Exception as e:
            logger.warning(f"Failed to load integrity database: {e}")

    def _save_integrity_database(self) -> None:
        """Save integrity database to disk."""
        try:
            data = {
                "saved_at": time.time(),
                "tracked_files": [info.to_dict() for info in self._tracked_files.values()],
                "corruption_events": [
                    event.to_dict() for event in self._corruption_events[-1000:]
                ],  # Keep last 1000 events
            }

            # Atomic write
            temp_path = self.integrity_db_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)

            temp_path.replace(self.integrity_db_path)

        except Exception as e:
            logger.warning(f"Failed to save integrity database: {e}")

    def _start_monitoring(self) -> None:
        """Start background monitoring thread."""
        self._monitor_thread = threading.Thread(target=self._monitoring_worker, name="integrity-monitor", daemon=True)
        self._monitor_thread.start()

    def _monitoring_worker(self) -> None:
        """Background worker for periodic integrity verification."""
        while not self._shutdown_event.wait(self.verification_interval):
            try:
                self._perform_periodic_verification()
            except Exception as e:
                logger.warning(f"Integrity monitoring error: {e}")

    def _perform_periodic_verification(self) -> None:
        """Perform periodic verification of tracked files."""
        current_time = time.time()
        files_to_verify = []

        with self._lock:
            for _file_path, info in self._tracked_files.items():
                # Check if file needs verification
                if (current_time - info.last_verified) >= self.verification_interval:
                    files_to_verify.append(info)

        if not files_to_verify:
            return

        logger.debug(f"Performing periodic verification of {len(files_to_verify)} files")

        for info in files_to_verify:
            try:
                self.verify_file_integrity(info.file_path, auto_repair=True)
            except Exception as e:
                logger.warning(f"Failed to verify {info.file_path}: {e}")

    def calculate_checksum(self, file_path: Path, algorithm: ChecksumAlgorithm = None) -> str:
        """Calculate checksum for a file."""
        if algorithm is None:
            algorithm = self.default_algorithm

        hash_func = hashlib.new(algorithm.value)

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)

            return hash_func.hexdigest()

        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            raise

    def add_file_tracking(
        self,
        file_path: Path,
        integrity_level: IntegrityLevel = IntegrityLevel.STANDARD,
        tags: list[str] | None = None,
        algorithm: ChecksumAlgorithm | None = None,
    ) -> FileIntegrityInfo:
        """Add a file to integrity tracking."""
        # Validate that file is within storage root
        try:
            file_path.relative_to(self.storage_root)
        except ValueError:
            msg = f"File must be within storage root directory: {file_path} not in {self.storage_root}"
            raise ValueError(msg)

        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        if algorithm is None:
            algorithm = self.default_algorithm

        if tags is None:
            tags = []

        # Calculate initial checksum
        checksum = self.calculate_checksum(file_path, algorithm)
        file_stat = file_path.stat()

        # Create integrity info
        info = FileIntegrityInfo(
            file_path=file_path,
            checksum=checksum,
            algorithm=algorithm,
            file_size=file_stat.st_size,
            last_modified=file_stat.st_mtime,
            last_verified=time.time(),
            integrity_level=integrity_level,
            tags=tags,
        )

        with self._lock:
            self._tracked_files[str(file_path)] = info
            self._save_integrity_database()

        logger.debug(f"Added integrity tracking for {file_path} with {integrity_level.value} level")
        return info

    def remove_file_tracking(self, file_path: Path) -> bool:
        """Remove a file from integrity tracking."""
        with self._lock:
            if str(file_path) in self._tracked_files:
                del self._tracked_files[str(file_path)]
                self._save_integrity_database()
                logger.debug(f"Removed integrity tracking for {file_path}")
                return True
            return False

    def verify_file_integrity(self, file_path: Path, auto_repair: bool = False) -> bool:
        """Verify the integrity of a tracked file."""
        with self._lock:
            info = self._tracked_files.get(str(file_path))
            if not info:
                msg = f"File not tracked: {file_path}"
                raise ValueError(msg)

        if not file_path.exists():
            self._handle_missing_file(info, auto_repair)
            return False

        # Check if file was modified
        file_stat = file_path.stat()
        if file_stat.st_mtime > info.last_modified:
            logger.info(f"File {file_path} was modified, updating checksum")
            new_checksum = self.calculate_checksum(file_path, info.algorithm)

            with self._lock:
                info.checksum = new_checksum
                info.file_size = file_stat.st_size
                info.last_modified = file_stat.st_mtime
                info.last_verified = time.time()
                info.verification_count += 1
                self._save_integrity_database()

            return True

        # Verify checksum
        current_checksum = self.calculate_checksum(file_path, info.algorithm)

        with self._lock:
            info.last_verified = time.time()
            info.verification_count += 1

        if current_checksum == info.checksum:
            # File is intact
            with self._lock:
                self._save_integrity_database()
            return True
        # Corruption detected
        self._handle_corruption(info, current_checksum, auto_repair)
        return False

    def _handle_missing_file(self, info: FileIntegrityInfo, auto_repair: bool) -> None:
        """Handle a missing file."""
        severity = (
            CorruptionSeverity.SEVERE
            if info.integrity_level == IntegrityLevel.CRITICAL
            else CorruptionSeverity.MODERATE
        )

        event = CorruptionEvent(
            file_path=info.file_path,
            detected_at=time.time(),
            severity=severity,
            expected_checksum=info.checksum,
            actual_checksum="FILE_MISSING",
            details="File is missing from expected location",
        )

        logger.error(f"Missing file detected: {info.file_path}")

        if auto_repair and info.integrity_level in [IntegrityLevel.CRITICAL, IntegrityLevel.IMPORTANT]:
            success = self._attempt_repair(info, event, "missing_file")
            event.repair_attempted = True
            event.repair_successful = success

        with self._lock:
            self._corruption_events.append(event)
            self._save_integrity_database()

    def _handle_corruption(self, info: FileIntegrityInfo, actual_checksum: str, auto_repair: bool) -> None:
        """Handle detected file corruption."""
        severity = self._assess_corruption_severity(info)

        event = CorruptionEvent(
            file_path=info.file_path,
            detected_at=time.time(),
            severity=severity,
            expected_checksum=info.checksum,
            actual_checksum=actual_checksum,
            details="Checksum mismatch detected",
        )

        logger.error(f"Corruption detected in {info.file_path}: expected {info.checksum}, got {actual_checksum}")

        with self._lock:
            info.corruption_count += 1

        if auto_repair and info.integrity_level in [IntegrityLevel.CRITICAL, IntegrityLevel.IMPORTANT]:
            success = self._attempt_repair(info, event, "corruption")
            event.repair_attempted = True
            event.repair_successful = success

        with self._lock:
            self._corruption_events.append(event)
            self._save_integrity_database()

    def _assess_corruption_severity(self, info: FileIntegrityInfo) -> CorruptionSeverity:
        """Assess the severity of corruption based on file importance."""
        if info.integrity_level == IntegrityLevel.CRITICAL:
            return CorruptionSeverity.SEVERE
        if info.integrity_level == IntegrityLevel.IMPORTANT:
            return CorruptionSeverity.MODERATE
        return CorruptionSeverity.MINOR

    def _attempt_repair(self, info: FileIntegrityInfo, event: CorruptionEvent, issue_type: str) -> bool:
        """Attempt to repair a corrupted or missing file."""
        logger.info(f"Attempting repair for {info.file_path}")

        # Try repair strategies in order of preference
        repair_strategies = ["backup_restore", "validate_and_fix", "regenerate"]

        for strategy in repair_strategies:
            if strategy in self._repair_strategies:
                try:
                    success = self._repair_strategies[strategy](info, issue_type)
                    if success:
                        event.repair_method = strategy
                        logger.info(f"Successfully repaired {info.file_path} using {strategy}")
                        return True
                except Exception as e:
                    logger.warning(f"Repair strategy {strategy} failed for {info.file_path}: {e}")

        logger.error(f"All repair strategies failed for {info.file_path}")
        return False

    def _repair_from_backup(self, info: FileIntegrityInfo, _issue_type: str) -> bool:
        """Attempt to restore file from backup."""
        backup_path = self._find_backup_file(info.file_path)

        if not backup_path or not backup_path.exists():
            return False

        try:
            # Verify backup integrity
            backup_checksum = self.calculate_checksum(backup_path, info.algorithm)
            if backup_checksum != info.checksum:
                logger.warning(f"Backup file {backup_path} also corrupted")
                return False

            # Restore from backup
            shutil.copy2(backup_path, info.file_path)

            # Verify restoration
            restored_checksum = self.calculate_checksum(info.file_path, info.algorithm)
            return restored_checksum == info.checksum

        except Exception as e:
            logger.warning(f"Backup restoration failed: {e}")
            return False

    def _repair_by_regeneration(self, info: FileIntegrityInfo, _issue_type: str) -> bool:
        """Attempt to regenerate file content (for generated files)."""
        # This would be implemented based on file type and generation logic
        # For now, just a placeholder that checks if file can be regenerated

        if "generated" not in info.tags:
            return False

        # Placeholder: would implement specific regeneration logic
        logger.debug(f"Regeneration not implemented for {info.file_path}")
        return False

    def _repair_validate_and_fix(self, info: FileIntegrityInfo, _issue_type: str) -> bool:
        """Attempt to validate and fix file content."""
        if not info.file_path.exists():
            return False

        # For JSON files, try to validate and fix
        if info.file_path.suffix == ".json":
            try:
                with open(info.file_path) as f:
                    data = json.load(f)

                # If we can load it, write it back to fix formatting
                with open(info.file_path, "w") as f:
                    json.dump(data, f, indent=2)

                # Verify the fix
                new_checksum = self.calculate_checksum(info.file_path, info.algorithm)
                if new_checksum == info.checksum:
                    return True

            except json.JSONDecodeError:
                logger.warning(f"JSON file {info.file_path} is corrupted beyond repair")

        return False

    def _find_backup_file(self, file_path: Path) -> Path | None:
        """Find the most recent backup for a file."""
        relative_path = file_path.relative_to(self.storage_root)
        backup_dir = self.backup_root / relative_path.parent

        if not backup_dir.exists():
            return None

        # Look for timestamped backups
        backup_pattern = f"{file_path.stem}_*{file_path.suffix}"
        backup_files = list(backup_dir.glob(backup_pattern))

        if not backup_files:
            return None

        # Return most recent backup
        return max(backup_files, key=lambda p: p.stat().st_mtime)

    def create_backup(self, file_path: Path) -> Path:
        """Create a timestamped backup of a file."""
        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        relative_path = file_path.relative_to(self.storage_root)
        backup_dir = self.backup_root / relative_path.parent
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped backup name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name

        # Copy file to backup location
        shutil.copy2(file_path, backup_path)

        logger.debug(f"Created backup: {backup_path}")
        return backup_path

    def get_integrity_report(self) -> dict[str, Any]:
        """Generate comprehensive integrity report."""
        with self._lock:
            total_files = len(self._tracked_files)
            corrupted_files = len([e for e in self._corruption_events if not e.repair_successful])

            # Calculate statistics
            verification_stats = {}
            corruption_stats = {}

            for info in self._tracked_files.values():
                level = info.integrity_level.value
                verification_stats[level] = verification_stats.get(level, 0) + 1

                if info.corruption_count > 0:
                    corruption_stats[level] = corruption_stats.get(level, 0) + 1

            recent_events = [
                event
                for event in self._corruption_events
                if time.time() - event.detected_at < 86400  # Last 24 hours
            ]

            return {
                "summary": {
                    "total_tracked_files": total_files,
                    "currently_corrupted": corrupted_files,
                    "total_corruption_events": len(self._corruption_events),
                    "recent_events_24h": len(recent_events),
                },
                "verification_stats": verification_stats,
                "corruption_stats": corruption_stats,
                "recent_events": [event.to_dict() for event in recent_events],
                "integrity_levels": {
                    level.value: len([f for f in self._tracked_files.values() if f.integrity_level == level])
                    for level in IntegrityLevel
                },
            }

    def cleanup_old_backups(self, retention_days: int | None = None) -> int:
        """Clean up old backup files."""
        if retention_days is None:
            retention_days = self.backup_retention_days

        cutoff_time = time.time() - (retention_days * 86400)
        removed_count = 0

        for backup_file in self.backup_root.rglob("*"):
            if backup_file.is_file() and backup_file.stat().st_mtime < cutoff_time:
                try:
                    backup_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove old backup {backup_file}: {e}")

        logger.debug(f"Cleaned up {removed_count} old backup files")
        return removed_count

    def shutdown(self) -> None:
        """Gracefully shutdown the integrity manager."""
        logger.debug("Shutting down data integrity manager")

        # Signal monitoring thread to stop
        self._shutdown_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        # Final save
        with self._lock:
            self._save_integrity_database()


# Global integrity manager
_global_integrity_manager: DataIntegrityManager | None = None


def get_integrity_manager(storage_root: Path | None = None) -> DataIntegrityManager:
    """Get the global integrity manager instance."""
    global _global_integrity_manager
    if _global_integrity_manager is None:
        if storage_root is None:
            storage_root = Path.cwd()
        _global_integrity_manager = DataIntegrityManager(storage_root)
    return _global_integrity_manager


# Convenience functions for common operations
def track_critical_file(file_path: Path, tags: list[str] | None = None) -> FileIntegrityInfo:
    """Track a critical file with highest integrity protection."""
    manager = get_integrity_manager()
    return manager.add_file_tracking(file_path, IntegrityLevel.CRITICAL, tags)


def verify_all_files(auto_repair: bool = False) -> dict[str, bool]:
    """Verify integrity of all tracked files."""
    manager = get_integrity_manager()
    results = {}

    for file_path in manager._tracked_files:
        try:
            results[file_path] = manager.verify_file_integrity(Path(file_path), auto_repair)
        except Exception as e:
            logger.error(f"Failed to verify {file_path}: {e}")
            results[file_path] = False

    return results


def create_critical_backup(file_path: Path) -> Path:
    """Create backup of a critical file before modification."""
    manager = get_integrity_manager()
    return manager.create_backup(file_path)
