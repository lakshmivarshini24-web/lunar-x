"""
Dataset Manifest Management

Maintains machine-readable JSON manifests of dataset files, metadata, validation status,
and summary statistics across sensor collections.
"""

import json
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class DatasetManifest:
    """
    Manages structured dataset manifests recording dataset file paths, metadata,
    validation statuses, and dataset-level statistics.
    """

    def __init__(self, manifest_name: str = "ohrc_manifest.json", manifests_dir: str = "data/manifests"):
        self.manifests_dir = Path(manifests_dir)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.manifests_dir / manifest_name

        self.data: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_files": 0,
            "valid_files": 0,
            "corrupted_files": 0,
            "files": [],
            "statistics": {
                "total_size_bytes": 0,
                "bit_depth_distribution": {},
                "format_distribution": {},
                "sensor_distribution": {},
            },
        }

        if self.manifest_path.exists():
            self.load()

    def add_entry(self, file_path: str, validation_result: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):

        path_obj = Path(file_path)
        file_size = path_obj.stat().st_size if path_obj.exists() else 0
        ext = path_obj.suffix.lower()

        entry = {
            "file_path": str(path_obj),
            "file_name": path_obj.name,
            "file_size_bytes": file_size,
            "extension": ext,
            "is_valid": validation_result.get("is_valid", False),
            "validation_errors": validation_result.get("errors", []),
            "validation_warnings": validation_result.get("warnings", []),
            "dimensions": validation_result.get("dimensions", None),
            "bit_depth": validation_result.get("bit_depth", None),
            "channels": validation_result.get("channels", 1),
            "sensor": validation_result.get("sensor", "OHRC"),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Remove previous entry for same path if exists
        self.data["files"] = [f for f in self.data["files"] if f["file_path"] != str(path_obj)]
        self.data["files"].append(entry)
        self.recalculate_statistics()

    def recalculate_statistics(self):

        files = self.data["files"]
        self.data["total_files"] = len(files)
        self.data["valid_files"] = sum(1 for f in files if f["is_valid"])
        self.data["corrupted_files"] = self.data["total_files"] - self.data["valid_files"]

        total_bytes = sum(f["file_size_bytes"] for f in files)
        bit_depths: Dict[str, int] = {}
        formats: Dict[str, int] = {}
        sensors: Dict[str, int] = {}

        for f in files:
            bd = str(f.get("bit_depth", "unknown"))
            bit_depths[bd] = bit_depths.get(bd, 0) + 1

            ext = f.get("extension", "unknown")
            formats[ext] = formats.get(ext, 0) + 1

            s = f.get("sensor", "OHRC")
            sensors[s] = sensors.get(s, 0) + 1

        self.data["statistics"] = {
            "total_size_bytes": total_bytes,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "bit_depth_distribution": bit_depths,
            "format_distribution": formats,
            "sensor_distribution": sensors,
        }
        self.data["updated_at"] = datetime.now().isoformat()

    def save(self):

        self.recalculate_statistics()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def load(self):

        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def get_summary(self) -> Dict[str, Any]:

        return {
            "manifest_file": str(self.manifest_path),
            "total_files": self.data["total_files"],
            "valid_files": self.data["valid_files"],
            "corrupted_files": self.data["corrupted_files"],
            "statistics": self.data["statistics"],
        }
