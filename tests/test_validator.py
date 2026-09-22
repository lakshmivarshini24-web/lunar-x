"""
Unit tests for Module 1: OHRC Data Validator and Dataset Manifest.
"""

import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest

from backend.sensors.ohrc.validator import OHRCValidator
from backend.core.manifest import DatasetManifest


def test_validator_valid_image(tmp_path):
    # Create valid synthetic 8-bit image
    img_path = tmp_path / "valid_ohrc.png"
    img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    validator = OHRCValidator()
    res = validator.validate_file(str(img_path))

    assert res["is_valid"] is True
    assert res["dimensions"] == (128, 128)
    assert res["bit_depth"] == 8
    assert res["channels"] == 1
    assert len(res["errors"]) == 0


def test_validator_nonexistent_file():
    validator = OHRCValidator()
    res = validator.validate_file("nonexistent_ohrc_image.tif")

    assert res["is_valid"] is False
    assert len(res["errors"]) > 0
    assert "File does not exist" in res["errors"][0]


def test_validator_too_small_dimensions(tmp_path):
    img_path = tmp_path / "tiny.png"
    img = np.zeros((16, 16), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    validator = OHRCValidator({"ohrc": {"min_dimensions": [64, 64]}})
    res = validator.validate_file(str(img_path))

    assert res["is_valid"] is False
    assert any("smaller than minimum" in err for err in res["errors"])


def test_dataset_manifest(tmp_path):
    manifest = DatasetManifest("test_manifest.json", manifests_dir=str(tmp_path))
    val_res = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "dimensions": (256, 256),
        "bit_depth": 8,
        "channels": 1,
        "sensor": "OHRC",
    }
    dummy_file = tmp_path / "dummy.tif"
    dummy_file.write_text("fake raster content")

    manifest.add_entry(str(dummy_file), val_res)
    manifest.save()

    assert manifest.manifest_path.exists()
    assert manifest.data["total_files"] == 1
    assert manifest.data["valid_files"] == 1
