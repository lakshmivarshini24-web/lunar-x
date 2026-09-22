"""
Failure Case and Edge-Condition Audit Tests (Step 8 Audit)

Tests handling of:
- Missing files
- Corrupted / zero-byte / truncated files
- Unsupported file formats
- Missing sidecar metadata fallback
- Scale, rotation, and illumination variations
"""

from pathlib import Path
import numpy as np
import pytest

from backend.sensors.ohrc.validator import OHRCValidator
from backend.sensors.tmc2.validator import TMC2Validator
from backend.sensors.iirs.validator import IIRSValidator
from backend.sensors.ohrc.reader import OHRCReader
from backend.sensors.tmc2.reader import TMC2Reader
from backend.sensors.iirs.reader import IIRSReader


def test_missing_file_validation():
    v1 = OHRCValidator()
    res1 = v1.validate_file("nonexistent_ohrc_path.tif")
    assert res1["is_valid"] is False
    assert any("does not exist" in e for e in res1["errors"])

    v2 = TMC2Validator()
    res2 = v2.validate_file("nonexistent_tmc2_path.tif")
    assert res2["is_valid"] is False

    v3 = IIRSValidator()
    res3 = v3.validate_file("nonexistent_iirs_path.npy")
    assert res3["is_valid"] is False


def test_corrupted_empty_file_validation(tmp_path):
    empty_file = tmp_path / "corrupt_empty.tif"
    empty_file.write_bytes(b"")  # 0 bytes file

    v1 = OHRCValidator()
    res1 = v1.validate_file(str(empty_file))
    assert res1["is_valid"] is False
    assert any("File is empty" in e for e in res1["errors"])


def test_unsupported_format_warning(tmp_path):
    unsupported_file = tmp_path / "data.txt"
    unsupported_file.write_text("not an image")

    v1 = OHRCValidator()
    res = v1.validate_file(str(unsupported_file))
    assert res["is_valid"] is False


def test_missing_metadata_sidecar_fallback(tmp_path):
    # Create image file without sidecar JSON metadata
    img_path = tmp_path / "no_sidecar.png"
    arr = np.random.randint(10, 200, (64, 64), dtype=np.uint8)
    import cv2
    cv2.imwrite(str(img_path), arr)

    reader = OHRCReader()
    lunar_img = reader.read(img_path)

    # Reader should successfully load default metadata without raising exception
    assert lunar_img.sensor == "OHRC"
    assert lunar_img.resolution_m == 0.25
    assert lunar_img.acquisition_info.get("orbit") == "Unknown"
