"""
Unit tests for Module 8: TMC-2 Validator, Reader, and Preprocessor.
"""

import json
from pathlib import Path
import numpy as np
import cv2
import pytest

from backend.core.lunar_image import LunarImage
from backend.sensors.tmc2.validator import TMC2Validator
from backend.sensors.tmc2.reader import TMC2Reader, load_tmc2
from backend.sensors.tmc2.preprocessor import TMC2Preprocessor


def test_tmc2_validator(tmp_path):
    img_path = tmp_path / "valid_tmc2.tif"
    img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    validator = TMC2Validator()
    res = validator.validate_file(str(img_path))

    assert res["is_valid"] is True
    assert res["dimensions"] == (128, 128)
    assert res["sensor"] == "TMC-2"


def test_tmc2_reader(tmp_path):
    img_path = tmp_path / "test_tmc2.png"
    img = np.random.randint(30, 220, (200, 200), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    meta_path = tmp_path / "test_tmc2.json"
    sidecar_data = {
        "acquisition_info": {"orbit": 105, "instrument": "TMC-2", "view_angle": "Fore"},
        "sun_angle_info": {"sun_elevation_deg": 30.0},
        "geographic_info": {"resolution_m": 5.0},
    }
    with open(meta_path, "w") as f:
        json.dump(sidecar_data, f)

    reader = TMC2Reader()
    lunar_tmc2 = reader.read(img_path)

    assert isinstance(lunar_tmc2, LunarImage)
    assert lunar_tmc2.shape == (200, 200)
    assert lunar_tmc2.sensor == "TMC-2"
    assert lunar_tmc2.resolution_m == 5.0
    assert lunar_tmc2.acquisition_info.get("view_angle") == "Fore"


def test_tmc2_preprocessor():
    raw_img = np.random.randint(20, 200, (100, 100), dtype=np.uint8)
    lunar_tmc2 = LunarImage(image=raw_img, sensor="TMC-2", resolution_m=5.0)

    preprocessor = TMC2Preprocessor()
    processed = preprocessor.process(lunar_tmc2)

    assert isinstance(processed, LunarImage)
    assert processed.shape == lunar_tmc2.shape
    assert processed.image.dtype == np.uint8
