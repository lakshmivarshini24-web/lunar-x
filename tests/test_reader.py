"""
Unit tests for Module 2: OHRC Data Reader and LunarImage Dataclass.
"""

import json
from pathlib import Path
import numpy as np
import cv2
import pytest

from backend.core.lunar_image import LunarImage
from backend.sensors.ohrc.reader import OHRCReader, load_ohrc


def test_lunar_image_dataclass():
    arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    lunar_img = LunarImage(image=arr, sensor="OHRC", resolution_m=0.25)

    assert lunar_img.shape == (100, 100)
    assert lunar_img.height == 100
    assert lunar_img.width == 100
    assert lunar_img.channels == 1
    assert lunar_img.sensor == "OHRC"

    # Test conversion methods
    f32 = lunar_img.to_float32()
    assert f32.dtype == np.float32
    assert np.max(f32) <= 1.0

    u8 = lunar_img.to_uint8()
    assert u8.dtype == np.uint8

    # Test deep copy
    copy_img = lunar_img.copy()
    assert copy_img.shape == lunar_img.shape
    assert copy_img.image is not lunar_img.image


def test_ohrc_reader(tmp_path):
    img_path = tmp_path / "test_ohrc.tif"
    img = np.random.randint(20, 200, (256, 256), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    meta_path = tmp_path / "test_ohrc.json"
    sidecar_data = {
        "acquisition_info": {"orbit": 999, "instrument": "OHRC"},
        "sun_angle_info": {"sun_elevation_deg": 42.0},
    }
    with open(meta_path, "w") as f:
        json.dump(sidecar_data, f)

    reader = OHRCReader()
    lunar_img = reader.read(img_path)

    assert isinstance(lunar_img, LunarImage)
    assert lunar_img.shape == (256, 256)
    assert lunar_img.sensor == "OHRC"
    assert lunar_img.acquisition_info.get("orbit") == 999
    assert lunar_img.sun_angle_info.get("sun_elevation_deg") == 42.0
