"""
Unit tests for Module 9: OHRC ↔ TMC-2 Cross-Sensor Matching.
"""

import numpy as np
import cv2
import pytest

from backend.core.lunar_image import LunarImage
from backend.matching.cross_sensor import CrossSensorMatcher, CrossSensorMatchResult


def test_cross_sensor_matching():
    # 1. High-resolution OHRC image (500 x 500)
    img_ohrc = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(img_ohrc, (120, 120), 40, (220,), -1)
    cv2.rectangle(img_ohrc, (300, 100), (380, 180), (200,), -1)
    cv2.circle(img_ohrc, (150, 350), 30, (240,), -1)
    cv2.putText(img_ohrc, "LUNAR_CROSS", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,), 2)

    # 2. Downsampled TMC-2 context image (100 x 100) representing 5x scale ratio
    img_tmc2 = cv2.resize(img_ohrc, (100, 100), interpolation=cv2.INTER_AREA)

    lunar_ohrc = LunarImage(image=img_ohrc, sensor="OHRC", resolution_m=0.25)
    lunar_tmc2 = LunarImage(image=img_tmc2, sensor="TMC-2", resolution_m=1.25)

    matcher = CrossSensorMatcher()
    result = matcher.match(lunar_ohrc, lunar_tmc2, method_override="sift")

    assert isinstance(result, CrossSensorMatchResult)
    assert result.sensor_a == "OHRC"
    assert result.sensor_b == "TMC-2"
    assert result.scale_ratio == 5.0
    assert result.num_inliers > 0
    assert len(result.inlier_pts_a_native) > 0
    assert len(result.inlier_pts_b_native) > 0
