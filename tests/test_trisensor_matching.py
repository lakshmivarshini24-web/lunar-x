"""
Unit tests for Modules 13, 14, & Tri-Sensor Correspondence Engine.
"""

import numpy as np
import cv2
import pytest

from backend.core.lunar_image import LunarImage
from backend.matching.cross_modal import CrossModalMatcher
from backend.matching.tri_sensor import TriSensorCorrespondenceEngine, TriSensorMatchResult
from backend.matching.cross_sensor import CrossSensorMatchResult


def test_cross_modal_matcher():
    # Synthetic optical image
    img_opt = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(img_opt, (100, 100), 30, (220,), -1)
    cv2.rectangle(img_opt, (200, 200), (300, 300), (200,), -1)
    cv2.putText(img_opt, "OPTICAL", (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,), 2)

    # Synthetic IIRS PC1 image (resampled)
    img_iirs = cv2.resize(img_opt, (100, 100), interpolation=cv2.INTER_AREA)

    lunar_opt = LunarImage(image=img_opt, sensor="OHRC", resolution_m=0.25)
    lunar_iirs = LunarImage(image=img_iirs, sensor="IIRS", resolution_m=1.0)

    matcher = CrossModalMatcher()
    res = matcher.match(lunar_opt, lunar_iirs, method_override="sift")

    assert isinstance(res, CrossSensorMatchResult)
    assert res.sensor_a == "OHRC"
    assert res.sensor_b == "IIRS"
    assert res.num_inliers >= 0


def test_tri_sensor_engine():
    img_ohrc = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(img_ohrc, (100, 100), 30, (220,), -1)
    cv2.rectangle(img_ohrc, (200, 200), (300, 300), (200,), -1)

    img_tmc2 = cv2.resize(img_ohrc, (200, 200), interpolation=cv2.INTER_AREA)
    img_iirs = cv2.resize(img_ohrc, (100, 100), interpolation=cv2.INTER_AREA)

    lunar_ohrc = LunarImage(image=img_ohrc, sensor="OHRC", resolution_m=0.25)
    lunar_tmc2 = LunarImage(image=img_tmc2, sensor="TMC-2", resolution_m=0.5)
    lunar_iirs = LunarImage(image=img_iirs, sensor="IIRS", resolution_m=1.0)

    engine = TriSensorCorrespondenceEngine()
    tri_res = engine.process_triplet(lunar_ohrc, lunar_tmc2, lunar_iirs)

    assert isinstance(tri_res, TriSensorMatchResult)
    assert tri_res.total_execution_time_sec > 0.0
    assert tri_res.tri_sensor_agreement_score >= 0.0
