"""
Unit tests for Module 5: Classical SIFT, ORB, AKAZE Correspondence Baselines.
"""

import numpy as np
import cv2
import pytest

from backend.core.lunar_image import LunarImage
from backend.correspondence.classical import ClassicalCorrespondence, MatchResult


def test_classical_sift_matching():
    # Generate synthetic image with geometric features (circles/rectangles)
    img_a = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img_a, (100, 100), 30, (200,), -1)
    cv2.rectangle(img_a, (180, 180), (240, 240), (220,), -1)
    cv2.putText(img_a, "LUNAR", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,), 2)

    # Shift Image B by (20, 10)
    M = np.float32([[1, 0, 20], [0, 1, 10]])
    img_b = cv2.warpAffine(img_a, M, (300, 300))

    lunar_a = LunarImage(image=img_a)
    lunar_b = LunarImage(image=img_b)

    matcher = ClassicalCorrespondence()
    result = matcher.match(lunar_a, lunar_b, method_override="sift")

    assert isinstance(result, MatchResult)
    assert result.num_keypoints_a > 0
    assert result.num_keypoints_b > 0
    assert result.num_inliers > 0
    assert result.homography is not None
    assert result.homography.shape == (3, 3)
    assert result.inlier_ratio > 0.0


def test_classical_orb_matching():
    img_a = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img_a, (100, 100), 30, (200,), -1)
    cv2.rectangle(img_a, (180, 180), (240, 240), (220,), -1)

    M = np.float32([[1, 0, 15], [0, 1, 5]])
    img_b = cv2.warpAffine(img_a, M, (300, 300))

    lunar_a = LunarImage(image=img_a)
    lunar_b = LunarImage(image=img_b)

    matcher = ClassicalCorrespondence()
    result = matcher.match(lunar_a, lunar_b, method_override="orb")

    assert isinstance(result, MatchResult)
    assert result.method == "orb"
