"""
Unit tests for Module 3: Preprocessing Core.
"""

import numpy as np
import pytest

from backend.core.lunar_image import LunarImage
from backend.preprocessing.core import OHRCPreprocessor


def test_ohrc_preprocessor():
    # Synthetic gradient image with illumination imbalance
    x = np.linspace(0, 255, 128)
    y = np.linspace(0, 255, 128)
    xx, yy = np.meshgrid(x, y)
    raw_img = (xx * 0.5 + yy * 0.5).astype(np.uint8)

    lunar_img = LunarImage(image=raw_img, sensor="OHRC")
    preprocessor = OHRCPreprocessor()

    processed = preprocessor.process(lunar_img)

    assert isinstance(processed, LunarImage)
    assert processed.shape == lunar_img.shape
    assert processed.image.dtype == np.uint8
    assert "preprocessing_applied" in processed.metadata


def test_percentile_stretch():
    img = np.array([[10, 20], [30, 200]], dtype=np.uint8)
    stretched = OHRCPreprocessor.percentile_stretch(img, p_low=5.0, p_high=95.0)

    assert stretched.shape == img.shape
    assert stretched.dtype == np.uint8
    assert np.max(stretched) == 255


def test_illumination_normalization():
    img = np.ones((64, 64), dtype=np.uint8) * 100
    normed = OHRCPreprocessor.normalize_illumination(img, method="high_pass", sigma=5.0)

    assert normed.shape == (64, 64)
    assert normed.dtype == np.uint8
