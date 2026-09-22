"""
Unit and integration tests for LUNAR-X Streamlit Dashboard application.
"""

import numpy as np
import pytest

from app import draw_match_canvas_ui, get_config


def test_app_config():
    cfg = get_config()
    assert isinstance(cfg, dict)
    assert cfg["project"]["name"] == "LUNAR-X"


def test_draw_match_canvas_ui():
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.zeros((100, 100), dtype=np.uint8)
    pts1 = np.array([[10, 10], [20, 20]], dtype=np.float32)
    pts2 = np.array([[15, 15], [25, 25]], dtype=np.float32)

    canvas = draw_match_canvas_ui(img1, img2, pts1, pts2)

    assert canvas.shape == (100, 200, 3)
    assert canvas.dtype == np.uint8
