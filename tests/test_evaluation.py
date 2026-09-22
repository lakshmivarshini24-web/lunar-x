"""
Unit tests for Module 6: Evaluation Infrastructure.
"""

import numpy as np
import pytest

from backend.core.lunar_image import LunarImage
from backend.correspondence.classical import MatchResult
from backend.evaluation.metrics import compute_metrics, compute_spatial_coverage, EvaluationMetrics
from backend.evaluation.benchmark import BenchmarkRunner


def test_spatial_coverage():
    points = np.array([[10, 10], [20, 20], [400, 400]], dtype=np.float32)
    image_shape = (512, 512)
    grid_size = (4, 4)

    cov_ratio, entropy = compute_spatial_coverage(points, image_shape, grid_size)

    assert cov_ratio > 0.0
    assert entropy > 0.0


def test_compute_metrics():
    match_result = MatchResult(
        method="sift",
        num_keypoints_a=100,
        num_keypoints_b=110,
        num_raw_matches=50,
        num_filtered_matches=30,
        num_inliers=25,
        inlier_ratio=25 / 30,
        confidence_score=0.8,
        keypoints_a=np.random.rand(100, 2) * 500,
        keypoints_b=np.random.rand(110, 2) * 500,
        matched_pts_a=np.random.rand(30, 2) * 500,
        matched_pts_b=np.random.rand(30, 2) * 500,
        inlier_pts_a=np.random.rand(25, 2) * 500,
        inlier_pts_b=np.random.rand(25, 2) * 500,
        inlier_mask=np.ones(30, dtype=bool),
        homography=np.eye(3),
        execution_time_sec=0.1,
    )

    metrics = compute_metrics(match_result, (512, 512))

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.num_inliers == 25
    assert metrics.precision == float(25 / 30)
    assert metrics.spatial_coverage_ratio > 0.0
    assert metrics.execution_time_sec == 0.1
