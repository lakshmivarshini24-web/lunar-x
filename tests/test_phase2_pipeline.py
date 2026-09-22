"""
Integration tests for Phase 2: OHRC Real Pipeline and Cross-Sensor Evaluation.
"""

from pathlib import Path
import numpy as np
import cv2
import pytest

from backend.sensors.ohrc.pipeline import OHRCRealCorrespondencePipeline, OHRCRealPipelineResult
from backend.matching.cross_sensor import CrossSensorMatchResult
from backend.evaluation.cross_sensor_eval import CrossSensorEvaluator
from backend.core.lunar_image import LunarImage


def test_ohrc_real_pipeline(tmp_path):
    img_a_path = tmp_path / "ohrc_a.png"
    img_b_path = tmp_path / "ohrc_b.png"

    img_a = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img_a, (150, 150), 40, (200,), -1)
    cv2.putText(img_a, "CRATER", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,), 2)

    M = np.float32([[1, 0, 10], [0, 1, 5]])
    img_b = cv2.warpAffine(img_a, M, (300, 300))

    cv2.imwrite(str(img_a_path), img_a)
    cv2.imwrite(str(img_b_path), img_b)

    pipeline = OHRCRealCorrespondencePipeline()
    res = pipeline.process_pair(img_a_path, img_b_path)

    assert isinstance(res, OHRCRealPipelineResult)
    assert res.match_result.num_inliers > 0
    assert res.metrics.inlier_ratio > 0.0


def test_cross_sensor_evaluator():
    cs_result = CrossSensorMatchResult(
        sensor_a="OHRC",
        sensor_b="TMC-2",
        scale_ratio=20.0,
        method="sift",
        num_keypoints_a_native=1000,
        num_keypoints_b_native=800,
        num_raw_matches=500,
        num_filtered_matches=200,
        num_inliers=180,
        inlier_ratio=180 / 200,
        confidence_score=0.9,
        matched_pts_a_native=np.random.rand(200, 2) * 1000,
        matched_pts_b_native=np.random.rand(200, 2) * 500,
        inlier_pts_a_native=np.random.rand(180, 2) * 1000,
        inlier_pts_b_native=np.random.rand(180, 2) * 500,
        inlier_mask=np.ones(200, dtype=bool),
        homography_native=np.eye(3),
        execution_time_sec=0.25,
    )

    lunar_ohrc = LunarImage(image=np.zeros((1000, 1000), dtype=np.uint8), sensor="OHRC", resolution_m=0.25)
    lunar_tmc2 = LunarImage(image=np.zeros((500, 500), dtype=np.uint8), sensor="TMC-2", resolution_m=5.0)

    evaluator = CrossSensorEvaluator()
    metrics = evaluator.evaluate(cs_result, lunar_ohrc, lunar_tmc2, dataset_type="SYNTHETIC_CROSS_SENSOR_PAIR")

    assert metrics.num_inliers == 180
    assert metrics.inlier_ratio == 0.9
    assert metrics.gpu_utilization.get("dataset_type") == "SYNTHETIC_CROSS_SENSOR_PAIR"
