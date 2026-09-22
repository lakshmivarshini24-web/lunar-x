"""
Integration tests for Phase 3 Pipeline Runner and Tri-Sensor Evaluator.
"""

from pathlib import Path
import numpy as np
import pytest

from backend.evaluation.tri_sensor_eval import TriSensorEvaluator
from backend.matching.tri_sensor import TriSensorCorrespondenceEngine
from backend.core.lunar_image import LunarImage


def test_tri_sensor_evaluator():
    lunar_ohrc = LunarImage(image=np.zeros((400, 400), dtype=np.uint8), sensor="OHRC", resolution_m=0.25)
    lunar_tmc2 = LunarImage(image=np.zeros((200, 200), dtype=np.uint8), sensor="TMC-2", resolution_m=0.5)
    lunar_iirs = LunarImage(image=np.zeros((100, 100), dtype=np.uint8), sensor="IIRS", resolution_m=1.0)

    engine = TriSensorCorrespondenceEngine()
    tri_res = engine.process_triplet(lunar_ohrc, lunar_tmc2, lunar_iirs)

    evaluator = TriSensorEvaluator()
    summary = evaluator.evaluate(tri_res, dataset_type="SYNTHETIC_TRISENSOR_TRIPLET")

    assert "total_execution_time_sec" in summary
    assert "tri_sensor_agreement_score" in summary
    assert "pairwise_metrics" in summary
    assert summary["dataset_type"] == "SYNTHETIC_TRISENSOR_TRIPLET"
