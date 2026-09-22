"""
Cross-Sensor Evaluation Pipeline (Module 10)

Calculates quantitative metrics for OHRC ↔ TMC-2 cross-sensor correspondence,
evaluates scale-aligned geometric error in native pixel units and ground meters,
and exports machine-readable JSON & CSV evaluation reports.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from backend.core.lunar_image import LunarImage
from backend.matching.cross_sensor import CrossSensorMatchResult
from backend.evaluation.metrics import EvaluationMetrics, compute_spatial_coverage


class CrossSensorEvaluator:
    """
    Evaluates OHRC ↔ TMC-2 cross-sensor correspondence performance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def evaluate(
        self,
        match_result: CrossSensorMatchResult,
        lunar_ohrc: LunarImage,
        lunar_tmc2: LunarImage,
        ground_truth_homography: Optional[np.ndarray] = None,
        dataset_type: str = "SYNTHETIC_CROSS_SENSOR_PAIR",
    ) -> EvaluationMetrics:
        """
        Computes evaluation metrics for cross-sensor match result.

        Args:
            match_result: CrossSensorMatchResult.
            lunar_ohrc: OHRC image.
            lunar_tmc2: TMC-2 image.
            ground_truth_homography: Optional GT homography mapping native OHRC -> native TMC-2.
            dataset_type: Tag distinguishing 'REAL_ISRO_PDS4' from 'SYNTHETIC_CROSS_SENSOR_PAIR'.

        Returns:
            EvaluationMetrics object.
        """
        num_raw = match_result.num_raw_matches
        num_filt = match_result.num_filtered_matches
        num_inliers = match_result.num_inliers

        inlier_ratio = match_result.inlier_ratio
        precision = float(num_inliers / num_filt) if num_filt > 0 else 0.0

        # Reprojection Error in native TMC-2 pixel space
        reproj_errors: List[float] = []
        H_eval = ground_truth_homography if ground_truth_homography is not None else match_result.homography_native

        if H_eval is not None and len(match_result.inlier_pts_a_native) > 0:
            pts_a = match_result.inlier_pts_a_native
            pts_b = match_result.inlier_pts_b_native

            pts_a_homo = np.hstack([pts_a, np.ones((len(pts_a), 1))])
            pts_b_proj_homo = (H_eval @ pts_a_homo.T).T

            valid_idx = np.abs(pts_b_proj_homo[:, 2]) > 1e-8
            if np.any(valid_idx):
                pts_b_proj = pts_b_proj_homo[valid_idx, :2] / pts_b_proj_homo[valid_idx, 2:3]
                target_b = pts_b[valid_idx]
                diffs = np.linalg.norm(pts_b_proj - target_b, axis=1)
                reproj_errors = diffs.tolist()

        mean_reproj_err = float(np.mean(reproj_errors)) if reproj_errors else 0.0
        rmse_val = float(np.sqrt(np.mean(np.square(reproj_errors)))) if reproj_errors else 0.0

        # Precision & Recall against GT if available
        recall_val: Optional[float] = None
        if ground_truth_homography is not None and len(match_result.matched_pts_a_native) > 0:
            pts_a = match_result.matched_pts_a_native
            pts_b = match_result.matched_pts_b_native
            pts_a_homo = np.hstack([pts_a, np.ones((len(pts_a), 1))])
            pts_b_gt_homo = (ground_truth_homography @ pts_a_homo.T).T
            pts_b_gt = pts_b_gt_homo[:, :2] / pts_b_gt_homo[:, 2:3]

            gt_diffs = np.linalg.norm(pts_b_gt - pts_b, axis=1)
            correct_matches = np.sum(gt_diffs <= 5.0)  # 5 pixel threshold for cross-sensor
            recall_val = float(correct_matches / len(pts_a)) if len(pts_a) > 0 else 0.0

        # Spatial Coverage across TMC-2 spatial domain
        cov_ratio, spatial_entropy = compute_spatial_coverage(match_result.inlier_pts_b_native, lunar_tmc2.shape[:2])

        exec_time = max(1e-5, match_result.execution_time_sec)
        fps = 1.0 / exec_time

        return EvaluationMetrics(
            algorithm=f"CrossSensor_{match_result.method.upper()}",
            num_kpts_a=match_result.num_keypoints_a_native,
            num_kpts_b=match_result.num_keypoints_b_native,
            num_raw_matches=num_raw,
            num_filtered_matches=num_filt,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            precision=precision,
            recall=recall_val,
            mean_reprojection_error_px=mean_reproj_err,
            rmse_px=rmse_val,
            spatial_coverage_ratio=cov_ratio,
            spatial_entropy=spatial_entropy,
            execution_time_sec=exec_time,
            fps=fps,
            gpu_utilization={"dataset_type": dataset_type, "scale_ratio": match_result.scale_ratio},
        )

    def save_report(
        self,
        metrics: EvaluationMetrics,
        output_dir: str = "outputs/benchmarks",
        filename_prefix: str = "cross_sensor_report",
    ) -> Tuple[str, str]:
        """
        Saves cross-sensor metrics report to JSON and CSV.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_path = out_path / f"{filename_prefix}.json"
        csv_path = out_path / f"{filename_prefix}.csv"

        dict_data = metrics.to_dict()

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dict_data, f, indent=2)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict_data.keys()))
            writer.writeheader()
            row_copy = dict_data.copy()
            if isinstance(row_copy.get("gpu_utilization"), dict):
                row_copy["gpu_utilization"] = json.dumps(row_copy["gpu_utilization"])
            writer.writerow(row_copy)

        return str(json_path), str(csv_path)
