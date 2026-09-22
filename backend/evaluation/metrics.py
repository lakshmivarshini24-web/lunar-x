"""
Evaluation Metrics Infrastructure (Module 6)

Computes quantitative metrics for feature correspondence and image registration:
- Keypoint & match counts
- Inlier ratio, precision, recall
- Reprojection error, RMSE, mean pixel error
- Spatial coverage score & entropy across spatial grid
- Execution time and GPU utilization stats
"""

from dataclasses import dataclass, field
import json
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from backend.correspondence.classical import MatchResult


@dataclass
class EvaluationMetrics:
    """
    Quantitative evaluation metrics summary for a correspondence algorithm run.
    """

    algorithm: str
    num_kpts_a: int
    num_kpts_b: int
    num_raw_matches: int
    num_filtered_matches: int
    num_inliers: int
    inlier_ratio: float
    precision: float
    recall: Optional[float]
    mean_reprojection_error_px: float
    rmse_px: float
    spatial_coverage_ratio: float
    spatial_entropy: float
    execution_time_sec: float
    fps: float
    gpu_utilization: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "algorithm": self.algorithm,
            "num_kpts_a": self.num_kpts_a,
            "num_kpts_b": self.num_kpts_b,
            "num_raw_matches": self.num_raw_matches,
            "num_filtered_matches": self.num_filtered_matches,
            "num_inliers": self.num_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4) if self.recall is not None else None,
            "mean_reprojection_error_px": round(self.mean_reprojection_error_px, 4),
            "rmse_px": round(self.rmse_px, 4),
            "spatial_coverage_ratio": round(self.spatial_coverage_ratio, 4),
            "spatial_entropy": round(self.spatial_entropy, 4),
            "execution_time_sec": round(self.execution_time_sec, 4),
            "fps": round(self.fps, 2),
            "gpu_utilization": self.gpu_utilization,
        }


def compute_metrics(
    match_result: MatchResult,
    image_shape: Tuple[int, int],
    ground_truth_homography: Optional[np.ndarray] = None,
    grid_size: Tuple[int, int] = (8, 8),
) -> EvaluationMetrics:
    """
    Computes complete quantitative metrics given a MatchResult and image dimensions.

    Args:
        match_result: Output from matcher.
        image_shape: Height, Width tuple of reference image.
        ground_truth_homography: Optional 3x3 ground truth homography matrix for synthetic experiments.
        grid_size: (rows, cols) grid for spatial coverage analysis.

    Returns:
        EvaluationMetrics object.
    """
    num_raw = match_result.num_raw_matches
    num_filt = match_result.num_filtered_matches
    num_inliers = match_result.num_inliers

    inlier_ratio = match_result.inlier_ratio
    precision = float(num_inliers / num_filt) if num_filt > 0 else 0.0

    # 1. Reprojection Error & RMSE
    reproj_errors: List[float] = []
    H_eval = ground_truth_homography if ground_truth_homography is not None else match_result.homography

    if H_eval is not None and len(match_result.inlier_pts_a) > 0:
        pts_a = match_result.inlier_pts_a
        pts_b = match_result.inlier_pts_b

        # Homogeneous transform pts_a by H_eval
        pts_a_homo = np.hstack([pts_a, np.ones((len(pts_a), 1))])
        pts_b_proj_homo = (H_eval @ pts_a_homo.T).T

        # Normalize homogeneous coordinates
        valid_idx = np.abs(pts_b_proj_homo[:, 2]) > 1e-8
        if np.any(valid_idx):
            pts_b_proj = pts_b_proj_homo[valid_idx, :2] / pts_b_proj_homo[valid_idx, 2:3]
            target_b = pts_b[valid_idx]
            diffs = np.linalg.norm(pts_b_proj - target_b, axis=1)
            reproj_errors = diffs.tolist()

    mean_reproj_err = float(np.mean(reproj_errors)) if reproj_errors else 0.0
    rmse_val = float(np.sqrt(np.mean(np.square(reproj_errors)))) if reproj_errors else 0.0

    # 2. Precision & Recall against Ground Truth (if provided)
    recall_val: Optional[float] = None
    if ground_truth_homography is not None and len(match_result.matched_pts_a) > 0:
        pts_a = match_result.matched_pts_a
        pts_b = match_result.matched_pts_b
        pts_a_homo = np.hstack([pts_a, np.ones((len(pts_a), 1))])
        pts_b_gt_homo = (ground_truth_homography @ pts_a_homo.T).T
        pts_b_gt = pts_b_gt_homo[:, :2] / pts_b_gt_homo[:, 2:3]

        gt_diffs = np.linalg.norm(pts_b_gt - pts_b, axis=1)
        correct_matches = np.sum(gt_diffs <= 3.0)  # 3 pixel threshold
        recall_val = float(correct_matches / len(pts_a)) if len(pts_a) > 0 else 0.0

    # 3. Spatial Coverage & Entropy
    cov_ratio, spatial_entropy = compute_spatial_coverage(match_result.inlier_pts_a, image_shape, grid_size)

    # 4. Processing Time & FPS
    exec_time = max(1e-5, match_result.execution_time_sec)
    fps = 1.0 / exec_time

    # 5. GPU utilization if PyTorch CUDA available
    gpu_stats: Dict[str, Any] = {"cuda_available": False}
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_stats = {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2),
        }

    return EvaluationMetrics(
        algorithm=match_result.method,
        num_kpts_a=match_result.num_keypoints_a,
        num_kpts_b=match_result.num_keypoints_b,
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
        gpu_utilization=gpu_stats,
    )


def compute_spatial_coverage(
    points: np.ndarray, image_shape: Tuple[int, int], grid_size: Tuple[int, int] = (8, 8)
) -> Tuple[float, float]:
    """
    Computes spatial coverage ratio and spatial distribution entropy across a Grid_H x Grid_W grid.

    Returns:
        (coverage_ratio, entropy_val)
    """
    if len(points) == 0:
        return 0.0, 0.0

    h, w = image_shape[:2]
    grid_r, grid_c = grid_size
    total_cells = grid_r * grid_c

    counts = np.zeros((grid_r, grid_c), dtype=int)

    for x, y in points:
        c_idx = int(np.clip(x / w * grid_c, 0, grid_c - 1))
        r_idx = int(np.clip(y / h * grid_r, 0, grid_r - 1))
        counts[r_idx, c_idx] += 1

    non_empty_cells = np.sum(counts > 0)
    coverage_ratio = float(non_empty_cells / total_cells)

    # Entropy calculation: H = -sum(p_i * log2(p_i))
    flat_counts = counts.flatten()
    total_pts = len(points)
    probs = flat_counts[flat_counts > 0] / total_pts
    entropy = -float(np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

    return coverage_ratio, entropy
