"""
Cross-Modal Feature Correspondence (Modules 13 & 14)

Implements modality-invariant cross-modal image matching between optical imagery (OHRC, TMC-2)
and hyperspectral infrared imagery (IIRS). Employs gradient-domain magnitude transformations to overcome
contrast polarity inversions, multi-scale scale-pyramid resampling, FLANN matching, and RANSAC geometric verification.
"""

import copy
from dataclasses import dataclass, field
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2

from backend.core.lunar_image import LunarImage
from backend.correspondence.classical import ClassicalCorrespondence
from backend.matching.cross_sensor import CrossSensorMatchResult


class CrossModalMatcher:
    """
    Cross-modal correspondence matcher for OHRC ↔ IIRS and TMC-2 ↔ IIRS pairs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cm_cfg = self.config.get("cross_modal", {})
        self.use_gradient = cm_cfg.get("use_gradient_magnitude", True)
        self.min_resample_dim = int(cm_cfg.get("min_resample_dim", 256))
        self.ratio_thresh = float(cm_cfg.get("ratio_threshold", 0.80))
        self.ransac_thresh = float(cm_cfg.get("ransac_threshold_px", 8.0))

        classical_config = copy.deepcopy(self.config)
        if "correspondence" not in classical_config:
            classical_config["correspondence"] = {}
        if "matching" not in classical_config["correspondence"]:
            classical_config["correspondence"]["matching"] = {}
        classical_config["correspondence"]["matching"]["ratio_threshold"] = self.ratio_thresh
        self.classical_matcher = ClassicalCorrespondence(classical_config)

    def match(
        self, lunar_optical: LunarImage, lunar_iirs: LunarImage, method_override: Optional[str] = None
    ) -> CrossSensorMatchResult:
        """
        Executes cross-modal matching between Optical (OHRC or TMC-2) and IIRS.

        Args:
            lunar_optical: Source optical image (OHRC or TMC-2).
            lunar_iirs: Source IIRS image (after PCA / preprocessor PC1 extraction).
            method_override: Optional algorithm override.

        Returns:
            CrossSensorMatchResult mapped to native pixel spaces.
        """
        start_time = time.time()
        method = method_override or self.config.get("correspondence", {}).get("method", "sift")

        res_opt = lunar_optical.resolution_m
        res_iirs = lunar_iirs.resolution_m
        scale_ratio = float(res_iirs / res_opt) if res_opt > 0 else 20.0

        img_opt_native = lunar_optical.to_uint8()
        img_iirs_native = lunar_iirs.to_uint8()

        # 1. Modality-Invariant Representation (Gradient Magnitude Transform)
        if self.use_gradient:
            grad_opt_native = compute_gradient_magnitude(img_opt_native)
            grad_iirs_native = compute_gradient_magnitude(img_iirs_native)
        else:
            grad_opt_native = img_opt_native
            grad_iirs_native = img_iirs_native

        # 2. Scale-Pyramid Resampling (Scale high-res optical to low-res IIRS domain with dimension floor)
        if scale_ratio > 1.2:
            target_h = max(self.min_resample_dim, int(round(grad_opt_native.shape[0] / scale_ratio)))
            target_w = max(self.min_resample_dim, int(round(grad_opt_native.shape[1] / scale_ratio)))
            target_h = min(grad_opt_native.shape[0], target_h)
            target_w = min(grad_opt_native.shape[1], target_w)
            grad_opt_resampled = cv2.resize(
                grad_opt_native, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            scale_x = grad_opt_native.shape[1] / target_w
            scale_y = grad_opt_native.shape[0] / target_h
        else:
            grad_opt_resampled = grad_opt_native
            scale_x, scale_y = 1.0, 1.0

        # Wrap in LunarImage
        lunar_opt_resampled = LunarImage(image=grad_opt_resampled, sensor=f"{lunar_optical.sensor}_Resampled", resolution_m=res_iirs)
        lunar_iirs_grad = LunarImage(image=grad_iirs_native, sensor="IIRS_Gradient", resolution_m=res_iirs)

        # 3. Match Features
        m_result = self.classical_matcher.match(lunar_opt_resampled, lunar_iirs_grad, method_override=method)

        # 4. Remap inliers back to native optical resolution
        if len(m_result.matched_pts_a) > 0:
            matched_pts_opt_native = m_result.matched_pts_a.copy()
            matched_pts_opt_native[:, 0] *= scale_x
            matched_pts_opt_native[:, 1] *= scale_y

            inlier_pts_opt_native = m_result.inlier_pts_a.copy() if len(m_result.inlier_pts_a) > 0 else np.empty((0, 2))
            if len(inlier_pts_opt_native) > 0:
                inlier_pts_opt_native[:, 0] *= scale_x
                inlier_pts_opt_native[:, 1] *= scale_y
        else:
            matched_pts_opt_native = np.empty((0, 2))
            inlier_pts_opt_native = np.empty((0, 2))

        matched_pts_iirs_native = m_result.matched_pts_b
        inlier_pts_iirs_native = m_result.inlier_pts_b

        # 5. Fit Native Homography Matrix (Native Optical -> Native IIRS)
        H_native = None
        if len(inlier_pts_opt_native) >= 4:
            H_native, _ = cv2.findHomography(
                inlier_pts_opt_native, inlier_pts_iirs_native, cv2.RANSAC, self.ransac_thresh
            )

        exec_time = time.time() - start_time

        return CrossSensorMatchResult(
            sensor_a=lunar_optical.sensor,
            sensor_b=lunar_iirs.sensor,
            scale_ratio=scale_ratio,
            method=f"CrossModal_{method}",
            num_keypoints_a_native=m_result.num_keypoints_a,
            num_keypoints_b_native=m_result.num_keypoints_b,
            num_raw_matches=m_result.num_raw_matches,
            num_filtered_matches=m_result.num_filtered_matches,
            num_inliers=m_result.num_inliers,
            inlier_ratio=m_result.inlier_ratio,
            confidence_score=m_result.confidence_score,
            matched_pts_a_native=matched_pts_opt_native,
            matched_pts_b_native=matched_pts_iirs_native,
            inlier_pts_a_native=inlier_pts_opt_native,
            inlier_pts_b_native=inlier_pts_iirs_native,
            inlier_mask=m_result.inlier_mask,
            homography_native=H_native,
            execution_time_sec=exec_time,
            metadata={
                "cross_modal": True,
                "use_gradient_magnitude": self.use_gradient,
                "scale_x": scale_x,
                "scale_y": scale_y,
            },
        )


def compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Computes Sobel gradient magnitude map for modality-invariant structural matching."""
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Normalize to uint8 [0, 255] and apply CLAHE to enhance edge contrast
    min_v, max_v = np.min(mag), np.max(mag)
    if max_v > min_v:
        normed = ((mag - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    else:
        normed = np.zeros(img.shape, dtype=np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(normed)
