"""
OHRC ↔ TMC-2 Cross-Sensor Matcher (Module 9)

Implements multi-scale cross-sensor feature correspondence between high-resolution OHRC (~0.25 m/px)
and low-resolution TMC-2 (~5.0 m/px) imagery. Handles ~20x resolution scale disparity via scale-pyramid
resampling, FLANN descriptor matching, Lowe ratio test, and RANSAC geometric verification.
"""

import copy
from dataclasses import dataclass, field
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2

from backend.core.lunar_image import LunarImage
from backend.correspondence.classical import MatchResult, ClassicalCorrespondence


@dataclass
class CrossSensorMatchResult:
    """Output container for OHRC ↔ TMC-2 cross-sensor correspondence."""

    sensor_a: str  # e.g., "OHRC"
    sensor_b: str  # e.g., "TMC-2"
    scale_ratio: float  # resolution_b / resolution_a (e.g. 20.0)
    method: str
    num_keypoints_a_native: int
    num_keypoints_b_native: int
    num_raw_matches: int
    num_filtered_matches: int
    num_inliers: int
    inlier_ratio: float
    confidence_score: float
    matched_pts_a_native: np.ndarray  # (M, 2) in full-res OHRC pixels
    matched_pts_b_native: np.ndarray  # (M, 2) in full-res TMC-2 pixels
    inlier_pts_a_native: np.ndarray  # (K, 2) in full-res OHRC pixels
    inlier_pts_b_native: np.ndarray  # (K, 2) in full-res TMC-2 pixels
    inlier_mask: np.ndarray
    homography_native: Optional[np.ndarray] = None  # 3x3 homography mapping native OHRC -> native TMC-2
    execution_time_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossSensorMatcher:
    """
    Multi-scale matcher for cross-sensor correspondence between OHRC and TMC-2.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cs_cfg = self.config.get("cross_sensor", {})
        self.scale_norm = cs_cfg.get("scale_normalization", True)
        self.min_resample_dim = int(cs_cfg.get("min_resample_dim", 256))
        self.ratio_thresh = float(cs_cfg.get("ratio_threshold", 0.80))
        self.ransac_thresh = float(cs_cfg.get("ransac_threshold_px", 5.0))

        classical_config = copy.deepcopy(self.config)
        if "correspondence" not in classical_config:
            classical_config["correspondence"] = {}
        if "matching" not in classical_config["correspondence"]:
            classical_config["correspondence"]["matching"] = {}
        classical_config["correspondence"]["matching"]["ratio_threshold"] = self.ratio_thresh
        self.classical_matcher = ClassicalCorrespondence(classical_config)

    def match(
        self, lunar_ohrc: LunarImage, lunar_tmc2: LunarImage, method_override: Optional[str] = None
    ) -> CrossSensorMatchResult:
        """
        Executes cross-sensor matching between an OHRC image and a TMC-2 image.

        Args:
            lunar_ohrc: High-resolution OHRC image (~0.25 m/px).
            lunar_tmc2: Lower-resolution TMC-2 image (~5.0 m/px).
            method_override: Optional method ('sift', 'orb').

        Returns:
            CrossSensorMatchResult with correspondences mapped to native pixel spaces.
        """
        start_time = time.time()
        method = method_override or self.config.get("correspondence", {}).get("method", "sift")

        res_ohrc = lunar_ohrc.resolution_m
        res_tmc2 = lunar_tmc2.resolution_m
        scale_ratio = float(res_tmc2 / res_ohrc) if res_ohrc > 0 else 20.0

        img_ohrc_native = lunar_ohrc.to_uint8()
        img_tmc2_native = lunar_tmc2.to_uint8()

        # 1. Scale Normalization (Downsample high-res OHRC image with a spatial dimension floor)
        if self.scale_norm and scale_ratio > 1.2:
            target_h = max(self.min_resample_dim, int(round(img_ohrc_native.shape[0] / scale_ratio)))
            target_w = max(self.min_resample_dim, int(round(img_ohrc_native.shape[1] / scale_ratio)))
            target_h = min(img_ohrc_native.shape[0], target_h)
            target_w = min(img_ohrc_native.shape[1], target_w)
            img_ohrc_resampled = cv2.resize(
                img_ohrc_native, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            scale_x = img_ohrc_native.shape[1] / target_w
            scale_y = img_ohrc_native.shape[0] / target_h
        else:
            img_ohrc_resampled = img_ohrc_native
            scale_x, scale_y = 1.0, 1.0

        # Create temporary LunarImage wrappers for feature extraction
        lunar_ohrc_resampled = LunarImage(image=img_ohrc_resampled, sensor="OHRC_Resampled", resolution_m=res_tmc2)

        # 2. Extract and Match Features in scale-aligned domain
        m_result = self.classical_matcher.match(lunar_ohrc_resampled, lunar_tmc2, method_override=method)

        # 3. Remap matched coordinates back to native OHRC resolution
        if len(m_result.matched_pts_a) > 0:
            matched_pts_ohrc_native = m_result.matched_pts_a.copy()
            matched_pts_ohrc_native[:, 0] *= scale_x
            matched_pts_ohrc_native[:, 1] *= scale_y

            inlier_pts_ohrc_native = m_result.inlier_pts_a.copy() if len(m_result.inlier_pts_a) > 0 else np.empty((0, 2))
            if len(inlier_pts_ohrc_native) > 0:
                inlier_pts_ohrc_native[:, 0] *= scale_x
                inlier_pts_ohrc_native[:, 1] *= scale_y
        else:
            matched_pts_ohrc_native = np.empty((0, 2))
            inlier_pts_ohrc_native = np.empty((0, 2))

        matched_pts_tmc2_native = m_result.matched_pts_b
        inlier_pts_tmc2_native = m_result.inlier_pts_b

        # 4. Re-fit Native Homography Matrix (Native OHRC -> Native TMC-2)
        H_native = None
        if len(inlier_pts_ohrc_native) >= 4:
            H_native, _ = cv2.findHomography(
                inlier_pts_ohrc_native, inlier_pts_tmc2_native, cv2.RANSAC, self.ransac_thresh
            )

        exec_time = time.time() - start_time

        return CrossSensorMatchResult(
            sensor_a=lunar_ohrc.sensor,
            sensor_b=lunar_tmc2.sensor,
            scale_ratio=scale_ratio,
            method=method,
            num_keypoints_a_native=m_result.num_keypoints_a,
            num_keypoints_b_native=m_result.num_keypoints_b,
            num_raw_matches=m_result.num_raw_matches,
            num_filtered_matches=m_result.num_filtered_matches,
            num_inliers=m_result.num_inliers,
            inlier_ratio=m_result.inlier_ratio,
            confidence_score=m_result.confidence_score,
            matched_pts_a_native=matched_pts_ohrc_native,
            matched_pts_b_native=matched_pts_tmc2_native,
            inlier_pts_a_native=inlier_pts_ohrc_native,
            inlier_pts_b_native=inlier_pts_tmc2_native,
            inlier_mask=m_result.inlier_mask,
            homography_native=H_native,
            execution_time_sec=exec_time,
            metadata={
                "scale_x": scale_x,
                "scale_y": scale_y,
                "resampled_shape_ohrc": img_ohrc_resampled.shape,
                "native_shape_ohrc": img_ohrc_native.shape,
                "native_shape_tmc2": img_tmc2_native.shape,
            },
        )
