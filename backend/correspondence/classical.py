"""
Classical Correspondence Baseline (Module 5)

Implements classical computer vision correspondence baselines using SIFT, ORB, and AKAZE
feature extractors, FLANN / BFMatcher, Lowe ratio test filtering, RANSAC homography estimation,
and robust confidence statistic calculations.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

from backend.core.lunar_image import LunarImage


@dataclass
class MatchResult:
    """
    Data structure containing feature correspondence and geometric registration outputs.
    """

    method: str
    num_keypoints_a: int
    num_keypoints_b: int
    num_raw_matches: int
    num_filtered_matches: int
    num_inliers: int
    inlier_ratio: float
    confidence_score: float
    keypoints_a: np.ndarray  # Shape (N, 2) [x, y]
    keypoints_b: np.ndarray  # Shape (N, 2) [x, y]
    matched_pts_a: np.ndarray  # Filtered matched points (M, 2)
    matched_pts_b: np.ndarray  # Filtered matched points (M, 2)
    inlier_pts_a: np.ndarray  # RANSAC inlier points (K, 2)
    inlier_pts_b: np.ndarray  # RANSAC inlier points (K, 2)
    inlier_mask: np.ndarray  # Boolean mask over matched_pts
    homography: Optional[np.ndarray] = None  # 3x3 matrix
    execution_time_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClassicalCorrespondence:
    """
    Pipeline for classical feature detection, description, matching, and geometric verification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        corr_cfg = self.config.get("correspondence", {})
        self.method = corr_cfg.get("method", "sift").lower()
        self.matching_cfg = corr_cfg.get("matching", {})
        self.ransac_cfg = corr_cfg.get("ransac", {})

    def match(self, lunar_a: LunarImage, lunar_b: LunarImage, method_override: Optional[str] = None) -> MatchResult:
        """
        Executes feature matching between two LunarImages.

        Args:
            lunar_a: Source image A.
            lunar_b: Source image B.
            method_override: Optional algorithm override ('sift', 'orb', 'akaze').

        Returns:
            MatchResult containing keypoints, matches, inliers, and homography matrix.
        """
        start_time = time.time()
        method = method_override.lower() if method_override else self.method

        img_a = lunar_a.to_uint8()
        img_b = lunar_b.to_uint8()

        # 1. Detect & Describe Keypoints
        kpts_a, descs_a = self._extract_features(img_a, method)
        kpts_b, descs_b = self._extract_features(img_b, method)

        num_kpts_a = len(kpts_a)
        num_kpts_b = len(kpts_b)

        if num_kpts_a == 0 or num_kpts_b == 0 or descs_a is None or descs_b is None:
            exec_time = time.time() - start_time
            return MatchResult(
                method=method,
                num_keypoints_a=num_kpts_a,
                num_keypoints_b=num_kpts_b,
                num_raw_matches=0,
                num_filtered_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                confidence_score=0.0,
                keypoints_a=np.empty((0, 2)),
                keypoints_b=np.empty((0, 2)),
                matched_pts_a=np.empty((0, 2)),
                matched_pts_b=np.empty((0, 2)),
                inlier_pts_a=np.empty((0, 2)),
                inlier_pts_b=np.empty((0, 2)),
                inlier_mask=np.array([], dtype=bool),
                homography=None,
                execution_time_sec=exec_time,
            )

        # 2. Descriptor Matching
        raw_matches, filtered_matches = self._match_descriptors(descs_a, descs_b, method)

        pts_a = np.float32([kpts_a[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 2) if filtered_matches else np.empty((0, 2))
        pts_b = np.float32([kpts_b[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 2) if filtered_matches else np.empty((0, 2))

        # 3. Geometric Verification via RANSAC Homography
        homography = None
        inlier_mask = np.zeros(len(pts_a), dtype=bool)
        inlier_pts_a = np.empty((0, 2))
        inlier_pts_b = np.empty((0, 2))
        num_inliers = 0
        inlier_ratio = 0.0

        if len(pts_a) >= 4:
            reproj_thresh = float(self.ransac_cfg.get("reproj_threshold", 3.0))
            max_iter = int(self.ransac_cfg.get("max_iter", 2000))
            conf = float(self.ransac_cfg.get("confidence", 0.99))

            H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, reproj_thresh, maxIters=max_iter, confidence=conf)

            if H is not None and mask is not None:
                homography = H
                inlier_mask = mask.ravel().astype(bool)
                inlier_pts_a = pts_a[inlier_mask]
                inlier_pts_b = pts_b[inlier_mask]
                num_inliers = int(np.sum(inlier_mask))
                inlier_ratio = float(num_inliers / len(pts_a)) if len(pts_a) > 0 else 0.0

        exec_time = time.time() - start_time

        # Calculate overall confidence score based on inlier count and ratio
        confidence_score = float(np.clip(inlier_ratio * np.tanh(num_inliers / 20.0), 0.0, 1.0))

        all_kpts_a = np.float32([kp.pt for kp in kpts_a]) if kpts_a else np.empty((0, 2))
        all_kpts_b = np.float32([kp.pt for kp in kpts_b]) if kpts_b else np.empty((0, 2))

        return MatchResult(
            method=method,
            num_keypoints_a=num_kpts_a,
            num_keypoints_b=num_kpts_b,
            num_raw_matches=len(raw_matches),
            num_filtered_matches=len(filtered_matches),
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            confidence_score=confidence_score,
            keypoints_a=all_kpts_a,
            keypoints_b=all_kpts_b,
            matched_pts_a=pts_a,
            matched_pts_b=pts_b,
            inlier_pts_a=inlier_pts_a,
            inlier_pts_b=inlier_pts_b,
            inlier_mask=inlier_mask,
            homography=homography,
            execution_time_sec=exec_time,
            metadata={"matcher": self.matching_cfg.get("matcher", "flann")},
        )

    def _extract_features(self, img: np.ndarray, method: str) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:

        sift_cfg = self.config.get("correspondence", {}).get("sift", {})
        orb_cfg = self.config.get("correspondence", {}).get("orb", {})
        akaze_cfg = self.config.get("correspondence", {}).get("akaze", {})

        if method == "sift":
            detector = cv2.SIFT_create(
                nfeatures=int(sift_cfg.get("n_features", 5000)),
                contrastThreshold=float(sift_cfg.get("contrast_threshold", 0.04)),
                edgeThreshold=float(sift_cfg.get("edge_threshold", 10.0)),
                sigma=float(sift_cfg.get("sigma", 1.6)),
            )
        elif method == "orb":
            detector = cv2.ORB_create(
                nfeatures=int(orb_cfg.get("n_features", 5000)),
                scaleFactor=float(orb_cfg.get("scale_factor", 1.2)),
                nlevels=int(orb_cfg.get("n_levels", 8)),
            )
        elif method == "akaze":
            threshold = float(akaze_cfg.get("threshold", 0.001))
            if hasattr(cv2, "AKAZE_create"):
                detector = cv2.AKAZE_create(threshold=threshold)
            elif hasattr(cv2, "AKAZE"):
                detector = cv2.AKAZE.create(threshold=threshold)
            else:
                raise AttributeError("OpenCV build does not support AKAZE")
        else:
            raise ValueError(f"Unsupported feature extraction method: '{method}'")

        keypoints, descriptors = detector.detectAndCompute(img, None)
        return keypoints or [], descriptors

    def _match_descriptors(
        self, descs_a: np.ndarray, descs_b: np.ndarray, method: str
    ) -> Tuple[List[cv2.DMatch], List[cv2.DMatch]]:

        matcher_type = self.matching_cfg.get("matcher", "flann").lower()
        ratio_thresh = float(self.matching_cfg.get("ratio_threshold", 0.75))

        # Binary descriptors (ORB, AKAZE) vs Floating point (SIFT)
        is_binary = method in ("orb", "akaze")

        if matcher_type == "flann":
            if is_binary:
                # LSH Index for Binary Descriptors
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            else:
                # KDTREE Index for SIFT
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            try:
                knn_matches = flann.knnMatch(descs_a, descs_b, k=2)
            except cv2.Error:
                # Fallback to BFMatcher if FLANN fails on edge cases
                norm_type = cv2.NORM_HAMMING if is_binary else cv2.NORM_L2
                bf = cv2.BFMatcher(norm_type)
                knn_matches = bf.knnMatch(descs_a, descs_b, k=2)
        else:
            norm_type = cv2.NORM_HAMMING if is_binary else cv2.NORM_L2
            bf = cv2.BFMatcher(norm_type)
            knn_matches = bf.knnMatch(descs_a, descs_b, k=2)

        raw_matches: List[cv2.DMatch] = []
        filtered_matches: List[cv2.DMatch] = []

        for match_pair in knn_matches:
            if len(match_pair) > 0:
                raw_matches.append(match_pair[0])
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    filtered_matches.append(m)

        return raw_matches, filtered_matches
