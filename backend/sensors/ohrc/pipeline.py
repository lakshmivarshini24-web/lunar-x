"""
OHRC Real Correspondence Pipeline (Module 7)

Provides an end-to-end operational pipeline for Chandrayaan-2 OHRC image matching,
handling image ingestion, sidecar metadata parsing, non-destructive preprocessing,
large-raster tiling, multi-scale feature correspondence, geometric verification,
confidence estimation, and output rendering.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Dict, Any, Optional, Union, List
import numpy as np

from backend.core.lunar_image import LunarImage
from backend.sensors.ohrc.validator import OHRCValidator
from backend.sensors.ohrc.reader import load_ohrc
from backend.preprocessing.core import OHRCPreprocessor
from backend.preprocessing.tiling import ImageTiler, TileInfo
from backend.correspondence.classical import ClassicalCorrespondence, MatchResult
from backend.evaluation.metrics import EvaluationMetrics, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class OHRCRealPipelineResult:
    """Output container for OHRC Real Correspondence Pipeline."""

    lunar_a: LunarImage
    lunar_b: LunarImage
    prep_a: LunarImage
    prep_b: LunarImage
    match_result: MatchResult
    metrics: EvaluationMetrics
    execution_time_sec: float
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)


class OHRCRealCorrespondencePipeline:
    """
    End-to-end operational correspondence pipeline for OHRC imagery.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validator = OHRCValidator(self.config)
        self.preprocessor = OHRCPreprocessor(self.config)
        self.matcher = ClassicalCorrespondence(self.config)

    def process_pair(
        self,
        image_a_path: Union[str, Path],
        image_b_path: Union[str, Path],
        metadata_a_path: Optional[Union[str, Path]] = None,
        metadata_b_path: Optional[Union[str, Path]] = None,
        method_override: Optional[str] = None,
    ) -> OHRCRealPipelineResult:
        """
        Executes complete OHRC-to-OHRC correspondence workflow.

        Args:
            image_a_path: File path to OHRC Image A.
            image_b_path: File path to OHRC Image B.
            metadata_a_path: Optional metadata JSON for Image A.
            metadata_b_path: Optional metadata JSON for Image B.
            method_override: Optional feature matcher override ('sift', 'orb', 'akaze').

        Returns:
            OHRCRealPipelineResult containing preprocessed images, matches, and metrics.
        """
        start_time = time.time()
        logger.info(f"Starting OHRC Real Pipeline for: {image_a_path} and {image_b_path}")

        # 1. Validation
        val_a = self.validator.validate_file(str(image_a_path))
        val_b = self.validator.validate_file(str(image_b_path))

        if not val_a["is_valid"]:
            raise ValueError(f"Image A validation failed: {'; '.join(val_a['errors'])}")
        if not val_b["is_valid"]:
            raise ValueError(f"Image B validation failed: {'; '.join(val_b['errors'])}")

        # 2. Reading
        lunar_a = load_ohrc(image_a_path, self.config)
        lunar_b = load_ohrc(image_b_path, self.config)

        # 3. Preprocessing
        prep_a = self.preprocessor.process(lunar_a)
        prep_b = self.preprocessor.process(lunar_b)

        # 4. Tiling check for large images (> 1024x1024 or configured)
        tiling_cfg = self.config.get("tiling", {})
        h_max = max(lunar_a.height, lunar_b.height)
        w_max = max(lunar_a.width, lunar_b.width)
        use_tiling = tiling_cfg.get("enabled", False) or (h_max > 2048 or w_max > 2048)

        if use_tiling:
            tile_size = int(tiling_cfg.get("tile_size", 512))
            overlap = int(tiling_cfg.get("overlap", 128))
            tiler = ImageTiler(tile_size=tile_size, overlap=overlap)
            logger.info(f"Tiling enabled: Splitting OHRC images with tile_size={tile_size}, overlap={overlap}")
            tiles_a = tiler.tile(prep_a)
            tiles_b = tiler.tile(prep_b)
            # Match tiles or run global matcher
            match_result = self._match_tiled(tiles_a, tiles_b, method_override)
        else:
            match_result = self.matcher.match(prep_a, prep_b, method_override=method_override)

        # 5. Evaluation Metrics
        gt_homography = None
        sidecar_b = lunar_b.metadata.get("sidecar_metadata", {})
        if "synthetic_ground_truth" in sidecar_b:
            gt_homography = np.array(sidecar_b["synthetic_ground_truth"].get("ground_truth_homography"), dtype=np.float64)

        metrics = compute_metrics(match_result, lunar_a.shape[:2], ground_truth_homography=gt_homography)

        exec_time = time.time() - start_time
        logger.info(f"OHRC Real Pipeline completed in {exec_time:.3f}s. Inliers: {match_result.num_inliers}")

        return OHRCRealPipelineResult(
            lunar_a=lunar_a,
            lunar_b=lunar_b,
            prep_a=prep_a,
            prep_b=prep_b,
            match_result=match_result,
            metrics=metrics,
            execution_time_sec=exec_time,
            pipeline_metadata={
                "tiling_used": use_tiling,
                "validation_a": val_a,
                "validation_b": val_b,
            },
        )

    def _match_tiled(
        self, tiles_a: List[TileInfo], tiles_b: List[TileInfo], method_override: Optional[str]
    ) -> MatchResult:
        """Matches tiles across images and accumulates global inliers."""
        all_pts_a: List[np.ndarray] = []
        all_pts_b: List[np.ndarray] = []

        for tile_a in tiles_a:
            for tile_b in tiles_b:
                m_res = self.matcher.match(tile_a.lunar_image, tile_b.lunar_image, method_override=method_override)
                if m_res.num_inliers > 0:
                    g_pts_a = ImageTiler.tile_to_global_keypoints(m_res.inlier_pts_a, tile_a)
                    g_pts_b = ImageTiler.tile_to_global_keypoints(m_res.inlier_pts_b, tile_b)
                    all_pts_a.append(g_pts_a)
                    all_pts_b.append(g_pts_b)

        if not all_pts_a:
            return MatchResult(
                method=method_override or "sift",
                num_keypoints_a=0,
                num_keypoints_b=0,
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
            )

        cat_a = np.vstack(all_pts_a)
        cat_b = np.vstack(all_pts_b)

        # Global RANSAC on accumulated tiled matches
        import cv2

        H, mask = cv2.findHomography(cat_a, cat_b, cv2.RANSAC, 3.0)
        inliers_count = int(np.sum(mask)) if mask is not None else 0

        return MatchResult(
            method=method_override or "sift",
            num_keypoints_a=len(cat_a),
            num_keypoints_b=len(cat_b),
            num_raw_matches=len(cat_a),
            num_filtered_matches=len(cat_a),
            num_inliers=inliers_count,
            inlier_ratio=float(inliers_count / len(cat_a)) if len(cat_a) > 0 else 0.0,
            confidence_score=float(inliers_count / (len(cat_a) + 1e-5)),
            keypoints_a=cat_a,
            keypoints_b=cat_b,
            matched_pts_a=cat_a,
            matched_pts_b=cat_b,
            inlier_pts_a=cat_a[mask.ravel().astype(bool)] if mask is not None else np.empty((0, 2)),
            inlier_pts_b=cat_b[mask.ravel().astype(bool)] if mask is not None else np.empty((0, 2)),
            inlier_mask=mask.ravel().astype(bool) if mask is not None else np.array([], dtype=bool),
            homography=H,
        )
