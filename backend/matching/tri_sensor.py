"""
Tri-Sensor Correspondence Consensus Framework

Unifies pairwise correspondences across all three Chandrayaan-2 instruments:
OHRC (0.25m) ↔ TMC-2 (5.0m) ↔ IIRS (100.0m).

Calculates geometric loop closure consistency errors across sensor triplets:
H_loop = H_(TMC-2->IIRS) @ H_(OHRC->TMC-2) @ inv(H_(OHRC->IIRS))
and determines the tri-sensor consensus agreement score and shared overlap regions.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from backend.core.lunar_image import LunarImage
from backend.matching.cross_sensor import CrossSensorMatcher, CrossSensorMatchResult
from backend.matching.cross_modal import CrossModalMatcher


@dataclass
class TriSensorMatchResult:
    """Container for Tri-Sensor correspondence consensus output."""

    res_ohrc_tmc2: CrossSensorMatchResult
    res_ohrc_iirs: CrossSensorMatchResult
    res_tmc2_iirs: CrossSensorMatchResult
    loop_closure_error_px: float
    tri_sensor_agreement_score: float
    is_consensus_valid: bool
    total_execution_time_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TriSensorCorrespondenceEngine:
    """
    Unified Tri-Sensor correspondence engine across OHRC, TMC-2, and IIRS.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        tri_cfg = self.config.get("tri_sensor", {})
        self.max_loop_err_px = float(tri_cfg.get("max_loop_closure_error_px", 10.0))
        self.cross_sensor_matcher = CrossSensorMatcher(self.config)
        self.cross_modal_matcher = CrossModalMatcher(self.config)

    def process_triplet(
        self, lunar_ohrc: LunarImage, lunar_tmc2: LunarImage, lunar_iirs: LunarImage
    ) -> TriSensorMatchResult:
        """
        Executes complete tri-sensor correspondence across OHRC, TMC-2, and IIRS.

        Args:
            lunar_ohrc: High-resolution OHRC image (~0.25 m/px).
            lunar_tmc2: Stereo context TMC-2 image (~5.0 m/px).
            lunar_iirs: Hyperspectral IIRS image (~100.0 m/px).

        Returns:
            TriSensorMatchResult containing pairwise outputs, loop closure error, and consensus score.
        """
        start_time = time.time()

        # 1. Pair 1: OHRC ↔ TMC-2
        m_ohrc_tmc2 = self.cross_sensor_matcher.match(lunar_ohrc, lunar_tmc2)

        # 2. Pair 2: OHRC ↔ IIRS
        m_ohrc_iirs = self.cross_modal_matcher.match(lunar_ohrc, lunar_iirs)

        # 3. Pair 3: TMC-2 ↔ IIRS
        m_tmc2_iirs = self.cross_modal_matcher.match(lunar_tmc2, lunar_iirs)

        # 4. Geometric Loop Closure Analysis
        loop_err_px, consensus_valid = self.evaluate_loop_closure(
            m_ohrc_tmc2.homography_native,
            m_ohrc_iirs.homography_native,
            m_tmc2_iirs.homography_native,
            lunar_ohrc.shape[:2],
        )

        # Calculate Tri-Sensor Agreement Consensus Score
        c1 = m_ohrc_tmc2.confidence_score
        c2 = m_ohrc_iirs.confidence_score
        c3 = m_tmc2_iirs.confidence_score
        mean_pairwise_conf = float((c1 + c2 + c3) / 3.0)

        loop_penalty = float(np.exp(-loop_err_px / 10.0)) if consensus_valid else 0.0
        agreement_score = float(np.clip(mean_pairwise_conf * loop_penalty, 0.0, 1.0))

        total_time = time.time() - start_time

        return TriSensorMatchResult(
            res_ohrc_tmc2=m_ohrc_tmc2,
            res_ohrc_iirs=m_ohrc_iirs,
            res_tmc2_iirs=m_tmc2_iirs,
            loop_closure_error_px=loop_err_px,
            tri_sensor_agreement_score=agreement_score,
            is_consensus_valid=consensus_valid,
            total_execution_time_sec=total_time,
            metadata={
                "pairwise_confidences": [c1, c2, c3],
                "max_loop_closure_allowed_px": self.max_loop_err_px,
            },
        )

    def evaluate_loop_closure(
        self,
        H_ohrc_tmc2: Optional[np.ndarray],
        H_ohrc_iirs: Optional[np.ndarray],
        H_tmc2_iirs: Optional[np.ndarray],
        image_shape: Tuple[int, int],
    ) -> Tuple[float, bool]:
        """
        Evaluates loop closure consistency:
        H_expected = H_(TMC-2->IIRS) @ H_(OHRC->TMC-2)
        H_direct   = H_(OHRC->IIRS)
        Calculates pixel deviation between H_expected and H_direct projections.

        Returns:
            (mean_loop_closure_error_px, is_consensus_valid)
        """
        if H_ohrc_tmc2 is None or H_ohrc_iirs is None or H_tmc2_iirs is None:
            return 999.0, False

        h, w = image_shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        corners_homo = np.hstack([corners, np.ones((4, 1))])

        # Project via direct path: OHRC -> IIRS
        p_direct_homo = (H_ohrc_iirs @ corners_homo.T).T
        p_direct = p_direct_homo[:, :2] / p_direct_homo[:, 2:3]

        # Project via indirect path: OHRC -> TMC-2 -> IIRS
        H_indirect = H_tmc2_iirs @ H_ohrc_tmc2
        p_indirect_homo = (H_indirect @ corners_homo.T).T
        p_indirect = p_indirect_homo[:, :2] / p_indirect_homo[:, 2:3]

        diffs = np.linalg.norm(p_direct - p_indirect, axis=1)
        mean_err = float(np.mean(diffs))
        is_valid = bool(mean_err <= self.max_loop_err_px)

        return round(mean_err, 4), is_valid
