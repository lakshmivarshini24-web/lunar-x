"""
Tri-Sensor Visualization Panel Generator

Renders a 4-way multi-panel visualization report showing:
1. OHRC ↔ TMC-2 Matches
2. OHRC ↔ IIRS Cross-Modal Matches
3. TMC-2 ↔ IIRS Cross-Modal Matches
4. Tri-Sensor Consensus Overlap & Loop Closure Statistics
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt

from backend.core.lunar_image import LunarImage
from backend.matching.tri_sensor import TriSensorMatchResult


def visualize_tri_sensor_correspondence(
    lunar_ohrc: LunarImage,
    lunar_tmc2: LunarImage,
    lunar_iirs: LunarImage,
    tri_result: TriSensorMatchResult,
    output_path: str,
    dataset_type: str = "SYNTHETIC_TRISENSOR_TRIPLET",
):
    """
    Renders 4-panel visual report for tri-sensor correspondence across OHRC, TMC-2, and IIRS.
    """
    img_ohrc = lunar_ohrc.to_uint8()
    img_tmc2 = lunar_tmc2.to_uint8()
    img_iirs = lunar_iirs.to_uint8()

    r1 = tri_result.res_ohrc_tmc2
    r2 = tri_result.res_ohrc_iirs
    r3 = tri_result.res_tmc2_iirs

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"LUNAR-X Unified Tri-Sensor Correspondence Consensus\n"
        f"OHRC ({lunar_ohrc.resolution_m}m) ↔ TMC-2 ({lunar_tmc2.resolution_m}m) ↔ IIRS ({lunar_iirs.resolution_m}m)\n"
        f"Data Source: {dataset_type} | Tri-Sensor Agreement Score: {tri_result.tri_sensor_agreement_score:.3f} | "
        f"Loop Closure Error: {tri_result.loop_closure_error_px:.2f}px | Consensus Valid: {tri_result.is_consensus_valid}",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: OHRC ↔ TMC-2 Matches
    canvas_1 = draw_match_canvas(img_ohrc, img_tmc2, r1.inlier_pts_a_native, r1.inlier_pts_b_native)
    axes[0, 0].imshow(cv2.cvtColor(canvas_1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"1. OHRC ↔ TMC-2 ({r1.num_inliers} Inliers | Ratio: {r1.inlier_ratio*100:.1f}%)")
    axes[0, 0].axis("off")

    # Panel 2: OHRC ↔ IIRS Cross-Modal Matches
    canvas_2 = draw_match_canvas(img_ohrc, img_iirs, r2.inlier_pts_a_native, r2.inlier_pts_b_native)
    axes[0, 1].imshow(cv2.cvtColor(canvas_2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"2. OHRC ↔ IIRS Cross-Modal ({r2.num_inliers} Inliers | Ratio: {r2.inlier_ratio*100:.1f}%)")
    axes[0, 1].axis("off")

    # Panel 3: TMC-2 ↔ IIRS Cross-Modal Matches
    canvas_3 = draw_match_canvas(img_tmc2, img_iirs, r3.inlier_pts_a_native, r3.inlier_pts_b_native)
    axes[1, 0].imshow(cv2.cvtColor(canvas_3, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"3. TMC-2 ↔ IIRS Cross-Modal ({r3.num_inliers} Inliers | Ratio: {r3.inlier_ratio*100:.1f}%)")
    axes[1, 0].axis("off")

    # Panel 4: Tri-Sensor Triangulation & Alignment Overview
    axes[1, 1].axis("off")
    summary_text = (
        f"TRI-SENSOR CONSENSUS METRICS\n"
        f"--------------------------------------------------\n"
        f"OHRC Resolution   : {lunar_ohrc.resolution_m} m/pixel\n"
        f"TMC-2 Resolution  : {lunar_tmc2.resolution_m} m/pixel\n"
        f"IIRS Resolution   : {lunar_iirs.resolution_m} m/pixel\n\n"
        f"Pairwise Inlier Counts:\n"
        f"  - OHRC ↔ TMC-2  : {r1.num_inliers} inliers (Conf: {r1.confidence_score:.3f})\n"
        f"  - OHRC ↔ IIRS   : {r2.num_inliers} inliers (Conf: {r2.confidence_score:.3f})\n"
        f"  - TMC-2 ↔ IIRS  : {r3.num_inliers} inliers (Conf: {r3.confidence_score:.3f})\n\n"
        f"Geometric Loop Closure Evaluation:\n"
        f"  - Loop Error    : {tri_result.loop_closure_error_px:.4f} pixels\n"
        f"  - Consensus     : {'PASSED' if tri_result.is_consensus_valid else 'FAILED'}\n"
        f"  - Agreement Score: {tri_result.tri_sensor_agreement_score:.4f}\n"
        f"  - Execution Time : {tri_result.total_execution_time_sec:.3f} seconds\n"
        f"--------------------------------------------------"
    )
    axes[1, 1].text(
        0.05,
        0.5,
        summary_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="gainsboro", alpha=0.8),
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(Path(output_path).resolve()), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Tri-Sensor Visualization Panel to: '{output_path}'")


def draw_match_canvas(img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Helper to draw side-by-side match canvas with green inlier lines."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1
    canvas[:h2, w1 : w1 + w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2

    if len(pts1) > 0 and len(pts2) > 0:
        for (x1, y1), (x2, y2) in zip(pts1, pts2):
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2 + w1), int(y2))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, (255, 0, 0), -1)
            cv2.circle(canvas, pt2, 3, (0, 0, 255), -1)

    return canvas
