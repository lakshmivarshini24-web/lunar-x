"""
Cross-Sensor Correspondence Visualization (Module 10)

Renders high-quality multi-panel visualization reports for OHRC ↔ TMC-2 cross-sensor matching:
1. Side-by-side raw input images with sensor metadata overlays.
2. Cross-sensor inlier feature match lines connecting OHRC & TMC-2.
3. Registered image overlay (Checkerboard / Alpha Blend).
4. Alignment error heatmap / difference visualization.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt

from backend.core.lunar_image import LunarImage
from backend.matching.cross_sensor import CrossSensorMatchResult


def visualize_cross_sensor_correspondence(
    lunar_ohrc: LunarImage,
    lunar_tmc2: LunarImage,
    match_result: CrossSensorMatchResult,
    metrics,
    output_path: str,
    dataset_type: str = "SYNTHETIC_CROSS_SENSOR_PAIR",
):
    """
    Renders 4-panel visual report for OHRC ↔ TMC-2 matching.
    """
    img_ohrc = lunar_ohrc.to_uint8()
    img_tmc2 = lunar_tmc2.to_uint8()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"LUNAR-X Cross-Sensor Correspondence: OHRC ({lunar_ohrc.resolution_m}m/px) ↔ TMC-2 ({lunar_tmc2.resolution_m}m/px)\n"
        f"Data Source: {dataset_type} | Algorithm: {match_result.method.upper()} | Inliers: {match_result.num_inliers} | "
        f"Inlier Ratio: {match_result.inlier_ratio*100:.1f}% | Reproj Error: {metrics.mean_reprojection_error_px:.2f}px",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: OHRC Raw Input
    axes[0, 0].imshow(img_ohrc, cmap="gray")
    axes[0, 0].set_title(f"OHRC (High-Res): {lunar_ohrc.width}x{lunar_ohrc.height} ({lunar_ohrc.resolution_m}m/px)")
    axes[0, 0].axis("off")

    # Panel 2: TMC-2 Raw Input
    axes[0, 1].imshow(img_tmc2, cmap="gray")
    axes[0, 1].set_title(f"TMC-2 (Context): {lunar_tmc2.width}x{lunar_tmc2.height} ({lunar_tmc2.resolution_m}m/px)")
    axes[0, 1].axis("off")

    # Panel 3: Cross-Sensor Inlier Match Overlay
    h_a, w_a = img_ohrc.shape
    h_b, w_b = img_tmc2.shape
    canvas_h = max(h_a, h_b)
    canvas_w = w_a + w_b
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[:h_a, :w_a] = cv2.cvtColor(img_ohrc, cv2.COLOR_GRAY2BGR)
    canvas[:h_b, w_a : w_a + w_b] = cv2.cvtColor(img_tmc2, cv2.COLOR_GRAY2BGR)

    pts_ohrc = match_result.inlier_pts_a_native
    pts_tmc2 = match_result.inlier_pts_b_native

    for (x1, y1), (x2, y2) in zip(pts_ohrc, pts_tmc2):
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2 + w_a), int(y2))
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, (255, 0, 0), -1)
        cv2.circle(canvas, pt2, 3, (0, 0, 255), -1)

    axes[1, 0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Cross-Sensor Inliers ({len(pts_ohrc)} matched pairs across 20x scale jump)")
    axes[1, 0].axis("off")

    # Panel 4: Registered Overlay (Warped OHRC aligned onto TMC-2 space)
    if match_result.homography_native is not None and len(pts_ohrc) >= 4:
        warped_ohrc = cv2.warpPerspective(img_ohrc, match_result.homography_native, (w_b, h_b))
        blend = cv2.addWeighted(img_tmc2, 0.5, warped_ohrc, 0.5, 0)
        axes[1, 1].imshow(blend, cmap="gray")
        axes[1, 1].set_title("Registered Alpha Blend (Warped OHRC onto TMC-2 Context)")
    else:
        diff_placeholder = np.zeros_like(img_tmc2)
        axes[1, 1].imshow(diff_placeholder, cmap="gray")
        axes[1, 1].set_title("Registration unavailable (insufficient inliers)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(Path(output_path).resolve()), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Cross-Sensor visualization to: '{output_path}'")
