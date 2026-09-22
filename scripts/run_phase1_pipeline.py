"""
Phase 1 Complete OHRC Correspondence Pipeline Runner

Executes end-to-end Phase 1 workflow:
1. Dataset Validation & Manifest Updating (Module 1)
2. Data Reading into LunarImage representation (Module 2)
3. Non-destructive Preprocessing (Module 3)
4. Tiling & Coordinate Tracking (Module 4)
5. Classical SIFT Baseline Correspondence & RANSAC Homography (Module 5)
6. Evaluation Metrics Computation & Benchmark Suite (Module 6)
7. Correspondence Visualization & Report Generation
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

from backend.core.lunar_image import LunarImage
from backend.core.manifest import DatasetManifest
from backend.sensors.ohrc.validator import OHRCValidator
from backend.sensors.ohrc.reader import load_ohrc
from backend.preprocessing.core import OHRCPreprocessor
from backend.preprocessing.tiling import ImageTiler
from backend.correspondence.classical import ClassicalCorrespondence
from backend.evaluation.metrics import compute_metrics
from backend.evaluation.benchmark import BenchmarkRunner


def load_config(config_path: str = "config.yaml") -> dict:
    """Loads central YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def visualize_correspondence(
    lunar_a: LunarImage,
    lunar_b: LunarImage,
    prep_a: LunarImage,
    prep_b: LunarImage,
    match_result,
    metrics,
    output_path: str,
):
    """Generates and saves a high-quality visualization panel for Phase 1 results."""
    img_a = lunar_a.to_uint8()
    img_b = lunar_b.to_uint8()
    p_img_a = prep_a.to_uint8()
    p_img_b = prep_b.to_uint8()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"LUNAR-X Phase 1 - OHRC Correspondence ({match_result.method.upper()} Baseline)\n"
        f"Inliers: {metrics.num_inliers}/{metrics.num_filtered_matches} | Inlier Ratio: {metrics.inlier_ratio:.2f} | "
        f"RMSE: {metrics.rmse_px:.2f}px | Coverage: {metrics.spatial_coverage_ratio:.2f}",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Raw Input Images
    axes[0, 0].imshow(img_a, cmap="gray")
    axes[0, 0].set_title(f"Image A Raw: {lunar_a.shape[1]}x{lunar_a.shape[0]}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_b, cmap="gray")
    axes[0, 1].set_title(f"Image B Raw: {lunar_b.shape[1]}x{lunar_b.shape[0]}")
    axes[0, 1].axis("off")

    # 2. Preprocessed Images
    axes[1, 0].imshow(p_img_a, cmap="gray")
    axes[1, 0].set_title("Image A Preprocessed (CLAHE + Illumination Norm)")
    axes[1, 0].axis("off")

    # 3. Match Visualization Overlay (Side-by-side)
    h_a, w_a = p_img_a.shape
    h_b, w_b = p_img_b.shape
    canvas_h = max(h_a, h_b)
    canvas_w = w_a + w_b
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[:h_a, :w_a] = cv2.cvtColor(p_img_a, cv2.COLOR_GRAY2BGR)
    canvas[:h_b, w_a : w_a + w_b] = cv2.cvtColor(p_img_b, cv2.COLOR_GRAY2BGR)

    pts_a = match_result.inlier_pts_a
    pts_b = match_result.inlier_pts_b

    # Draw inlier correspondence lines
    for (x1, y1), (x2, y2) in zip(pts_a, pts_b):
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2 + w_a), int(y2))
        color = (0, 255, 0)  # Green for inliers
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, (255, 0, 0), -1)
        cv2.circle(canvas, pt2, 3, (0, 0, 255), -1)

    axes[1, 1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Inlier Correspondences ({len(pts_a)} matched pairs)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(Path(output_path).resolve()), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correspondence visualization panel to: '{output_path}'")


def run_pipeline(config_path: str = "config.yaml", image_a_path: str = None, image_b_path: str = None):
    """Executes Phase 1 Pipeline end-to-end."""
    print("=================================================================")
    print(" LUNAR-X (ISRO SIH 2026) - Phase 1 Pipeline Execution")
    print("=================================================================")

    # 1. Load Configuration
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    manifest_dir = paths_cfg.get("manifests_dir", "data/manifests")
    out_dir = Path(paths_cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Determine Image Input Paths
    raw_dir = Path(paths_cfg.get("raw_dir", "data/raw"))
    synthetic_dir = raw_dir / "synthetic_ohrc"

    if image_a_path is None or image_b_path is None:
        path_a = synthetic_dir / "ohrc_synthetic_01_A.tif"
        path_b = synthetic_dir / "ohrc_synthetic_01_B.tif"

        if not path_a.exists() or not path_b.exists():
            print("\n[!] Synthetic dataset not found. Generating synthetic OHRC pairs...")
            from scripts.generate_synthetic_data import create_synthetic_pair

            create_synthetic_pair(synthetic_dir, pair_id=1)
            create_synthetic_pair(synthetic_dir, pair_id=2)

        image_a_path = str(path_a)
        image_b_path = str(path_b)

    print(f"Input Image A: {image_a_path}")
    print(f"Input Image B: {image_b_path}\n")

    # 3. Module 1: Validation & Dataset Manifesting
    print("--> Module 1: Validating OHRC files & updating dataset manifest...")
    validator = OHRCValidator(cfg)
    manifest = DatasetManifest(manifest_name="ohrc_manifest.json", manifests_dir=manifest_dir)

    val_a = validator.validate_file(image_a_path)
    val_b = validator.validate_file(image_b_path)

    manifest.add_entry(image_a_path, val_a)
    manifest.add_entry(image_b_path, val_b)
    manifest.save()
    print(f"    Manifest updated at: {manifest.manifest_path}")

    # 4. Module 2: OHRC Reader & LunarImage Data Structure
    print("\n--> Module 2: Loading OHRC datasets into LunarImage representation...")
    lunar_a = load_ohrc(image_a_path, cfg)
    lunar_b = load_ohrc(image_b_path, cfg)
    print(f"    Image A summary: {lunar_a.summary()}")
    print(f"    Image B summary: {lunar_b.summary()}")

    # 5. Module 3: Preprocessing Core
    print("\n--> Module 3: Applying non-destructive preprocessing pipeline...")
    preprocessor = OHRCPreprocessor(cfg)
    prep_a = preprocessor.process(lunar_a)
    prep_b = preprocessor.process(lunar_b)
    print("    Applied CLAHE, Illumination Normalization, and Percentile Stretch.")

    # 6. Module 4: Tiling (Optional check)
    tiling_cfg = cfg.get("tiling", {})
    if tiling_cfg.get("enabled", False):
        print("\n--> Module 4: Decomposing images into sub-tiles...")
        tiler = ImageTiler(tile_size=tiling_cfg.get("tile_size", 512), overlap=tiling_cfg.get("overlap", 128))
        tiles_a = tiler.tile(prep_a)
        print(f"    Extracted {len(tiles_a)} tiles from Image A.")

    # 7. Module 5: Classical SIFT Correspondence Baseline
    print("\n--> Module 5: Running SIFT Feature Correspondence Baseline...")
    matcher = ClassicalCorrespondence(cfg)
    match_result = matcher.match(prep_a, prep_b)
    print(f"    Keypoints A: {match_result.num_keypoints_a} | Keypoints B: {match_result.num_keypoints_b}")
    print(f"    Raw Matches: {match_result.num_raw_matches} | Filtered Ratio Test: {match_result.num_filtered_matches}")
    print(f"    RANSAC Inliers: {match_result.num_inliers} ({match_result.inlier_ratio*100:.1f}%)")
    print(f"    Match Confidence: {match_result.confidence_score:.3f}")

    # 8. Module 6: Quantitative Evaluation & Benchmarking
    print("\n--> Module 6: Computing Quantitative Evaluation Metrics & Benchmarking...")
    # Load ground truth homography if available in sidecar metadata
    gt_homography = None
    sidecar_meta_b = lunar_b.metadata.get("sidecar_metadata", {})
    gt_info = sidecar_meta_b.get("synthetic_ground_truth", {})
    if "ground_truth_homography" in gt_info:
        gt_homography = np.array(gt_info["ground_truth_homography"], dtype=np.float64)
        print("    Found synthetic ground-truth homography in sidecar metadata.")

    metrics = compute_metrics(match_result, lunar_a.shape[:2], ground_truth_homography=gt_homography)
    print("\n---------------- PHASE 1 EVALUATION METRICS ----------------")
    for k, v in metrics.to_dict().items():
        print(f"  {k:30s}: {v}")
    print("------------------------------------------------------------\n")

    # Run Benchmark across SIFT, ORB, AKAZE
    benchmarks_dir = out_dir / "benchmarks"
    runner = BenchmarkRunner(cfg)
    bench_results = runner.run_benchmark(
        prep_a, prep_b, algorithms=["sift", "orb", "akaze"], ground_truth_homography=gt_homography
    )
    json_rep, csv_rep = runner.save_reports(bench_results, output_dir=str(benchmarks_dir))
    print(f"    Benchmark reports exported to: '{json_rep}' and '{csv_rep}'")

    # 9. Visualization Output
    vis_path = out_dir / "phase1_correspondence_result.png"
    visualize_correspondence(lunar_a, lunar_b, prep_a, prep_b, match_result, metrics, str(vis_path))

    print("\n[+] Phase 1 Pipeline execution completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LUNAR-X Phase 1 Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--image_a", type=str, default=None, help="Path to Image A")
    parser.add_argument("--image_b", type=str, default=None, help="Path to Image B")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, image_a_path=args.image_a, image_b_path=args.image_b)
