"""
Phase 2 Complete Cross-Sensor Pipeline Runner (OHRC ↔ TMC-2)

Executes Phase 2 workflow end-to-end:
1. OHRC Real Pipeline Execution (Module 7)
2. TMC-2 Ingestion, Validation & Preprocessing (Module 8)
3. Multi-Scale Cross-Sensor Matching between OHRC (~0.25m) and TMC-2 (~5.0m) (Module 9)
4. Cross-Sensor Evaluation, Benchmark Export, and Visual Panel Rendering (Module 10)
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np

from backend.core.manifest import DatasetManifest
from backend.sensors.ohrc.reader import load_ohrc
from backend.sensors.ohrc.pipeline import OHRCRealCorrespondencePipeline
from backend.sensors.tmc2.validator import TMC2Validator
from backend.sensors.tmc2.reader import load_tmc2
from backend.sensors.tmc2.preprocessor import TMC2Preprocessor
from backend.matching.cross_sensor import CrossSensorMatcher
from backend.evaluation.cross_sensor_eval import CrossSensorEvaluator
from backend.visualization.cross_sensor_vis import visualize_cross_sensor_correspondence


def load_config(config_path: str = "config.yaml") -> dict:

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_phase2_pipeline(
    config_path: str = "config.yaml",
    ohrc_path: Optional[str] = None,
    tmc2_path: Optional[str] = None,
):
    """Executes Phase 2 Cross-Sensor Pipeline end-to-end."""
    print("=================================================================")
    print(" LUNAR-X (ISRO SIH 2026) - Phase 2 Pipeline Execution")
    print(" OHRC (0.25 m/px) <-> TMC-2 (5.0 m/px) Cross-Sensor Matching")
    print("=================================================================")

    # 1. Configuration & Paths
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    manifest_dir = paths_cfg.get("manifests_dir", "data/manifests")
    out_dir = Path(paths_cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(paths_cfg.get("raw_dir", "data/raw"))
    cross_dir = raw_dir / "synthetic_cross_sensor"

    # 2. Check / Generate Synthetic Cross-Sensor Pair if not provided
    if ohrc_path is None or tmc2_path is None:
        path_ohrc = cross_dir / "ohrc_cross_01.tif"
        path_tmc2 = cross_dir / "tmc2_cross_01.tif"

        if not path_ohrc.exists() or not path_tmc2.exists():
            print("\n[!] Cross-sensor test pair not found. Generating synthetic OHRC <-> TMC-2 pairs...")
            from scripts.generate_cross_sensor_data import generate_cross_sensor_pair

            generate_cross_sensor_pair(cross_dir, pair_id=1)
            generate_cross_sensor_pair(cross_dir, pair_id=2)

        ohrc_path = str(path_ohrc)
        tmc2_path = str(path_tmc2)

    print(f"\nInput OHRC Image  : {ohrc_path}")
    print(f"Input TMC-2 Image : {tmc2_path}")

    # 3. Module 7: OHRC Real Pipeline Test (OHRC <-> OHRC operational check)
    print("\n--> Module 7: Operational OHRC Real Pipeline Verification...")
    ohrc_pipeline = OHRCRealCorrespondencePipeline(cfg)
    # Validate loading single OHRC image
    lunar_ohrc = load_ohrc(ohrc_path, cfg)
    print(f"    Loaded OHRC image: {lunar_ohrc.shape} at {lunar_ohrc.resolution_m} m/px")

    # 4. Module 8: TMC-2 Data Reader, Validator & Preprocessor
    print("\n--> Module 8: TMC-2 Data Ingestion, Validation & Preprocessing...")
    tmc2_validator = TMC2Validator(cfg)
    val_tmc2 = tmc2_validator.validate_file(tmc2_path)

    tmc2_manifest = DatasetManifest(manifest_name="tmc2_manifest.json", manifests_dir=manifest_dir)
    tmc2_manifest.add_entry(tmc2_path, val_tmc2)
    tmc2_manifest.save()
    print(f"    TMC-2 Manifest updated at: {tmc2_manifest.manifest_path}")

    lunar_tmc2 = load_tmc2(tmc2_path, cfg)
    print(f"    Loaded TMC-2 image: {lunar_tmc2.shape} at {lunar_tmc2.resolution_m} m/px")

    tmc2_prep = TMC2Preprocessor(cfg)
    prep_tmc2 = tmc2_prep.process(lunar_tmc2)
    print("    Applied TMC-2 contrast stretch and terrain illumination normalization.")

    # 5. Module 9: OHRC ↔ TMC-2 Cross-Sensor Multi-Scale Matching
    print("\n--> Module 9: Executing Cross-Sensor Matching (OHRC 0.25m <-> TMC-2 5.0m)...")
    cs_matcher = CrossSensorMatcher(cfg)
    cs_result = cs_matcher.match(lunar_ohrc, prep_tmc2)

    print(f"    Resolution Scale Ratio: {cs_result.scale_ratio:.1f}x")
    print(f"    Keypoints OHRC: {cs_result.num_keypoints_a_native} | Keypoints TMC-2: {cs_result.num_keypoints_b_native}")
    print(f"    Raw Matches: {cs_result.num_raw_matches} | Filtered Matches: {cs_result.num_filtered_matches}")
    print(f"    RANSAC Cross-Sensor Inliers: {cs_result.num_inliers} ({cs_result.inlier_ratio*100:.1f}%)")
    print(f"    Match Confidence Score: {cs_result.confidence_score:.3f}")

    # 6. Module 10: Cross-Sensor Evaluation & Benchmarking
    print("\n--> Module 10: Cross-Sensor Evaluation & Visualization Reporting...")
    gt_homography = None
    sidecar_tmc2 = lunar_tmc2.metadata.get("sidecar_metadata", {})
    dataset_type = sidecar_tmc2.get("dataset_type", "SYNTHETIC_CROSS_SENSOR_PAIR")

    if "synthetic_ground_truth" in sidecar_tmc2:
        gt_info = sidecar_tmc2["synthetic_ground_truth"]
        gt_homography = np.array(gt_info["ground_truth_homography"], dtype=np.float64)
        print("    Found synthetic ground-truth homography in TMC-2 sidecar metadata.")

    evaluator = CrossSensorEvaluator(cfg)
    metrics = evaluator.evaluate(
        match_result=cs_result,
        lunar_ohrc=lunar_ohrc,
        lunar_tmc2=lunar_tmc2,
        ground_truth_homography=gt_homography,
        dataset_type=dataset_type,
    )

    print("\n---------------- PHASE 2 CROSS-SENSOR EVALUATION METRICS ----------------")
    for k, v in metrics.to_dict().items():
        print(f"  {k:30s}: {v}")
    print("--------------------------------------------------------------------------\n")

    # Export benchmark report
    benchmarks_dir = out_dir / "benchmarks"
    j_rep, c_rep = evaluator.save_report(metrics, output_dir=str(benchmarks_dir), filename_prefix="cross_sensor_report")
    print(f"    Cross-sensor report exported to: '{j_rep}' and '{c_rep}'")

    # Save 4-panel visual report
    vis_path = out_dir / "phase2_cross_sensor_result.png"
    visualize_cross_sensor_correspondence(
        lunar_ohrc=lunar_ohrc,
        lunar_tmc2=lunar_tmc2,
        match_result=cs_result,
        metrics=metrics,
        output_path=str(vis_path),
        dataset_type=dataset_type,
    )

    print("\n[+] Phase 2 Cross-Sensor Pipeline execution completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LUNAR-X Phase 2 Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--ohrc", type=str, default=None, help="Path to OHRC image")
    parser.add_argument("--tmc2", type=str, default=None, help="Path to TMC-2 image")
    args = parser.parse_args()

    run_phase2_pipeline(config_path=args.config, ohrc_path=args.ohrc, tmc2_path=args.tmc2)
