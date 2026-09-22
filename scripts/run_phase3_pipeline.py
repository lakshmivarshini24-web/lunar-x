"""
Phase 3 Tri-Sensor Correspondence Pipeline Runner (OHRC + TMC-2 + IIRS)

Executes Phase 3 workflow:
1. IIRS Hyperspectral Ingestion & Validation (Module 11)
2. Bad-Band Detection & SVD-based PCA Reduction (Module 12)
3. OHRC ↔ IIRS & TMC-2 ↔ IIRS Cross-Modal Correspondence (Modules 13 & 14)
4. Unified Tri-Sensor Consensus Framework & Geometric Loop Closure Analysis
5. Tri-Sensor Evaluation Report Export & 4-Panel Visualization
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
from backend.sensors.tmc2.reader import load_tmc2
from backend.sensors.iirs.validator import IIRSValidator
from backend.sensors.iirs.reader import load_iirs
from backend.sensors.iirs.preprocessor import IIRSPreprocessor
from backend.matching.tri_sensor import TriSensorCorrespondenceEngine
from backend.evaluation.tri_sensor_eval import TriSensorEvaluator
from backend.visualization.tri_sensor_vis import visualize_tri_sensor_correspondence


def load_config(config_path: str = "config.yaml") -> dict:

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_phase3_pipeline(
    config_path: str = "config.yaml",
    ohrc_path: Optional[str] = None,
    tmc2_path: Optional[str] = None,
    iirs_path: Optional[str] = None,
):
    """Executes Phase 3 Tri-Sensor Pipeline end-to-end."""
    print("=================================================================")
    print(" LUNAR-X (ISRO SIH 2026) - Phase 3 Tri-Sensor Pipeline Execution")
    print(" OHRC (0.25m) <-> TMC-2 (5.0m) <-> IIRS (100.0m Hyperspectral)")
    print("=================================================================")

    # 1. Configuration & Paths
    cfg = load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    manifest_dir = paths_cfg.get("manifests_dir", "data/manifests")
    out_dir = Path(paths_cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(paths_cfg.get("raw_dir", "data/raw"))
    tri_dir = raw_dir / "synthetic_trisensor"

    # 2. Check / Generate Synthetic Tri-Sensor Triplet if not provided
    if ohrc_path is None or tmc2_path is None or iirs_path is None:
        p_ohrc = tri_dir / "ohrc_tri_01.tif"
        p_tmc2 = tri_dir / "tmc2_tri_01.tif"
        p_iirs = tri_dir / "iirs_tri_01.npy"

        if not p_ohrc.exists() or not p_tmc2.exists() or not p_iirs.exists():
            print("\n[!] Synthetic tri-sensor triplet not found. Generating synthetic test datasets...")
            from scripts.generate_trisensor_data import generate_trisensor_triplet

            generate_trisensor_triplet(tri_dir, triplet_id=1)
            generate_trisensor_triplet(tri_dir, triplet_id=2)

        ohrc_path = str(p_ohrc)
        tmc2_path = str(p_tmc2)
        iirs_path = str(p_iirs)

    print(f"\nInput OHRC Image : {ohrc_path}")
    print(f"Input TMC-2 Image: {tmc2_path}")
    print(f"Input IIRS Raster: {iirs_path}")

    # 3. Module 11: IIRS Ingestion, Validation & Dataset Manifesting
    print("\n--> Module 11: Validating IIRS hyperspectral raster & updating manifest...")
    iirs_validator = IIRSValidator(cfg)
    val_iirs = iirs_validator.validate_file(iirs_path)

    iirs_manifest = DatasetManifest(manifest_name="iirs_manifest.json", manifests_dir=manifest_dir)
    iirs_manifest.add_entry(iirs_path, val_iirs)
    iirs_manifest.save()
    print(f"    IIRS Manifest updated at: {iirs_manifest.manifest_path}")

    lunar_ohrc = load_ohrc(ohrc_path, cfg)
    lunar_tmc2 = load_tmc2(tmc2_path, cfg)
    lunar_iirs_raw = load_iirs(iirs_path, cfg)
    print(f"    Loaded OHRC : {lunar_ohrc.shape} at {lunar_ohrc.resolution_m} m/px")
    print(f"    Loaded TMC-2: {lunar_tmc2.shape} at {lunar_tmc2.resolution_m} m/px")
    print(f"    Loaded IIRS : {lunar_iirs_raw.shape} ({val_iirs['num_bands']} bands) at {lunar_iirs_raw.resolution_m} m/px")

    # 4. Module 12: IIRS Bad-Band Filtering & PCA Dimensionality Reduction
    print("\n--> Module 12: Executing IIRS Spectral Preprocessing & PCA Dimensionality Reduction...")
    iirs_preprocessor = IIRSPreprocessor(cfg)
    lunar_iirs_pc1, iirs_diag = iirs_preprocessor.process(lunar_iirs_raw)
    print(f"    PCA Cumulative Explained Variance: {iirs_diag['cumulative_explained_variance']*100:.2f}%")
    print(f"    Filtered {iirs_diag['num_bad_bands']} bad bands out of {iirs_diag['num_original_bands']} original bands.")

    # 5. Modules 13, 14 & Tri-Sensor Consensus Engine
    print("\n--> Modules 13 & 14 + Tri-Sensor Engine: Executing Unified Correspondence...")
    tri_engine = TriSensorCorrespondenceEngine(cfg)
    tri_result = tri_engine.process_triplet(lunar_ohrc, lunar_tmc2, lunar_iirs_pc1)

    print(f"\n---------------- TRI-SENSOR MATCHING RESULTS ----------------")
    print(f"  OHRC <-> TMC-2  : {tri_result.res_ohrc_tmc2.num_inliers} inliers ({tri_result.res_ohrc_tmc2.inlier_ratio*100:.1f}%)")
    print(f"  OHRC <-> IIRS   : {tri_result.res_ohrc_iirs.num_inliers} inliers ({tri_result.res_ohrc_iirs.inlier_ratio*100:.1f}%)")
    print(f"  TMC-2 <-> IIRS  : {tri_result.res_tmc2_iirs.num_inliers} inliers ({tri_result.res_tmc2_iirs.inlier_ratio*100:.1f}%)")
    print(f"  Loop Closure Error : {tri_result.loop_closure_error_px:.4f} pixels")
    print(f"  Agreement Score    : {tri_result.tri_sensor_agreement_score:.4f}")
    print(f"  Consensus Status   : {'PASSED' if tri_result.is_consensus_valid else 'FAILED'}")
    print("--------------------------------------------------------------")

    # 6. Evaluation Reporting & Visual Panel
    print("\n--> Module 10 / Phase 3: Exporting Benchmark Reports & Visual Panel...")
    sidecar_iirs = lunar_iirs_raw.metadata.get("sidecar_metadata", {})
    dataset_type = sidecar_iirs.get("dataset_type", "SYNTHETIC_TRISENSOR_TRIPLET")

    evaluator = TriSensorEvaluator(cfg)
    metrics_summary = evaluator.evaluate(tri_result, dataset_type=dataset_type)

    benchmarks_dir = out_dir / "benchmarks"
    j_rep, c_rep = evaluator.save_report(metrics_summary, output_dir=str(benchmarks_dir), filename_prefix="trisensor_report")
    print(f"    Tri-Sensor report exported to: '{j_rep}' and '{c_rep}'")

    vis_path = out_dir / "phase3_trisensor_result.png"
    visualize_tri_sensor_correspondence(
        lunar_ohrc=lunar_ohrc,
        lunar_tmc2=lunar_tmc2,
        lunar_iirs=lunar_iirs_pc1,
        tri_result=tri_result,
        output_path=str(vis_path),
        dataset_type=dataset_type,
    )

    print("\n[+] Phase 3 Tri-Sensor Pipeline execution completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LUNAR-X Phase 3 Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--ohrc", type=str, default=None, help="Path to OHRC image")
    parser.add_argument("--tmc2", type=str, default=None, help="Path to TMC-2 image")
    parser.add_argument("--iirs", type=str, default=None, help="Path to IIRS raster")
    args = parser.parse_args()

    run_phase3_pipeline(
        config_path=args.config,
        ohrc_path=args.ohrc,
        tmc2_path=args.tmc2,
        iirs_path=args.iirs,
    )
