"""
LUNAR-X Complete System End-to-End Execution Entrypoint

Single end-to-end master command executing the full LUNAR-X pipeline:
OHRC + TMC-2 + IIRS
        ↓
  Preprocessing & PCA
        ↓
  Multi-Scale Feature Extraction
        ↓
  Pairwise & Cross-Modal Matching
        ↓
  Geometric Verification & RANSAC Homography
        ↓
  Tri-Sensor Correspondence Consensus
        ↓
  Quantitative Evaluation Metrics
        ↓
  Visualization Panels & Reports Export
"""

import argparse
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_phase1_pipeline import run_pipeline as run_phase1
from scripts.run_phase2_pipeline import run_phase2_pipeline as run_phase2
from scripts.run_phase3_pipeline import run_phase3_pipeline as run_phase3


def main():
    parser = argparse.ArgumentParser(description="LUNAR-X Complete End-to-End Master System Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to central config.yaml")
    args = parser.parse_args()

    start_total = time.time()
    print("=======================================================================")
    print(" LUNAR-X: AI-Based Multi-Modal Chandrayaan-2 Correspondence System")
    print(" ISRO SIH 2026 Problem Statement SIH26166")
    print("=======================================================================\n")

    # Step 1: Execute Phase 1 Pipeline (OHRC ↔ OHRC Baseline)
    print(">>> STEP 1 / 3: PHASE 1 (OHRC Optical Pipeline & Classical SIFT Baseline)")
    run_phase1(config_path=args.config)

    # Step 2: Execute Phase 2 Pipeline (OHRC ↔ TMC-2 Cross-Sensor)
    print("\n>>> STEP 2 / 3: PHASE 2 (OHRC 0.25m <-> TMC-2 5.0m Cross-Sensor Pipeline)")
    run_phase2(config_path=args.config)

    # Step 3: Execute Phase 3 Pipeline (OHRC ↔ TMC-2 ↔ IIRS Tri-Sensor Framework)
    print("\n>>> STEP 3 / 3: PHASE 3 (OHRC + TMC-2 + IIRS Hyperspectral Tri-Sensor Pipeline)")
    run_phase3(config_path=args.config)

    elapsed_total = time.time() - start_total
    print("\n=======================================================================")
    print(f" [+] LUNAR-X Full System End-to-End Execution Completed in {elapsed_total:.2f}s!")
    print("     Artifacts & Benchmark Reports saved to: 'outputs/' and 'data/manifests/'")
    print("=======================================================================")


if __name__ == "__main__":
    main()
