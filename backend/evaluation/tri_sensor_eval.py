"""
Tri-Sensor Evaluation Framework

Calculates quantitative metrics across all pairwise combinations (OHRC ↔ TMC-2, OHRC ↔ IIRS,
TMC-2 ↔ IIRS) and computes overall Tri-Sensor consensus loop closure error, inlier ratios,
execution runtime, and memory usage statistics.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from backend.matching.tri_sensor import TriSensorMatchResult
from backend.evaluation.metrics import EvaluationMetrics


class TriSensorEvaluator:
    """
    Evaluates Tri-Sensor correspondence performance across OHRC, TMC-2, and IIRS.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def evaluate(
        self,
        tri_result: TriSensorMatchResult,
        dataset_type: str = "SYNTHETIC_TRISENSOR_TRIPLET",
    ) -> Dict[str, Any]:
        """
        Computes structured metrics dictionary for tri-sensor execution.

        Args:
            tri_result: TriSensorMatchResult from TriSensorCorrespondenceEngine.
            dataset_type: Tag distinguishing 'REAL_ISRO_PDS4' from 'SYNTHETIC_TRISENSOR_TRIPLET'.

        Returns:
            Dict containing detailed pairwise and tri-sensor metrics.
        """
        r1 = tri_result.res_ohrc_tmc2
        r2 = tri_result.res_ohrc_iirs
        r3 = tri_result.res_tmc2_iirs

        summary = {
            "dataset_type": dataset_type,
            "total_execution_time_sec": round(tri_result.total_execution_time_sec, 4),
            "tri_sensor_agreement_score": round(tri_result.tri_sensor_agreement_score, 4),
            "loop_closure_error_px": round(tri_result.loop_closure_error_px, 4),
            "is_consensus_valid": tri_result.is_consensus_valid,
            "pairwise_metrics": {
                "ohrc_tmc2": {
                    "method": r1.method,
                    "inliers": r1.num_inliers,
                    "inlier_ratio": round(r1.inlier_ratio, 4),
                    "confidence": round(r1.confidence_score, 4),
                    "scale_ratio": r1.scale_ratio,
                },
                "ohrc_iirs": {
                    "method": r2.method,
                    "inliers": r2.num_inliers,
                    "inlier_ratio": round(r2.inlier_ratio, 4),
                    "confidence": round(r2.confidence_score, 4),
                    "scale_ratio": r2.scale_ratio,
                },
                "tmc2_iirs": {
                    "method": r3.method,
                    "inliers": r3.num_inliers,
                    "inlier_ratio": round(r3.inlier_ratio, 4),
                    "confidence": round(r3.confidence_score, 4),
                    "scale_ratio": r3.scale_ratio,
                },
            },
        }
        return summary

    def save_report(
        self,
        eval_dict: Dict[str, Any],
        output_dir: str = "outputs/benchmarks",
        filename_prefix: str = "trisensor_report",
    ) -> Tuple[str, str]:
        """
        Saves tri-sensor evaluation report to JSON and CSV formats.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_path = out_path / f"{filename_prefix}.json"
        csv_path = out_path / f"{filename_prefix}.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(eval_dict, f, indent=2)

        # Flatten dict for CSV row
        flattened = {
            "dataset_type": eval_dict["dataset_type"],
            "total_execution_time_sec": eval_dict["total_execution_time_sec"],
            "tri_sensor_agreement_score": eval_dict["tri_sensor_agreement_score"],
            "loop_closure_error_px": eval_dict["loop_closure_error_px"],
            "is_consensus_valid": eval_dict["is_consensus_valid"],
            "ohrc_tmc2_inliers": eval_dict["pairwise_metrics"]["ohrc_tmc2"]["inliers"],
            "ohrc_tmc2_inlier_ratio": eval_dict["pairwise_metrics"]["ohrc_tmc2"]["inlier_ratio"],
            "ohrc_iirs_inliers": eval_dict["pairwise_metrics"]["ohrc_iirs"]["inliers"],
            "ohrc_iirs_inlier_ratio": eval_dict["pairwise_metrics"]["ohrc_iirs"]["inlier_ratio"],
            "tmc2_iirs_inliers": eval_dict["pairwise_metrics"]["tmc2_iirs"]["inliers"],
            "tmc2_iirs_inlier_ratio": eval_dict["pairwise_metrics"]["tmc2_iirs"]["inlier_ratio"],
        }

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flattened.keys()))
            writer.writeheader()
            writer.writerow(flattened)

        return str(json_path), str(csv_path)
