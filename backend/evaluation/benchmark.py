"""
Benchmarking Infrastructure (Module 6)

Executes benchmark suites comparing classical (SIFT, ORB, AKAZE) and learned matchers
across image pairs and saves machine-readable JSON and CSV reports.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from backend.core.lunar_image import LunarImage
from backend.correspondence.classical import ClassicalCorrespondence
from backend.evaluation.metrics import EvaluationMetrics, compute_metrics


class BenchmarkRunner:
    """
    Executes algorithmic benchmarks on lunar image pairs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def run_benchmark(
        self,
        lunar_a: LunarImage,
        lunar_b: LunarImage,
        algorithms: Optional[List[str]] = None,
        ground_truth_homography: Optional[np.ndarray] = None,
    ) -> List[EvaluationMetrics]:
        """
        Runs benchmark across specified algorithms.

        Args:
            lunar_a: First LunarImage.
            lunar_b: Second LunarImage.
            algorithms: List of methods to benchmark (e.g. ['sift', 'orb', 'akaze']).
            ground_truth_homography: Optional GT matrix.

        Returns:
            List[EvaluationMetrics]: Comparative results for each algorithm.
        """
        if algorithms is None:
            algorithms = ["sift", "orb", "akaze"]

        results: List[EvaluationMetrics] = []
        matcher = ClassicalCorrespondence(self.config)

        for algo in algorithms:
            try:
                match_res = matcher.match(lunar_a, lunar_b, method_override=algo)
                metrics = compute_metrics(
                    match_result=match_res,
                    image_shape=lunar_a.shape[:2],
                    ground_truth_homography=ground_truth_homography,
                )
                results.append(metrics)
            except Exception as e:
                print(f"Error benchmarking algorithm '{algo}': {e}")

        return results

    def save_reports(
        self,
        metrics_list: List[EvaluationMetrics],
        output_dir: str = "outputs/benchmarks",
        filename_prefix: str = "benchmark_report",
    ) -> Tuple[str, str]:
        """
        Saves benchmark results to JSON and CSV formats.

        Returns:
            Tuple[str, str]: (json_file_path, csv_file_path)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_path = out_path / f"{filename_prefix}.json"
        csv_path = out_path / f"{filename_prefix}.csv"

        # 1. Export JSON
        dict_data = [m.to_dict() for m in metrics_list]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dict_data, f, indent=2)

        # 2. Export CSV
        if dict_data:
            fieldnames = list(dict_data[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in dict_data:
                    # Format dict values safely for CSV
                    row_copy = row.copy()
                    if isinstance(row_copy.get("gpu_utilization"), dict):
                        row_copy["gpu_utilization"] = json.dumps(row_copy["gpu_utilization"])
                    writer.writerow(row_copy)

        return str(json_path), str(csv_path)
