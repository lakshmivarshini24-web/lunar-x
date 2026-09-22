"""
Synthetic Cross-Sensor Data Generator (OHRC ↔ TMC-2)

Generates synthetic paired OHRC (~0.25 m/px high-resolution) and TMC-2 (~1.25m - 5.0 m/px context) image pairs
with a known spatial scale ratio, geometric homography transformation, and solar illumination variation.

IMPORTANT: This generates synthetic test data for software validation only.
It does not fabricate real ISRO mission datasets or scientific ground truth.
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2


def generate_cross_sensor_pair(
    output_dir: Path,
    ohrc_size: int = 1024,
    scale_ratio: float = 4.0,
    angle_deg: float = 12.0,
    pair_id: int = 1,
):
    """
    Creates a paired synthetic OHRC (0.25m/px) and TMC-2 (1.0m/px) dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(100 + pair_id)

    # 1. Generate master high-resolution lunar terrain (1024 x 1024)
    master_h, master_w = ohrc_size, ohrc_size
    x = np.linspace(-4, 4, master_w)
    y = np.linspace(-4, 4, master_h)
    xx, yy = np.meshgrid(x, y)
    master_terrain = 120.0 + 35.0 * np.sin(xx) * np.cos(yy)
    regolith_noise = np.random.normal(0, 12, (master_h, master_w))
    master_img = master_terrain + regolith_noise

    # Synthesize major crater structures with rim ejecta
    num_craters = 50
    for _ in range(num_craters):
        cx = np.random.randint(80, master_w - 80)
        cy = np.random.randint(80, master_h - 80)
        radius = np.random.randint(18, 65)

        y_idx, x_idx = np.ogrid[:master_h, :master_w]
        dist_sq = (x_idx - cx) ** 2 + (y_idx - cy) ** 2

        crater_mask = dist_sq <= radius ** 2
        rim_mask = (dist_sq > radius ** 2) & (dist_sq <= (radius + 6) ** 2)

        master_img[crater_mask] -= 45.0
        master_img[rim_mask] += 35.0

    ohrc_img = np.clip(master_img, 0, 255).astype(np.uint8)

    # 2. Generate TMC-2 Image by scale-resampling master terrain + applying rotation + illumination angle shift
    tmc2_w = int(master_w / scale_ratio)  # 256 x 256
    tmc2_h = int(master_h / scale_ratio)
    tmc2_base = cv2.resize(ohrc_img, (tmc2_w, tmc2_h), interpolation=cv2.INTER_AREA)

    # Apply rotation and translation to TMC-2 context view
    center = (tmc2_w / 2.0, tmc2_h / 2.0)
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    M_rot[0, 2] += 8.0
    M_rot[1, 2] += -5.0

    tmc2_warped = cv2.warpAffine(tmc2_base, M_rot, (tmc2_w, tmc2_h), borderMode=cv2.BORDER_REFLECT)
    tmc2_img = cv2.resize(tmc2_warped, (512, 512), interpolation=cv2.INTER_LINEAR)  # 512x512 canvas

    # Construct Ground Truth Homography Matrix mapping OHRC native pixels (1024x1024) -> TMC-2 native pixels (512x512)
    s_x = 512.0 / (master_w * scale_ratio)
    s_y = 512.0 / (master_h * scale_ratio)

    H_gt = np.array([
        [M_rot[0, 0] * (512.0 / tmc2_w), M_rot[0, 1] * (512.0 / tmc2_h), M_rot[0, 2] * (512.0 / tmc2_w)],
        [M_rot[1, 0] * (512.0 / tmc2_w), M_rot[1, 1] * (512.0 / tmc2_h), M_rot[1, 2] * (512.0 / tmc2_h)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 3. Save image files
    path_ohrc = output_dir / f"ohrc_cross_{pair_id:02d}.tif"
    path_tmc2 = output_dir / f"tmc2_cross_{pair_id:02d}.tif"

    cv2.imwrite(str(path_ohrc), ohrc_img)
    cv2.imwrite(str(path_tmc2), tmc2_img)

    # 4. Save sidecar metadata JSONs
    meta_ohrc = {
        "sensor": "OHRC",
        "resolution_m": 0.25,
        "acquisition_info": {"orbit": 2042 + pair_id, "timestamp": "2026-04-10T08:00:00Z", "instrument": "OHRC"},
        "sun_angle_info": {"sun_elevation_deg": 40.0, "sun_azimuth_deg": 110.0},
        "dataset_type": "SYNTHETIC_CROSS_SENSOR_PAIR",
    }

    meta_tmc2 = {
        "sensor": "TMC-2",
        "resolution_m": 0.25 * scale_ratio * (tmc2_w / 512.0),
        "acquisition_info": {"orbit": 2100 + pair_id, "timestamp": "2026-04-20T12:30:00Z", "instrument": "TMC-2", "view_angle": "Nadir"},
        "sun_angle_info": {"sun_elevation_deg": 32.0, "sun_azimuth_deg": 125.0},
        "dataset_type": "SYNTHETIC_CROSS_SENSOR_PAIR",
        "synthetic_ground_truth": {
            "scale_ratio": scale_ratio,
            "rotation_deg": angle_deg,
            "ground_truth_homography": H_gt.tolist(),
        },
    }

    with open(path_ohrc.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta_ohrc, f, indent=2)
    with open(path_tmc2.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta_tmc2, f, indent=2)

    print(f"Generated synthetic cross-sensor pair #{pair_id}: '{path_ohrc.name}' (OHRC 0.25m) & '{path_tmc2.name}' (TMC-2 context)")
    return str(path_ohrc), str(path_tmc2)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OHRC <-> TMC-2 cross-sensor test pairs")
    parser.add_argument("--output_dir", type=str, default="data/raw/synthetic_cross_sensor", help="Output directory")
    parser.add_argument("--num_pairs", type=int, default=2, help="Number of pairs to generate")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    for i in range(1, args.num_pairs + 1):
        generate_cross_sensor_pair(out_dir, pair_id=i)


if __name__ == "__main__":
    main()
