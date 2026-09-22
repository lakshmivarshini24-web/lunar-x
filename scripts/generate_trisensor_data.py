"""
Synthetic Tri-Sensor Data Generator (OHRC + TMC-2 + IIRS)

Generates synchronized synthetic dataset triplets simulating:
- OHRC Optical High-Resolution (~0.25 m/px, 1024 x 1024 px)
- TMC-2 Terrain Mapping Context (~5.0 m/px, 512 x 512 px)
- IIRS Imaging Infrared Hyperspectral (~100.0 m/px, 256 x 256 x 32 spectral bands)

Includes known synthetic ground-truth homography matrices and sidecar metadata JSONs.
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


def generate_trisensor_triplet(
    output_dir: Path,
    ohrc_size: int = 1024,
    triplet_id: int = 1,
):
    """
    Generates a synchronized synthetic triplet of OHRC, TMC-2, and IIRS datasets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(200 + triplet_id)

    # 1. Master Terrain Elevation (1024 x 1024)
    x = np.linspace(-4, 4, ohrc_size)
    y = np.linspace(-4, 4, ohrc_size)
    xx, yy = np.meshgrid(x, y)
    master_terrain = 120.0 + 35.0 * np.sin(xx) * np.cos(yy)
    regolith_noise = np.random.normal(0, 12, (ohrc_size, ohrc_size))
    master_img = master_terrain + regolith_noise

    # Synthesize major crater structures
    for _ in range(50):
        cx = np.random.randint(80, ohrc_size - 80)
        cy = np.random.randint(80, ohrc_size - 80)
        radius = np.random.randint(18, 65)

        y_idx, x_idx = np.ogrid[:ohrc_size, :ohrc_size]
        dist_sq = (x_idx - cx) ** 2 + (y_idx - cy) ** 2

        crater_mask = dist_sq <= radius ** 2
        rim_mask = (dist_sq > radius ** 2) & (dist_sq <= (radius + 6) ** 2)

        master_img[crater_mask] -= 45.0
        master_img[rim_mask] += 35.0

    ohrc_u8 = np.clip(master_img, 0, 255).astype(np.uint8)

    # 2. TMC-2 Context Image (512x512 with scale factor & rotation)
    tmc2_base = cv2.resize(ohrc_u8, (256, 256), interpolation=cv2.INTER_AREA)
    M_rot = cv2.getRotationMatrix2D((128, 128), 10.0, 1.0)
    M_rot[0, 2] += 6.0
    M_rot[1, 2] += -4.0
    tmc2_warped = cv2.warpAffine(tmc2_base, M_rot, (256, 256), borderMode=cv2.BORDER_REFLECT)
    tmc2_u8 = cv2.resize(tmc2_warped, (512, 512), interpolation=cv2.INTER_LINEAR)

    # 3. IIRS 3D Hyperspectral Cube (256 x 256 x 32 bands)
    iirs_base_2d = cv2.resize(ohrc_u8, (256, 256), interpolation=cv2.INTER_AREA)
    num_bands = 32
    wavelengths_nm = np.linspace(800, 3000, num_bands).tolist()

    iirs_cube = np.zeros((256, 256, num_bands), dtype=np.float32)

    for b in range(num_bands):
        wl = wavelengths_nm[b]
        # Simulate pyroxene mineral absorption feature near 1000nm and 2000nm
        abs_factor = 1.0 - 0.25 * np.exp(-((wl - 1000.0) ** 2) / (2 * 150.0 ** 2)) - 0.20 * np.exp(-((wl - 2000.0) ** 2) / (2 * 200.0 ** 2))
        band_noise = np.random.normal(0, 3.0, (256, 256)).astype(np.float32)
        iirs_cube[:, :, b] = np.clip(iirs_base_2d.astype(np.float32) * abs_factor + band_noise, 0, 255)

    # 4. Save image files
    path_ohrc = output_dir / f"ohrc_tri_{triplet_id:02d}.tif"
    path_tmc2 = output_dir / f"tmc2_tri_{triplet_id:02d}.tif"
    path_iirs_npy = output_dir / f"iirs_tri_{triplet_id:02d}.npy"
    path_iirs_tif = output_dir / f"iirs_tri_{triplet_id:02d}.tif"

    cv2.imwrite(str(path_ohrc), ohrc_u8)
    cv2.imwrite(str(path_tmc2), tmc2_u8)
    np.save(str(path_iirs_npy), iirs_cube.astype(np.float32))

    # Save PC1 band as display TIFF for IIRS
    cv2.imwrite(str(path_iirs_tif), np.clip(iirs_base_2d, 0, 255).astype(np.uint8))

    # 5. Homographies
    H_ohrc_tmc2 = np.array([
        [M_rot[0, 0] * 2.0, M_rot[0, 1] * 2.0, M_rot[0, 2] * 2.0],
        [M_rot[1, 0] * 2.0, M_rot[1, 1] * 2.0, M_rot[1, 2] * 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    H_ohrc_iirs = np.array([
        [0.25, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Sidecar JSON metadata
    meta_ohrc = {
        "sensor": "OHRC",
        "resolution_m": 0.25,
        "acquisition_info": {"orbit": 3042 + triplet_id, "timestamp": "2026-05-01T06:00:00Z", "instrument": "OHRC"},
        "sun_angle_info": {"sun_elevation_deg": 45.0, "sun_azimuth_deg": 105.0},
        "dataset_type": "SYNTHETIC_TRISENSOR_TRIPLET",
    }

    meta_tmc2 = {
        "sensor": "TMC-2",
        "resolution_m": 5.0,
        "acquisition_info": {"orbit": 3100 + triplet_id, "timestamp": "2026-05-05T10:15:00Z", "instrument": "TMC-2", "view_angle": "Nadir"},
        "sun_angle_info": {"sun_elevation_deg": 38.0, "sun_azimuth_deg": 115.0},
        "dataset_type": "SYNTHETIC_TRISENSOR_TRIPLET",
        "synthetic_ground_truth": {
            "scale_ratio": 20.0,
            "ground_truth_homography": H_ohrc_tmc2.tolist(),
        },
    }

    meta_iirs = {
        "sensor": "IIRS",
        "resolution_m": 100.0,
        "num_bands": num_bands,
        "wavelengths_nm": wavelengths_nm,
        "acquisition_info": {"orbit": 3150 + triplet_id, "timestamp": "2026-05-10T14:45:00Z", "instrument": "IIRS", "spectral_range_nm": [800, 3000]},
        "sun_angle_info": {"sun_elevation_deg": 30.0, "sun_azimuth_deg": 130.0},
        "dataset_type": "SYNTHETIC_TRISENSOR_TRIPLET",
        "synthetic_ground_truth": {
            "ground_truth_homography_ohrc_to_iirs": H_ohrc_iirs.tolist(),
        },
    }

    with open(path_ohrc.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta_ohrc, f, indent=2)
    with open(path_tmc2.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta_tmc2, f, indent=2)
    with open(path_iirs_npy.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta_iirs, f, indent=2)

    print(f"Generated synthetic Tri-Sensor triplet #{triplet_id}: '{path_ohrc.name}' (OHRC), '{path_tmc2.name}' (TMC-2), '{path_iirs_npy.name}' (IIRS 32-band)")
    return str(path_ohrc), str(path_tmc2), str(path_iirs_npy)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OHRC + TMC-2 + IIRS tri-sensor test triplets")
    parser.add_argument("--output_dir", type=str, default="data/raw/synthetic_trisensor", help="Output directory")
    parser.add_argument("--num_triplets", type=int, default=2, help="Number of triplets to generate")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    for i in range(1, args.num_triplets + 1):
        generate_trisensor_triplet(out_dir, triplet_id=i)


if __name__ == "__main__":
    main()
