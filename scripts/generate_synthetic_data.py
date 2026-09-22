"""
Synthetic Lunar Data Generator (Phase 1 Pipeline Testing)

Generates synthetic OHRC-like cratered lunar surface image pairs with known geometric transformations
(rotation, translation, scaling, affine warp), sun illumination variations, and synthetic ground-truth
homography matrices for automated verification and evaluation.

IMPORTANT: This generates synthetic test data for software validation only.
It does not fabricate real ISRO mission datasets or scientific ground truth.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2


def generate_synthetic_lunar_surface(height: int = 512, width: int = 512, seed: int = 42) -> np.ndarray:

    np.random.seed(seed)

    # Base low-frequency regolith elevation map
    x = np.linspace(-3, 3, width)
    y = np.linspace(-3, 3, height)
    xx, yy = np.meshgrid(x, y)
    base_terrain = 120.0 + 30.0 * np.sin(xx) * np.cos(yy)

    # High-frequency noise texture (simulating fine regolith detail)
    regolith_noise = np.random.normal(0, 15, (height, width))

    img = base_terrain + regolith_noise

    # Synthesize procedural lunar impact craters
    num_craters = 40
    for _ in range(num_craters):
        cx = np.random.randint(40, width - 40)
        cy = np.random.randint(40, height - 40)
        radius = np.random.randint(8, 45)

        y_indices, x_indices = np.ogrid[:height, :width]
        dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2

        crater_mask = dist_sq <= radius ** 2
        rim_mask = (dist_sq > radius ** 2) & (dist_sq <= (radius + 4) ** 2)

        # Crater floor depression + shadow on left inner rim
        img[crater_mask] -= 45.0
        # Bright crater rim ejecta
        img[rim_mask] += 35.0

    # Add directional lighting shadow gradient (Simulating solar illumination angle)
    light_gradient = np.tile(np.linspace(1.2, 0.7, width), (height, 1))
    img = img * light_gradient

    # Clip & normalize to 8-bit image
    img_norm = np.clip(img, 0, 255).astype(np.uint8)
    return img_norm


def create_synthetic_pair(
    output_dir: Path,
    image_size: int = 512,
    angle_deg: float = 12.0,
    scale: float = 1.05,
    tx: float = 18.0,
    ty: float = -12.0,
    pair_id: int = 1,
):
    """
    Creates a synthetic image pair (A & B) where B is transformed from A by a known homography.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate base surface Image A
    img_a = generate_synthetic_lunar_surface(height=image_size, width=image_size, seed=42 + pair_id)

    # 2. Define known transformation matrix (Rotation + Scale + Translation)
    center = (image_size / 2.0, image_size / 2.0)
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty

    # 3x3 Homography matrix representation
    H_gt = np.eye(3, dtype=np.float64)
    H_gt[:2, :] = M_rot

    # 3. Warp Image A to produce Image B
    img_b = cv2.warpAffine(img_a, M_rot, (image_size, image_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Add slight independent noise & sun angle contrast shift to Image B
    noise = np.random.normal(0, 5, img_b.shape).astype(np.float32)
    img_b_float = np.clip(img_b.astype(np.float32) * 1.08 + noise, 0, 255)
    img_b = img_b_float.astype(np.uint8)

    # 4. Write image files
    path_a = output_dir / f"ohrc_synthetic_{pair_id:02d}_A.tif"
    path_b = output_dir / f"ohrc_synthetic_{pair_id:02d}_B.tif"

    cv2.imwrite(str(path_a), img_a)
    cv2.imwrite(str(path_b), img_b)

    # 5. Write metadata sidecars
    meta_a = {
        "sensor": "OHRC",
        "resolution_m": 0.25,
        "acquisition_info": {"orbit": 1042 + pair_id, "timestamp": "2026-03-15T10:00:00Z", "instrument": "OHRC"},
        "sun_angle_info": {"sun_elevation_deg": 35.0, "sun_azimuth_deg": 120.0, "incidence_angle_deg": 55.0},
        "geographic_info": {"crs": "MOON_2000", "resolution_m": 0.25},
    }

    meta_b = {
        "sensor": "OHRC",
        "resolution_m": 0.25,
        "acquisition_info": {"orbit": 1088 + pair_id, "timestamp": "2026-03-28T14:30:00Z", "instrument": "OHRC"},
        "sun_angle_info": {"sun_elevation_deg": 28.0, "sun_azimuth_deg": 135.0, "incidence_angle_deg": 62.0},
        "geographic_info": {"crs": "MOON_2000", "resolution_m": 0.25},
        "synthetic_ground_truth": {
            "rotation_deg": angle_deg,
            "scale": scale,
            "translation_xy": [tx, ty],
            "ground_truth_homography": H_gt.tolist(),
        },
    }

    with open(path_a.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta_a, f, indent=2)
    with open(path_b.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta_b, f, indent=2)

    print(f"Generated synthetic pair #{pair_id}: '{path_a.name}' and '{path_b.name}'")
    return str(path_a), str(path_b)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OHRC test image pairs")
    parser.add_argument("--output_dir", type=str, default="data/raw/synthetic_ohrc", help="Output directory")
    parser.add_argument("--num_pairs", type=int, default=2, help="Number of image pairs to generate")
    parser.add_argument("--image_size", type=int, default=512, help="Image size (width and height)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    for i in range(1, args.num_pairs + 1):
        angle = 8.0 * i
        scale = 1.0 + 0.03 * i
        tx = 15.0 * i
        ty = -10.0 * i
        create_synthetic_pair(
            out_dir, image_size=args.image_size, angle_deg=angle, scale=scale, tx=tx, ty=ty, pair_id=i
        )


if __name__ == "__main__":
    main()
