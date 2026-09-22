"""
OHRC Tiling System (Module 4)

Decomposes large OHRC lunar rasters into overlapping sub-tiles while tracking tile coordinates
and providing tile reconstruction / global keypoint coordinate remapping utilities.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from backend.core.lunar_image import LunarImage


@dataclass
class TileInfo:
    """
    Metadata for a single sub-tile extracted from a larger LunarImage.
    """

    tile_id: int
    row_idx: int
    col_idx: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    tile_height: int
    tile_width: int
    lunar_image: LunarImage


class ImageTiler:
    """
    Handles decomposition of large LunarImages into overlapping tiles and global coordinate remapping.
    """

    def __init__(self, tile_size: int = 512, overlap: int = 128, min_valid_pixel_ratio: float = 0.1):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must be >= 0 and < tile_size")

        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.min_valid_pixel_ratio = min_valid_pixel_ratio

    def tile(self, lunar_img: LunarImage) -> List[TileInfo]:
        """
        Decomposes a LunarImage into overlapping sub-tiles.

        Args:
            lunar_img: Source LunarImage object.

        Returns:
            List[TileInfo]: Extracted sub-tiles with coordinate metadata.
        """
        img = lunar_img.image
        h, w = img.shape[:2]

        tiles: List[TileInfo] = []
        tile_id = 0

        # Calculate grid steps
        y_steps = range(0, max(1, h - self.overlap), self.stride)
        x_steps = range(0, max(1, w - self.overlap), self.stride)

        for row_idx, y_min in enumerate(y_steps):
            y_max = min(y_min + self.tile_size, h)
            # Adjust y_min if y_max reached border to ensure uniform tile_size when possible
            if y_max - y_min < self.tile_size and y_min > 0:
                y_min = max(0, y_max - self.tile_size)

            for col_idx, x_min in enumerate(x_steps):
                x_max = min(x_min + self.tile_size, w)
                if x_max - x_min < self.tile_size and x_min > 0:
                    x_min = max(0, x_max - self.tile_size)

                # Extract sub-raster slice
                sub_img = img[y_min:y_max, x_min:x_max]

                # Create tile metadata
                tile_meta = lunar_img.metadata.copy()
                tile_meta["tile_info"] = {
                    "tile_id": tile_id,
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "y_min": y_min,
                    "y_max": y_max,
                    "x_min": x_min,
                    "x_max": x_max,
                }

                tile_lunar = LunarImage(
                    image=sub_img,
                    sensor=lunar_img.sensor,
                    resolution_m=lunar_img.resolution_m,
                    acquisition_info=lunar_img.acquisition_info.copy(),
                    sun_angle_info=lunar_img.sun_angle_info.copy(),
                    metadata=tile_meta,
                    geographic_info=lunar_img.geographic_info.copy(),
                    file_path=lunar_img.file_path,
                )

                tile_info = TileInfo(
                    tile_id=tile_id,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    y_min=y_min,
                    y_max=y_max,
                    x_min=x_min,
                    x_max=x_max,
                    tile_height=sub_img.shape[0],
                    tile_width=sub_img.shape[1],
                    lunar_image=tile_lunar,
                )

                tiles.append(tile_info)
                tile_id += 1

        return tiles

    @staticmethod
    def tile_to_global_keypoints(keypoints: np.ndarray, tile_info: TileInfo) -> np.ndarray:
        """
        Remaps local keypoint coordinates (x_local, y_local) within a tile back to global image coordinates (x_global, y_global).

        Args:
            keypoints: (N, 2) array of (x, y) float coordinates.
            tile_info: TileInfo metadata containing x_min, y_min offsets.

        Returns:
            (N, 2) array of global coordinates.
        """
        if len(keypoints) == 0:
            return np.empty((0, 2), dtype=np.float32)

        global_kpts = keypoints.copy().astype(np.float32)
        global_kpts[:, 0] += tile_info.x_min
        global_kpts[:, 1] += tile_info.y_min
        return global_kpts

    @staticmethod
    def reconstruct_image(tiles: List[TileInfo], full_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstructs original image from list of tiles using linear blending across overlap regions.
        """
        h, w = full_shape[:2]
        reconstructed = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)

        for tile in tiles:
            img = tile.lunar_image.image.astype(np.float32)
            th, tw = img.shape[:2]

            reconstructed[tile.y_min : tile.y_min + th, tile.x_min : tile.x_min + tw] += img
            weights[tile.y_min : tile.y_min + th, tile.x_min : tile.x_min + tw] += 1.0

        weights[weights == 0] = 1.0
        reconstructed /= weights
        return reconstructed.astype(np.uint8)
