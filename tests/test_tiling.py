"""
Unit tests for Module 4: OHRC Tiling.
"""

import numpy as np
import pytest

from backend.core.lunar_image import LunarImage
from backend.preprocessing.tiling import ImageTiler, TileInfo


def test_image_tiler():
    arr = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    lunar_img = LunarImage(image=arr, sensor="OHRC")

    tiler = ImageTiler(tile_size=512, overlap=128)
    tiles = tiler.tile(lunar_img)

    assert len(tiles) > 0
    assert isinstance(tiles[0], TileInfo)
    assert tiles[0].lunar_image.shape[0] == 512
    assert tiles[0].lunar_image.shape[1] == 512


def test_tile_to_global_keypoints():
    tile_info = TileInfo(
        tile_id=0,
        row_idx=0,
        col_idx=1,
        y_min=100,
        y_max=612,
        x_min=384,
        x_max=896,
        tile_height=512,
        tile_width=512,
        lunar_image=LunarImage(image=np.zeros((512, 512), dtype=np.uint8)),
    )

    local_kpts = np.array([[10.0, 20.0], [50.0, 60.0]], dtype=np.float32)
    global_kpts = ImageTiler.tile_to_global_keypoints(local_kpts, tile_info)

    assert global_kpts[0, 0] == 10.0 + 384
    assert global_kpts[0, 1] == 20.0 + 100
    assert global_kpts[1, 0] == 50.0 + 384
    assert global_kpts[1, 1] == 60.0 + 100
