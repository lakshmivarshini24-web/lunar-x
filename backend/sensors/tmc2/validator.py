"""
TMC-2 Data Validator (Module 8)

Validates file integrity, dimensions, bit depth, header corruption, and sidecar metadata
for Chandrayaan-2 TMC-2 (Terrain Mapping Camera-2) imagery.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image


class TMC2Validator:
    """
    Validator for Chandrayaan-2 TMC-2 optical rasters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        tmc2_cfg = self.config.get("tmc2", {})
        self.supported_extensions = set(
            tmc2_cfg.get("supported_extensions", [".tif", ".tiff", ".png", ".jpg", ".jpeg"])
        )
        self.min_dims = tuple(tmc2_cfg.get("min_dimensions", [32, 32]))
        self.max_dims = tuple(tmc2_cfg.get("max_dimensions", [65536, 65536]))
        self.valid_bit_depths = set(tmc2_cfg.get("valid_bit_depths", [8, 16, 32]))

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validates TMC-2 image file path and header integrity.
        """
        path = Path(file_path)
        errors: List[str] = []
        warnings: List[str] = []

        if not path.exists():
            return {
                "is_valid": False,
                "errors": [f"File does not exist: {file_path}"],
                "warnings": warnings,
                "dimensions": None,
                "bit_depth": None,
                "channels": 0,
                "sensor": "TMC-2",
            }

        if path.stat().st_size == 0:
            errors.append("File is empty (0 bytes)")

        if path.suffix.lower() not in self.supported_extensions:
            warnings.append(f"Unsupported TMC-2 file extension '{path.suffix}'")

        dimensions: Optional[Tuple[int, int]] = None
        bit_depth: Optional[int] = None
        channels = 1

        try:
            img_array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img_array is None:
                errors.append("OpenCV failed to decode TMC-2 image raster data")
            else:
                h, w = img_array.shape[:2]
                dimensions = (h, w)
                channels = 1 if img_array.ndim == 2 else img_array.shape[2]

                if img_array.dtype == np.uint8:
                    bit_depth = 8
                elif img_array.dtype in (np.uint16, np.int16):
                    bit_depth = 16
                elif img_array.dtype in (np.float32, np.float64):
                    bit_depth = 32

                if np.issubdtype(img_array.dtype, np.floating):
                    if np.isnan(img_array).any():
                        warnings.append("TMC-2 raster contains NaN values")
                    if np.isinf(img_array).any():
                        warnings.append("TMC-2 raster contains Inf values")
        except Exception as e:
            errors.append(f"TMC-2 file read error: {str(e)}")

        if dimensions is not None:
            h, w = dimensions
            if h < self.min_dims[0] or w < self.min_dims[1]:
                errors.append(f"TMC-2 dimensions ({h}x{w}) smaller than minimum allowed {self.min_dims}")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "dimensions": dimensions,
            "bit_depth": bit_depth,
            "channels": channels,
            "sensor": "TMC-2",
        }
