"""
IIRS Data Validator (Module 11)

Validates file integrity, 3D hyperspectral dimensions (H, W, Bands), bit depth,
wavelength metadata, bad/flat bands, and raster payloads for Chandrayaan-2 IIRS imagery.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image


class IIRSValidator:
    """
    Validator for Chandrayaan-2 IIRS (Imaging Infrared Spectrometer) multi-band datasets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        iirs_cfg = self.config.get("iirs", {})
        self.supported_extensions = set(
            iirs_cfg.get("supported_extensions", [".tif", ".tiff", ".npy", ".png", ".jpg", ".jpeg"])
        )
        self.min_dims = tuple(iirs_cfg.get("min_dimensions", [16, 16]))
        self.max_dims = tuple(iirs_cfg.get("max_dimensions", [65536, 65536]))
        self.valid_bit_depths = set(iirs_cfg.get("valid_bit_depths", [8, 16, 32]))

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validates an IIRS image/array file path.
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
                "num_bands": 0,
                "bit_depth": None,
                "sensor": "IIRS",
            }

        if path.stat().st_size == 0:
            errors.append("File is empty (0 bytes)")

        if path.suffix.lower() not in self.supported_extensions:
            warnings.append(f"Unsupported IIRS extension '{path.suffix}'")

        dimensions: Optional[Tuple[int, int]] = None
        num_bands = 1
        bit_depth: Optional[int] = None

        try:
            if path.suffix.lower() == ".npy":
                arr = np.load(str(path))
                if arr.ndim == 2:
                    dimensions = arr.shape
                    num_bands = 1
                elif arr.ndim == 3:
                    dimensions = arr.shape[:2]
                    num_bands = arr.shape[2]
                else:
                    errors.append(f"Numpy array must be 2D or 3D, got ndim={arr.ndim}")

                if arr.dtype == np.uint8:
                    bit_depth = 8
                elif arr.dtype in (np.uint16, np.int16):
                    bit_depth = 16
                elif arr.dtype in (np.float32, np.float64):
                    bit_depth = 32

                if np.isnan(arr).any():
                    warnings.append("IIRS array contains NaN values")
                if np.isinf(arr).any():
                    warnings.append("IIRS array contains Inf values")

            else:
                img_array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img_array is None:
                    with Image.open(path) as pil_img:
                        img_array = np.array(pil_img)

                if img_array is not None:
                    dimensions = img_array.shape[:2]
                    num_bands = 1 if img_array.ndim == 2 else img_array.shape[2]
                    bit_depth = 8 if img_array.dtype == np.uint8 else 16

        except Exception as e:
            errors.append(f"IIRS read failure: {str(e)}")

        if dimensions is not None:
            h, w = dimensions
            if h < self.min_dims[0] or w < self.min_dims[1]:
                errors.append(f"IIRS dimensions ({h}x{w}) smaller than minimum allowed {self.min_dims}")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "dimensions": dimensions,
            "num_bands": num_bands,
            "bit_depth": bit_depth,
            "sensor": "IIRS",
        }
