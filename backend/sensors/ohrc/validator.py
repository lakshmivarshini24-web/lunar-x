"""
OHRC Data Validator (Module 1)

Validates file integrity, image dimensions, bit depth, header corruption,
missing metadata, and array payload validity for Chandrayaan-2 OHRC images.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image


class OHRCValidator:
    """
    Validator for OHRC (Orbiter High Resolution Camera) optical images.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        ohrc_cfg = self.config.get("ohrc", {})
        self.supported_extensions = set(
            ohrc_cfg.get("supported_extensions", [".tif", ".tiff", ".png", ".jpg", ".jpeg"])
        )
        self.min_dims = tuple(ohrc_cfg.get("min_dimensions", [64, 64]))
        self.max_dims = tuple(ohrc_cfg.get("max_dimensions", [65536, 65536]))
        self.valid_bit_depths = set(ohrc_cfg.get("valid_bit_depths", [8, 16, 32]))

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Executes complete validation suite on an OHRC file path.

        Returns:
            Dict containing validation details:
            - is_valid (bool)
            - errors (List[str])
            - warnings (List[str])
            - dimensions (Tuple[int, int])
            - bit_depth (int)
            - channels (int)
            - sensor (str)
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
                "sensor": "OHRC",
            }

        if path.stat().st_size == 0:
            errors.append("File is empty (0 bytes)")

        if path.suffix.lower() not in self.supported_extensions:
            warnings.append(
                f"Unsupported extension '{path.suffix}'. Supported: {sorted(list(self.supported_extensions))}"
            )

        dimensions: Optional[Tuple[int, int]] = None
        bit_depth: Optional[int] = None
        channels = 1
        img_array: Optional[np.ndarray] = None

        # Attempt decoding with PIL first for header info
        try:
            with Image.open(path) as img:
                img.verify()  # Verify image integrity / corrupt header
            with Image.open(path) as img:
                width, height = img.size
                dimensions = (height, width)
                mode = img.mode
                if mode in ("L", "P"):
                    bit_depth = 8
                    channels = 1
                elif mode in ("I;16", "I;16B", "I;16L", "I"):
                    bit_depth = 16
                    channels = 1
                elif mode in ("F",):
                    bit_depth = 32
                    channels = 1
                elif mode in ("RGB", "RGBA"):
                    bit_depth = 8
                    channels = len(mode)
                    warnings.append(f"Image has {channels} channels; OHRC optical images are typically single-band grayscale.")
        except Exception as e:
            errors.append(f"Header/PIL decode failure (corrupted file): {str(e)}")

        # Attempt OpenCV read for deep payload verification
        try:
            img_array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img_array is None:
                errors.append("OpenCV failed to decode image array (corrupt or unreadable raster data)")
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
                else:
                    warnings.append(f"Unusual image numpy dtype: {img_array.dtype}")

                # Check for NaNs/Infs
                if np.issubdtype(img_array.dtype, np.floating):
                    if np.isnan(img_array).any():
                        warnings.append("Image array contains NaN values")
                    if np.isinf(img_array).any():
                        warnings.append("Image array contains Infinite values")

                # Check flat image (all zero or constant)
                if img_array.size > 0:
                    if np.max(img_array) == np.min(img_array):
                        warnings.append("Image payload has zero variance (constant pixel values)")

        except Exception as e:
            errors.append(f"Array decoding failure: {str(e)}")

        # Validate dimensions against bounds
        if dimensions is not None:
            h, w = dimensions
            if h < self.min_dims[0] or w < self.min_dims[1]:
                errors.append(f"Image dimensions ({h}x{w}) smaller than minimum allowed {self.min_dims}")
            if h > self.max_dims[0] or w > self.max_dims[1]:
                errors.append(f"Image dimensions ({h}x{w}) larger than maximum allowed {self.max_dims}")

        # Validate bit depth
        if bit_depth is not None and bit_depth not in self.valid_bit_depths:
            warnings.append(f"Bit depth {bit_depth} not in standard set {sorted(list(self.valid_bit_depths))}")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "dimensions": dimensions,
            "bit_depth": bit_depth,
            "channels": channels,
            "sensor": "OHRC",
        }
