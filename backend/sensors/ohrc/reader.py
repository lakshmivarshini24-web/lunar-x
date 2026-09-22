"""
OHRC Data Reader (Module 2)

Reads Chandrayaan-2 OHRC raster imagery, extracts header/auxiliary metadata,
safely normalizes pixel representations, and returns standardized `LunarImage` objects.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import cv2
from PIL import Image

from backend.core.lunar_image import LunarImage
from backend.sensors.ohrc.validator import OHRCValidator


class OHRCReader:
    """
    Reader for Chandrayaan-2 Orbiter High Resolution Camera (OHRC) datasets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validator = OHRCValidator(self.config)
        self.default_resolution = self.config.get("ohrc", {}).get("default_resolution_m", 0.25)

    def read(self, file_path: Union[str, Path], metadata_path: Optional[Union[str, Path]] = None) -> LunarImage:
        """
        Reads an OHRC image file and returns a LunarImage instance.

        Args:
            file_path: Path to OHRC image file.
            metadata_path: Optional path to associated metadata JSON or sidecar file.

        Returns:
            LunarImage: Standardized image representation.
        """
        path = Path(file_path)

        # 1. Validate input file
        val_res = self.validator.validate_file(str(path))
        if not val_res["is_valid"]:
            raise ValueError(f"Invalid OHRC file '{path}': {'; '.join(val_res['errors'])}")

        # 2. Read image raster array
        img_array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img_array is None:
            # Fallback to PIL if OpenCV fails
            with Image.open(path) as pil_img:
                img_array = np.array(pil_img)

        # Convert BGR/RGB to grayscale if multi-channel
        if img_array.ndim == 3:
            if img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2GRAY)

        # 3. Read metadata if sidecar exists or provided
        sidecar_meta: Dict[str, Any] = {}
        target_meta_path = Path(metadata_path) if metadata_path else path.with_suffix(".json")
        if target_meta_path.exists():
            try:
                with open(target_meta_path, "r", encoding="utf-8") as f:
                    sidecar_meta = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load sidecar metadata '{target_meta_path}': {e}")

        # Extract acquisition & sun angle metadata
        acq_info = sidecar_meta.get("acquisition_info", {
            "orbit": sidecar_meta.get("orbit", "Unknown"),
            "timestamp": sidecar_meta.get("timestamp", "Unknown"),
            "instrument": "OHRC",
        })

        sun_info = sidecar_meta.get("sun_angle_info", {
            "sun_elevation_deg": sidecar_meta.get("sun_elevation_deg", None),
            "sun_azimuth_deg": sidecar_meta.get("sun_azimuth_deg", None),
            "incidence_angle_deg": sidecar_meta.get("incidence_angle_deg", None),
        })

        geo_info = sidecar_meta.get("geographic_info", {
            "crs": sidecar_meta.get("crs", "MOON_2000"),
            "resolution_m": sidecar_meta.get("resolution_m", self.default_resolution),
        })

        resolution_m = float(geo_info.get("resolution_m", self.default_resolution))

        meta = {
            "original_dtype": str(img_array.dtype),
            "bit_depth": val_res.get("bit_depth", 8),
            "original_shape": img_array.shape,
            "validation_warnings": val_res.get("warnings", []),
            "sidecar_metadata": sidecar_meta,
        }

        return LunarImage(
            image=img_array,
            sensor="OHRC",
            resolution_m=resolution_m,
            acquisition_info=acq_info,
            sun_angle_info=sun_info,
            metadata=meta,
            geographic_info=geo_info,
            file_path=str(path),
        )


def load_ohrc(path: Union[str, Path], config: Optional[Dict[str, Any]] = None) -> LunarImage:
    """Convenience helper function to load an OHRC image."""
    reader = OHRCReader(config)
    return reader.read(path)
