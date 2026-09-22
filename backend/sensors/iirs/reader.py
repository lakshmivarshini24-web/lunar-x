"""
IIRS Data Reader (Module 11)

Reads Chandrayaan-2 IIRS (Imaging Infrared Spectrometer) multi-band/hyperspectral imagery,
parses associated sidecar wavelength metadata, and constructs standardized `LunarImage` objects
with sensor='IIRS' and default resolution 100.0 m/px.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import cv2
from PIL import Image

from backend.core.lunar_image import LunarImage
from backend.sensors.iirs.validator import IIRSValidator


class IIRSReader:
    """
    Reader for Chandrayaan-2 Imaging Infrared Spectrometer (IIRS) multi-band datasets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validator = IIRSValidator(self.config)
        self.default_resolution = self.config.get("iirs", {}).get("default_resolution_m", 100.0)

    def read(self, file_path: Union[str, Path], metadata_path: Optional[Union[str, Path]] = None) -> LunarImage:
        """
        Reads an IIRS image file (.tif, .npy) and returns a LunarImage object.

        Args:
            file_path: Path to IIRS raster array.
            metadata_path: Optional sidecar JSON path.

        Returns:
            LunarImage: Standardized image representation for IIRS.
        """
        path = Path(file_path)

        val_res = self.validator.validate_file(str(path))
        if not val_res["is_valid"]:
            raise ValueError(f"Invalid IIRS file '{path}': {'; '.join(val_res['errors'])}")

        if path.suffix.lower() == ".npy":
            img_array = np.load(str(path))
        else:
            img_array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img_array is None:
                with Image.open(path) as pil_img:
                    img_array = np.array(pil_img)

        # Parse sidecar metadata if present
        sidecar_meta: Dict[str, Any] = {}
        target_meta_path = Path(metadata_path) if metadata_path else path.with_suffix(".json")
        if target_meta_path.exists():
            try:
                with open(target_meta_path, "r", encoding="utf-8") as f:
                    sidecar_meta = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load IIRS sidecar metadata '{target_meta_path}': {e}")

        acq_info = sidecar_meta.get("acquisition_info", {
            "orbit": sidecar_meta.get("orbit", "Unknown"),
            "timestamp": sidecar_meta.get("timestamp", "Unknown"),
            "instrument": "IIRS",
            "spectral_range_nm": sidecar_meta.get("spectral_range_nm", [800, 5000]),
        })

        sun_info = sidecar_meta.get("sun_angle_info", {
            "sun_elevation_deg": sidecar_meta.get("sun_elevation_deg", None),
            "sun_azimuth_deg": sidecar_meta.get("sun_azimuth_deg", None),
        })

        geo_info = sidecar_meta.get("geographic_info", {
            "crs": sidecar_meta.get("crs", "MOON_2000"),
            "resolution_m": sidecar_meta.get("resolution_m", self.default_resolution),
        })

        wavelengths_nm = sidecar_meta.get("wavelengths_nm", None)

        resolution_m = float(geo_info.get("resolution_m", self.default_resolution))

        meta = {
            "original_dtype": str(img_array.dtype),
            "num_bands": val_res.get("num_bands", 1),
            "original_shape": img_array.shape,
            "wavelengths_nm": wavelengths_nm,
            "validation_warnings": val_res.get("warnings", []),
            "sidecar_metadata": sidecar_meta,
        }

        return LunarImage(
            image=img_array,
            sensor="IIRS",
            resolution_m=resolution_m,
            acquisition_info=acq_info,
            sun_angle_info=sun_info,
            metadata=meta,
            geographic_info=geo_info,
            file_path=str(path),
        )


def load_iirs(path: Union[str, Path], config: Optional[Dict[str, Any]] = None) -> LunarImage:
    """Convenience function to load an IIRS raster into LunarImage."""
    reader = IIRSReader(config)
    return reader.read(path)
