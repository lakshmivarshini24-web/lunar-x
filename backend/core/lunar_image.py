"""
LunarImage Data Structure

Standardized internal representation for lunar remote sensing images from
Chandrayaan-2 payloads (OHRC, TMC-2, IIRS).
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class LunarImage:
    """
    Standardized internal representation of a lunar surface raster image.

    Attributes:
        image (np.ndarray): 2D grayscale or 3D multi-band image array.
        sensor (str): Instrument identifier ('OHRC', 'TMC-2', 'IIRS').
        resolution_m (float): Spatial resolution in meters/pixel (e.g., 0.25 for OHRC).
        acquisition_info (Dict[str, Any]): Orbit number, timestamp, integration time, etc.
        sun_angle_info (Dict[str, Any]): Sun elevation, azimuth, incidence, phase angle.
        metadata (Dict[str, Any]): File metadata (bit depth, original dtype, file format).
        geographic_info (Dict[str, Any]): Georeferencing info (CRS, affine transform, bounds).
        file_path (Optional[str]): Source file path.
    """

    image: np.ndarray
    sensor: str = "OHRC"
    resolution_m: float = 0.25
    acquisition_info: Dict[str, Any] = field(default_factory=dict)
    sun_angle_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    geographic_info: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None

    def __post_init__(self):

        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy ndarray")
        if self.image.ndim not in (2, 3):
            raise ValueError(f"Image array must be 2D or 3D, got shape {self.image.shape}")

    @property
    def shape(self) -> Tuple[int, ...]:

        return self.image.shape

    @property
    def height(self) -> int:

        return self.image.shape[0]

    @property
    def width(self) -> int:

        return self.image.shape[1]

    @property
    def channels(self) -> int:

        return 1 if self.image.ndim == 2 else self.image.shape[2]

    def to_uint8(self) -> np.ndarray:
        """Return 8-bit unsigned integer array scaled [0, 255]."""
        img = self.image
        if img.dtype == np.uint8:
            return img.copy()

        img_float = img.astype(np.float32)
        min_val, max_val = np.min(img_float), np.max(img_float)
        if max_val > min_val:
            normalized = (img_float - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(img_float)

        return (normalized * 255.0).astype(np.uint8)

    def to_float32(self) -> np.ndarray:
        """Return float32 array normalized to [0.0, 1.0]."""
        img = self.image.astype(np.float32)
        if self.image.dtype == np.uint8:
            return img / 255.0
        elif self.image.dtype == np.uint16:
            return img / 65535.0

        min_val, max_val = np.min(img), np.max(img)
        if max_val > min_val and (min_val < 0.0 or max_val > 1.0):
            return (img - min_val) / (max_val - min_val)
        return img

    def copy(self) -> "LunarImage":

        return LunarImage(
            image=self.image.copy(),
            sensor=self.sensor,
            resolution_m=self.resolution_m,
            acquisition_info=copy.deepcopy(self.acquisition_info),
            sun_angle_info=copy.deepcopy(self.sun_angle_info),
            metadata=copy.deepcopy(self.metadata),
            geographic_info=copy.deepcopy(self.geographic_info),
            file_path=self.file_path,
        )

    def summary(self) -> Dict[str, Any]:

        return {
            "sensor": self.sensor,
            "shape": self.shape,
            "dtype": str(self.image.dtype),
            "resolution_m": self.resolution_m,
            "min_pixel": float(np.min(self.image)),
            "max_pixel": float(np.max(self.image)),
            "mean_pixel": float(np.mean(self.image)),
            "std_pixel": float(np.std(self.image)),
            "sun_angle_info": self.sun_angle_info,
            "file_path": self.file_path,
        }
