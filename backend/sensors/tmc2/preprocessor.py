"""
TMC-2 Preprocessor (Module 8)

Preprocessing operations tailored specifically for TMC-2 stereo / orthorectified imagery:
radiometric stretch, adaptive CLAHE contrast enhancement, terrain illumination normalization,
and multi-scale smoothing.
"""

from typing import Dict, Any, Optional
import numpy as np
import cv2

from backend.core.lunar_image import LunarImage
from backend.preprocessing.core import OHRCPreprocessor


class TMC2Preprocessor:
    """
    Preprocessing pipeline for TMC-2 rasters.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.prep_cfg = self.config.get("preprocessing", {})

    def process(self, lunar_img: LunarImage, custom_cfg: Optional[Dict[str, Any]] = None) -> LunarImage:
        """
        Processes a TMC-2 LunarImage non-destructively.

        Args:
            lunar_img: Source TMC-2 LunarImage.
            custom_cfg: Optional override configuration.

        Returns:
            LunarImage: New preprocessed copy.
        """
        if lunar_img.sensor != "TMC-2":
            print(f"Notice: TMC2Preprocessor received image with sensor='{lunar_img.sensor}'")

        cfg = custom_cfg if custom_cfg is not None else self.prep_cfg
        if not cfg.get("enabled", True):
            return lunar_img.copy()

        img_u8 = lunar_img.to_uint8()

        # 1. Percentile Stretch
        percentiles = cfg.get("stretch_percentiles", [1.0, 99.0])
        img_u8 = OHRCPreprocessor.percentile_stretch(img_u8, p_low=percentiles[0], p_high=percentiles[1])

        # 2. High Pass Illumination Normalization for low-frequency shading
        illum_cfg = cfg.get("illumination_norm", {})
        if illum_cfg.get("enabled", True):
            sigma = float(illum_cfg.get("sigma", 12.0))
            img_u8 = OHRCPreprocessor.normalize_illumination(img_u8, method="high_pass", sigma=sigma)

        # 3. CLAHE Contrast Normalization
        clahe_cfg = cfg.get("clahe", {})
        if clahe_cfg.get("enabled", True):
            clip_limit = float(clahe_cfg.get("clip_limit", 2.5))
            grid_size = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
            img_u8 = OHRCPreprocessor.apply_clahe(img_u8, clip_limit=clip_limit, tile_grid_size=grid_size)

        # 4. Optional mild Denoise
        img_u8 = cv2.GaussianBlur(img_u8, (3, 3), 0)

        out_lunar = lunar_img.copy()
        out_lunar.image = img_u8
        out_lunar.metadata["preprocessing_applied"] = cfg

        return out_lunar
