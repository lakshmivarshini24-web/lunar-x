"""
OHRC Preprocessing Core (Module 3)

Non-destructive preprocessing pipeline for OHRC optical imagery, providing
radiometric normalization, CLAHE contrast enhancement, denoising, illumination
normalization, shadow enhancement, and edge enhancement operations.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from backend.core.lunar_image import LunarImage


class OHRCPreprocessor:
    """
    Modular, non-destructive preprocessing pipeline for Lunar images.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.prep_cfg = self.config.get("preprocessing", {})

    def process(self, lunar_img: LunarImage, custom_cfg: Optional[Dict[str, Any]] = None) -> LunarImage:
        """
        Applies configured preprocessing steps to input LunarImage non-destructively.

        Args:
            lunar_img: Source LunarImage object.
            custom_cfg: Optional override configuration dictionary.

        Returns:
            LunarImage: New preprocessed LunarImage object.
        """
        cfg = custom_cfg if custom_cfg is not None else self.prep_cfg
        if not cfg.get("enabled", True):
            return lunar_img.copy()

        # Convert to 8-bit uint8 normalized image array for CV operations
        img_u8 = lunar_img.to_uint8()

        # 1. Radiometric Normalization / Percentile Stretch
        if cfg.get("radiometric_normalization", True):
            percentiles = cfg.get("stretch_percentiles", [1.0, 99.0])
            img_u8 = self.percentile_stretch(img_u8, p_low=percentiles[0], p_high=percentiles[1])

        # 2. Illumination Normalization
        illum_cfg = cfg.get("illumination_norm", {})
        if illum_cfg.get("enabled", True):
            method = illum_cfg.get("method", "high_pass")
            sigma = float(illum_cfg.get("sigma", 15.0))
            img_u8 = self.normalize_illumination(img_u8, method=method, sigma=sigma)

        # 3. CLAHE Contrast Normalization
        clahe_cfg = cfg.get("clahe", {})
        if clahe_cfg.get("enabled", True):
            clip_limit = float(clahe_cfg.get("clip_limit", 3.0))
            grid_size = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
            img_u8 = self.apply_clahe(img_u8, clip_limit=clip_limit, tile_grid_size=grid_size)

        # 4. Denoising
        denoise_cfg = cfg.get("denoise", {})
        if denoise_cfg.get("enabled", True):
            method = denoise_cfg.get("method", "gaussian")
            ksize = int(denoise_cfg.get("kernel_size", 3))
            img_u8 = self.denoise(img_u8, method=method, kernel_size=ksize)

        # 5. Edge Enhancement
        edge_cfg = cfg.get("edge_enhancement", {})
        if edge_cfg.get("enabled", False):
            alpha = float(edge_cfg.get("alpha", 0.5))
            img_u8 = self.enhance_edges(img_u8, alpha=alpha)

        # Create output preprocessed LunarImage copy
        processed_lunar_img = lunar_img.copy()
        processed_lunar_img.image = img_u8
        processed_lunar_img.metadata["preprocessing_applied"] = cfg

        return processed_lunar_img

    @staticmethod
    def percentile_stretch(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:

        img_f = img.astype(np.float32)
        v_min, v_max = np.percentile(img_f, (p_low, p_high))
        if v_max > v_min:
            img_clipped = np.clip(img_f, v_min, v_max)
            stretched = (img_clipped - v_min) / (v_max - v_min) * 255.0
            return stretched.astype(np.uint8)
        return img.astype(np.uint8)

    @staticmethod
    def apply_clahe(img: np.ndarray, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:

        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

    @staticmethod
    def denoise(img: np.ndarray, method: str = "gaussian", kernel_size: int = 3) -> np.ndarray:

        if kernel_size % 2 == 0:
            kernel_size += 1

        if method == "gaussian":
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif method == "median":
            return cv2.medianBlur(img, kernel_size)
        elif method == "non_local_means":
            return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return img

    @staticmethod
    def normalize_illumination(img: np.ndarray, method: str = "high_pass", sigma: float = 15.0) -> np.ndarray:
        """
        Removes low-frequency illumination variations (sun angle gradient across terrain).
        """
        img_f = img.astype(np.float32)

        if method == "high_pass":
            # Subtract Gaussian low-pass filter (background shading model)
            low_pass = gaussian_filter(img_f, sigma=sigma)
            high_pass = img_f - low_pass + 128.0
            return np.clip(high_pass, 0, 255).astype(np.uint8)

        elif method == "homomorphic":
            # Log transform -> high pass -> exp transform
            img_log = np.log1p(img_f)
            low_pass = gaussian_filter(img_log, sigma=sigma)
            high_pass = img_log - low_pass
            result = np.expm1(high_pass)
            norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
            return norm.astype(np.uint8)

        elif method == "divide_median":
            kernel_size = int(sigma * 2) | 1
            med = cv2.medianBlur(img, kernel_size).astype(np.float32) + 1.0
            norm = (img_f / med) * 128.0
            return np.clip(norm, 0, 255).astype(np.uint8)

        return img

    @staticmethod
    def enhance_edges(img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Unsharp masking edge enhancement."""
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(img, 1.0 + alpha, blurred, -alpha, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
