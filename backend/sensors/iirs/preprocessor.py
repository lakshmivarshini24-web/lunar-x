"""
IIRS Preprocessor (Module 12)

Executes spectral preprocessing, bad-band detection/filtering, spectral L2 vector normalization,
PCA dimensionality reduction, and diagnostic statistics generation for IIRS hyperspectral rasters.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2

from backend.core.lunar_image import LunarImage
from backend.sensors.iirs.pca import IIRSDataReducer, PCAResult


class IIRSPreprocessor:
    """
    Hyperspectral preprocessing and PCA reduction pipeline for Chandrayaan-2 IIRS datasets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        iirs_cfg = self.config.get("iirs", {})
        self.n_components = int(iirs_cfg.get("pca_n_components", 3))
        self.bad_band_thresh = float(iirs_cfg.get("bad_band_threshold_std", 1e-4))
        self.pca_reducer = IIRSDataReducer(n_components=self.n_components)

    def process(self, lunar_img: LunarImage, custom_components: Optional[int] = None) -> Tuple[LunarImage, Dict[str, Any]]:
        """
        Preprocesses hyperspectral LunarImage and generates PCA component representations.

        Args:
            lunar_img: Input IIRS LunarImage (2D or 3D).
            custom_components: Optional override for PCA n_components.

        Returns:
            Tuple[LunarImage, Dict[str, Any]]: (Preprocessed LunarImage, Diagnostic Statistics)
        """
        raw_arr = lunar_img.image.copy()

        # Handle NaNs / Infs
        if np.issubdtype(raw_arr.dtype, np.floating):
            raw_arr = np.nan_to_num(raw_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Band Statistics & Bad-Band Detection
        num_bands = 1 if raw_arr.ndim == 2 else raw_arr.shape[2]
        bad_bands: List[int] = []
        band_stats: List[Dict[str, float]] = []

        if raw_arr.ndim == 3:
            for b in range(num_bands):
                band_slice = raw_arr[:, :, b]
                std_v = float(np.std(band_slice))
                mean_v = float(np.mean(band_slice))
                band_stats.append({"band_idx": b, "mean": mean_v, "std": std_v})
                if std_v < self.bad_band_thresh:
                    bad_bands.append(b)

            # Filter bad bands if some valid bands remain
            valid_band_indices = [b for b in range(num_bands) if b not in bad_bands]
            if valid_band_indices:
                filtered_arr = raw_arr[:, :, valid_band_indices]
            else:
                filtered_arr = raw_arr
        else:
            filtered_arr = raw_arr
            band_stats.append({"band_idx": 0, "mean": float(np.mean(raw_arr)), "std": float(np.std(raw_arr))})

        # 2. PCA Dimensionality Reduction
        n_comp = custom_components if custom_components is not None else self.n_components
        reducer = IIRSDataReducer(n_components=n_comp) if n_comp != self.n_components else self.pca_reducer
        pca_res = reducer.fit_transform(filtered_arr)

        # Extract 2D single-channel raster for feature matching (PC1 or grayscale component)
        if pca_res.reduced_image.ndim == 3:
            pc1_u8 = pca_res.reduced_image[:, :, 0]
        else:
            pc1_u8 = pca_res.reduced_image

        # Create output LunarImage with PC1 2D raster while retaining full 3D hyperspectral payload in metadata
        out_lunar = lunar_img.copy()
        out_lunar.image = pc1_u8
        out_lunar.metadata["iirs_hyperspectral_payload"] = filtered_arr
        out_lunar.metadata["pca_result"] = {
            "n_components": pca_res.n_components,
            "explained_variance_ratio": pca_res.explained_variance_ratio.tolist(),
            "cumulative_explained_variance": float(pca_res.cumulative_explained_variance),
            "num_original_bands": num_bands,
            "num_bad_bands": len(bad_bands),
        }

        diagnostics = {
            "num_original_bands": num_bands,
            "num_bad_bands": len(bad_bands),
            "bad_bands": bad_bands,
            "band_statistics": band_stats[:10],  # sample first 10 for log
            "pca_explained_variance_ratio": pca_res.explained_variance_ratio.tolist(),
            "cumulative_explained_variance": float(pca_res.cumulative_explained_variance),
        }

        return out_lunar, diagnostics
