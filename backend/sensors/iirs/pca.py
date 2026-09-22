"""
IIRS PCA Dimensionality Reduction (Module 12)

Computes SVD-based Principal Component Analysis (PCA) on multi-band hyperspectral 3D arrays (H, W, Bands),
reducing spectral dimensionality to principal component spatial rasters while preserving max variance.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


@dataclass
class PCAResult:
    """Output container for PCA reduction."""

    reduced_image: np.ndarray  # Shape (H, W) if n_components==1 else (H, W, n_components)
    explained_variance_ratio: np.ndarray  # Percentage variance per component
    cumulative_explained_variance: float
    num_original_bands: int
    n_components: int
    components: np.ndarray  # Eigenvector matrix (n_components, B)
    mean_spectrum: np.ndarray  # Mean spectrum across pixels (B,)


class IIRSDataReducer:
    """
    SVD / Eig-based PCA dimensionality reduction for IIRS hyperspectral rasters.
    """

    def __init__(self, n_components: int = 3):
        if n_components <= 0:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components

    def fit_transform(self, hyperspectral_array: np.ndarray) -> PCAResult:
        """
        Fits PCA model on 3D array (H, W, B) and transforms into reduced space.

        Args:
            hyperspectral_array: (H, W, B) or (H, W) numpy array.

        Returns:
            PCAResult object.
        """
        if hyperspectral_array.ndim == 2:
            # Already 2D grayscale
            return PCAResult(
                reduced_image=hyperspectral_array,
                explained_variance_ratio=np.array([1.0]),
                cumulative_explained_variance=1.0,
                num_original_bands=1,
                n_components=1,
                components=np.array([[1.0]]),
                mean_spectrum=np.array([np.mean(hyperspectral_array)]),
            )

        h, w, b = hyperspectral_array.shape
        n_pixels = h * w

        # Reshape (H, W, B) to (n_pixels, B)
        X = hyperspectral_array.reshape(n_pixels, b).astype(np.float64)

        # Remove zero variance / NaN bands
        std_per_band = np.std(X, axis=0)
        valid_bands_mask = std_per_band > 1e-6
        if not np.any(valid_bands_mask):
            # Fallback if flat spectrum
            valid_bands_mask[0] = True

        X_valid = X[:, valid_bands_mask]
        num_valid_bands = X_valid.shape[1]
        k_comp = min(self.n_components, num_valid_bands)

        # 1. Mean-center spectral vectors
        mean_vec = np.mean(X_valid, axis=0)
        X_centered = X_valid - mean_vec

        # 2. Singular Value Decomposition (SVD)
        # X_centered = U @ S @ Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Explained variance calculation
        eigenvalues = (S ** 2) / (n_pixels - 1)
        total_var = np.sum(eigenvalues)
        var_ratio = (eigenvalues[:k_comp] / total_var) if total_var > 0 else np.ones(k_comp) / k_comp

        # Transform pixels into principal component space
        components = Vt[:k_comp, :]
        X_pca = X_centered @ components.T  # (n_pixels, k_comp)

        # Reshape back to spatial image (H, W, k_comp) or (H, W) if k_comp==1
        if k_comp == 1:
            img_pca = X_pca[:, 0].reshape(h, w)
            # Normalize to 8-bit range [0, 255] for feature extractors
            img_norm = cv2_normalize_u8(img_pca)
        else:
            img_pca = X_pca.reshape(h, w, k_comp)
            img_norm = np.zeros((h, w, k_comp), dtype=np.uint8)
            for c in range(k_comp):
                img_norm[:, :, c] = cv2_normalize_u8(img_pca[:, :, c])

        cum_var = float(np.sum(var_ratio))

        return PCAResult(
            reduced_image=img_norm,
            explained_variance_ratio=var_ratio,
            cumulative_explained_variance=cum_var,
            num_original_bands=b,
            n_components=k_comp,
            components=components,
            mean_spectrum=mean_vec,
        )


def cv2_normalize_u8(arr: np.ndarray) -> np.ndarray:
    """Normalizes 2D float array to uint8 [0, 255]."""
    min_v, max_v = np.min(arr), np.max(arr)
    if max_v > min_v:
        normed = (arr - min_v) / (max_v - min_v) * 255.0
        return normed.astype(np.uint8)
    return np.zeros(arr.shape, dtype=np.uint8)
