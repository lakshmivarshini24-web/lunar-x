"""
Unit tests for Module 11 & 12: IIRS Data Validator, Reader, PCA Data Reducer, and Preprocessor.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from backend.core.lunar_image import LunarImage
from backend.sensors.iirs.validator import IIRSValidator
from backend.sensors.iirs.reader import IIRSReader, load_iirs
from backend.sensors.iirs.pca import IIRSDataReducer, PCAResult
from backend.sensors.iirs.preprocessor import IIRSPreprocessor


def test_iirs_validator(tmp_path):
    arr_path = tmp_path / "valid_iirs.npy"
    arr = np.random.rand(64, 64, 16).astype(np.float32)
    np.save(str(arr_path), arr)

    validator = IIRSValidator()
    res = validator.validate_file(str(arr_path))

    assert res["is_valid"] is True
    assert res["dimensions"] == (64, 64)
    assert res["num_bands"] == 16
    assert res["sensor"] == "IIRS"


def test_iirs_reader(tmp_path):
    arr_path = tmp_path / "test_iirs.npy"
    arr = np.random.rand(50, 50, 32).astype(np.float32)
    np.save(str(arr_path), arr)

    meta_path = tmp_path / "test_iirs.json"
    sidecar_data = {
        "acquisition_info": {"orbit": 500, "instrument": "IIRS"},
        "wavelengths_nm": np.linspace(800, 3000, 32).tolist(),
    }
    with open(meta_path, "w") as f:
        json.dump(sidecar_data, f)

    reader = IIRSReader()
    lunar_iirs = reader.read(arr_path)

    assert isinstance(lunar_iirs, LunarImage)
    assert lunar_iirs.shape == (50, 50, 32)
    assert lunar_iirs.sensor == "IIRS"
    assert lunar_iirs.resolution_m == 100.0
    assert len(lunar_iirs.metadata["wavelengths_nm"]) == 32


def test_iirs_pca_reducer():
    arr = np.random.rand(40, 40, 20).astype(np.float32)
    reducer = IIRSDataReducer(n_components=3)
    pca_res = reducer.fit_transform(arr)

    assert isinstance(pca_res, PCAResult)
    assert pca_res.reduced_image.shape == (40, 40, 3)
    assert len(pca_res.explained_variance_ratio) == 3
    assert pca_res.cumulative_explained_variance > 0.0


def test_iirs_preprocessor():
    arr = np.random.rand(40, 40, 16).astype(np.float32)
    lunar_iirs = LunarImage(image=arr, sensor="IIRS", resolution_m=100.0)

    preprocessor = IIRSPreprocessor()
    out_lunar, diag = preprocessor.process(lunar_iirs, custom_components=1)

    assert isinstance(out_lunar, LunarImage)
    assert out_lunar.shape == (40, 40)
    assert out_lunar.image.dtype == np.uint8
    assert "pca_result" in out_lunar.metadata
    assert diag["num_original_bands"] == 16
