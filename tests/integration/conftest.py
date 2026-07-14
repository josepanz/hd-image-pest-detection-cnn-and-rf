"""Fixtures compartidas por los tests de integración: no hay imágenes .tif reales en
el repo (ni el dataset TTADDA ni predict-test/ las incluyen), así que se generan
sintéticas acá mismo con cv2 - alcanza porque el motor de inferencia vigente
(inference_utils.py) lee todas las bandas con cv2.imread, no con rasterio.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BEST_MODELS_DIR = REPO_ROOT / "best_models"

# Debe coincidir con BAND_SUFFIXES en pest_detection/evaluation/inference_utils.py
_MS_BAND_SUFFIXES = ["_blue.tif", "_green.tif", "_red.tif", "_red edge.tif", "_nir.tif"]


@pytest.fixture
def ms_sample_dir(tmp_path):
    """Carpeta con las 5 bandas multiespectrales de una muestra sintética."""
    sample_dir = tmp_path / "2021-05-25"
    sample_dir.mkdir()
    rng = np.random.default_rng(42)
    for suffix in _MS_BAND_SUFFIXES:
        band = (rng.random((40, 40)) * 255).astype("uint8")
        cv2.imwrite(str(sample_dir / f"sample{suffix}"), band)
    return sample_dir


@pytest.fixture
def rgb_sample_file(tmp_path):
    """Un único archivo *rgb.tif sintético (RGB de 3 canales)."""
    rgb_path = tmp_path / "20210525_rgb.tif"
    rng = np.random.default_rng(7)
    img = (rng.random((40, 40, 3)) * 255).astype("uint8")
    cv2.imwrite(str(rgb_path), img)
    return rgb_path
