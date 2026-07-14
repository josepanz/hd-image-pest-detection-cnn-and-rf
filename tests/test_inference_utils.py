import os

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.evaluation.inference_utils import get_all_sample_folders, run_inference_on_path


def _write_rgb_tif(path):
    cv2.imwrite(path, (np.random.rand(20, 20, 3) * 255).astype("uint8"))


def test_get_all_sample_folders_accepts_a_bare_rgb_file(tmp_path):
    """Regresión: antes esto devolvía [] (0 muestras, sin error) para un archivo
    rgb.tif suelto, aunque load_and_preprocess_image sí sabía leerlo directamente."""
    rgb_file = tmp_path / "20210525_rgb.tif"
    _write_rgb_tif(str(rgb_file))

    assert get_all_sample_folders(str(rgb_file), is_ms=False) == [str(rgb_file)]


def test_get_all_sample_folders_accepts_a_folder_containing_the_rgb_file(tmp_path):
    rgb_file = tmp_path / "20210525_rgb.tif"
    _write_rgb_tif(str(rgb_file))

    assert get_all_sample_folders(str(tmp_path), is_ms=False) == [str(tmp_path)]


class _FakeFeatureExtractor:
    def __init__(self, features):
        self._features = features

    def predict(self, x, verbose=0):
        return self._features


class _RecordingRF:
    """Sustituye al RandomForestClassifier real: solo registra qué features
    recibió, para poder verificar si estaban escaladas o no."""

    def __init__(self):
        self.last_x_input = None

    def predict_proba(self, x_input):
        self.last_x_input = x_input
        return np.array([[0.3, 0.7]])


def test_run_inference_on_path_applies_the_bundled_scaler_to_rf_features(tmp_path):
    """Regresión: run_inference_on_path extraía el RF del bundle {rf_model, scaler,
    feature_extractor} pero nunca extraía ni aplicaba el scaler, pasándole al RF
    features crudas de la CNN en vez de las escaladas que vio en entrenamiento."""
    rgb_file = tmp_path / "sample_rgb.tif"
    _write_rgb_tif(str(rgb_file))

    raw_features = np.array([[100.0, -50.0, 30.0]])
    scaler = StandardScaler().fit(np.array([[10.0, -5.0, 3.0], [90.0, -45.0, 27.0]]))
    rf = _RecordingRF()
    bundle = {"rf_model": rf, "scaler": scaler}

    run_inference_on_path(
        model=bundle,
        feature_extractor_rf=_FakeFeatureExtractor(raw_features),
        path=str(rgb_file),
        threshold=0.5,
        img_size=(20, 20),
        model_name="fake_rf.joblib",
        is_multiespectral=False,
        is_random_forest=True,
    )

    expected = scaler.transform(raw_features)
    np.testing.assert_allclose(rf.last_x_input, expected)


def test_run_inference_on_path_works_without_a_scaler_in_the_bundle(tmp_path):
    """Bundles viejos sin 'scaler' no deben romper: se le pasan las features tal cual."""
    rgb_file = tmp_path / "sample_rgb.tif"
    _write_rgb_tif(str(rgb_file))

    raw_features = np.array([[1.0, 2.0, 3.0]])
    rf = _RecordingRF()
    bundle = {"rf_model": rf}  # sin scaler

    run_inference_on_path(
        model=bundle,
        feature_extractor_rf=_FakeFeatureExtractor(raw_features),
        path=str(rgb_file),
        threshold=0.5,
        img_size=(20, 20),
        model_name="fake_rf.joblib",
        is_multiespectral=False,
        is_random_forest=True,
    )

    np.testing.assert_allclose(rf.last_x_input, raw_features)
