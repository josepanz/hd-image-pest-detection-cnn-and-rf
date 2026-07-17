"""Tests de integración end-to-end: cargan los checkpoints reales ya entrenados en
best_models/ y corren cli/infer.py::run_unified_inference de punta a punta sobre
muestras sintéticas (ver conftest.py). Son lentos (cargan TensorFlow y modelos
reales) - correr con `pytest -m integration` o `pytest` sin filtrar (van incluidos).
Para iterar rápido sin ellos: `pytest -m "not integration"`.
"""

import json

import pytest

from pest_detection.cli import infer
from tests.integration.conftest import BEST_MODELS_DIR

pytestmark = pytest.mark.integration


def _model_path(filename):
    path = BEST_MODELS_DIR / filename
    if not path.exists():
        pytest.skip(f"No está el checkpoint esperado: {path}")
    return str(path)


def _latest_rf_joblib(prefix):
    """Los .joblib de RF incluyen fecha/hora de entrenamiento en el nombre (ver
    train.py::run_rf_training) - hardcodear un timestamp puntual hace que el test se
    salte silenciosamente en vez de correr apenas se reentrena. Se toma el más
    reciente que matchee el prefijo (RGB o MULTIESPECTRAL)."""
    candidates = sorted(BEST_MODELS_DIR.glob(f"best_model_final_random_forest_{prefix}_*.joblib"))
    if not candidates:
        pytest.skip(f"No hay ningún .joblib de RF {prefix} en {BEST_MODELS_DIR}")
    return str(candidates[-1])


def _run_and_load_json(tmp_path, path, model_path, arch_type, loss="fl"):
    infer.run_unified_inference(str(path), model_path, threshold=0.5, arch_type=arch_type, loss=loss, base_dir=str(tmp_path))

    result_jsons = list(tmp_path.rglob("results_*.json"))
    assert len(result_jsons) == 1, f"Se esperaba 1 JSON de resultados, se encontraron {len(result_jsons)}"
    return json.loads(result_jsons[0].read_text())


def test_cnn_multiespectral_inference_end_to_end(tmp_path, ms_sample_dir):
    model_path = _model_path("best_model_final_MULTIESPECTRAL_focal_loss.keras")

    results = _run_and_load_json(tmp_path, ms_sample_dir, model_path, arch_type="cnn")

    assert len(results) == 1
    r = results[0]
    assert r["prediccion"] in ("Plaga", "Sana")
    assert 0.0 <= r["prob_sana"] <= 1.0


def test_cnn_rgb_inference_end_to_end_with_bare_file_path(tmp_path, rgb_sample_file):
    """Ejercita también la regresión ya cubierta a nivel unitario en
    test_inference_utils.py, pero acá de punta a punta con un modelo real."""
    model_path = _model_path("best_model_final_RGB_binary_crossentropy.keras")

    results = _run_and_load_json(tmp_path, rgb_sample_file, model_path, arch_type="cnn")

    assert len(results) == 1
    assert results[0]["prediccion"] in ("Plaga", "Sana")


def test_random_forest_multiespectral_inference_end_to_end(tmp_path, ms_sample_dir):
    model_path = _latest_rf_joblib("MULTIESPECTRAL")
    # requiere su CNN "hermana" (misma banda RGB/MS, loss 'fl') presente en best_models/
    _model_path("best_model_final_MULTIESPECTRAL_focal_loss.keras")

    results = _run_and_load_json(tmp_path, ms_sample_dir, model_path, arch_type="rf", loss="fl")

    assert len(results) == 1
    assert results[0]["prediccion"] in ("Plaga", "Sana")


def test_random_forest_rgb_inference_end_to_end(tmp_path, rgb_sample_file):
    model_path = _latest_rf_joblib("RGB")
    _model_path("best_model_final_RGB_focal_loss.keras")

    results = _run_and_load_json(tmp_path, rgb_sample_file, model_path, arch_type="rf", loss="fl")

    assert len(results) == 1
    assert results[0]["prediccion"] in ("Plaga", "Sana")
