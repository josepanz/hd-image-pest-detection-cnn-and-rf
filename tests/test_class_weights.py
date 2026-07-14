import pytest

from pest_detection.datasets.class_weights import calculate_class_weights


def test_zero_samples_falls_back_to_neutral_weights():
    assert calculate_class_weights(0, 0) == {0: 1.0, 1: 1.0}


def test_balanced_classes_only_gets_the_factor_sensibilidad_boost_on_plaga():
    weights = calculate_class_weights(100, 100)

    # sin desbalance, el peso base de ambas clases es 1.0; "Plaga" además se
    # multiplica por factor_sensibilidad (1.5) a propósito.
    assert weights[0] == pytest.approx(1.5)
    assert weights[1] == pytest.approx(1.0)


def test_matches_real_values_observed_in_training_logs():
    """Caso real tomado de BITACORA_clean.md: Plaga=360, Sana=172 ->
    'Pesos de Clase Calculados: Plaga (0): 1.11, Sana (1): 1.55'."""
    weights = calculate_class_weights(360, 172)

    assert weights[0] == pytest.approx(1.11, abs=0.005)
    assert weights[1] == pytest.approx(1.55, abs=0.005)


def test_missing_one_class_does_not_divide_by_zero():
    weights = calculate_class_weights(0, 50)

    assert weights[0] == pytest.approx(1.5)  # fallback 1.0 * factor_sensibilidad
    assert weights[1] == pytest.approx(0.5)  # 50 / (2 * 50)
