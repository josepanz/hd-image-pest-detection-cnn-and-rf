import numpy as np
from sklearn.metrics import roc_curve

from src.utils.utils_train import encontrar_umbral_optimo


class _FakeModel:
    """Sustituye a un modelo Keras real: solo necesita .predict(x) -> array de
    probabilidades, que es lo único que encontrar_umbral_optimo usa del modelo."""

    def __init__(self, probs):
        self._probs = np.asarray(probs).reshape(-1, 1)

    def predict(self, x_val):
        return self._probs


def test_perfectly_separable_data_yields_a_threshold_with_perfect_youden_j():
    y_val = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    model = _FakeModel(probs)

    best_threshold = encontrar_umbral_optimo(model, x_val=None, y_val=y_val)

    y_pred = (probs >= best_threshold).astype(int)
    np.testing.assert_array_equal(y_pred, y_val)  # separa perfectamente

    fpr, tpr, thresholds = roc_curve(y_val, probs)
    j_at_best = (tpr - fpr)[list(thresholds).index(best_threshold)]
    assert j_at_best == 1.0


def test_returned_threshold_is_one_of_the_roc_curve_thresholds():
    y_val = np.array([0, 1, 0, 1])
    probs = np.array([0.3, 0.8, 0.4, 0.6])
    model = _FakeModel(probs)

    best_threshold = encontrar_umbral_optimo(model, x_val=None, y_val=y_val)

    _, _, thresholds = roc_curve(y_val, probs)
    assert best_threshold in thresholds
