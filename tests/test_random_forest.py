import numpy as np
from sklearn.datasets import make_classification

from src.utils.models.random_forest import model_random_forest, entrenar_rf, evaluar_rf


def _synthetic_dataset():
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=5, n_redundant=0,
        weights=[0.7, 0.3], random_state=42,
    )
    split = 150
    return X[:split], X[split:], y[:split], y[split:]


def test_trains_and_predicts_on_synthetic_data():
    X_train, X_test, y_train, y_test = _synthetic_dataset()

    rf = model_random_forest(n_estimators=50, max_depth=5, random_state=42, class_weight="balanced")
    rf = entrenar_rf(rf, X_train, y_train)
    y_pred, y_prob = evaluar_rf(rf, X_test, y_test)

    assert y_pred.shape == y_test.shape
    assert y_prob.shape == y_test.shape
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))
    # dataset separable a propósito: un RF de 50 árboles debería hacerlo bastante bien
    accuracy = (y_pred == y_test).mean()
    assert accuracy > 0.7


def test_accepts_an_explicit_class_weight_dict_like_calculate_class_weights_returns():
    X_train, X_test, y_train, y_test = _synthetic_dataset()

    rf = model_random_forest(class_weight={0: 1.0, 1: 1.5})
    rf = entrenar_rf(rf, X_train, y_train)
    y_pred, _ = evaluar_rf(rf, X_test, y_test)

    assert y_pred.shape == y_test.shape
