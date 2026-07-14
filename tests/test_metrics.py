import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.metrics import evaluar_modelo


def test_confusion_matrix_and_auc_match_hand_computed_case():
    y_true = np.array([0, 0, 1, 1])
    y_pred_probs = np.array([0.1, 0.4, 0.35, 0.8])

    cm, report, auc, fpr, tpr = evaluar_modelo(y_true, y_pred_probs, threshold=0.5)

    # y_pred con threshold=0.5: [0, 0, 0, 1] -> TN=2, FP=0, FN=1, TP=1
    np.testing.assert_array_equal(cm, np.array([[2, 0], [1, 1]]))
    assert auc == roc_auc_score(y_true, y_pred_probs)
    assert "precision" in report
    assert len(fpr) == len(tpr)


def test_perfect_separation_gives_auc_of_one():
    y_true = np.array([0, 0, 1, 1])
    y_pred_probs = np.array([0.05, 0.1, 0.9, 0.95])

    _, _, auc, _, _ = evaluar_modelo(y_true, y_pred_probs)

    assert auc == 1.0


def test_threshold_shifts_which_predictions_count_as_positive():
    y_true = np.array([0, 1])
    y_pred_probs = np.array([0.6, 0.6])

    cm_low, *_ = evaluar_modelo(y_true, y_pred_probs, threshold=0.5)
    cm_high, *_ = evaluar_modelo(y_true, y_pred_probs, threshold=0.9)

    # con threshold 0.5 ambas se clasifican como 1; con threshold 0.9 ambas como 0
    assert cm_low.tolist() != cm_high.tolist()
