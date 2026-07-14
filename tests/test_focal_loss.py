import numpy as np
import tensorflow as tf

from src.utils.focal_loss import focal_loss


def test_output_shape_matches_batch_size_not_batch_squared():
    """Regresión: alpha_factor/modulating_factor en forma (batch, 1) contra bce ya
    reducido a (batch,) hacían broadcast a (batch, batch) en vez de (batch,)."""
    loss_fn = focal_loss(alpha=0.25, gamma=2.0)
    y_true = tf.constant([[1.0], [0.0], [1.0]])
    y_pred = tf.constant([[0.9], [0.1], [0.6]])

    out = loss_fn(y_true, y_pred)

    assert out.shape == (3,)


def test_confident_correct_prediction_has_lower_loss_than_unsure_one():
    loss_fn = focal_loss(alpha=0.25, gamma=2.0)
    y_true = tf.constant([[1.0], [0.0]])

    loss_confident = loss_fn(y_true, tf.constant([[0.99], [0.01]])).numpy()
    loss_unsure = loss_fn(y_true, tf.constant([[0.55], [0.45]])).numpy()

    assert np.all(loss_confident < loss_unsure)


def test_alpha_half_weighs_both_classes_equally():
    loss_fn = focal_loss(alpha=0.5, gamma=2.0)

    loss_pos = loss_fn(tf.constant([[1.0]]), tf.constant([[0.3]])).numpy()
    loss_neg = loss_fn(tf.constant([[0.0]]), tf.constant([[0.7]])).numpy()

    np.testing.assert_allclose(loss_pos, loss_neg, rtol=1e-5)


def test_higher_gamma_downweights_already_well_classified_examples_more():
    y_true = tf.constant([[1.0]])
    y_pred = tf.constant([[0.9]])  # ya bien clasificado

    loss_gamma_low = focal_loss(alpha=0.25, gamma=0.0)(y_true, y_pred).numpy()
    loss_gamma_high = focal_loss(alpha=0.25, gamma=5.0)(y_true, y_pred).numpy()

    assert loss_gamma_high < loss_gamma_low
