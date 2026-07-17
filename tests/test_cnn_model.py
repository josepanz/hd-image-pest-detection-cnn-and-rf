import numpy as np
import pytest

from pest_detection.models.cnn_model import crear_modelo_cnn


@pytest.mark.parametrize("channels", [3, 5])  # RGB=3, multiespectral=5 bandas
def test_builds_and_predicts_with_expected_shapes(channels):
    input_shape = (32, 32, channels)  # tamaño chico para que el test sea rápido
    model = crear_modelo_cnn(input_shape=input_shape, loss_type="focal_loss")

    assert model.input_shape == (None, *input_shape)
    assert model.output_shape == (None, 1)

    batch = np.random.rand(2, *input_shape).astype("float32")
    preds = model.predict(batch, verbose=0)

    assert preds.shape == (2, 1)
    assert np.all((preds >= 0.0) & (preds <= 1.0))  # sigmoid


def test_binary_crossentropy_loss_type_also_builds_a_working_model():
    model = crear_modelo_cnn(input_shape=(32, 32, 3), loss_type="binary_crossentropy")

    batch = np.random.rand(1, 32, 32, 3).astype("float32")
    preds = model.predict(batch, verbose=0)

    assert preds.shape == (1, 1)
