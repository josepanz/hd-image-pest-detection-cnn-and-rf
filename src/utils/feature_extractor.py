import numpy as np
from keras.models import Model

def extraer_features_cnn(model, X):
    """
    Extrae features quitando la última capa (sigmoid)
    """
    feature_model = Model(
        inputs=model.input,
        outputs=model.layers[-2].output  # antes de la última capa
    )

    features = feature_model.predict(X)
    return features