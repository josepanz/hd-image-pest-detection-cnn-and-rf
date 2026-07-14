import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from keras import layers, Model, Input
from keras.metrics import Precision, Recall, AUC, BinaryAccuracy
from src.utils.focal_loss import focal_loss

def crear_modelo_cnn(input_shape, loss_type='focal_loss', alpha=0.25, gamma=2.0):
    """CNN simple (3 bloques Conv+BN+MaxPool -> GAP -> Dense -> sigmoid) para
    clasificación binaria Plaga/Sana. input_shape debe ser (224, 224, 3) para RGB o
    (224, 224, 5) para multiespectral (bandas: red, red edge, nir, blue, green, ver
    BAND_SUFFIXES en extract_data_to_img.py); la arquitectura es la misma en ambos
    casos, solo cambia el número de canales de entrada.

    NOTA sobre normalización: esta capa Rescaling(1./255) divide la entrada por 255
    DE NUEVO, sin importar cómo haya llegado normalizada desde extract_data_to_img.py
    (que ya la deja en ~0-1 antes de esto). Para RGB es consistente porque tanto el
    entrenamiento como inference_utils.py normalizan igual antes de llegar acá
    (doble división por 255, pero igual en ambos lados). Para multiespectral NO es
    consistente entre entrenamiento (/max por imagen) e inferencia (/255 fijo) - ver
    el comentario de bug documentado en evaluation/inference_utils.py.
    """
    inp = Input(shape=input_shape)

    x = layers.Rescaling(1./255)(inp)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=focal_loss(alpha, gamma) if loss_type == 'focal_loss' else 'binary_crossentropy',
        metrics=[
            BinaryAccuracy(name='accuracy'), 
            Precision(name='precision'), 
            Recall(name='recall'), 
            AUC(name='auc')
            ]
    )

    return model