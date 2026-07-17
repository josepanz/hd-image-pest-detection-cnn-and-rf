import tensorflow as tf
from keras import layers, Model, Input
from keras.metrics import Precision, Recall, AUC, BinaryAccuracy
from pest_detection.focal_loss import focal_loss
from pest_detection.models.plaga_metrics import RecallPlaga, F2Plaga, F2Macro

def crear_modelo_cnn(input_shape, loss_type='focal_loss', alpha=0.25, gamma=2.0):
    """CNN simple (3 bloques Conv+BN+MaxPool -> GAP -> Dense -> sigmoid) para
    clasificación binaria Plaga/Sana. input_shape debe ser (224, 224, 3) para RGB o
    (224, 224, 5) para multiespectral (bandas: red, red edge, nir, blue, green, ver
    BAND_SUFFIXES en extract_data_to_img.py); la arquitectura es la misma en ambos
    casos, solo cambia el número de canales de entrada.

    NOTA sobre normalización: esta capa Rescaling(1./255) divide la entrada por 255
    DE NUEVO, sin importar cómo haya llegado normalizada desde extract_data_to_img.py
    (que ya la deja en ~0-1 antes de esto). Es redundante pero consistente: tanto
    entrenamiento como inference_utils.py normalizan la entrada de la misma forma
    antes de llegar acá (RGB: /255; multiespectral: /max de la imagen), así que esta
    segunda división por 255 se aplica igual en ambos lados (ver
    evaluation/inference_utils.py para el detalle de esa normalización previa).
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
            Recall(name='recall'), # BUG: mide recall de Sana (clase 1), no de Plaga - ver plaga_metrics.py. Se deja por compatibilidad/diagnóstico, ya no se usa para nada crítico.
            AUC(name='auc'),
            RecallPlaga(),
            F2Plaga(),
            F2Macro(),
            ]
    )

    return model