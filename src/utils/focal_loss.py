"""Implementación canónica de Focal Loss del proyecto.

Es la única usada para compilar/entrenar las CNN (models/cnn_model.py) y también la
que se pasa como custom_objects al recargar un .keras guardado para evaluación o
inferencia (evaluation/model_loading.py). Antes existía una segunda implementación
con fórmula ligeramente distinta en models/function_losses.py, usada solo al
recargar el modelo; se unificó en esta para que loss reportado en evaluate.py sea
consistente con el que efectivamente se usó al entrenar (ver alpha/gamma abajo).
"""

import tensorflow as tf

def focal_loss(alpha=0.25, gamma=2.0):
    """alpha pondera la clase minoritaria (ver train.py: con focal_loss no se aplica
    además class_weight, alpha ya cumple ese rol). gamma controla cuánto se
    down-weightea a los ejemplos ya bien clasificados (p_t alto), enfocando el
    gradiente en los casos difíciles. Los valores por defecto acá NO son
    necesariamente los usados para entrenar un checkpoint dado: esos se pasan por
    CLI en train.py (-a/-g) y no quedan persistidos junto al .keras."""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow((1 - p_t), gamma)

        return alpha_factor * modulating_factor * bce

    return loss