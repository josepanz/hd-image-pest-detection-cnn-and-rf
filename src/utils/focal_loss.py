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
        # binary_crossentropy reduce el último eje: de (batch, 1) pasa a (batch,).
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # BUG HISTÓRICO (corregido): p_t/alpha_factor/modulating_factor se calculaban
        # con y_true/y_pred todavía en forma (batch, 1). Al multiplicarlos por bce en
        # forma (batch,), broadcasting los combinaba en una matriz (batch, batch) en vez
        # de un vector (batch,): la diagonal tenía el valor por-muestra correcto, pero el
        # resto eran productos cruzados sin sentido (factor de la muestra i × bce de la
        # muestra j) que diluían/corrompían el promedio usado como gradiente. Se
        # resuelve reshapeando a la misma forma que bce antes de combinar.
        y_true_flat = tf.reshape(y_true, tf.shape(bce))
        y_pred_flat = tf.reshape(y_pred, tf.shape(bce))

        p_t = y_true_flat * y_pred_flat + (1 - y_true_flat) * (1 - y_pred_flat)
        alpha_factor = y_true_flat * alpha + (1 - y_true_flat) * (1 - alpha)
        modulating_factor = tf.pow((1 - p_t), gamma)

        return alpha_factor * modulating_factor * bce

    return loss