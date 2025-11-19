# src/models/losses.py

import tensorflow as tf

def focal_loss(alpha: float = 0.50, gamma: float = 3.0):
    """
    Genera una función de pérdida focal configurada con alpha y gamma.
    alpha=0.5 (por defecto) significa balance de clases por pérdida;
    para Focal Loss pura, a menudo se usa alpha=0.25 (o 0.15 como sugeriste).
    
    Referencia: Duplicado en modelfl.py, modelbc.py, model_multiespectral.py
    """
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Binary Cross-Entropy (BCE)
        bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Alpha weighting factor (para balance de clases)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

        # Modulating factor (p_t es la probabilidad del verdadero valor)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.math.pow((1 - p_t), gamma)
        
        # Pérdida focal
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    
    return loss_fn