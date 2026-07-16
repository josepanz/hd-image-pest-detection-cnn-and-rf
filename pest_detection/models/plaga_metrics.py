"""Métricas custom para la clase "Plaga" (0), usadas por ModelCheckpoint en
callbacks.py y expuestas en el entrenamiento (train.py).

BUG CORREGIDO: el 'recall' de Keras (Recall(name='recall') en cnn_model.py) mide la
clase 1 sobre la salida sigmoide cruda - y como LabelEncoder asigna Plaga=0/Sana=1
(orden alfabético), eso es en realidad recall de SANA, no de Plaga. La tesis del
proyecto es explícita en que ModelCheckpoint debe maximizar el recall de PLAGA
("minimizar los falsos negativos, asegurando que el sistema sea capaz de detectar la
mayor cantidad posible de plantas afectadas [por plaga]" - el costo de no detectar
una plaga es mayor que el de una falsa alarma). RecallPlaga corrige esto.

F2Plaga (F-beta, beta=2) pesa el recall 4x más que la precisión en la fórmula
- se usa como criterio de ModelCheckpoint en vez de recall puro porque recall puro
tiene techo trivial: un modelo que predice "Plaga" para todo saca recall_plaga=1.0
sin ser útil. F2 sigue premiando recall alto (la prioridad real, según la tesis)
pero penaliza ese colapso porque su precisión sería pésima.
"""

import tensorflow as tf


class RecallPlaga(tf.keras.metrics.Metric):
    def __init__(self, name='recall_plaga', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_bin = tf.cast(tf.reshape(y_pred, [-1]) >= 0.5, tf.float32)
        is_plaga = 1.0 - y_true
        pred_plaga = 1.0 - y_pred_bin
        self.tp.assign_add(tf.reduce_sum(is_plaga * pred_plaga))
        self.fn.assign_add(tf.reduce_sum(is_plaga * (1.0 - pred_plaga)))

    def result(self):
        return self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)


class F2Plaga(tf.keras.metrics.Metric):
    def __init__(self, name='f2_plaga', beta=2.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_bin = tf.cast(tf.reshape(y_pred, [-1]) >= 0.5, tf.float32)
        is_plaga = 1.0 - y_true
        pred_plaga = 1.0 - y_pred_bin
        self.tp.assign_add(tf.reduce_sum(is_plaga * pred_plaga))
        self.fp.assign_add(tf.reduce_sum((1.0 - is_plaga) * pred_plaga))
        self.fn.assign_add(tf.reduce_sum(is_plaga * (1.0 - pred_plaga)))

    def result(self):
        eps = tf.keras.backend.epsilon()
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        b2 = self.beta ** 2
        return (1 + b2) * precision * recall / (b2 * precision + recall + eps)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
