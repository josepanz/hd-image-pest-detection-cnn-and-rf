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


class F2Macro(tf.keras.metrics.Metric):
    """Promedio del F2 de Plaga y el F2 de Sana - usado como criterio de
    ModelCheckpoint en vez de F2Plaga a secas.

    BUG ENCONTRADO EN LA PRÁCTICA (validado contra el dataset real, ronda 1 del
    reentrenamiento de este fix): F2Plaga sola no evita el colapso a "predecir
    Plaga para todo", porque Plaga ya es la clase mayoritaria (67.7% del dataset) -
    ese modelo degenerado saca precision_plaga~0.68 gratis y recall_plaga=1.0,
    dando F2Plaga~0.91 sin ser útil (ignora a Sana por completo). Promediando con
    F2 de Sana (que sería 0 en ese caso degenerado, al no acertar ningún Sana) el
    colapso queda penalizado (F2 macro cae a ~0.46), mientras se sigue premiando
    recall alto en ambas clases (beta=2 en cada una)."""
    def __init__(self, name='f2_macro', beta=2.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.tp_p = self.add_weight(name='tp_p', initializer='zeros')
        self.fp_p = self.add_weight(name='fp_p', initializer='zeros')
        self.fn_p = self.add_weight(name='fn_p', initializer='zeros')
        self.tp_s = self.add_weight(name='tp_s', initializer='zeros')
        self.fp_s = self.add_weight(name='fp_s', initializer='zeros')
        self.fn_s = self.add_weight(name='fn_s', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_bin = tf.cast(tf.reshape(y_pred, [-1]) >= 0.5, tf.float32)

        is_plaga = 1.0 - y_true
        pred_plaga = 1.0 - y_pred_bin
        self.tp_p.assign_add(tf.reduce_sum(is_plaga * pred_plaga))
        self.fp_p.assign_add(tf.reduce_sum((1.0 - is_plaga) * pred_plaga))
        self.fn_p.assign_add(tf.reduce_sum(is_plaga * (1.0 - pred_plaga)))

        is_sana = y_true
        pred_sana = y_pred_bin
        self.tp_s.assign_add(tf.reduce_sum(is_sana * pred_sana))
        self.fp_s.assign_add(tf.reduce_sum((1.0 - is_sana) * pred_sana))
        self.fn_s.assign_add(tf.reduce_sum(is_sana * (1.0 - pred_sana)))

    def _f2(self, tp, fp, fn):
        eps = tf.keras.backend.epsilon()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        b2 = self.beta ** 2
        return (1 + b2) * precision * recall / (b2 * precision + recall + eps)

    def result(self):
        f2_plaga = self._f2(self.tp_p, self.fp_p, self.fn_p)
        f2_sana = self._f2(self.tp_s, self.fp_s, self.fn_s)
        return (f2_plaga + f2_sana) / 2.0

    def reset_state(self):
        for w in (self.tp_p, self.fp_p, self.fn_p, self.tp_s, self.fp_s, self.fn_s):
            w.assign(0.0)
