# src/cnn/focal_loss/model.py
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from keras.applications import MobileNetV2
from keras.metrics import Precision, Recall, BinaryAccuracy
import tensorflow as tf # Asegúrate de tenerlo importado

def focal_loss(alpha: float = 0.50, gamma: float = 3.0): # <<<<<<<<<< CAMBIO AQUI: alpha=0.15, gamma=3.0
    """
    Genera una función de pérdida focal configurada con alpha y gamma.
    """
    def loss_fn(y_true, y_pred):
        # y_pred para evitar log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Cálculo del BCE (Binary Cross-Entropy)
        bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        
        # Peso alpha para balancear clases:
        # y_true es 1 para "Sana" y 0 para "Plaga".
        # Queremos penalizar más los errores de "Plaga" (cuando es 0)
        # alpha_factor debe ser (1 - alpha) cuando y_true es 0 (Plaga)
        # alpha_factor debe ser alpha cuando y_true es 1 (Sana)
        # Con alpha = 0.15, cuando y_true es 0 (Plaga), el factor es 0.85
        # Cuando y_true es 1 (Sana), el factor es 0.15
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha) # Esta línea es correcta para el efecto deseado

        # Modulating factor (p_t es la probabilidad del verdadero valor)
        # Si y_true es 1 (Sana), p_t es y_pred (probabilidad de Sana)
        # Si y_true es 0 (Plaga), p_t es 1 - y_pred (probabilidad de Plaga)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.math.pow((1 - p_t), gamma)
        
        # Pérdida focal
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    return loss_fn


def crear_modelo(
    input_shape: tuple[int,int,int] = (224,224,3),
    dropout_rate: float = 0.3, 
    alpha: float = 0.50, # <<<<<<<<<< CAMBIO AQUI: alpha=0.15
    gamma: float = 3.0 # <<<<<<<<<< CAMBIO AQUI: gamma=3.0
) -> Model:
    """
    - Backbone MobileNetV2 (pre-entrenado en ImageNet) congelado.
    - GlobalAveragePooling + Dropout + Dense(sigmoid).
    - Focal Loss y métricas precision/recall.
    """
    # 1) Backbone
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    # 2) Cabeza de clasificación
    inp = Input(shape=input_shape, name="input_image")
    x = base(inp, training=False)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dropout(dropout_rate, name="dropout")(x)
    out = Dense(1, activation="sigmoid", name="prediction")(x)

    model = Model(inp, out, name="hd_plagas_mobilenet_focal")

    # 3) Compilación con Focal Loss
    model.compile(
        optimizer="adam", 
        loss=focal_loss(alpha=alpha, gamma=gamma), # Se usa el alpha y gamma pasados a la función
        metrics=[ 
            BinaryAccuracy(name='accuracy'), # Precisión binaria general
            # Para Precision y Recall, por defecto se calculan para la clase "positiva" (la que tiene y_true=1).
            # En nuestro caso, hemos definido "Sana" como 1 y "Plaga" como 0.
            # Si queremos ver el recall/precision para Plaga, podemos especificar `class_id=0` o `name='precision_plaga'`.
            # Sin embargo, para un clasificador binario simple, los valores de Precision/Recall están inherentemente vinculados.
            # Mantendremos estos así, ya que son estándar y la matriz de confusión nos dará el detalle.
            Precision(name='precision'),     
            Recall(name='recall')             
        ]
    )

    return model


if __name__ == "__main__":
    m = crear_modelo()
    m.summary()