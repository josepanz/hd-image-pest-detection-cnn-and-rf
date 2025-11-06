import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
from keras.applications import MobileNetV2 # O la arquitectura que uses (VGG, ResNet, etc.)

from keras.metrics import Precision, Recall, BinaryAccuracy

# --- Focal Loss Implementation (Asumiendo que ya la tienes en tu entorno) ---
# Si Focal Loss es un archivo separado, asegúrate de importarlo.
# from focal_loss import BinaryFocalLoss 
# Si usas una implementación estándar, la tendrás disponible.
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


def crear_modelo_multiespectral(
    input_shape: tuple[int, int, int] = (224, 224, 5),  # ¡5 CANALES!
    num_classes: int = 1,
    fine_tune_layers: bool = False,
    dropout_rate: float = 0.5,
    l2_reg: float = 0.0001,
    loss_function: str = 'binary_crossentropy',
    alpha: float = 0.50, # <<<<<<<<<< CAMBIO AQUI: alpha=0.15
    gamma: float = 3.0 # <<<<<<<<<< CAMBIO AQUI: gamma=3.0
) -> Model:
    """
    Crea un modelo CNN basado en MobileNetV2 adaptado para 5 canales de entrada 
    y diseñado para la clasificación binaria (Plaga/Sana).

    Args:
        input_shape: Forma de la imagen (Alto, Ancho, Canales). Debe ser (224, 224, 5).
        num_classes: Número de clases de salida (1 para clasificación binaria con sigmoid).
        fine_tune_layers: Si se debe permitir el entrenamiento de las capas base.
        dropout_rate: Tasa de Dropout en la capa de clasificación.
        l2_reg: Tasa de regularización L2 para las capas densas.

    Returns:
        Modelo Keras compilado.
    """
    
    # 1. Definir la entrada con N canales
    input_tensor = Input(shape=input_shape) 
    
    # 2. Cargar el modelo base (Transfer Learning)
    # IMPORTANTE: Usamos weights=None porque los pesos de ImageNet (RGB=3 canales)
    # no son válidos para 5 canales (incluyendo NIR y Red Edge).
    # ¡El modelo base se entrenará desde cero o con una inicialización aleatoria!
    base_model = MobileNetV2(
        input_tensor=input_tensor,
        include_top=False,  # Excluir la capa final de clasificación original
        weights=None,       # Entrenar desde cero (o re-entrenar)
        pooling='avg'
    )
    
    # 3. Congelar/Descongelar Capas (si se desea Transfer Learning parcial)
    if not fine_tune_layers:
        base_model.trainable = False
    
    # 4. Crear la cabeza de clasificación (Classification Head)
    x = base_model.output
    
    # Capa Densa con Regularización
    x = Dense(
        128, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    
    # Dropout para evitar sobreajuste
    x = Dropout(dropout_rate)(x)
    
    # Capa de Salida Binaria
    # Usamos activación sigmoid y 1 neurona para clasificación binaria
    output_tensor = Dense(num_classes, activation='sigmoid', name='output_layer')(x)
    
    # 5. Construir y compilar el modelo
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # Nota: Si estás comparando con el modelo de Focal Loss, debes usar la misma pérdida aquí.
    # Necesitarás la implementación de BinaryFocalLoss.
    # Asumo que la tienes disponible para la compilación.
    if loss_function == 'focal_loss':
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
    else:
      model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
          # Usar la pérdida que estás comparando (asumiendo que es Focal Loss)
          loss=tf.keras.losses.BinaryCrossentropy(), # Usa tu BinaryFocalLoss si la tienes
          metrics=[
              'accuracy', 
              tf.keras.metrics.Precision(name='precision'), 
              tf.keras.metrics.Recall(name='recall')
          ]
      )

    # 6. Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        # Usar la pérdida que estás comparando (asumiendo que es Focal Loss)
        loss=tf.keras.losses.BinaryCrossentropy(), # Usa tu BinaryFocalLoss si la tienes
        metrics=[
            'accuracy', 
            tf.keras.metrics.Precision(name='precision'), 
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model

if __name__ == '__main__':
    # Prueba rápida de la arquitectura con 5 canales
    modelo_ms = crear_modelo_multiespectral(
        input_shape=(224, 224, 5), 
        fine_tune_layers=False 
    )
    modelo_ms.summary()