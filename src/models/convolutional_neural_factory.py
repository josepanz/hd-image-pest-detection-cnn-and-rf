# src/models/cnn_factory.py

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from keras.applications import MobileNetV2
from keras.metrics import Precision, Recall, BinaryAccuracy
from typing import Tuple
from .function_losses import focal_loss # Importamos la función centralizada

def crear_modelo_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    loss_type: str = 'binary_crossentropy', # 'bce' (Binary Cross-Entropy) o 'focal'
    learning_rate: float = 0.0001,
    alpha: float = 0.50, # Para Focal Loss
    gamma: float = 3.0,  # Para Focal Loss
    l2_reg: float = 0.0001 # Para el modelo multiespectral
) -> Model:
    """
    Fábrica unificada para crear modelos CNN con MobileNetV2 (3 o 5 canales).

    Gestiona:
    1. Base MobileNetV2 (congelada).
    2. Input Shape (3 canales con pesos ImageNet, 5 canales sin pesos).
    3. Pérdida (BCE o Focal Loss).
    """
    
    num_channels = input_shape[-1]
    
    # --- 1) Backbone MobileNetV2 ---
    if num_channels == 3:
        # RGB (3 Canales): Usamos pesos pre-entrenados de ImageNet (Transfer Learning)
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False
        
        inp = Input(shape=input_shape, name="input_image")
        x = base_model(inp, training=False)
        x = GlobalAveragePooling2D(name="gap")(x)

    elif num_channels == 5:
        # Multiespectral (5 Canales): No se usan pesos ImageNet, se inicializa aleatoriamente
        # La lógica es más similar a model_multiespectral.py
        inp = Input(shape=input_shape)
        
        # Creamos el MobileNetV2 con la nueva entrada
        base_model = MobileNetV2(
            input_tensor=inp,
            include_top=False,
            weights=None, # IMPORTANTE: No usar pesos ImageNet para 5 canales
            pooling='avg'
        )
        
        # Si queremos la cabeza de clasificación como en model_multiespectral.py:
        x = base_model.output
        
        # Adaptación de la cabeza para el modelo multiespectral (con L2)
        x = Dense(
            128, 
            activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(x)
        # El pooling ya está en base_model(pooling='avg'), si no, se haría GlobalAveragePooling2D()

    else:
        raise ValueError(f"Canales no soportados: {num_channels}. Use 3 (RGB) o 5 (MS).")


    # --- 2) Cabeza de clasificación común (si no se aplicó en 5ch) ---
    if num_channels == 3:
        # La cabeza del modelo 3ch es más simple, pero es el mismo principio que 5ch sin L2/128
        x = Dropout(dropout_rate, name="dropout")(x)
        out = Dense(1, activation="sigmoid", name="prediction")(x)
    elif num_channels == 5:
        # En el modelo 5ch, ya aplicamos la capa 128 y pooling='avg'
        x = Dropout(dropout_rate)(x)
        out = Dense(1, activation='sigmoid', name='output_layer')(x)
        
    model = Model(inp, out, name=f"hd_mobilenet_{num_channels}ch_{loss_type}")

    # --- 3) Compilación ---
    if loss_type == 'focal_loss':
        loss_fn = focal_loss(alpha=alpha, gamma=gamma)
    elif loss_type == 'binary_crossentropy':
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    else:
        raise ValueError("loss_type debe ser 'binary_crossentropy' o 'focal_loss'.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )

    return model

if __name__ == "__main__":
    m = crear_modelo_cnn()
    m.summary()