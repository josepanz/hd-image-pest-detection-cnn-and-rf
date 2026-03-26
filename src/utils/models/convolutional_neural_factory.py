# src/models/cnn_factory.py

from keras import layers
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from keras.applications import MobileNetV2, EfficientNetB0
from keras.metrics import Precision, Recall, BinaryAccuracy, AUC
from typing import Tuple
from .function_losses import focal_loss # Importamos la función centralizada

def crear_modelo_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.4,
    loss_type: str = 'binary_crossentropy', # 'bce' (Binary Cross-Entropy) o 'focal'
    learning_rate: float = 0.0001,
    alpha: float = 0.50, # Para Focal Loss
    gamma: float = 3.0,  # Para Focal Loss
    l2_reg: float = 0.01, # Para el modelo multiespectral
    threshold: float = 0.5,
    isRgb: bool = True
) -> Model:
    """
    Fábrica unificada para crear modelos CNN con MobileNetV2 (3 o 5 canales).

    Gestiona:
    1. Base MobileNetV2 (congelada).
    2. Input Shape (3 canales con pesos ImageNet, 5 canales sin pesos).
    3. Pérdida (BCE o Focal Loss).
    """
    
    num_channels = input_shape[-1]
    print('numero de canales:', num_channels, ' isRgb:', isRgb)
    # --- 0) CAPAS DE DATA AUGMENTATION (Dentro del modelo) ---
    # Esto crea variaciones de la imagen en cada época de entrenamiento automáticamente
    data_augmentation = tf.keras.Sequential([
        # layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1), # Muy útil para cambios de luz en drones
    ], name="data_augmentation")
    # --- 1) Backbone MobileNetV2 ---
    if num_channels == 3:
        # RGB (3 Canales): Usamos pesos pre-entrenados de ImageNet (Transfer Learning)
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet" if isRgb else None
        )
        if isRgb:
          base_model.trainable = False
          training_mode = False
        else:
          # base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
          
          # ESTRATEGIA DE TESIS: Descongelar a partir de la capa 100 
          # para que las capas base mantengan formas básicas pero las profundas aprendan agronomía.
          base_model.trainable = True
          for layer in base_model.layers[:100]:
              layer.trainable = False
          training_mode = True
        
        inp = Input(shape=input_shape, name="input_image")
        x = data_augmentation(inp)
        x = base_model(x, training=training_mode)
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
            224, 
            activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(x)
        # El pooling ya está en base_model(pooling='avg'), si no, se haría GlobalAveragePooling2D()

    else:
        raise ValueError(f"Canales no soportados: {num_channels}. Use 3 (RGB) o 5 (MS).")


    # --- 2) Cabeza de clasificación común (si no se aplicó en 5ch) ---
    if num_channels == 3:
        # La cabeza del modelo 3ch es más simple, pero es el mismo principio que 5ch sin L2/128
        if isRgb:
            x = Dropout(dropout_rate, name="dropout")(x)
            out = Dense(1, activation="sigmoid", name="prediction")(x)
        else:
          x = BatchNormalization()(x)
          x = Dropout(dropout_rate, name="dropout")(x)
          x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
          out = Dense(1, activation="sigmoid", name="prediction", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    elif num_channels == 5:
        # En el modelo 5ch, ya aplicamos la capa 128 y pooling='avg'
        x = Dropout(dropout_rate)(x)
        out = Dense(1, activation='sigmoid', name='output_layer')(x)
        
    model = Model(inp, out, name=f"hd_mobilenet_{num_channels}ch_{loss_type}")
    
    # final_lr = learning_rate if isRgb else learning_rate * 2

    # --- 3) Compilación ---
    if loss_type == 'focal_loss':
        loss_fn = focal_loss(alpha=alpha, gamma=gamma)
    elif loss_type == 'binary_crossentropy':
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    else:
        raise ValueError("loss_type debe ser 'binary_crossentropy' o 'focal_loss'.")

    model.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate=final_lr),
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss_fn,
        metrics=[
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )

    return model

def crear_modelo_cnnv2_bk(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.5, # Subido a 0.5 para combatir el overfitting detectado
    loss_type: str = 'binary_crossentropy',
    learning_rate: float = 0.0001,
    alpha: float = 0.25, # Ajustado para desbalanceo (Sanas vs Plagas)
    gamma: float = 2.0,
    l2_reg: float = 0.01,
    isRgb: bool = True
) -> Model:
    """
    V2: Reemplaza MobileNetV2 por EfficientNetB0 para mejorar AUC > 0.70.
    Implementa GlobalMax + Average Pooling y regularización agresiva.
    """
    
    num_channels = input_shape[-1]
    
    # 1. DATA AUGMENTATION MEJORADA
    # Crucial para compensar el dataset pequeño (665 imágenes)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
    ], name="augmentation_v2")

    # 2. SELECCIÓN DE BACKBONE (EfficientNetB0)
    # EfficientNet es superior en captar patrones multiespectrales sutiles
    inp = Input(shape=input_shape, name="input_v2")
    x = data_augmentation(inp)

    if num_channels == 3:
        # Caso RGB: Transfer Learning con Pesos ImageNet
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet" if isRgb else None
        )
        # Descongelamos solo las últimas 20 capas para no romper el AUC
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    else:
        # Caso Multiespectral (5 canales): Entrenamiento desde cero
        # No usamos pesos ImageNet porque la firma espectral NIR/RE es distinta
        base_model = EfficientNetB0(
            input_tensor=x,
            include_top=False,
            weights=None
        )
        base_model.trainable = True

    # 3. EXTRACCIÓN DE CARACTERÍSTICAS AVANZADA
    if num_channels == 3:
        x = base_model(x)
    else:
        x = base_model.output

    # Combinamos Global Average y Max Pooling para captar texturas de plagas
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    # 4. CABEZA DE CLASIFICACIÓN (DENSE)
    # Agregamos BatchNormalization para estabilizar el aprendizaje
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Capa de salida
    out = layers.Dense(1, activation="sigmoid", name="prediction_v2")(x)

    model = Model(inp, out, name=f"efficientnet_v2_{num_channels}ch")

    # 5. CONFIGURACIÓN DE PÉRDIDA Y OPTIMIZADOR
    if loss_type == 'focal_loss':
        # Asegúrate de tener definida tu función focal_loss
        loss_fn = focal_loss(alpha=alpha, gamma=gamma)
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Recomendación: Usar un LR ligeramente más alto para MS que para RGB
    actual_lr = learning_rate if isRgb else learning_rate * 1.5

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=actual_lr),
        loss=loss_fn,
        metrics=[
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc') # Agregado para monitorear tu objetivo > 0.70
        ]
    )

    return model

def crear_modelo_cnnv2_bk2(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.5,
    loss_type: str = 'focal_loss',
    learning_rate: float = .0001, # LR más bajo para no "saltarse" el mínimo
    alpha: float = 0.25, # Ajustado para desbalanceo (Sanas vs Plagas)
    gamma: float = 2.0,
    isRgb: bool = True
) -> Model:
    
    num_channels = input_shape[-1]
    
    # 1. AUMENTO DE DATOS AGRESIVO (Indispensable para 665 datos)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.3),
        layers.RandomContrast(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ])

    inp = Input(shape=input_shape)
    x = data_augmentation(inp)

    # 2. BACKBONE CON ESTRATEGIA DE CONGELADO TOTAL
    # Si es MS (5ch), usamos una arquitectura más pequeña para evitar overfitting
    if num_channels == 5:
        # Mini-CNN robusta para Multiespectral
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
    else:
        # Para RGB: EfficientNetB0 CONGELADO al 100% (solo entrenamos la cabeza)
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=False, weights="imagenet" if isRgb else None
        )
        if isRgb:
          base_model.trainable = False 
          x = base_model(x, training=False)
        else:
          x = layers.Rescaling(1./255)
          base_model.trainable = True 
          for layer in base_model.layers[-30:]:
            layer.trainable = True
          x = base_model(x, training=True)
            
        x = layers.GlobalAveragePooling2D()(x)

    # 3. CABEZA DE CLASIFICACIÓN (DENSE)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inp, out)

    # 4. COMPILACIÓN CON MÉTRICA DE ENFOQUE
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=focal_loss(alpha=alpha, gamma=gamma) if loss_type == 'focal_loss' else 'binary_crossentropy',
        metrics=[
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc') # Agregado para monitorear tu objetivo > 0.70
        ]
    )
    return model

def crear_modelo_cnnv2(
    input_shape=(224, 224, 3),
    dropout_rate=0.5,
    loss_type='focal_loss',
    learning_rate=1e-4,
    alpha=0.25,
    gamma=2.0,
    isRgb=True
):

    num_channels = input_shape[-1]
    print(f"Numero de canales {num_channels} | isRgb {isRgb}")
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.1),
        layers.RandomZoom(0.1),
    ])

    inp = Input(shape=input_shape)
    x = data_augmentation(inp)

    if isRgb:
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False
        x = base_model(x, training=False)

    else:
        # CNN para multiespectral (ajustada a 5 canales)
        x = layers.Conv2D(32, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inp, out)

    loss = focal_loss(alpha=alpha, gamma=gamma) if loss_type == 'focal_loss' else 'binary_crossentropy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc') # Agregado para monitorear tu objetivo > 0.70
        ]
    )

    return model

if __name__ == "__main__":
    m = crear_modelo_cnnv2()
    m.summary()