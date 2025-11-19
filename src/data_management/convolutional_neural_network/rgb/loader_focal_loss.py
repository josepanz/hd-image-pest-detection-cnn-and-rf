# src/data_management/loader_cnn_fl.py

import tensorflow as tf
import os
import math
from typing import Tuple, List
from src.data_management.base_loader import get_data_augmentation, preprocess_image # Asumiendo que esta es la estructura

# Configuración de Clases
CLASSES = ["Plaga", "Sana"] 

def crear_datasets_cnn_fl(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 123
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], int]:
    """
    Carga datos y aplica una estrategia de Undersampling/Over-sampling
    para crear un dataset de entrenamiento balanceado.

    Returns:
        (train_ds, val_ds, class_names, n_minority_samples_train)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"El directorio de datos '{data_dir}' no existe.")

    data_augmentation = get_data_augmentation() # Se obtiene del módulo base
    
    # 1. Carga del dataset RAW (sin batch, sin split aún)
    ds_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int', # Etiquetas como 0 o 1
        class_names=CLASSES,
        image_size=img_size,
        interpolation='nearest',
        batch_size=None, # IMPORTANTE: Cargar sin batch para el split
        shuffle=True,
        seed=seed
    ).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) # Preprocesar (rescale)

    data_size = tf.data.experimental.cardinality(ds_raw).numpy()
    train_size = math.ceil(data_size * (1 - val_split))
    
    # 2. Split en entrenamiento y validación
    ds_train = ds_raw.take(train_size)
    ds_val = ds_raw.skip(train_size)

    # 3. Separar clases en el dataset de entrenamiento
    
    # La clase minoritaria y mayoritaria deben ser identificadas.
    # Asumiendo 'Plaga' (0) es la minoritaria y 'Sana' (1) es la mayoritaria, 
    # según la implementación típica de desbalance.
    
    # Filtramos la clase minoritaria (Plaga=0)
    ds_minority = ds_train.filter(lambda x, y: y == 0)
    # Filtramos la clase mayoritaria (Sana=1)
    ds_majority = ds_train.filter(lambda x, y: y == 1)
    
    n_minority_samples = tf.data.experimental.cardinality(ds_minority).numpy()
    n_majority_samples = tf.data.experimental.cardinality(ds_majority).numpy()
    
    print(f"\n--- Estrategia Focal Loss (Undersampling) ---")
    print(f"Muestras de clase minoritaria (Plaga=0) en train: {n_minority_samples}")
    print(f"Muestras de clase mayoritaria (Sana=1) en train: {n_majority_samples}")

    # 4. Aplicar Submuestreo (Undersampling) a la clase mayoritaria
    
    # Repetir la clase minoritaria indefinidamente (over-sampling lógico)
    ds_minority_repeated = ds_minority.shuffle(buffer_size=1000).repeat()
    
    # Submuestrear la clase mayoritaria al tamaño de la minoritaria (undersampling)
    ds_majority_undersampled = ds_majority.take(n_minority_samples)
    
    # El nuevo tamaño total del dataset de entrenamiento será ~ 2 * n_minority_samples
    
    # 5. Combinar, Aumentar y Batch
    train_ds = tf.data.Dataset.sample_from_datasets(
        [ds_minority_repeated, ds_majority_undersampled]
    )
    # Aplicar la aumentación SOLO al dataset de entrenamiento
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 6. Procesar el set de validación
    val_ds = ds_val.batch(batch_size)
    # IMPORTANTE: Aplicar .repeat() al set de validación (coherente con trainfl.py)
    # Esto es necesario si no se sabe el tamaño exacto o si se desea usar un número fijo de steps.
    val_ds = val_ds.cache().repeat().prefetch(tf.data.AUTOTUNE) 

    return train_ds, val_ds, CLASSES, n_minority_samples