# src/cnn/binary_crossentropy/data_loader.py

import tensorflow as tf
import numpy as np
import os

# --- INICIO: Añadir capa de aumento de datos ---
# Definimos la aumentación de datos como una capa secuencial
# Coincidiendo con la tesis: "rotación, volteo, zoom, ajuste de contraste" 
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")
# --- FIN: Añadir capa de aumento de datos ---


def crear_datasets(
    data_dir: str,
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 123
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], int, int]: # <<< CAMBIO: Se devuelven más valores
    """
    Carga imágenes y crea tf.data.Datasets para entrenamiento y validación.
    Aplica aumento de datos al set de entrenamiento.
    Calcula el número de muestras por clase para el class_weighting.

    Returns:
        Una tupla que contiene:
        - train_ds: tf.data.Dataset para entrenamiento (con aumentación).
        - val_ds: tf.data.Dataset para validación.
        - class_names: Lista de nombres de las clases.
        - n_plagas_train: Número de muestras 'Plaga' en el set de entrenamiento.
        - n_sanas_train: Número de muestras 'Sana' en el set de entrenamiento.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"El directorio de datos '{data_dir}' no existe.")

    # Crear el dataset de entrenamiento
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset='training' # Especificar que este es el set de entrenamiento
    )

    # Crear el dataset de validación
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=False, # No barajar la validación
        seed=seed,
        validation_split=val_split,
        subset='validation' # Especificar que este es el set de validación
    )

    class_names = train_ds.class_names
    print(f"Clases detectadas: {class_names}") # Asumimos ['Plaga', 'Sana']

    # --- INICIO: Cálculo de muestras para class_weight ---
    train_ds = train_ds.cache()
    # Necesitamos iterar sobre el train_ds (sin batch) para contar las clases
    # Esto es necesario para calcular los pesos en train.py
    n_plagas_train = 0
    n_sanas_train = 0
    
    # Iteramos sobre los batches de entrenamiento
    for _, labels_batch in train_ds:
        n_plagas_train += tf.reduce_sum(tf.cast(labels_batch == 0, tf.int32)).numpy()
        n_sanas_train += tf.reduce_sum(tf.cast(labels_batch == 1, tf.int32)).numpy()

    print(f"Muestras de Plaga en entrenamiento: {n_plagas_train}")
    print(f"Muestras de Sana en entrenamiento: {n_sanas_train}")
    # --- FIN: Cálculo de muestras ---


    # --- INICIO: Aplicar Aumentación y Prefetch ---
    # Aplicar la aumentación SÓLO al dataset de entrenamiento
    # Usamos .map() para aplicar la capa de aumentación
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Configurar prefetch para ambos datasets para optimizar el rendimiento
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    # --- FIN: Aplicar Aumentación y Prefetch ---
    
    # No usamos .repeat() aquí, Keras lo manejará automáticamente

    # Devolvemos los recuentos para que train.py calcule los pesos
    return train_ds, val_ds, class_names, n_plagas_train, n_sanas_train


if __name__ == "__main__":
    # Prueba rápida
    import sys
    ruta = sys.argv[1] if len(sys.argv) > 1 else "data"
    
    train_ds, val_ds, clases, n_plaga, n_sana = crear_datasets(ruta, batch_size=32)
    print("Clases detectadas:", clases)
    print(f"Plaga (train): {n_plaga}, Sana (train): {n_sana}")
    
    for batch_x, batch_y in train_ds.take(1):
        print(f"Forma de las imágenes en un batch de entrenamiento: {batch_x.shape}")
        print(f"Forma de las etiquetas en un batch de entrenamiento: {batch_y.shape}")
        break