# src/cnn/focal_loss/data_loader.py

import tensorflow as tf
import numpy as np
import os
import math # Para calcular el tamaño del split

# --- Capa de aumento de datos ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
], name="data_augmentation")


def crear_datasets(
    data_dir: str,
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 123
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], int]: # Retorna 4 valores
    """
    Carga datos, realiza el split, aplica Undersampling y devuelve 4 valores.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"El directorio de datos '{data_dir}' no existe.")

    # 1. Carga del dataset RAW (sin batch)
    ds_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        interpolation='nearest',
        shuffle=True,
        seed=seed,
        batch_size=None # Importante: sin batch para el split y filter
    )
    class_names = ds_raw.class_names
    print(f"Clases detectadas: {class_names}")

    # 2. Split en entrenamiento (80%) y validación (20%)
    ds_size = tf.data.experimental.cardinality(ds_raw).numpy()
    train_size = int(math.floor((1 - val_split) * ds_size))

    ds_train_raw = ds_raw.take(train_size)
    ds_val = ds_raw.skip(train_size)

    # 3. Undersampling en el set de entrenamiento
    
    # FIX CLAVE: tf.squeeze(y) convierte la etiqueta (1,) en escalar ()
    ds_plaga = ds_train_raw.filter(lambda x, y: tf.squeeze(y) == 0) # Clase 0: Plaga
    ds_sana = ds_train_raw.filter(lambda x, y: tf.squeeze(y) == 1)  # Clase 1: Sana

    # --- FIX CLAVE: Contar samples iterando si cardinality no funciona ---
    # Contamos la clase minoritaria (Sana) para determinar el tamaño del Undersampling
    # n_sanas_in_train = tf.data.experimental.cardinality(ds_sana).numpy()
    
    # En lugar de cardinality, contamos iterando:
    n_sanas_in_train = 0
    # Iteramos sobre el dataset de la clase Sana. Esto fuerza a TensorFlow a contar.
    # El overhead es mínimo ya que solo recorre las etiquetas de la partición de entrenamiento.
    for _ in ds_sana: 
        n_sanas_in_train += 1
    # n_sanas_in_train ahora será un entero positivo correcto.
    # ----------------------------------------------------------------------
    if n_sanas_in_train <= 0:
        raise ValueError(f"El conteo de muestras Sana en entrenamiento ({n_sanas_in_train}) es inválido. Revise la estructura de su directorio de datos o el split.")
    
    # Undersample (tomar n_sanas muestras de Plaga) y repetir
    ds_plaga_undersampled = ds_plaga.shuffle(buffer_size=1000).take(n_sanas_in_train).repeat() 
    ds_sana_repeated = ds_sana.shuffle(buffer_size=1000).repeat() # Repetir clase minoritaria

    # Combinar, barajar, aumentar y batch
    train_ds = tf.data.Dataset.sample_from_datasets([ds_plaga_undersampled, ds_sana_repeated])
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    # 4. Procesar el set de validación
    val_ds = ds_val.batch(batch_size)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE) 
    val_ds = val_ds.repeat() # Mantenido para cálculos de steps_per_epoch

    print(f"Clases detectadas: {class_names}")
    print(f"Muestras de Sana en entrenamiento (base undersampling): {n_sanas_in_train}")
    
    # Devolvemos 4 valores
    return train_ds, val_ds, class_names, n_sanas_in_train
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