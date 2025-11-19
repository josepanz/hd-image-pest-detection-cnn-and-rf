# src/data_management/loader_cnn_rgb.py

import tensorflow as tf
import os
import math
from typing import Tuple, List, Dict
from ...base_loader import get_data_augmentation_layer # Importar la función base

data_augmentation = get_data_augmentation_layer()

def crear_datasets_cnn_rgb(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 123,
    mode: str = 'class_weight' # 'class_weight' o 'undersample'
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], Dict[int, int]]:
    """
    Carga imágenes RGB, realiza el split, aplica la estrategia de balanceo
    (Class Weighting o Undersampling) y devuelve los datasets.
    
    Returns: (train_ds, val_ds, class_names, train_counts)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"El directorio de datos '{data_dir}' no existe.")

    # 1. Carga y Split (Igual para ambos modos)
    # Crear el dataset de entrenamiento, carga y split
    if mode == 'class_weight':
        # Modo Binary Crossentropy: tf.data.Dataset por lotes y aumento
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, labels='inferred', label_mode='binary', image_size=img_size, 
            # interpolation al hacer resize completa con pilexes inventados, en este caso:
            interpolation='nearest', #  nearest tomando los mas cercanos como referencia
            batch_size=batch_size, shuffle=True, seed=seed, validation_split=val_split, 
            subset='training'
        )

        # Crear el dataset de validación
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, labels='inferred', label_mode='binary', image_size=img_size, 
            # interpolation al hacer resize completa con pilexes inventados, en este caso:
            interpolation='nearest', #  nearest tomando los mas cercanos como referencia
            batch_size=batch_size, shuffle=False, seed=seed, validation_split=val_split, subset='validation'
        )
        class_names = train_ds.class_names
        print(f"Clases detectadas: {class_names}") # Asumimos ['Plaga', 'Sana']
        
        # Conteo de clases para Class Weighting (Lógica de data_loaderbc.py)
        n_plagas_train, n_sanas_train = 0, 0
        for _, labels_batch in train_ds.cache().unbatch(): # Itera sobre muestras individuales
            n_plagas_train += tf.reduce_sum(tf.cast(labels_batch == 0, tf.int32)).numpy()
            n_sanas_train += tf.reduce_sum(tf.cast(labels_batch == 1, tf.int32)).numpy()
            
        print(f"Muestras de Plaga en entrenamiento: {n_plagas_train}")
        print(f"Muestras de Sana en entrenamiento: {n_sanas_train}")

        train_counts = {0: n_plagas_train, 1: n_sanas_train}

        # Aplicar la aumentación SÓLO al dataset de entrenamiento
        # Usamos .map() para aplicar la capa de aumentación
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y), 
            num_parallel_calls=tf.data.AUTOTUNE
        )

    elif mode == 'undersample':
        # Modo Focal Loss: Carga sin batch para Undersampling (Lógica de data_loaderfl.py)
        ds_raw = tf.keras.utils.image_dataset_from_directory(
            data_dir, labels='inferred', label_mode='binary', image_size=img_size,
            # interpolation al hacer resize completa con pilexes inventados, en este caso:
            interpolation='nearest', #  nearest tomando los mas cercanos como referencia
            shuffle=True, seed=seed, batch_size=None # Sin batch para filter y split
        )
        class_names = ds_raw.class_names
        ds_size = tf.data.experimental.cardinality(ds_raw).numpy()
        train_size = int(math.floor((1 - val_split) * ds_size))

        ds_train_raw = ds_raw.take(train_size).cache()
        ds_val = ds_raw.skip(train_size)

        # 2. Undersampling
        ds_plaga = ds_train_raw.filter(lambda x, y: tf.squeeze(y) == 0)
        ds_sana = ds_train_raw.filter(lambda x, y: tf.squeeze(y) == 1)

        # Contar la clase minoritaria (asumimos 'Sana' es minoritaria en el set original)
        n_sanas_in_train = sum(1 for _ in ds_sana)
        if n_sanas_in_train <= 0:
          raise ValueError(f"El conteo de muestras Sana en entrenamiento ({n_sanas_in_train}) es inválido. Revise la estructura de su directorio de datos o el split.")
        
        train_counts = {0: n_sanas_in_train, 1: n_sanas_in_train}

        # Combinar, barajar, aumentar y batch (ds_plaga repetirá los menores)
        ds_plaga_undersampled = ds_plaga.shuffle(buffer_size=1000).take(n_sanas_in_train).repeat() 
        ds_sana_repeated = ds_sana.shuffle(buffer_size=1000).repeat()

        train_ds = tf.data.Dataset.sample_from_datasets([ds_plaga_undersampled, ds_sana_repeated])
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(batch_size)
        
        val_ds = ds_val.batch(batch_size).cache()
        val_ds = val_ds.repeat() # Repetir para steps_per_epoch en train_fl.py

        print(f"Clases detectadas: {class_names}")
        print(f"Muestras de Sana en entrenamiento (base undersampling): {n_sanas_in_train}")
        
    else:
        raise ValueError("Modo debe ser 'class_weight' o 'undersample'.")

    # Final: Prefetch
    # Configurar prefetch para ambos datasets para optimizar el rendimiento
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    # return train_ds, val_ds, class_names, n_plagas_train, n_sanas_train
    return train_ds, val_ds, class_names, train_counts

if __name__ == "__main__":
    # Prueba rápida
    import sys
    ruta = sys.argv[1] if len(sys.argv) > 1 else "data"
    
    train_ds, val_ds, clases, n = crear_datasets_cnn_rgb(ruta, batch_size=32)
    print("Clases detectadas:", clases)
    print(f"Plaga (train): {n[0]}, Sana (train): {n[1]}")
    
    for batch_x, batch_y in train_ds.take(1):
        print(f"Forma de las imágenes en un batch de entrenamiento: {batch_x.shape}")
        print(f"Forma de las etiquetas en un batch de entrenamiento: {batch_y.shape}")
        break