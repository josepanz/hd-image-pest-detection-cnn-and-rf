# src/data_management/loader_rf.py

import numpy as np
import os
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from typing import Tuple, List

CLASSES = ["Plaga", "Sana"] 
CLASS_MAPPING = {name: i for i, name in enumerate(CLASSES)}


def crear_feature_extractor(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """
    Crea la base MobileNetV2 pre-entrenada para extraer características.
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights="imagenet"
    )
    model = tf.keras.Model(
        inputs=base.input,
        outputs=tf.keras.layers.GlobalAveragePooling2D()(base.output)
    )
    return model


def load_and_extract_features(data_dir: str, feature_extractor: tf.keras.Model, img_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga, preprocesa y extrae características de todas las imágenes.
    """
    # ... (cuerpo de load_and_extract_features original) ...
    X_features = []
    y_labels = []
    
    for class_name, class_id in CLASS_MAPPING.items():
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Extrayendo características para la clase: {class_name} ({class_id})")
        
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_name)
                
                try:
                    img = load_img(img_path, target_size=img_size)
                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    
                    features = feature_extractor.predict(img_array, verbose=0)
                    
                    X_features.append(features[0])
                    y_labels.append(class_id)
                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")
                    
    return np.array(X_features), np.array(y_labels)


def crear_datasets_rf(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    test_split: float = 0.2,
    seed: int = 123
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Pipeline completo: Extracción de características y split de datos para Random Forest.

    Returns: (X_train, X_val, y_train, y_val, class_names)
    """
    print("1. Creando extractor de características (MobileNetV2)...")
    feature_extractor = crear_feature_extractor(input_shape=(img_size[0], img_size[1], 3))
    
    print("2. Cargando imágenes y extrayendo características...")
    X, y = load_and_extract_features(data_dir, feature_extractor, img_size)

    if len(X) == 0:
        raise ValueError("No se pudieron extraer características.")

    print(f"Total de muestras extraídas: {len(X)}")

    print("3. Realizando split de Train/Validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )

    return X_train, X_val, y_train, y_val, CLASSES