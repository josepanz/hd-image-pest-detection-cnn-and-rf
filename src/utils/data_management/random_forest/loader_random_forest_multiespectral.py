# src/rf/loader_rf_ms.py

import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
from skimage.transform import resize

# Importaciones específicas para Multiespectral
# Asumiendo que las funciones de bajo nivel de carga TIFF están en un módulo accesible
try:
    # Usaremos las funciones de carga de datos del módulo multiespectral
    from src.multiespectral.data_loader_multiespectral import (
        load_bands_and_mask, 
        load_multiespectral_data, # Reutilizaremos load_multiespectral_data
        BASE_DATA_DIR,
        CLASSES
    )
except ImportError:
    print("Error: Asegúrate de que 'src/data_management/data_loader_multiespectral.py' es accesible.")
    sys.exit(1)


# --- 1. Funciones de Cálculo de Índices Espectrales ---

def calculate_indices(stacked_image: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcula índices clave (NDVI, NDRE, EVI) a partir de una imagen apilada de 5 bandas.
    
    Asume el orden de bandas: [Blue, Green, Red, Red_Edge, NIR]
    """
    # Se debe manejar la posibilidad de NaN o división por cero
    
    # Índices (Asumiendo que las bandas están en el canal 0 a 4)
    # Bandas: B=0, G=1, R=2, RE=3, NIR=4
    RED = stacked_image[..., 2]
    NIR = stacked_image[..., 4]
    RED_EDGE = stacked_image[..., 3]
    
    # NDVI (Normalized Difference Vegetation Index)
    # (NIR - Red) / (NIR + Red)
    numerator_ndvi = NIR - RED
    denominator_ndvi = NIR + RED
    # Evitar división por cero
    ndvi = np.divide(numerator_ndvi, denominator_ndvi, out=np.zeros_like(numerator_ndvi), where=denominator_ndvi!=0)
    
    # NDRE (Normalized Difference Red Edge)
    # (NIR - Red Edge) / (NIR + Red Edge)
    numerator_ndre = NIR - RED_EDGE
    denominator_ndre = NIR + RED_EDGE
    ndre = np.divide(numerator_ndre, denominator_ndre, out=np.zeros_like(numerator_ndre), where=denominator_ndre!=0)
    
    # Promedio sobre el parche (simplificación del feature)
    mean_ndvi = np.mean(ndvi)
    mean_ndre = np.mean(ndre)
    
    # Promedio de los valores de las 5 bandas como Features simples
    mean_band_values = np.mean(stacked_image, axis=(0, 1)) # Promedio en H y W
    
    # El vector final de features serán 5 valores de bandas promedio + 2 índices
    return mean_band_values, mean_ndvi, mean_ndre

# --- 2. Función de Extracción de Características para RF ---

def extract_rf_features_from_multispectral(
    X_multiespectral: np.ndarray, 
    Y_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Toma los parches multiespectrales (N, H, W, 5) y extrae un vector de características
    para cada parche, combinando promedios de banda e índices.
    
    Args:
        X_multiespectral: Array de NumPy de parches (N, H, W, C).
        Y_labels: Array de NumPy de etiquetas (N, 1).

    Returns:
        (X_features, Y_labels_flat) donde X_features es (N, 7).
    """
    X_features = []
    
    print(f"Extrayendo características espectrales para {len(X_multiespectral)} parches...")

    for i, patch in enumerate(X_multiespectral):
        # 1. Calcular Features
        mean_band_values, mean_ndvi, mean_ndre = calculate_indices(patch)
        
        # 2. Construir el Vector de Características (Feature Vector)
        # Vector final: [B_avg, G_avg, R_avg, RE_avg, NIR_avg, NDVI_avg, NDRE_avg] (Total 7 features)
        feature_vector = np.concatenate([
            mean_band_values,
            np.array([mean_ndvi, mean_ndre])
        ])
        
        X_features.append(feature_vector)
    
    print(f"Extracción completada. Forma de las características: ({len(X_features)}, {feature_vector.shape[0]})")
    
    # Las etiquetas deben ser 1D para scikit-learn
    Y_labels_flat = Y_labels.flatten()
    
    return np.array(X_features), Y_labels_flat

# --- 3. Pipeline Principal (Reemplazo de crear_datasets) ---

def crear_datasets_rf_ms(
    patch_size: int = 224,
    test_split: float = 0.2,
    seed: int = 123
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Pipeline completo para Random Forest Multiespectral:
    1. Carga los parches de datos multiespectrales (X_raw, Y_raw).
    2. Extrae características (5 Promedios de Banda + 2 Índices).
    3. Realiza el split estratificado.

    Returns:
        (X_train, X_val, y_train, y_val, class_names) como arrays de NumPy.
    """
    
    # 1. Carga del dataset RAW (usando la función del loader MS)
    # Esta función carga TODOS los parches y las etiquetas de tus TIFF/Shapefiles
    print("1. Cargando datos Multiespectrales RAW (parches) para extracción de features...")
    X_raw, Y_raw = load_multiespectral_data(patch_size=patch_size)
    
    if len(X_raw) == 0:
        raise ValueError("No se pudieron cargar datos multiespectrales.")

    # 2. Extracción de Características
    X_features, y_labels_flat = extract_rf_features_from_multispectral(X_raw, Y_raw)
    
    # 3. Split en entrenamiento y validación (stratify=y para mantener el balance)
    print(f"Realizando split estratificado (Test Split: {test_split})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_features, 
        y_labels_flat, 
        test_size=test_split, 
        random_state=seed, 
        stratify=y_labels_flat # Esencial para mantener el balance de clases
    )
    
    print(f"\n--- Resumen de Datos ---")
    print(f"Tamaño de Entrenamiento (Features): {X_train.shape}")
    print(f"Tamaño de Validación (Features): {X_val.shape}")
    print(f"Número de Features por muestra: {X_train.shape[1]}") # Debe ser 7
    
    return X_train, X_val, y_train, y_val, CLASSES

# Ejemplo de uso (opcional para pruebas)
if __name__ == '__main__':
    # Asume que 'data' existe y contiene la estructura MS necesaria.
    try:
        X_train, X_val, y_train, y_val, clases = crear_datasets_rf_ms(patch_size=224)
    except Exception as e:
        print(f"Error al ejecutar la prueba: {e}")