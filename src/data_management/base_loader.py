# src/data_management/base_loader.py

import tensorflow as tf
from typing import Tuple, List, Dict
import math

# --- CONFIGURACIÓN GLOBAL ---
CLASSES = ["Plaga", "Sana"] 

# --- 1. AUMENTO DE DATOS ---
# Definimos la aumentación de datos como una capa secuencial
# Coincidiendo con la tesis: "rotación, volteo, zoom, ajuste de contraste" 
def get_data_augmentation_layer() -> tf.keras.Model:
    """
    Define la capa secuencial de aumentación de datos (rotación, volteo, zoom, contraste).
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="data_augmentation")

# --- 2. CÁLCULO DE PESOS PARA CLASES ---

def calculate_class_weights(n_plagas: int, n_sanas: int) -> Dict[int, float]:
    """
    Calcula los pesos de clase para manejar el desbalance de datos.
    Se usa la fórmula del total de muestras / (2 * conteo de clase).
    """
    total_samples = n_plagas + n_sanas
    if total_samples == 0:
        return {0: 1.0, 1: 1.0}

    # Calcula el peso: peso_clase = total / (num_clases * conteo_clase)
    weight_plaga = total_samples / (len(CLASSES) * n_plagas) if n_plagas > 0 else 1.0
    weight_sana = total_samples / (len(CLASSES) * n_sanas) if n_sanas > 0 else 1.0

    print(f"Pesos de Clase Calculados: Plaga (0): {weight_plaga:.2f}, Sana (1): {weight_sana:.2f}")

    return {0: weight_plaga, 1: weight_sana}