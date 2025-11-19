# src/cnn/evaluate_multiespectral.py

import argparse
import os
import numpy as np

# Importaciones de Módulos Centralizados
# IMPORTANTE: Cambiar la importación del data loader a la versión de 5 canales
from src.data_management.convolutional_neural_network.multiespectral.loader_multiespectral import load_multiespectral_data 
from src.evaluation.utils_metrics import save_report_and_plot_cm, CLASSES
from src.evaluation.utils_inference import load_model_for_inference

def run_evaluation(patch_size: int, model_path: str, threshold: float, base_dir: str) -> None:
    
    # 1. Carga de Datos (Extracción de features y split)
    print("1. Cargando y Extrayendo parches Multiespectrales (5 canales)...")
    # Este loader devuelve arrays de numpy, no tf.data.Dataset
    _, X_val, _, Y_val = load_multiespectral_data(patch_size=patch_size, test_split=0.2)
    
    # 2. Carga del Modelo
    model = load_model_for_inference(model_path)
    model_name = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
    
    # 3. Predicciones (Predicción directa en arrays de NumPy)
    print("\n3. Iniciando predicción del modelo Multiespectral...")
    # El modelo Keras espera un array de forma (N, H, W, 5)
    # Las etiquetas verdaderas (Y_val) deben ser 0 o 1 (se asume que load_multiespectral_data lo devuelve)
    y_pred_proba = model.predict(X_val, verbose=1).flatten()
    y_true = Y_val.flatten()
    
    # 4. Asignación de Clase usando el umbral
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 5. Guardar Reporte y Plotear Matriz de Confusión
    RESULTS_DIR = os.path.join(base_dir, 'evaluation_results')
    save_report_and_plot_cm(
        y_true, 
        y_pred, 
        CLASSES, 
        RESULTS_DIR, 
        model_name, 
        threshold
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa el modelo CNN Multiespectral")
    # Usar un tamaño de parche como argumento, coherente con el entrenamiento
    parser.add_argument("patch_size", type=int, default=224, help="Tamaño del parche para el modelo (ej. 224)") 
    parser.add_argument("-m", "--model", help="Ruta al archivo del modelo Keras", required=True)
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Umbral de decisión (0.0 a 1.0)")
    # CAMBIAR DIRECTORIO BASE
    parser.add_argument("-b", "--base_dir", default="results/multiespectral", help="Directorio base para guardar resultados (multiespectral)")
    args = parser.parse_args()
    
    run_evaluation(args.patch_size, args.model, args.threshold, args.base_dir)