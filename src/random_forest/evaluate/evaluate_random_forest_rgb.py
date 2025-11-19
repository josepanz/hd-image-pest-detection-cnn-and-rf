# src/rf/evaluate_rf.py

import argparse
import os
import numpy as np

# Importaciones de Módulos Centralizados
from src.data_management.random_forest.loader_random_forest_rgb import crear_datasets_rf
from src.evaluation.utils_metrics import save_report_and_plot_cm, CLASSES
from src.evaluation.utils_inference import load_model_for_inference, predict_rf

def run_evaluation_rf(data_dir: str, model_path: str, base_dir: str) -> None:
    
    # 1. Carga de Datos (Extracción de features y split)
    print("1. Extrayendo características y cargando datos de Validación...")
    # Usamos test_split=0.2, pero solo usamos X_val y y_val
    _, X_val, _, y_val, _ = crear_datasets_rf(data_dir, test_split=0.2)
    
    # 2. Carga del Modelo Joblib
    model = load_model_for_inference(model_path)
    model_name = os.path.basename(model_path).replace('.joblib', '')
    
    # 3. Predicciones (RF predice clases directamente)
    y_pred, y_true = predict_rf(model, X_val, y_val)
    
    # 4. Guardar Reporte y Plotear Matriz de Confusión
    # El umbral no aplica a RF, se usa 0.5 por convención en el nombre del archivo.
    RESULTS_DIR = os.path.join(base_dir, 'evaluation_results')
    save_report_and_plot_cm(
        y_true, 
        y_pred, 
        CLASSES, 
        RESULTS_DIR, 
        model_name, 
        threshold=0.5
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa el modelo Random Forest")
    parser.add_argument("data_dir", help="Ruta al directorio de datos (raíz)")
    parser.add_argument("-m", "--model", help="Ruta al archivo del modelo Joblib", required=True)
    parser.add_argument("-b", "--base_dir", default="results/rf", help="Directorio base para guardar resultados")
    args = parser.parse_args()
    
    run_evaluation_rf(args.data_dir, args.model, args.base_dir)