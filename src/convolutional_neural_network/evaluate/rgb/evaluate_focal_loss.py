# src/cnn/evaluate_fl.py

import argparse
import os
import tensorflow as tf

# Importaciones de Módulos Centralizados
# IMPORTANTE: Cambiar la importación del data loader al que usa Undersampling/Repeat
from src.data_management.convolutional_neural_network.rgb.loader_focal_loss import crear_datasets_cnn_fl 
from src.evaluation.utils_metrics import save_report_and_plot_cm, CLASSES
from src.evaluation.utils_inference import load_model_for_inference, predict_cnn

def run_evaluation(data_dir: str, model_path: str, threshold: float, base_dir: str) -> None:
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # 1. Carga de Datos (solo validación)
    print("1. Cargando datos de Validación (Undersampled mode)...")
    # El loader_cnn_fl devuelve un val_ds con .repeat()
    _, val_ds, _, _ = crear_datasets_cnn_fl(
        data_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE
    )
    
    # El número de pasos de validación debe ser coherente con el entrenamiento
    # Usamos un número fijo de batches de validación si val_ds tiene .repeat()
    validation_steps = 14  # <-- AJUSTAR basado en tu configuración de entrenamiento (trainfl.py)
    
    # 2. Carga del Modelo
    # El loader_model_for_inference maneja automáticamente la focal_loss
    model = load_model_for_inference(model_path)
    model_name = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
    
    # 3. Predicciones
    y_pred_proba, y_true = predict_cnn(model, val_ds, steps=validation_steps)
    
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
    parser = argparse.ArgumentParser(description="Evalúa el modelo CNN entrenado con Focal Loss")
    parser.add_argument("data_dir", help="Ruta al directorio de datos (raíz)")
    parser.add_argument("-m", "--model", help="Ruta al archivo del modelo Keras", required=True)
    parser.add_argument("-t", "--threshold", type=float, default=0.65, help="Umbral de decisión (se usa 0.65 como óptimo para F1-Score)") # Umbral por defecto optimizado
    # CAMBIAR DIRECTORIO BASE
    parser.add_argument("-b", "--base_dir", default="results/focal_loss", help="Directorio base para guardar resultados (focal_loss)") 
    args = parser.parse_args()
    
    run_evaluation(args.data_dir, args.model, args.threshold, args.base_dir)