# src/cnn/evaluate_bce.py
# (Plantilla para evaluatefl.py y evaluate_multiespectral.py con ajustes menores)

import argparse
import os
from math import ceil
import tensorflow as tf

# Importaciones de Módulos Centralizados
from src.data_management.convolutional_neural_network.rgb.loader_binary_crossentropy_rgb import crear_datasets_cnn_rgb # <-- CAMBIAR para FL/MS
from src.evaluation.utils_metrics import save_report_and_plot_cm, CLASSES
from src.evaluation.utils_inference import load_model_for_inference, predict_cnn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_evaluation(data_dir: str, model_path: str, threshold: float, type: str, base_dir: str = BASE_DIR) -> None:
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # 1. Carga de Datos (solo validación)
    print("1. Cargando datos de Validación...")
    # Usamos 'class_weight' mode para unificar la carga de datos RGB
    if type == 'bce':
      _, val_ds, _, _ = crear_datasets_cnn_rgb(
          data_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, mode='class_weight', 
          val_split=0.2 # Se usa val_split para asegurar que se cargue el subset de validación
      )
    
    # Calculamos el número de batches a predecir
    val_cardinality = tf.data.experimental.cardinality(val_ds).numpy()
    validation_steps = val_cardinality if val_cardinality > 0 else 1
    
    # 2. Carga del Modelo
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
    parser = argparse.ArgumentParser(description="Evalúa el modelo CNN con BCE/Focal Loss/MS")
    parser.add_argument("data_dir", help="Ruta al directorio de datos (raíz)")
    parser.add_argument("-m", "--model", help="Ruta al archivo del modelo Keras", required=True)
    parser.add_argument("-tp", "--type", help="tipo de algoritmo", required=True)
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Umbral de decisión (0.0 a 1.0)")
    parser.add_argument("-b", "--base_dir", default="results/bce", help="Directorio base para guardar resultados")
    args = parser.parse_args()
    
    run_evaluation(args.data_dir, args.model, args.threshold, args.type, args.base_dir)