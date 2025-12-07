import argparse
import os
from math import ceil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# agregar a path la carpeta src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

# Importaciones de Módulos Centralizados
from src.utils.evaluation.utils_metrics import save_report_and_plot_cm, CLASSES
from src.utils.evaluation.utils_inference import load_model_for_inference, predict_cnn
from src.utils.data_management.extract_data_to_img import crear_datasets_cnn_multiespectral
from src.utils.print_utils import print_time_and_step

import time
from datetime import datetime
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

def run_evaluation(data_dir: str, model_path: str, threshold: float, base_dir: str = BASE_DIR, batch_size: int = 32) -> None:
    IMG_SIZE = (224, 224)
    SEED = 42
    VAL_SPLIT = 0.2

    imgType = 'RGB' if model_path.lower().find('rgb') != -1 else 'MULTIESPECTRAL'
    loss_type = 'focal_loss' if model_path.lower().find('focal') != -1 else 'binary_crossentropy'
    isRgb = True if model_path.lower().find('rgb') != -1 else False

    print_time_and_step('init', f'Iniciando evaluacion del modelo {imgType} con perdida {"Focal Loss" if loss_type == "focal_loss" else "Binary Crossentropy"}', timestamp=timestamp, start_time=start_time)
    # 1. Carga de Datos (solo validación)
    print_time_and_step('1', 'Cargando datos de Validación...', timestamp=timestamp, start_time=start_time)
    _, val_ds, _, _ = crear_datasets_cnn_multiespectral(data_dir=data_dir, isRgb=isRgb, img_size=IMG_SIZE, val_split=VAL_SPLIT, seed=SEED, batch_size=batch_size)

    val_cardinality = len(val_ds[1]) 
    validation_steps = ceil(val_cardinality / 32) if val_cardinality > 0 else 1
    
    # 2. Carga del Modelo
    print_time_and_step("2", "Cargando modelo...", timestamp=timestamp, start_time=start_time)
    model = load_model_for_inference(model_path)
    model_name = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
    
    # 3. Predicciones
    print_time_and_step("3", "Realizando predicciones...", timestamp=timestamp, start_time=start_time)
    (X_test, y_test) = val_ds
    y_pred_proba = predict_cnn(model, X_test, steps=validation_steps)
    
    # 4. Asignación de Clase usando el umbral
    print_time_and_step("4", "Asignando clases usando el umbral...", timestamp=timestamp, start_time=start_time)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 5. Guardar Reporte y Plotear Matriz de Confusión
    print_time_and_step("5", "Guardando reporte y ploteando matriz de confusión...", timestamp=timestamp, start_time=start_time)
    RESULTS_DIR = os.path.join(base_dir, f'evaluation_results/{imgType}')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_report_and_plot_cm(
        y_test, 
        y_pred, 
        CLASSES, 
        RESULTS_DIR, 
        model_name, 
        threshold
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa el modelo CNN con BCE/Focal RGB/MS")
    parser.add_argument("data_dir", help="Ruta al directorio de datos (raíz)")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo Keras")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Umbral de decisión (0.0 a 1.0)")
    parser.add_argument("-b", "--base_dir", default=BASE_DIR, help="Directorio base para guardar resultados")
    args = parser.parse_args()
    
    run_evaluation(args.data_dir, args.model, args.threshold, args.base_dir)