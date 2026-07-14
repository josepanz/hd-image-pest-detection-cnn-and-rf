"""Script CLI de inferencia vigente (reemplaza al viejo inference.py, ya eliminado).

Soporta CNN (.keras) y Random Forest (.joblib) sobre RGB o multiespectral, sobre una
imagen/carpeta de muestra individual o una carpeta que contenga varias. Para RF,
además del .joblib del RF necesita encontrar en el mismo best_models/ la CNN "hermana"
(mismo tipo RGB/MS, misma -l/--loss) para extraer las features; si no la encuentra,
avisa y sigue igual pasando los píxeles crudos aplanados (ver run_unified_inference).
"""

import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf
import joblib
import json
from datetime import datetime
from typing import List, Dict, Any

# Configuración de Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../..'))
sys.path.append(os.path.join(BASE_DIR, '../evaluation'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))

# Imports de la Tesis
from src.utils.evaluation.inference_utils import load_model_for_inference, run_inference_on_path, save_inference_results
from src.utils.print_utils import print_time_and_step
from src.utils.print_utils import plot_inference_results 

# Constantes Globales
IMG_SIZE = (224, 224)
CLASSES = ['Plaga', 'Sana']

def main():
    parser = argparse.ArgumentParser(description="Inferencia unificada para modelos CNN y Random Forest (RGB/MS).")
    parser.add_argument("path", help="Ruta a imagen (RGB) o carpeta de muestra (MS).")
    parser.add_argument("-m", "--model", required=True, help="Ruta al modelo (.keras o .joblib).")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Umbral de decisión (t).")
    parser.add_argument("-mt", "--model_type", type=str, required=True, choices=["cnn", "rf"], help="Tipo de arquitectura.")
    parser.add_argument("-l", "--loss", type=str, required=False, choices=["fl", "bce"], help="Tipo de perdida.")
    args = parser.parse_args()

    run_unified_inference(args.path, args.model, args.threshold, args.model_type, args.loss)

def run_unified_inference(path, model_path, threshold, arch_type, loss: str = 'fl'):
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 1. Identificación automática de configuración
    model_path_lower = model_path.lower()
    is_ms = "multiespectral" in model_path_lower
    img_mode = "MULTIESPECTRAL" if is_ms else "RGB"
    
    # Identificar función de pérdida/algoritmo para el log
    if arch_type == "rf":
        loss_info = "Focal Loss" if "fl" in loss else "BCE"
    else:
        loss_info = "Focal Loss" if "focal" in model_path_lower else "BCE"

    print_time_and_step('init', f'🚀 Modo: {img_mode} | Arq: {arch_type.upper()} | Config: {loss_info}', timestamp=timestamp, start_time=start_time)

    # 2. Carga del Modelo
    print_time_and_step('1', f"⏳ Cargando modelo: {os.path.basename(model_path)}", timestamp=timestamp, start_time=start_time)
    
    model = None
    feature_extractor = None # Por si en el futuro usas extracción de features para RF

    if arch_type == "cnn":
        model = load_model_for_inference(model_path)
    else:
        # Cargar RF (.joblib)
        model = joblib.load(model_path)
        
        # --- LÓGICA DEL EXTRACTOR PARA RF ---
        # Buscamos el modelo CNN correspondiente para extraer las 64 features
        # Asumimos que están en la misma carpeta 'best_models'
        best_models_dir = os.path.dirname(model_path)
        
        # El RF necesita que la CNN use la misma base (MS o RGB)
        # Intentamos cargar la versión Focal Loss por defecto que es la más robusta
        cnn_suffix = "focal_loss.keras" if "fl" in loss else "binary_crossentropy.keras"
        cnn_for_rf_path = os.path.join(best_models_dir, f"best_model_final_{img_mode}_{cnn_suffix}")

        if os.path.exists(cnn_for_rf_path):
            print_time_and_step('1.1', f"⏳ Cargando extractor de features desde: {os.path.basename(cnn_for_rf_path)}", timestamp=timestamp, start_time=start_time)
            full_cnn = tf.keras.models.load_model(cnn_for_rf_path, compile=False)
            
            # Creamos el extractor: entrada de la CNN hasta la capa Dense de 64 (índice -2)
            feature_extractor = tf.keras.Model(
                inputs=full_cnn.input,
                outputs=full_cnn.layers[-2].output
            )
        else:
            print_time_and_step('WARN', f"⚠️ No se encontró la CNN base en {cnn_for_rf_path}. El RF podría fallar si espera 64 features.", timestamp=timestamp, start_time=start_time)

    # 3. Ejecución de Inferencia
    # Usamos la utilidad centralizada que ya maneja la lógica de carpetas MS y archivos RGB
    print_time_and_step('2', "🔎 Procesando imágenes y realizando predicción...", timestamp=timestamp, start_time=start_time)
    
    try:
        results = run_inference_on_path(
            model=model,
            feature_extractor_rf=feature_extractor,
            path=path,
            threshold=threshold,
            img_size=IMG_SIZE,
            model_name=os.path.basename(model_path),
            is_multiespectral=is_ms,
            is_random_forest=(arch_type == "rf")
        )
    except Exception as e:
        print_time_and_step('ERROR', f"Fallo crítico en inferencia: {e}", timestamp=timestamp, start_time=start_time)
        return

    # 4. Post-procesamiento y Guardado
    if not results:
        print_time_and_step('WARN', "No se encontraron resultados válidos en la ruta proporcionada.", timestamp=timestamp, start_time=start_time)
        return

    print_time_and_step('3', f"✅ Inferencia completada. Procesados: {len(results)} items.", timestamp=timestamp, start_time=start_time)

    # Crear directorios de salida
    output_base = os.path.join(BASE_DIR, f'inference-results/{arch_type}/{loss_info}/{img_mode}/{threshold}')
    os.makedirs(output_base, exist_ok=True)

    # Guardar JSON y Gráfico
    save_inference_results(results, output_base, threshold, img_mode, loss_info.replace(" ", "_").lower())
    plot_inference_results(results, output_base, timestamp, is_ms, arch_type.upper())
    
    print_time_and_step('END', f"✨ Proceso finalizado. Resultados en: {output_base}", timestamp=timestamp, start_time=start_time)

if __name__ == "__main__":
    main()