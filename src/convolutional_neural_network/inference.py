# src/cnn/inference_bce.py (Usado también como plantilla para inference_fl.py)

import argparse
import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# agregar a path la carpeta src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from src.utils.evaluation.utils_inference import load_model_for_inference, run_inference_on_path, save_inference_results
from src.utils.print_utils import print_time_and_step

import time
from datetime import datetime
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

def main():
    parser = argparse.ArgumentParser(description="Realiza inferencia con modelo CNN (RGB/MS).")
    parser.add_argument("path", help="Ruta a un archivo de imagen o carpeta con imágenes RGB/MS.")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo Keras (.keras).")
    parser.add_argument("-t", "--threshold", type=float, default=0.65, help="Umbral de decisión.")
    args = parser.parse_args()

    IMG_SIZE = (224, 224)
    img_type = 'RGB' if args.model.find('RGB') != -1 else 'MULTIESPECTRAL'
    model_type = 'focal_loss' if args.model.lower().find('focal') != -1 else ('random_forest' if args.model.lower().find('random_forest') != -1 else 'binary_crossentropy')
    is_multiespectral = True if args.model.lower().find('multiespectral') != -1 else False
    is_random_forest = True if args.model.lower().find('random_forest') != -1 else False

    print_time_and_step('init', f'Iniciando inferencia del modelo {img_type} {"Focal Loss" if model_type == "focal_loss" else ("random_forest" if model_type == "random_forest" else "Binary Crossentropy")}', timestamp=timestamp, start_time=start_time)

    # 1. Carga del Modelo
    print_time_and_step('1', f"⏳ Cargando modelo desde: {args.model}", timestamp=timestamp, start_time=start_time)
    model = load_model_for_inference(args.model)
    model_name = os.path.basename(args.model).replace('.keras', '').replace('.h5', '')
    
    # 2. Ejecutar Inferencia
    print_time_and_step('2', "⏳ Realizando inferencia...", timestamp=timestamp, start_time=start_time)
    results = run_inference_on_path(
        model=model,
        feature_extractor_rf=None,
        path=args.path,
        threshold=args.threshold,
        img_size=IMG_SIZE,
        model_name=model_name,
        is_multiespectral=is_multiespectral,
        is_random_forest=is_random_forest,
    )

    # 3. Guardar Resultados
    print_time_and_step('3', "⏳ Guardando resultados...", timestamp=timestamp, start_time=start_time)
    if results:
        save_inference_results(results, BASE_DIR, args.threshold, img_type, model_type)
        

if __name__ == "__main__":
    main()