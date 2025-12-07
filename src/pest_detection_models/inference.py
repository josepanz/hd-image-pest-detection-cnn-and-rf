# src/cnn/inference_bce.py (Usado también como plantilla para inference_fl.py)

import argparse
import os
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from inference_random_forest import plot_inference_results, run_rf_inference
# agregar a path la carpeta src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from src.utils.evaluation.utils_inference import load_model_for_inference, run_inference_on_path, save_inference_results
from src.utils.print_utils import print_time_and_step

import time
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Realiza inferencia con modelo CNN (RGB/MS).")
    parser.add_argument("path", help="Ruta a un archivo de imagen o carpeta con imágenes RGB/MS.")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo Keras (.keras).")
    parser.add_argument("-t", "--threshold", type=float, default=0.65, help="Umbral de decisión.")
    parser.add_argument("-mt", "--model_type", type=str, required=True, choices=["cnn", "rf"], help="Tipo de modelo a entrenar (cnn o rf)")
    parser.add_argument("-s", "--size", nargs=2, type=int, default=[224, 224], 
                      help="Tamaño de la imagen (alto ancho) usado en el entrenamiento. Debe ser (224, 224) para 150528 features.")
    args = parser.parse_args()
    run_inference(args.path, args.model, args.threshold, args.model_type, args.size)

def run_inference(path, model, threshold, model_type, size):
  if model_type == 'cnn':
      run_cnn_inference(path, model, threshold)
  elif model_type == 'rf':
      run_rf_inference(path, model, size)
  else:
      raise ValueError(f"Tipo de modelo '{model_type}' no soportado.")
  
def run_cnn_inference(path, model_path, threshold):
  start_time = time.time()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  IMG_SIZE = (224, 224)
  img_type = 'RGB' if model_path.find('RGB') != -1 else 'MULTIESPECTRAL'
  model_type = 'focal_loss' if model_path.lower().find('focal') != -1 else ('random_forest' if model_path.lower().find('random_forest') != -1 else 'binary_crossentropy')
  is_multiespectral = True if model_path.lower().find('multiespectral') != -1 else False
  is_random_forest = True if model_path.lower().find('random_forest') != -1 else False

  print_time_and_step('init', f'Iniciando inferencia del modelo {img_type} {"Focal Loss" if model_type == "focal_loss" else ("random_forest" if model_type == "random_forest" else "Binary Crossentropy")}', timestamp=timestamp, start_time=start_time)

  # 1. Carga del Modelo
  print_time_and_step('1', f"⏳ Cargando modelo desde: {model_path}", timestamp=timestamp, start_time=start_time)
  model = load_model_for_inference(model_path)
  model_name = os.path.basename(model_path).replace('.keras', '').replace('.h5', '')
  
  # 2. Ejecutar Inferencia
  print_time_and_step('2', "⏳ Realizando inferencia...", timestamp=timestamp, start_time=start_time)
  results = run_inference_on_path(
      model=model,
      feature_extractor_rf=None,
      path=path,
      threshold=threshold,
      img_size=IMG_SIZE,
      model_name=model_name,
      is_multiespectral=is_multiespectral,
      is_random_forest=is_random_forest,
  )

  # 3. Guardar Resultados
  print_time_and_step('3', "⏳ Guardando resultados...", timestamp=timestamp, start_time=start_time)
  if results:
      save_inference_results(results, BASE_DIR, threshold, img_type, model_type)
      # Pasamos el flag is_multiespectral al ploteo
      OUTPUT_DIR = os.path.join(BASE_DIR, f'inference-results/{img_type}')
      os.makedirs(OUTPUT_DIR, exist_ok=True)
      plot_inference_results(results, OUTPUT_DIR, timestamp, is_multiespectral, '')
        

if __name__ == "__main__":
    main()