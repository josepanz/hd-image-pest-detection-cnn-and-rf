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
from src.utils.data_management.extract_data_to_img import crear_datasets_cnn_multiespectral, crear_datasets_rf_multiespectral
from src.utils.print_utils import print_time_and_step

from sklearn.metrics import classification_report, confusion_matrix 
import joblib # Usamos joblib para guardar modelos sklearn
from src.utils.evaluation.utils_metrics import plot_confusion
import json

import time
from datetime import datetime

def run_evaluation(data_dir: str, model_path: str, threshold: float, model_type: str, base_dir: str = BASE_DIR, batch_size: int = 32) -> None:
  if model_type == 'cnn':
      run_evaluation_cnn(data_dir, model_path, threshold, base_dir, batch_size)
  elif model_type == 'rf':
      # Llamar a la nueva función especializada para RF
      run_evaluation_rf(data_dir, model_path) 
  else:
      raise ValueError(f"Tipo de modelo '{model_type}' no soportado.")
  
def run_evaluation_rf(data_dir: str, model_path: str) -> None:
  start_time = time.time()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  imgType = 'RGB' if model_path.lower().find('rgb') != -1 else 'MULTIESPECTRAL'
  isRgb = True if model_path.lower().find('rgb') != -1 else False

  print_time_and_step('init', f'Iniciando evaluacion del modelo RANDOM FOREST {imgType}', timestamp=timestamp, start_time=start_time)
  # Definir el subdirectorio de resultados
  RESULTS_DIR = os.path.join(BASE_DIR, 'results')

  # Crear el directorio 'results' si no existe
  # El argumento exist_ok=True evita un error si la carpeta ya existe.
  os.makedirs(RESULTS_DIR, exist_ok=True)
  
  # 1. Cargar Datos de Validación (Features)
  _, X_val, _, y_val, class_names, _ = crear_datasets_rf_multiespectral(data_dir=data_dir, isRgb=isRgb)
  
  # 2. Cargar Modelo Random Forest
  try:
      model = joblib.load(model_path)
  except Exception as e:
      print(f"Error cargando el modelo RF: {e}")
      return

  # 3. Predicción en el conjunto de prueba
  y_pred = model.predict(X_val) 

  # 4. Reporte de clasificación y Matriz de Confusión
  cm = confusion_matrix(y_val, y_pred)
  # Generar el nombre para el gráfico basado en el nombre del reporte JSON
  # Se reemplaza la extensión .json por .png
  final_plot_path = os.path.join(BASE_DIR, f'evaluation_results/RANDOM_FOREST/')
  os.makedirs(final_plot_path, exist_ok=True)
  plot_confusion(cm, class_names, final_plot_path, name=f'report_best_model_RANDOM_FOREST_{"RGB" if isRgb else "MULTIESPECTRAL"}_{timestamp}.png')

  # Generar el reporte de clasificación y matriz de confusión (asumiendo que tienes utils para esto)
  report_dict = classification_report(
      y_val, y_pred, target_names=class_names, output_dict=True, zero_division=0 
  )
  
  # 5. Guardar Reporte
  final_save_path = os.path.join(final_plot_path, f'report_best_model_RANDOM_FOREST_{"RGB" if isRgb else "MULTIESPECTRAL"}_{timestamp}.json')
  with open(final_save_path, "w") as f:
      json.dump(report_dict, f, indent=2)
  print(f"\nReporte guardado en '{final_save_path}'")

  # 4. Evaluación y Reporte
  print_time_and_step('4', 'Evaluación y Reporte de Random Forest...', timestamp=timestamp, start_time=start_time)

  # 5. Guardado (Ajusta esta función según cómo guardes los reportes RF)

def run_evaluation_cnn(data_dir: str, model_path: str, threshold: float, base_dir: str, batch_size: int) -> None:
  start_time = time.time()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M")
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
  parser.add_argument("-mt", "--model_type", type=str, required=True, choices=["cnn", "rf"], help="Tipo de modelo a entrenar (cnn o rf)")
  args = parser.parse_args()
  
  run_evaluation(args.data_dir, args.model, args.threshold, args.model_type, args.base_dir)