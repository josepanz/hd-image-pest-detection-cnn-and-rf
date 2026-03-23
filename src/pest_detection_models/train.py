import numpy as np
import tensorflow as tf
import random
import os

SEED = 42
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Si usas operaciones muy específicas de GPU:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seeds(SEED) # Usa el número que quieras, pero mantenlo

import argparse
import os
from math import ceil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# agregar a path la carpeta src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

# Importaciones de Módulos
from src.utils.data_management.base_loader import calculate_class_weights
from src.utils.models.convolutional_neural_factory import crear_modelo_cnn, crear_modelo_cnnv2
from src.utils.data_management.extract_data_to_img import crear_datasets_cnn_multiespectral, crear_datasets_rf_multiespectral
from src.utils.utils_train import create_cnn_callbacks, encontrar_umbral_optimo, save_history_and_plot
from src.utils.print_utils import print_time_and_step

import joblib # Usamos joblib para guardar modelos sklearn
from src.utils.models.model_random_forest import crear_modelo_rf

import time
from datetime import datetime

def run_training(data_dir: str, epochs: int, loss_type: str, isRgb: bool, alpha: float, gamma: float, model_type: str, threshold: float = 0.5, batch_size: int = 32, base_dir: str = BASE_DIR) -> None:
  if model_type == 'cnn':
      run_cnn_training(data_dir, epochs, loss_type, isRgb, alpha, gamma, threshold, batch_size, base_dir)
  elif model_type == 'rf':
      # Llamar a la nueva función especializada para RF
      run_rf_training(data_dir, isRgb, base_dir) 
  else:
      raise ValueError(f"Tipo de modelo '{model_type}' no soportado.")
    
def run_rf_training(data_dir: str, isRgb: bool, base_dir: str) -> None:
  start_time = time.time()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  # RF no necesita IMG_SIZE, BATCH_SIZE ni EPOCHS
  
  print_time_and_step('1', f'Iniciando entrenamiento Random Forest {"RGB" if isRgb else "MULTIESPECTRAL"}', timestamp=timestamp, start_time=start_time)
  # 1. Carga de Datos (Formato Vectorial N x F)
  # Se utiliza la función que devuelve X_train, X_test, y_train, y_test como arrays N x F
  print_time_and_step('1', "Cargando datos vectoriales para Random Forest...", timestamp=timestamp, start_time=start_time)
  X_train, X_test, y_train, y_test, class_names, _ = crear_datasets_rf_multiespectral(data_dir=data_dir, isRgb=isRgb)
  
  # 2. Construcción del Modelo (Ajusta los hiperparámetros de RF aquí)
  print_time_and_step('2', "Creando Modelo Random Forest (n_estimators=100)...", timestamp=timestamp, start_time=start_time)
  
  # Asumo que crear_modelo_rf es una función que envuelve RandomForestClassifier
  model = crear_modelo_rf(n_estimators=100, random_state=42) 
  
  # 3. Entrenamiento
  print_time_and_step("3", "Iniciando entrenamiento (RF)...", timestamp=timestamp, start_time=start_time)
  model.fit(X_train, y_train)

  print_time_and_step('Finish', "Entrenamiento completado.", timestamp=timestamp, start_time=start_time)
  
  # 4. Guarda el modelo con joblib
  MODEL_DIR = os.path.join(base_dir, 'best_models')
  os.makedirs(MODEL_DIR, exist_ok=True)
  
  timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  model_file_name = f"best_model_random_forest_{timestamp}_{'RGB' if isRgb else 'MULTIESPECTRAL'}.joblib"
  final_save_path = os.path.join(MODEL_DIR, model_file_name)
  
  joblib.dump(model, final_save_path)
  print_time_and_step('Saves', f"\nModelo Random Forest guardado en '{final_save_path}'")

def run_cnn_training(data_dir: str, epochs: int, loss_type: str, isRgb: bool, alpha: float, gamma: float, threshold: float, batch_size: int, base_dir: str) -> None:
  start_time = time.time()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  IMG_SIZE = (224, 224)
  VAL_SPLIT = 0.2

  print_time_and_step('init', f'Iniciando entrenamiento {"RGB" if isRgb else "MULTIESPECTRAL"} con perdida {"Focal Loss" if loss_type == "focal_loss" else "Binary Crossentropy"}, umbral: {threshold}', timestamp=timestamp, start_time=start_time)
  # 1. Carga de Datos y Cálculo de Pesos
  print_time_and_step('1', f"1. Cargando datos con {'Undersampling' if loss_type == 'focal_loss' else 'Class Weighting'}...", timestamp=timestamp, start_time=start_time)
  train_ds, val_ds, _, train_counts = crear_datasets_cnn_multiespectral(data_dir=data_dir, isRgb=isRgb, img_size=IMG_SIZE, val_split=VAL_SPLIT, seed=SEED, batch_size=batch_size)
  class_weight = calculate_class_weights(train_counts[0], train_counts[1])
  
  # 2. Construcción del Modelo (3 canales, BCE)
  lr_to_use = 0.0001 # if loss_type == 'focal_loss' else 0.00001
  print_time_and_step('2', f"Creando y Compilando Modelo (MobileNetV2 + {'Focal' if loss_type == 'focal_loss' else 'BCE'})...", timestamp=timestamp, start_time=start_time)
  if loss_type == 'focal_loss':
    model = crear_modelo_cnnv2(input_shape=(*IMG_SIZE, 3 if isRgb else 5), loss_type='focal_loss', learning_rate=lr_to_use, alpha=alpha, gamma=gamma, isRgb=isRgb)
  else:
    model = crear_modelo_cnnv2(input_shape=(*IMG_SIZE, 3 if isRgb else 5), loss_type='binary_crossentropy', learning_rate=lr_to_use, isRgb=isRgb)
  
  # 3. Callbacks y Pasos
  print_time_and_step('3', "Configurando Callbacks y Pasos...", timestamp=timestamp, start_time=start_time)
  callbacks, _ = create_cnn_callbacks(base_dir, isRgb, loss_type, monitor='val_loss')
  train_size = sum(train_counts.values()) 
  steps_per_epoch = ceil(train_size / batch_size)
  
  val_total_samples = len(val_ds[1]) 
  validation_steps = ceil(val_total_samples / batch_size) if val_total_samples > 0 else 1

  # 4. Entrenamiento
  print_time_and_step("4", "Iniciando entrenamiento...", timestamp=timestamp, start_time=start_time)
  # Desempaquetamos train_ds en X_train (train_ds[0]) y y_train (train_ds[1])
  # Nota: val_ds es una tupla (X_test, y_test) y funciona correctamente en validation_data.
  history = model.fit(
      train_ds[0],  # X_train (Features/Imágenes)
      train_ds[1],  # y_train (Labels/Etiquetas)
      epochs=epochs,
      batch_size=batch_size,
      shuffle=True,
      steps_per_epoch=steps_per_epoch,
      validation_data=val_ds,
      validation_steps=validation_steps,
      callbacks=callbacks,
      class_weight= None if loss_type == 'focal_loss' else class_weight,
      verbose=1
  )
  
  # 5. Guardado y Ploteo (Usando utils_train)
  print_time_and_step('5', 'Guardado y Ploteo (Usando utils_train)', timestamp=timestamp, start_time=start_time)
  suffix = f"_Focal_a{alpha}_g{gamma}" if loss_type == 'focal_loss' else "_BCE"
  suffix += f"_t{threshold}"
  save_history_and_plot(history, base_dir, epochs, suffix=suffix, isRgb=isRgb)

  umbral_maestro = encontrar_umbral_optimo(model, val_ds[0], val_ds[1])
  print('Umbral mas optimo: ', umbral_maestro)
    
def main():
  parser = argparse.ArgumentParser(description="Entrena el modelo HD-only para detección de plagas")
  parser.add_argument("data_dir", help="Directorio raíz con subcarpetas de clases")
  parser.add_argument("-e", "--epochs", type=int, default=20, help="Número máximo de épocas")
  parser.add_argument("-a", "--alpha", type=float, default=0.50, help="Alpha")
  parser.add_argument("-g", "--gamma", type=float, default=3.0, help="Gamma")
  parser.add_argument("-lt", "--loss_type", type=str, choices=["focal_loss", "binary_crossentropy"], help="Tipo de funcion de perdida")
  parser.add_argument("-rgb", "--rgb", action='store_true', default=False, help="Es RGB?")
  parser.add_argument("-mt", "--model_type", type=str, required=True, choices=["cnn", "rf"], help="Tipo de modelo a entrenar (cnn o rf)")
  parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Umbral de decisión (0.0 a 1.0)")
  args = parser.parse_args()
  run_training(args.data_dir, args.epochs, args.loss_type, args.rgb, args.alpha, args.gamma, args.model_type, args.threshold)

if __name__ == "__main__":
  main()