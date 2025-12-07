# src/rf/inference.py

import argparse
import numpy as np
import os
import sys
import joblib 
import json
import matplotlib.pyplot as plt # Necesario para plot_inference_results
from datetime import datetime
import rasterio 
from rasterio.mask import mask
import cv2 # Necesario para resize
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier

import time
from datetime import datetime

# agregar a path la carpeta src
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from src.utils.print_utils import print_time_and_step

# --- Configuraciones y Rutas ---
# Asegura que las funciones de utilidad (como plot_inference_results) sean accesibles
# Si 'plot_inference_results' está en un archivo de utilidad, asegúrate de importarlo
# Por simplicidad, incluiré una versión básica de plot_inference_results al final.
# NOTA: En este flujo, NO NECESITAS importar TensorFlow/Keras.

# --- Constantes ---
BAND_SUFFIXES = ['red.tif', 'red edge.tif', 'nir.tif']
CLASSES = ["Plaga", "Sana"] 
RASTER_EXTENSIONS = ('.tif', '.tiff')
# Tamaño de features que coincide con 224 * 224 * 3
TARGET_RF_FEATURES = 150528 

# --- Funciones de Utilidad (Asegúrate que coincidan con tus originales) ---

def plot_inference_results(results: List[Dict[str, Any]], output_dir: str, timestamp: str, is_multiespectral: bool):
  """Crea y guarda un gráfico de confianza de predicción."""
  if not results:
      print("No hay resultados para plotear.")
      return

  file_names = [res['file_name'] for res in results]
  # Asumimos que la probabilidad de "Sana" está en 'prob_sana'
  prob_sana = np.array([res['prob_sana'] for res in results])
  predictions = [res['prediccion'] for res in results]
  
  # Crear la figura
  plt.figure(figsize=(15, 6))
  x_pos = np.arange(len(file_names))
  
  # Colores: Rojo para 'Plaga', Verde para 'Sana'
  colors = ['red' if pred == 'Plaga' else 'green' for pred in predictions]
  
  plt.bar(x_pos, prob_sana, color=colors)
  
  # Usar el umbral del primer resultado (asumiendo que es constante)
  umbral = results[0].get('umbral', 0.5) 
  plt.axhline(umbral, color='gray', linestyle='--', linewidth=1, label=f'Umbral ({umbral:.2f})')
  
  # Nombres de eje X acortados para mejor visualización
  short_names = [name[0:6] + "..." + name[-15:] if len(name) > 25 else name for name in file_names]
  
  plt.ylabel(f'Probabilidad de ser "{CLASSES[1]}"')
  model_name = "MULTIESPECTRAL" if is_multiespectral else "RGB"
  plt.title(f'Confianza de la Predicción RF ({model_name}) (Umbral: {umbral:.2f})')
  plt.xticks(x_pos, short_names, rotation=45, ha='right')
  plt.ylim(0, 1)
  plt.legend()
  plt.tight_layout()
  
  plot_path = os.path.join(output_dir, f"inference_confidence_plot_RANDOM_FOREST_{model_name}_{timestamp}.png")
  plt.savefig(plot_path)
  plt.close()
  
  print(f"📈 Gráfico de confianza guardado en: {plot_path}")


# --- Función Principal de Preprocesamiento y Vectorización ---

def load_and_preprocess_image(img_path: str, img_size: tuple[int, int], is_multiespectral: bool) -> np.ndarray | None:
  """
  Carga, corrige a 3 canales, redimensiona a IMG_SIZE (224x224x3) y APLANA
  el array de imagen para que coincida con el formato (1, 150528) esperado por RF.
  """
  try:
      if is_multiespectral:
          # 1. Carga y Apilamiento para Multiespectral (Red, Red Edge, NIR)
          # img_path es la carpeta de la muestra (ej: 2021-05-25)
          tif_folder = img_path 
          all_bands_clipped = []
          
          for suffix in BAND_SUFFIXES:
              # Buscamos el archivo de la banda dentro de la carpeta
              band_files = [f for f in os.listdir(tif_folder) if f.endswith(suffix)]
              if not band_files:
                  print(f"Advertencia: Falta la banda MS {suffix} en {tif_folder}.")
                  return None
                  
              band_path = os.path.join(tif_folder, band_files[0]) 

              with rasterio.open(band_path) as src:
                  out_image = src.read() 
                  # out_image tiene forma (C, H, W). C debería ser 1 para bandas individuales
                  if out_image.shape[0] > 1:
                      out_image = out_image[0:1, :, :] 
                      
                  out_band = np.transpose(out_image, (1, 2, 0)) # (H, W, C=1)
                  all_bands_clipped.append(out_band)

          # Apilar las 3 bandas (H, W, 1) -> (H, W, 3)
          stacked_image = np.concatenate(all_bands_clipped, axis=-1)
          X = cv2.resize(stacked_image.astype(np.float32), img_size, interpolation=cv2.INTER_LINEAR)
          
      else:
          # 2. Carga Simple para RGB (Asumiendo que es un único archivo RGB.tif)
          # img_path es el archivo .tif
          with rasterio.open(img_path) as src:
              out_image = src.read() # (C, H, W). Puede ser 4 bandas (RGBA)
              
              # CORRECCIÓN CLAVE: Seleccionamos explícitamente solo las primeras 3 bandas (RGB)
              if src.count >= 3:
                  out_image = out_image[0:3, :, :] 
              else:
                  print(f"Error: El archivo RGB tiene menos de 3 bandas: {src.count}")
                  return None
                  
              X = np.transpose(out_image, (1, 2, 0)) # (H, W, C=3)
          
          # Redimensionar la imagen a IMG_SIZE (ej: 224x224x3)
          X = cv2.resize(X.astype(np.float32), img_size, interpolation=cv2.INTER_LINEAR)
      
      # 3. APLANAMIENTO DIRECTO (Vectorización)
      # Replicando la lógica de X_images_array.reshape(num_samples, total_features)
      
      total_features = X.size
      
      if total_features != TARGET_RF_FEATURES:
          print(f"Error: Features esperadas: {TARGET_RF_FEATURES}, Features encontradas: {total_features}. Revise el tamaño de la imagen.")
          return None
          
      # El RF espera un array 2D de forma (N_samples=1, N_features=150528)
      X_vectorized = X.reshape(1, total_features)
      
      return X_vectorized

  except Exception as e:
      print(f"Error al cargar/procesar imagen/bandas {img_path}: {e}")
      return None


# --- Función de Predicción ---

def predict_single_image(
    rf_model: RandomForestClassifier, 
    img_path: str,
    img_size: tuple[int, int] = (224, 224),
    is_multiespectral: bool = False,
) -> tuple[str, np.ndarray]:
  """
  Carga, aplana la imagen y realiza la predicción directa con el modelo Random Forest.
  """
  
  # 1. Cargar y Preprocesar la imagen (Devuelve el vector aplanado 1 x 150528)
  vector_features = load_and_preprocess_image(img_path, img_size, is_multiespectral)
  
  if vector_features is None:
      return f"Error al cargar/procesar imagen", np.array([0.0, 0.0])

  # 2. Predicción con Random Forest (Directamente sobre el vector)
  try:
      probabilities = rf_model.predict_proba(vector_features)[0] 
      prediction = rf_model.predict(vector_features)[0] 
      etiqueta = CLASSES[prediction]
  except ValueError as e:
      print(f"Error en la predicción RF: {e}. Confirma que el modelo espera {vector_features.shape[1]} features.")
      return f"Error de Features", np.array([0.0, 0.0])
  
  return etiqueta, probabilities


# --- Función Main ---
def main():
  parser = argparse.ArgumentParser(description="Realiza inferencia con el modelo Random Forest")
  parser.add_argument("path", help="Ruta a un archivo TIF (RGB) o a una carpeta que contiene las bandas (MS).")
  parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo Random Forest (.joblib)")
  parser.add_argument("-s", "--size", nargs=2, type=int, default=[224, 224], 
                      help="Tamaño de la imagen (alto ancho) usado en el entrenamiento. Debe ser (224, 224) para 150528 features.")
  args = parser.parse_args()
  run_rf_inference(args.path, args.model, args.size)


def run_rf_inference(path, model, size):
  start_time = time.time()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M")
  # --- Inicialización ---
  IMG_SIZE_TUPLE = tuple(size)
  
  img_type = 'RGB' if model.find('RGB') != -1 else 'MULTIESPECTRAL'
  model_type = 'focal_loss' if model.lower().find('focal') != -1 else ('random_forest' if model.lower().find('random_forest') != -1 else 'binary_crossentropy')
  multiespectral = True if model.lower().find('multiespectral') != -1 else False
  
  print_time_and_step('init', f'Iniciando inferencia del modelo {img_type} {"Focal Loss" if model_type == "focal_loss" else ("random_forest" if model_type == "random_forest" else "Binary Crossentropy")}', timestamp=timestamp, start_time=start_time)
  if IMG_SIZE_TUPLE != (224, 224):
      print_time_and_step('WARN', f"Advertencia: El RF fue entrenado con (224, 224) (150528 features). Usando {IMG_SIZE_TUPLE} podría causar errores si no coincide.", timestamp=timestamp, start_time=start_time)
  
  try:
      rf_model = joblib.load(model)
  except Exception as e:
      print_time_and_step('Error', f"Error: No se pudo cargar el modelo RF. {e}", timestamp=timestamp, start_time=start_time)
      sys.exit(1)

  inference_results = []
  
  # --- Identificación de Casos a Procesar ---
  paths_to_process = []
  
  if os.path.isfile(path) and path.lower().endswith(RASTER_EXTENSIONS) and not multiespectral:
      # Caso A: Archivo RGB único
      paths_to_process = [path]
  
  elif os.path.isdir(path):
      if multiespectral:
          # Caso B: Carpeta única MS o Carpeta que contiene múltiples casos MS
          # Si el usuario pasa una carpeta que contiene subcarpetas (los casos), las procesamos.
          # Si el usuario pasa la carpeta de la muestra (con las bandas dentro), la procesamos.
          
          subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
          
          # Chequeo si la carpeta raíz contiene las bandas MS (caso de una única muestra)
          contains_bands = any(os.path.exists(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(tuple(BAND_SUFFIXES)))
          
          if contains_bands:
              paths_to_process = [path] # Carpeta única es el caso
          elif subdirs:
              paths_to_process = subdirs # Subcarpetas son los casos
          else:
              print_time_and_step('WARN', f"Advertencia: La carpeta {path} no parece contener bandas MS ni subcarpetas de casos MS válidos.", timestamp=timestamp, start_time=start_time)

      else:
          # Caso C: Carpeta con múltiples archivos TIF (RGB)
          paths_to_process = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(RASTER_EXTENSIONS)]
  
  else:
      print_time_and_step("Error: La ruta proporcionada no es válida o no coincide con el modo de inferencia.")
      sys.exit(1)
  
  print_time_and_step('1', f"🔎 Se encontraron {len(paths_to_process)} casos para inferencia...", timestamp=timestamp, start_time=start_time)
  
  # --- Ejecución de la Inferencia ---
  for full_image_path in sorted(paths_to_process):
      
      # full_image_path puede ser el archivo RGB o la carpeta MS.
      etiqueta, probs = predict_single_image(rf_model, full_image_path, IMG_SIZE_TUPLE, multiespectral)

      prob_plaga_np = probs[0]
      prob_sana_np = probs[1] 
      
      # Convertir a float nativo de Python para serialización JSON
      prob_plaga = float(prob_plaga_np)
      prob_sana = float(prob_sana_np)
      
      # El file_name es el nombre del archivo o de la carpeta/muestra
      file_id = os.path.basename(full_image_path)
      
      result = {
          "file_name": file_id,
          "path": full_image_path,
          "prob_sana": round(prob_sana, 4),
          "prob_plaga": round(prob_plaga, 4),
          "prediccion": etiqueta,
          "umbral": 0.5,
          "modelo": os.path.basename(model) 
      }
      inference_results.append(result)
      
      print_time_and_step('2', f"\n--- Caso: {file_id} --- Predicción: {etiqueta} (Sana: {prob_sana:.4f})", timestamp=timestamp, start_time=start_time)

  # --- Guardado de Resultados ---
  if not inference_results:
      print_time_and_step('Error', "No se pudieron procesar casos. Finalizando.", timestamp=timestamp, start_time=start_time)
      sys.exit(0)

  model_type_str = "MULTIESPECTRAL" if multiespectral else "RGB"
  output_filename = f"inference_results_RANDOM_FOREST_{model_type_str}_{timestamp}.json"
  
  # Asumimos que BASE_DIR es la carpeta donde reside este script
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  OUTPUT_DIR = os.path.join(BASE_DIR, 'inference-results/RANDOM_FOREST')
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  
  final_json_path = os.path.join(OUTPUT_DIR, output_filename)
  
  with open(final_json_path, 'w') as f:
      json.dump(inference_results, f, indent=4)
      
  print_time_and_step('3', f"\n\n✅ Resultados de la inferencia guardados en: {final_json_path}", timestamp=timestamp, start_time=start_time)

  # Pasamos el flag is_multiespectral al ploteo
  plot_inference_results(inference_results, OUTPUT_DIR, timestamp, multiespectral)

if __name__ == "__main__":
    main()