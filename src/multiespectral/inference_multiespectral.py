import argparse
import os
import sys
from typing import Tuple
import numpy as np
import tensorflow as tf
from keras.models import load_model
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Importa el data loader adaptado y el ploteo (asumiendo que está en el PATH)
try:
    from data_loader_multiespectral import load_single_multispectral_image
    from model_multiespectral import focal_loss
    # Asume que plot_inference_results está en un módulo de utilidades o aquí mismo.
    # Usaremos una versión simplificada aquí.
except ImportError as e:
    print(f"Error de importación: {e}. Asegúrate de que el data_loader multiespectral es accesible.")
    sys.exit(1)

# Asumimos que la función plot_inference_results está accesible
# ... (debes copiar la función plot_inference_results que ajustamos anteriormente)

CLASSES = ["Plaga", "Sana"] 
THRESHOLD_DEFAULT = 0.5 
DATA_PARENT_DIR = os.path.join('data', 'multispectral_images') # Carpeta a iterar

def plot_inference_results(results: list, output_dir: str, timestamp: str = datetime.now().strftime("%Y%m%d_%H%M"), threshold: float = 0.65, batch: bool = False):
    """
    Crea un gráfico de confianza de predicción y lo guarda.
    """
    file_names = [res['file_name'] for res in results]
    prob_sana = np.array([res['prob_sana'] for res in results])
    predictions = [res['prediccion'] for res in results]
    
    # Asignar color basado en la predicción final
    colors = ['red' if pred == 'Plaga' else 'green' for pred in predictions]
    
    #plt.figure(figsize=(20, 8)) # Usar el nuevo tamaño ancho
    plt.figure(figsize=(15, 6))
    
    # Crear un índice numérico para las imágenes
    x_pos = np.arange(len(file_names))
    
    # Dibujar barras o puntos (Barras son mejores si hay pocas imágenes)
    plt.bar(x_pos, prob_sana, color=colors)
    
    # Dibujar la línea de umbral (referencia de decisión)
    umbral = results[0]['umbral'] 
    plt.axhline(umbral, color='gray', linestyle='--', linewidth=1, label=f'Umbral ({umbral:.2f})')

    # Usar los nombres acortados en el eje X:
    short_names = [name[0:6] + "..." + name[-15:] for name in file_names]
    
    plt.ylabel('Probabilidad de ser "Sana"')
    plt.title(f'Confianza de la Predicción por Imagen (Umbral: {umbral:.2f})')
    #plt.xticks(x_pos, file_names, rotation=90) # Rotar nombres para que quepan
    plt.xticks(x_pos, short_names, rotation=45, ha='right')
    plt.ylim(0, 1) # Escala de probabilidad de 0 a 1
    
    plt.legend()
    plt.tight_layout() # Ajusta automáticamente los parámetros de la subtrama
    
    # Guardar el gráfico
    prefix = "batch_" if batch else ""
    plot_path = os.path.join(output_dir, f"{prefix}inference_confidence_plot_{timestamp}_t{threshold}.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    print(f"📈 Gráfico de confianza guardado en: {plot_path}")

def predict_multispectral_image(
    model: tf.keras.Model, 
    image_dir: str, 
    img_size: Tuple[int, int],
    threshold: float
) -> Tuple[str, np.ndarray]:
    """
    Carga, procesa y predice una imagen multiespectral (5 bandas) de una carpeta.
    """
    try:
        # Cargar y preprocesar la imagen (H, W, 5)
        processed_image = load_single_multispectral_image(image_dir, img_size)
        
        # Expandir dimensión para el modelo (1, H, W, 5)
        input_data = np.expand_dims(processed_image, axis=0)
        
    except Exception as e:
        return f"Error al cargar/procesar imagen: {e}", np.array([0.0, 0.0])

    # 2. Predicción
    # Probabilidad de la clase 1 (Sana)
    prob_sana = model.predict(input_data, verbose=0)[0][0] 
    
    # Generar array de probabilidades (Plaga, Sana)
    probabilities = np.array([1.0 - prob_sana, prob_sana])
    
    # 3. Clasificación con Umbral
    prediction = 1 if prob_sana >= threshold else 0
    etiqueta = CLASSES[prediction]
    
    return etiqueta, probabilities


def main():
    parser = argparse.ArgumentParser(description="Realiza inferencia con el modelo CNN Multiespectral")
    parser.add_argument("-p", "--path", default=None, help="Ruta a la carpeta que contiene las 5 bandas TIF (Ej: data/multispectral_images/2022_06_15__eko_ecobreed)")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo CNN (.keras)")
    parser.add_argument("-s", "--size", nargs=2, type=int, #default=[224, 224], # demasiado grande para procesar, usa 10.9 GB
                        default=[128, 128], 
                        help="Tamaño de la imagen (alto ancho).")
    parser.add_argument("-f", "--loss_function", choices=['focal_loss', 'binary_crossentropy'], default="binary_crossentropy", help="Función de pérdida")
    parser.add_argument("-t", "--threshold", type=float, default=THRESHOLD_DEFAULT, 
                        help="Umbral de clasificación. Por defecto: 0.5.")
    parser.add_argument("-all", "--all_data", default=False, help="Si trae todo los objetos o no")
    args = parser.parse_args()
    inference(args)

def inference(args):
    if args.all_data:
      # --- Cargar Modelo ---
      custom_objects = {}
      try:
          model = load_model(args.model, custom_objects=custom_objects)
          input_shape = model.input_shape[1:]
          
          if input_shape[-1] != 5:
              print(f"❌ Error: El modelo espera {input_shape[-1]} canales, pero este script es para 5 canales.")
              sys.exit(1)
              
          print(f"✅ Modelo Multiespectral cargado. Espera input: {input_shape}")
      except Exception as e:
          print(f"❌ Error al cargar el modelo: {e}")
          sys.exit(1)

      # --- Inicialización ---
      IMG_SIZE_TUPLE = tuple(args.size)
      inference_results = []
      timestamp = datetime.now().strftime("%Y%m%d_%H%M")
      
      # --- 1. Iterar sobre las carpetas de imágenes ---
      print(f"\nIniciando inferencia por lote en: {DATA_PARENT_DIR}")
      
      # Obtener todas las subcarpetas dentro del directorio principal de imágenes MS
      image_dirs = sorted([
          os.path.join(DATA_PARENT_DIR, d) 
          for d in os.listdir(DATA_PARENT_DIR) 
          if os.path.isdir(os.path.join(DATA_PARENT_DIR, d))
      ])

      if not image_dirs:
          print(f"❌ No se encontraron carpetas de imágenes TIF en {DATA_PARENT_DIR}. Revise la estructura de archivos.")
          sys.exit(0)

      # --- 2. Procesar cada carpeta ---
      for image_dir_path in image_dirs:
          folder_name = os.path.basename(image_dir_path)
          
          print(f"-> Prediciendo para carpeta: {folder_name}...")
          
          etiqueta, probs = predict_multispectral_image(model, image_dir_path, IMG_SIZE_TUPLE, args.threshold)

          if etiqueta != "Error":
              # Manejar la conversión de tipos para JSON
              prob_plaga = float(probs[0])
              prob_sana = float(probs[1])
              
              # 3. Guardar el resultado en la lista
              result = {
                  "file_name": folder_name,
                  "prob_plaga": round(prob_plaga, 4), 
                  "prob_sana": round(prob_sana, 4),
                  "prediccion": etiqueta,
                  "umbral": args.threshold,
                  "modelo": os.path.basename(args.model) 
              }
              inference_results.append(result)
              print(f"   Resultado: {etiqueta} | Confianza Sana: {prob_sana:.4f}")

      # --- 4. Guardar Reporte Final y Ploteo ---
      
      BASE_DIR = os.path.dirname(os.path.abspath(__file__))
      RESULTS_BASE_DIR = os.path.join(BASE_DIR, 'results_multispectral')
      OUTPUT_DIR = os.path.join(RESULTS_BASE_DIR, 'inference-results-ms')
      os.makedirs(OUTPUT_DIR, exist_ok=True)
      
      output_filename = f"batch_inference_report_ms_{timestamp}_t{args.threshold:.2f}.json"
      final_json_path = os.path.join(OUTPUT_DIR, output_filename)
      
      if inference_results:
          with open(final_json_path, 'w') as f:
              json.dump(inference_results, f, indent=4)
              
          print(f"\n\n✅ Reporte de Inferencias Multiespectrales (Total: {len(inference_results)}) guardado en: {final_json_path}")

          # Ploteo de resultados
          plot_inference_results(inference_results, OUTPUT_DIR, timestamp, args.threshold, True)
      else:
          print("No se pudieron generar resultados válidos.")
    else:
      # --- Cargar Modelo ---
      custom_objects = {}
      # Añadir tu clase de Focal Loss si es necesaria para cargar el modelo
      if 'focal_loss' in args.loss_function:
        custom_objects['focal_loss'] = focal_loss() 

      try:
          model = load_model(args.model, custom_objects=custom_objects)
          print(f"✅ Modelo Multiespectral cargado: {os.path.basename(args.model)}")
      except Exception as e:
          print(f"❌ Error al cargar el modelo: {e}")
          sys.exit(1)

      # --- Preparación ---
      IMG_SIZE_TUPLE = tuple(args.size)
      inference_results = []
      timestamp = datetime.now().strftime("%Y%m%d_%H%M")
      
      # El path debe ser la carpeta que contiene las 5 bandas TIF, no la carpeta padre.
      
      if os.path.isdir(args.path):
          # Asumimos que el path es la carpeta que contiene los archivos TIF
          image_dir_path = args.path 
          
          etiqueta, probs = predict_multispectral_image(model, image_dir_path, IMG_SIZE_TUPLE, args.threshold)

          # Manejar la conversión de tipos para JSON
          prob_plaga = float(probs[0])
          prob_sana = float(probs[1])
          
          # 2. Guardar el resultado
          result = {
              "file_name": os.path.basename(image_dir_path), # Usamos el nombre de la carpeta como identificador
              "prob_plaga": round(prob_plaga, 4), 
              "prob_sana": round(prob_sana, 4),
              "prediccion": etiqueta,
              "umbral": args.threshold,
              "modelo": os.path.basename(args.model) 
          }
          inference_results.append(result)
          
          # Imprimir el resultado
          print(f"\n--- Resultados para: {result['file_name']} ---")
          print(f"Prob. '{CLASSES[0]}': {result['prob_plaga']:.4f}")
          print(f"Prob. '{CLASSES[1]}': {result['prob_sana']:.4f}")
          print(f"Predicción final (con umbral {args.threshold:.2f}): {etiqueta}")

      else:
          print(f"❌ Error: La ruta debe ser a la carpeta que contiene las 5 bandas TIFF (Ej. data/multispectral_images/2022_06_15__eko_ecobreed)")
          sys.exit(1)


      # 3. Guardar el archivo JSON
      # Directorio de resultados
      BASE_DIR = os.path.dirname(os.path.abspath(__file__))
      RESULTS_BASE_DIR = os.path.join(BASE_DIR, 'results_multispectral')
      OUTPUT_DIR = os.path.join(RESULTS_BASE_DIR, 'inference-results-ms')
      os.makedirs(OUTPUT_DIR, exist_ok=True)
      
      output_filename = f"inference_results_ms_{timestamp}_t{args.threshold}.json"
      final_json_path = os.path.join(OUTPUT_DIR, output_filename)
      
      with open(final_json_path, 'w') as f:
          json.dump(inference_results, f, indent=4)
          
      print(f"\n\n✅ Resultados de la inferencia guardados en: {final_json_path}")

      plot_inference_results(inference_results, OUTPUT_DIR, timestamp, args.threshold)

      # No se puede plotear una sola inferencia, pero si se hiciera un loop, se podría.
      # Si quieres plotear todas las carpetas, deberías modificar la lógica del 'main' para iterar
      # sobre la carpeta padre (data/multispectral_images).


if __name__ == "__main__":
    main()