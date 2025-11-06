# src/rf/inference.py

import argparse
import numpy as np
import tensorflow as tf
import os
import sys
import joblib 
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input

# Ajustar la ruta de importación si es necesario
# ... (código de ajuste de ruta, omitido para brevedad)

import json
from datetime import datetime
import matplotlib.pyplot as plt

from data_loader import crear_feature_extractor 

CLASSES = ["Plaga", "Sana"] 

def predict_single_image(
    rf_model, 
    feature_extractor,
    img_path: str,
    img_size: tuple[int, int] = (224, 224),
) -> tuple[str, np.ndarray]:
    
    # 1. Cargar y Preprocesar la imagen
    try:
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
    except Exception as e:
        return f"Error al cargar/procesar imagen: {e}", np.array([0.0, 0.0])

    # 2. Extracción de características
    features = feature_extractor.predict(img_array, verbose=0)
    
    # 3. Predicción con Random Forest
    probabilities = rf_model.predict_proba(features)[0] # Probabilidades de cada clase
    prediction = rf_model.predict(features)[0]          # Etiqueta predicha (0 o 1)
    etiqueta = CLASSES[prediction]
    
    return etiqueta, probabilities

def plot_inference_results(results: list, output_dir: str, timestamp: str = datetime.now().strftime("%Y%m%d_%H%M")):
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
    plot_path = os.path.join(output_dir, f"inference_confidence_plot_{timestamp}.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
    print(f"📈 Gráfico de confianza guardado en: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Realiza inferencia con el modelo Random Forest")
    parser.add_argument("path", help="Ruta a una imagen o carpeta de imágenes")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo Random Forest (.joblib)")
    parser.add_argument("-s", "--size", nargs=2, type=int, default=[224, 224], 
                        help="Tamaño de la imagen (alto ancho)")
    args = parser.parse_args()

    # --- Cargar Extractor de Features y Modelo RF ---
    IMG_SIZE_TUPLE = tuple(args.size)
    feature_extractor = crear_feature_extractor(input_shape=(IMG_SIZE_TUPLE[0], IMG_SIZE_TUPLE[1], 3))
    
    try:
        rf_model = joblib.load(args.model)
    except Exception as e:
        print(f"Error: No se pudo cargar el modelo RF. {e}")
        sys.exit(1)

    # 1. Inicializar la lista de resultados
    inference_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if os.path.isfile(args.path):
        etiqueta, probs = predict_single_image(rf_model, feature_extractor, args.path, IMG_SIZE_TUPLE)

        prob_plaga_np = probs[0] # probs[0] es la probabilidad de Plaga (Clase 0)
        prob_sana_np = probs[1]  # probs[1] es la probabilidad de Sana (Clase 1)
        
        # Convertir a float nativo de Python para serialización JSON
        prob_plaga = float(prob_plaga_np)
        prob_sana = float(prob_sana_np)
        # 2. Guardar el resultado en la lista
        result = {
            "file_name": os.path.basename(full_image_path),
            "prob_sana": round(prob_sana, 4),  # Probabilidad de la clase 1 (Sana)
            "prob_plaga": round(prob_plaga, 4),  # Probabilidad de la clase 1 (Sana)
            "prediccion": etiqueta,
            "umbral": 0.5,
            "modelo": os.path.basename(args.model) 
        }
        inference_results.append(result)

        print(f"\n--- Resultados para: {os.path.basename(args.path)} ---")
        print(f"Probabilidad de ser '{CLASSES[1]}': {probs[1]:.4f}")
        print(f"Predicción final: {etiqueta}")
        
    elif os.path.isdir(args.path):
      print(f"\nProcesando todas las imágenes en la carpeta: {args.path}")
      image_files = [f for f in os.listdir(args.path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
      
      if not image_files:
          print(f"No se encontraron imágenes válidas en la carpeta: {args.path}")
          sys.exit(0)
      
      # Iterar sobre las imágenes y predecir
      for image_file in sorted(image_files):
          full_image_path = os.path.join(args.path, image_file)
          etiqueta, probs = predict_single_image(rf_model, feature_extractor, full_image_path, IMG_SIZE_TUPLE)

          prob_plaga_np = probs[0] # probs[0] es la probabilidad de Plaga (Clase 0)
          prob_sana_np = probs[1]  # probs[1] es la probabilidad de Sana (Clase 1)
          
          # Convertir a float nativo de Python para serialización JSON
          prob_plaga = float(prob_plaga_np)
          prob_sana = float(prob_sana_np)
          # 2. Guardar el resultado en la lista
          result = {
              "file_name": os.path.basename(full_image_path),
              "prob_sana": round(prob_sana, 4),  # Probabilidad de la clase 1 (Sana)
              "prob_plaga": round(prob_plaga, 4),  # Probabilidad de la clase 1 (Sana)
              "prediccion": etiqueta,
              "umbral": 0.5,
              "modelo": os.path.basename(args.model) 
          }
          inference_results.append(result)
          
          # Imprimir el resultado
          print(f"\n--- Resultados para: {os.path.basename(full_image_path)} ---")
          print(f"Probabilidad de ser 'Sana' (Clase 1): {probs[1]:.4f}")
          print(f"Predicción final: {etiqueta}")
    else:
        return "error"


    # 3. Guardar el archivo JSON
    output_filename = f"inference_results_{timestamp}.json"
    
    # Asegúrate de definir un directorio para guardar los resultados
    # Usaremos una carpeta 'inference_results' dentro del directorio del script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'inference-results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    final_json_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(final_json_path, 'w') as f:
        json.dump(inference_results, f, indent=4)
        
    print(f"\n\n✅ Resultados de la inferencia guardados en: {final_json_path}")

    plot_inference_results(inference_results, OUTPUT_DIR, timestamp)

if __name__ == "__main__":
    main()