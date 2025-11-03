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


    if os.path.isfile(args.path):
        etiqueta, probs = predict_single_image(rf_model, feature_extractor, args.path, IMG_SIZE_TUPLE)
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
          
          # Imprimir el resultado
          print(f"\n--- Resultados para: {os.path.basename(full_image_path)} ---")
          print(f"Probabilidad de ser 'Sana' (Clase 1): {probs[1]:.4f}")
          print(f"Predicción final: {etiqueta}")

if __name__ == "__main__":
    main()