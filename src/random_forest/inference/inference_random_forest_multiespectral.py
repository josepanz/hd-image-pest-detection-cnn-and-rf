# src/rf/inferencerf_multiespectral.py

import argparse
import numpy as np
import os
import joblib 
import json
import sys
from datetime import datetime
from typing import List, Dict, Tuple

# --- Importaciones de utilidades Multiespectrales ---
# Necesitamos la función que carga una imagen TIFF y la transforma a 5 bandas normalizadas
from src.multiespectral.data_loader_multiespectral import load_single_multispectral_image 

# Necesitamos la función que calcula los índices (NDVI, NDRE)
# La copiamos de loader_rf_ms.py o la ponemos en un utils.
def calculate_indices(stacked_image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Calcula índices clave (NDVI, NDRE) y promedios de banda a partir de una imagen (H, W, 5).
    Asume el orden de bandas: [Blue, Green, Red, Red_Edge, NIR]
    """
    # Bandas: B=0, G=1, R=2, RE=3, NIR=4
    RED = stacked_image[..., 2]
    NIR = stacked_image[..., 4]
    RED_EDGE = stacked_image[..., 3]
    
    # NDVI (Normalized Difference Vegetation Index)
    numerator_ndvi = NIR - RED
    denominator_ndvi = NIR + RED
    ndvi = np.divide(numerator_ndvi, denominator_ndvi, out=np.zeros_like(numerator_ndvi), where=denominator_ndvi!=0)
    
    # NDRE (Normalized Difference Red Edge)
    numerator_ndre = NIR - RED_EDGE
    denominator_ndre = NIR + RED_EDGE
    ndre = np.divide(numerator_ndre, denominator_ndre, out=np.zeros_like(numerator_ndre), where=denominator_ndre!=0)
    
    # Promedios de Features
    mean_band_values = np.mean(stacked_image, axis=(0, 1)) # Promedio en H y W (5 valores)
    mean_ndvi = np.mean(ndvi)
    mean_ndre = np.mean(ndre)
    
    # Vector final de 7 features: [B_avg, G_avg, R_avg, RE_avg, NIR_avg, NDVI_avg, NDRE_avg]
    feature_vector = np.concatenate([
        mean_band_values,
        np.array([mean_ndvi, mean_ndre])
    ])
    
    # Se devuelve la imagen (opcional) y el vector de características (necesario)
    return feature_vector 
# --- Fin de utilidades ---


CLASSES = ["Plaga", "Sana"] 

def predict_single_multiespectral_folder(
    rf_model, 
    path: str, # path es la RUTA a la carpeta que contiene las 5 bandas TIFF
    img_size: tuple[int, int] = (224, 224),
) -> tuple[str, np.ndarray, np.ndarray] | tuple[str, None, None]:
    """
    Carga los TIFFs de la carpeta, calcula las 7 características y predice con RF.

    Returns:
        etiqueta: La clase predicha.
        probabilities: Array de probabilidades [prob_clase_0, prob_clase_1].
        feature_vector: El vector de 7 features utilizado.
    """
    
    # 1. Cargar la imagen Multiespectral (H, W, 5)
    # Se usa la función del data_loader multiespectral que apila y normaliza
    try:
        # load_single_multispectral_image devuelve (H, W, 5) normalizado (0-1)
        stacked_image = load_single_multispectral_image(path, img_size=img_size)
    except Exception as e:
        print(f"Error al cargar/procesar carpeta TIFF en {path}: {e}")
        return "Error de Carga", None, None

    # 2. Extracción de características (Cálculo de Índices)
    # feature_vector es (7,)
    feature_vector = calculate_indices(stacked_image)
    
    # El modelo RF espera un array 2D: (N_muestras, N_features)
    features_2d = np.expand_dims(feature_vector, axis=0) # Ahora es (1, 7)
    
    # 3. Predicción con Random Forest
    probabilities = rf_model.predict_proba(features_2d)[0] # Probabilidades de cada clase (2,)
    
    # 4. Clasificación (RF usa 0.5 por defecto, predict_proba nos da la confianza)
    predicted_class_id = rf_model.predict(features_2d)[0]
    etiqueta = CLASSES[predicted_class_id]

    return etiqueta, probabilities, feature_vector


def run_inference_on_path(model_path: str, target_path: str, img_size: tuple[int, int]) -> List[Dict]:
    """
    Función principal de inferencia que maneja la carga del modelo y los datos.
    """
    # 1. Carga del Modelo RF
    print(f"Cargando modelo Random Forest desde: {model_path}")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Error cargando el modelo RF: {e}")
        return []

    inference_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Comprobar si la ruta es una carpeta (esperamos la carpeta que contiene las 5 bandas)
    if os.path.isdir(target_path):
        # El nombre del archivo se identifica con el nombre de la carpeta que contiene las bandas
        file_name = os.path.basename(target_path) 
        
        # Realizar la predicción
        etiqueta, probs, features = predict_single_multiespectral_folder(
            rf_model=model,
            path=target_path,
            img_size=img_size
        )

        if probs is not None:
            # probs[0] es Plaga, probs[1] es Sana
            result = {
                "file_name": file_name,
                "prob_plaga": round(probs[0], 4),  
                "prob_sana": round(probs[1], 4),
                "prediccion": etiqueta,
                "features_used": features.tolist(), # Guardar los 7 features
                "modelo": os.path.basename(model_path) 
            }
            inference_results.append(result)
            
            print(f"\n--- Resultados para: {file_name} ---")
            print(f"Características (7): {result['features_used']}")
            print(f"Prob. 'Sana': {result['prob_sana']:.4f}")
            print(f"Predicción final: {etiqueta}")
        else:
            print(f"❌ Error al procesar la carpeta: {target_path}")

    else:
        print(f"Error: La ruta '{target_path}' debe ser la carpeta que contiene los 5 archivos TIFF de bandas.")
        return []

    # 2. Guardar Resultados JSON
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'inference-results-rf-ms')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_filename = f"inference_results_rf_ms_{timestamp}.json"
    final_json_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(final_json_path, 'w') as f:
        json.dump(inference_results, f, indent=4)
        
    print(f"\n✅ Resultados de la inferencia guardados en: {final_json_path}")
    return inference_results


def main():
    parser = argparse.ArgumentParser(description="Realiza inferencia con modelo Random Forest Multiespectral (7 features).")
    parser.add_argument("path", help="Ruta a la CARPETA que contiene los 5 archivos TIFF de bandas.")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo Random Forest (.joblib).")
    args = parser.parse_args()

    run_inference_on_path(
        model_path=args.model,
        target_path=args.path,
        img_size=(224, 224) 
    )

if __name__ == "__main__":
    main()