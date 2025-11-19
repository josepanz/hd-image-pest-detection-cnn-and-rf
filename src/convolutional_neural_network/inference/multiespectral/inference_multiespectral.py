# src/cnn/inference_multiespectral.py

import argparse
import os
import tensorflow as tf
from src.evaluation.utils_inference import load_model_for_inference, run_inference_on_path, save_inference_results

def main():
    parser = argparse.ArgumentParser(description="Realiza inferencia con modelo CNN Multiespectral.")
    # La ruta debe ser a la carpeta que contiene las 5 bandas TIFF
    parser.add_argument("path", help="Ruta a la carpeta que contiene las 5 bandas TIFF para inferencia.")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo Keras Multiespectral (.keras).")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Umbral de decisión.")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Carga del Modelo
    model = load_model_for_inference(args.model)
    model_name = os.path.basename(args.model).replace('.keras', '').replace('.h5', '')
    
    # 2. Ejecutar Inferencia
    # La función run_inference_on_path se encarga de llamar a load_single_multispectral_image
    results = run_inference_on_path(
        model=model,
        feature_extractor_rf=None,
        path=args.path,
        threshold=args.threshold,
        img_size=(224, 224), # Asegúrate de que este tamaño coincida con el entrenamiento
        model_name=model_name,
        is_multiespectral=True, # Bandera crucial
        is_random_forest=False
    )

    # 3. Guardar Resultados
    if results:
        save_inference_results(results, BASE_DIR, args.threshold, 'multiespectral')
        
if __name__ == "__main__":
    main()