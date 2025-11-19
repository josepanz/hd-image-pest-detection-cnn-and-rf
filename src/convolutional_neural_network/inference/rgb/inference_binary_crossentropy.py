# src/cnn/inference_bce.py (Usado también como plantilla para inference_fl.py)

import argparse
import os
import tensorflow as tf
from src.evaluation.utils_inference import load_model_for_inference, run_inference_on_path, save_inference_results

def main():
    parser = argparse.ArgumentParser(description="Realiza inferencia con modelo CNN (RGB).")
    parser.add_argument("path", help="Ruta a un archivo de imagen o carpeta con imágenes RGB.")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo Keras (.keras).")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Umbral de decisión.")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Carga del Modelo
    model = load_model_for_inference(args.model)
    model_name = os.path.basename(args.model).replace('.keras', '').replace('.h5', '')
    
    # 2. Ejecutar Inferencia
    results = run_inference_on_path(
        model=model,
        feature_extractor_rf=None,
        path=args.path,
        threshold=args.threshold,
        img_size=(224, 224),
        model_name=model_name,
        is_multiespectral=False,
        is_random_forest=False
    )

    # 3. Guardar Resultados
    if results:
        # Usamos 'bce' o 'fl' como tipo de modelo para la carpeta de resultados
        model_type = 'bce' if 'bce' in model_name.lower() else 'fl' 
        save_inference_results(results, BASE_DIR, args.threshold, model_type)
        

if __name__ == "__main__":
    main()