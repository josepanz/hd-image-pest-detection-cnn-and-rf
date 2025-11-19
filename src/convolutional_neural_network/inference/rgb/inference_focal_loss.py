# src/cnn/inference_fl.py

import argparse
import os
from src.evaluation.utils_inference import load_model_for_inference, run_inference_on_path, save_inference_results

def main():
    parser = argparse.ArgumentParser(description="Realiza inferencia con modelo CNN (Focal Loss).")
    parser.add_argument("path", help="Ruta a un archivo de imagen o carpeta con imágenes RGB.")
    parser.add_argument("-m", "--model", required=True, help="Ruta al archivo del modelo Keras (.keras).")
    # Umbral más óptimo por defecto para modelos balanceados/Focal Loss
    parser.add_argument("-t", "--threshold", type=float, default=0.65, help="Umbral de decisión (típicamente > 0.5 para Focal Loss).") 
    args = parser.parse_args()

    # Directorio base para guardar los resultados (puede ser el directorio del script)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Carga del Modelo
    # load_model_for_inference maneja automáticamente la 'focal_loss'
    model = load_model_for_inference(args.model)
    model_name = os.path.basename(args.model).replace('.keras', '').replace('.h5', '')
    
    # 2. Ejecutar Inferencia
    # run_inference_on_path maneja la carga de imágenes, preprocesamiento y predicción.
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
        # Usamos 'fl' como tipo de modelo para el nombre de la carpeta de resultados
        save_inference_results(results, BASE_DIR, args.threshold, 'fl')
        

if __name__ == "__main__":
    main()